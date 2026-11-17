"""Acceptance tests for the drift monitor and notification outbox (v4).

Coverage (mapped to the data contract):

Engine (pure, exact Fraction core)
  * exact weighted residual boundaries, trigger streak backtracking
  * empty/insufficient windows are emitted and reset pending streaks
  * hysteresis (in-between residuals), clearing and reopening
  * insufficient windows never close an open episode
  * peak selection with earliest-window tie breaking
  * correcting a past sample moves an episode start
  * a reference correction changes the PAVA staircase and hence the episode

Notifications / durable outbox
  * first appearance upsert, payload-change upsert, disappearance retract
  * never-sent disappearance cancels silently (no retract)
  * A -> B -> A with the middle notice unsent cancels B silently
  * sent disappearance retracts; retract revoked on reappearance with payload compare
  * stale notices are never physically deleted; notice ids are monotonic
  * replay receipts across close/reopen; same-id/different-content whole rejection
  * new snapshot id with identical content still advances the revision

Leases / claims / acks
  * two workers racing one notice, expired lease reclaimed, old token rejected
  * idempotent ack replay ignores current now/lease state; invalid tokens rejected
  * empty queue returns null with exit 0

Concurrency and crash semantics
  * ack-before-sync leaves a required retract; sync-before-ack rejects the stale ack
  * out-of-transaction computation against a competing sync never overwrites new state
  * pre-commit failure leaves no half revision; post-commit failure recovers by replay
  * status is one consistent read snapshot, sorted JSON, symlink rejected
"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import unittest
from fractions import Fraction

from score_calibration import monitor
from score_calibration.monitor import contract as mcontract
from score_calibration.monitor import engine, store

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLI = [sys.executable, '-m', 'score_calibration.cli']
ENV = {**os.environ, 'PYTHONPATH': str(ROOT / 'src')}


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #

def base_snapshot(snapshot_id='s1', observations=None, reference=None,
                  start_date='2026-01-01', period_days=10, period_count=6,
                  min_weight=1, trigger_delta='0.5', clear_delta='0.1',
                  trigger_periods=2, clear_periods=2):
    return {
        'snapshot_id': snapshot_id,
        'reference': reference if reference is not None else [
            {'id': 'r1', 'score': 10, 'label': 0, 'weight': 1},
            {'id': 'r2', 'score': 20, 'label': 1, 'weight': 1},
        ],
        'observations': observations if observations is not None else [],
        'start_date': start_date, 'period_days': period_days,
        'period_count': period_count, 'min_weight': min_weight,
        'trigger_delta': trigger_delta, 'clear_delta': clear_delta,
        'trigger_periods': trigger_periods,
        'clear_periods': clear_periods,
    }


def obs(oid, cohort, day, score=10, label=1, weight=1):
    """Default observation: score 10 predicts p=0, so label 1 gives residual 1."""
    return {'id': oid, 'cohort': cohort, 'date': day, 'score': score,
            'label': label, 'weight': weight}


def day(window_index, period_days=10, start=(2026, 1, 1), offset=0):
    import datetime
    first = datetime.date(*start) + datetime.timedelta(
        days=window_index * period_days + offset)
    return first.isoformat()


def residuals_for(conn_snapshot):
    result = engine.compute(mcontract.validate_snapshot(conn_snapshot))
    return {
        (w['cohort'], w['window']): (w['status'], w['residual'],
                                     w['total_weight'])
        for w in result['windows']
    }


def episode_keys(result):
    return {(e['cohort'], e['start']) for e in result['episodes']}


class MonitorDbCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.tmp.name)
        self.db = self.dir / 'monitor.db'
        self.conn = monitor.connect(self.db)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def sync(self, snapshot, **kwargs):
        return monitor.sync(self.conn, snapshot, **kwargs)

    def reopen(self):
        self.conn.close()
        self.conn = monitor.connect(self.db)
        return self.conn

    def status(self, name='status.json'):
        return monitor.status_report(self.conn, self.dir / name)

    def notices(self):
        return self.conn.execute(
            'SELECT * FROM notices ORDER BY notice_id').fetchall()

    def claim(self, worker='w1', now=100, lease_seconds=60):
        return monitor.claim(self.conn, worker, now, lease_seconds)

    def ack(self, notice_id, token, now=101):
        return monitor.ack(self.conn, notice_id, token, now)


# --------------------------------------------------------------------------- #
# Contract validation
# --------------------------------------------------------------------------- #

class ContractTests(unittest.TestCase):
    def assert_invalid(self, data):
        with self.assertRaises((monitor.MonitorError, ValueError)):
            mcontract.validate_snapshot(data)

    def test_strict_fields_and_duplicate_keys(self):
        good = base_snapshot()
        mcontract.validate_snapshot(good)
        bad = dict(good, extra=1)
        self.assert_invalid(bad)
        missing = dict(good)
        del missing['min_weight']
        self.assert_invalid(missing)
        # duplicate JSON keys are rejected by the shared strict JSON reader
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / 'dup.json'
            path.write_text(
                '{"snapshot_id":"s","reference":[],"observations":[],'
                '"start_date":"2026-01-01","period_days":10,"period_count":2,'
                '"min_weight":1,"trigger_delta":"0.5","clear_delta":"0.1",'
                '"trigger_periods":1,"clear_periods":1,"period_count":3}')
            from score_calibration.common import read
            with self.assertRaises(ValueError):
                read(path)

    def test_reference_contract_nonempty_and_id_disjoint(self):
        self.assert_invalid(base_snapshot(reference=[]))
        overlap = base_snapshot(
            reference=[{'id': 'x', 'score': 10, 'label': 0, 'weight': 1}],
            observations=[obs('x', 'A', day(0))])
        self.assert_invalid(overlap)
        dup_ref = base_snapshot(reference=[
            {'id': 'x', 'score': 10, 'label': 0, 'weight': 1},
            {'id': 'x', 'score': 20, 'label': 1, 'weight': 1}])
        self.assert_invalid(dup_ref)
        dup_obs = base_snapshot(observations=[
            obs('o1', 'A', day(0)), obs('o1', 'A', day(1))])
        self.assert_invalid(dup_obs)

    def test_integer_bounds_and_booleans_rejected(self):
        for field, bad in (
                ('period_days', 0), ('period_days', 32),
                ('period_count', 0), ('period_count', 367),
                ('min_weight', 0), ('min_weight', 10 ** 9 + 1),
                ('trigger_periods', 0), ('clear_periods', 0)):
            data = base_snapshot(**{field: bad})
            self.assert_invalid(data)
        for field in ('period_days', 'period_count', 'min_weight',
                      'trigger_periods', 'clear_periods'):
            data = base_snapshot(**{field: True})
            self.assert_invalid(data)
        # streak lengths are bounded by period_count
        self.assert_invalid(base_snapshot(trigger_periods=7, period_count=6))
        # observation row uses the existing v1 integer bounds
        self.assert_invalid(base_snapshot(observations=[
            obs('o1', 'A', day(0), score=1_000_001)]))
        self.assert_invalid(base_snapshot(observations=[
            obs('o1', 'A', day(0), weight=0)]))
        self.assert_invalid(base_snapshot(observations=[
            obs('o1', 'A', day(0), label=2)]))
        self.assert_invalid(base_snapshot(observations=[
            obs('o1', 'A', day(0), label=True)]))

    def test_identifiers_and_dates(self):
        self.assert_invalid(base_snapshot(snapshot_id=' '))
        self.assert_invalid(base_snapshot(observations=[
            obs('o1', ' bad', day(0))]))
        self.assert_invalid(base_snapshot(observations=[
            obs('o1', 'A', '2026-02-30')]))
        self.assert_invalid(base_snapshot(observations=[
            obs('o1', 'A', '2026-1-1')]))
        # observation outside the continuous left-closed/right-open horizon
        outside = base_snapshot(
            period_days=10, period_count=2,
            observations=[obs('o1', 'A', '2026-01-21')])  # window end excluded
        self.assert_invalid(outside)
        boundary = base_snapshot(
            period_days=10, period_count=2,
            observations=[obs('o1', 'A', '2026-01-20')])  # last valid day
        mcontract.validate_snapshot(boundary)

    def test_delta_decimal_contract_and_ordering(self):
        for bad in ('0.5.1', '1.0', '1e-1', '.5', '00.5', 0.5, 2, '-0.1',
                    'NaN', 'Infinity'):
            self.assert_invalid(base_snapshot(trigger_delta=bad))
        for bad in ('0.1.1', '1.00', 0.1):
            self.assert_invalid(base_snapshot(clear_delta=bad))
        self.assert_invalid(base_snapshot(
            trigger_delta='0.1', clear_delta='0.1'))
        self.assert_invalid(base_snapshot(
            trigger_delta='0.1', clear_delta='0.2'))
        # boundary decimals are legal and exact
        snap = mcontract.validate_snapshot(
            base_snapshot(trigger_delta='1', clear_delta='0'))
        self.assertEqual(snap['trigger_delta'], Fraction(1))
        self.assertEqual(snap['clear_delta'], Fraction(0))

    def test_row_limits(self):
        big = base_snapshot(observations=[
            obs(f'o{i}', 'A', day(i % 6)) for i in range(20001)])
        self.assert_invalid(big)
        big_ref = base_snapshot(reference=[
            {'id': f'r{i}', 'score': i % 1_000_000, 'label': i % 2,
             'weight': 1} for i in range(20001)])
        self.assert_invalid(big_ref)
        # exactly 20000 is legal
        ok = base_snapshot(observations=[
            obs(f'o{i}', 'A', day(i % 6)) for i in range(20000)])
        mcontract.validate_snapshot(ok)

    def test_content_digest_excludes_snapshot_id_and_is_canonical(self):
        a = base_snapshot(snapshot_id='s1', observations=[
            obs('b', 'A', day(1)), obs('a', 'A', day(0))])
        b = base_snapshot(snapshot_id='s2', observations=[
            obs('a', 'A', day(0)), obs('b', 'A', day(1))])
        va, vb = mcontract.validate_snapshot(a), mcontract.validate_snapshot(b)
        self.assertEqual(mcontract.content_digest(va),
                         mcontract.content_digest(vb))
        # changing any frozen configuration changes the digest
        c = mcontract.validate_snapshot(base_snapshot(
            snapshot_id='s1', min_weight=2))
        self.assertNotEqual(mcontract.content_digest(va),
                            mcontract.content_digest(c))
        d = mcontract.validate_snapshot(base_snapshot(
            snapshot_id='s1', trigger_delta='0.6'))
        self.assertNotEqual(mcontract.content_digest(va),
                            mcontract.content_digest(d))

    def test_configuration_freezes_after_first_sync(self):
        case = MonitorDbCase()
        case.setUp()
        try:
            case.sync(base_snapshot())
            with self.assertRaises(monitor.MonitorError):
                case.sync(base_snapshot(
                    snapshot_id='s2', period_days=7))
            # replaceable parts (reference/observations) remain free to change
            receipt = case.sync(base_snapshot(
                snapshot_id='s2', observations=[obs('o1', 'A', day(0))]))
            self.assertEqual(receipt['revision'], 2)
        finally:
            case.tearDown()


# --------------------------------------------------------------------------- #
# Pure engine: exact boundaries, streaks, hysteresis, peak, corrections
# --------------------------------------------------------------------------- #

class EngineTests(unittest.TestCase):
    def test_exact_weighted_residual_and_insufficient_windows(self):
        # window 0: weights 1 (label 1) and 3 (label 0) at score 10 (p=0)
        # numerator = 1*(1-0) + 3*(0-0) = 1; residual = 1/4
        snap = base_snapshot(
            min_weight=2, trigger_periods=1, clear_periods=1,
            observations=[
                obs('o1', 'A', day(0), label=1, weight=1),
                obs('o2', 'A', day(0, offset=1), label=0, weight=3),
                obs('o3', 'A', day(1), label=1, weight=1)])  # weight 1 < min
        windows = residuals_for(snap)
        self.assertEqual(windows[('A', 0)], ('eligible', '1/4', 4))
        self.assertEqual(windows[('A', 1)], ('insufficient', None, 1))
        self.assertEqual(windows[('A', 2)], ('insufficient', None, 0))
        self.assertEqual(len(windows), 6)  # empty windows are still emitted

    def test_trigger_and_clear_boundary_equality(self):
        # residual exactly trigger_delta opens; exactly clear_delta clears
        snap = base_snapshot(
            min_weight=1, trigger_periods=1, clear_periods=1,
            trigger_delta='0.5', clear_delta='0.1',
            observations=[
                # window 0 residual 1/2 (>= trigger)
                obs('a', 'A', day(0), label=1, weight=1),
                obs('b', 'A', day(0, offset=1), label=0, weight=1),
                # window 1 residual 1/10 (<= clear): one score-10 label-1
                # weighted against nine perfect observations
                *[obs(f'c{i}', 'A', day(1, offset=i % 5), label=0, weight=1)
                  for i in range(9)],
            ])
        result = engine.compute(mcontract.validate_snapshot(snap))
        self.assertEqual(len(result['episodes']), 1)
        ep = result['episodes'][0]
        self.assertEqual((ep['start'], ep['end'], ep['open']), (0, 1, False))
        self.assertEqual(ep['peak_value'], '1/2')

    def test_trigger_streak_backtracks_to_first_crossing_window(self):
        # trigger_periods=2; windows 0 and 1 cross -> episode start = 0 (not 1)
        snap = base_snapshot(
            trigger_periods=2, clear_periods=1,
            observations=[
                obs('a', 'A', day(0)),
                obs('b', 'A', day(1)),
                obs('c', 'A', day(2), label=0),  # perfect -> clears immediately
            ])
        result = engine.compute(mcontract.validate_snapshot(snap))
        ep = result['episodes'][0]
        self.assertEqual((ep['start'], ep['end']), (0, 2))

    def test_empty_window_resets_trigger_streak_without_gluing(self):
        # windows 0 crosses, window 1 empty/insufficient, window 2 crosses:
        # with trigger_periods=2 no episode may open (streak never reaches 2).
        snap = base_snapshot(
            min_weight=5, trigger_periods=2, clear_periods=1,
            observations=[
                obs('a', 'A', day(0), weight=5),
                obs('b', 'A', day(2), weight=5)])
        result = engine.compute(mcontract.validate_snapshot(snap))
        self.assertEqual(result['episodes'], [])

    def test_hysteresis_in_between_residual_resets_both_streaks(self):
        # open at windows 0-1 (residual 1). Window 2 residual 0.25 sits strictly
        # between clear 0.1 and trigger 0.5: it must reset the clearing streak.
        mid = [
            obs(f'm{i}', 'A', day(2, offset=i), label=1, weight=1)
            for i in range(3)] + [
            obs('m9', 'A', day(2, offset=4), label=0, weight=9)]
        snap = base_snapshot(
            trigger_periods=2, clear_periods=2,
            observations=[
                obs('a', 'A', day(0)), obs('b', 'A', day(1)), *mid,
                obs('c', 'A', day(3), label=0),
                obs('d', 'A', day(4), label=0)])
        result = engine.compute(mcontract.validate_snapshot(snap))
        windows = {w['window']: w for w in result['windows'] if w['cohort'] == 'A'}
        self.assertEqual(windows[2]['residual'], '1/4')
        ep = result['episodes'][0]
        # clearing only completes at window 4; window 2 did not count toward it
        self.assertEqual((ep['start'], ep['end']), (0, 4))

    def test_insufficient_window_never_closes_open_episode(self):
        snap = base_snapshot(
            min_weight=5, trigger_periods=1, clear_periods=1,
            observations=[
                obs('a', 'A', day(0), weight=5),          # opens
                obs('b', 'A', day(1), weight=1),          # insufficient, no close
                obs('c', 'A', day(2), label=0, weight=5)])  # clears
        result = engine.compute(mcontract.validate_snapshot(snap))
        ep = result['episodes'][0]
        self.assertEqual((ep['start'], ep['end']), (0, 2))

    def test_clear_then_reopen_uses_next_window_and_new_identity(self):
        snap = base_snapshot(
            trigger_periods=1, clear_periods=1,
            observations=[
                obs('a', 'A', day(0)),                       # open start 0
                obs('b', 'A', day(1), label=0),              # clear end 1
                obs('c', 'A', day(2)),                       # reopen start 2
                obs('d', 'A', day(3), label=0)])             # clear end 3
        result = engine.compute(mcontract.validate_snapshot(snap))
        self.assertEqual(
            [(e['start'], e['end']) for e in result['episodes']],
            [(0, 1), (2, 3)])

    def test_open_episode_payload_and_peak_tie_keeps_earliest(self):
        snap = base_snapshot(
            trigger_periods=1, clear_periods=1, period_count=3,
            observations=[
                obs('a', 'A', day(0)),                       # residual 1
                obs('b', 'A', day(1)),                       # residual 1 (tie)
                obs('c', 'A', day(2))])                      # stays open
        result = engine.compute(mcontract.validate_snapshot(snap))
        ep = result['episodes'][0]
        self.assertTrue(ep['open'])
        self.assertIsNone(ep['end'])
        self.assertEqual(ep['peak_window'], 0)  # earliest window on tie
        self.assertEqual(ep['peak_value'], '1')
        self.assertEqual(ep['payload'], {
            'start': 0, 'end': None, 'open': True,
            'peak_window': 0, 'peak_value': '1'})

    def test_past_sample_correction_moves_episode_start(self):
        def build(window0_labels):
            observations = []
            idx = 0
            for labels in window0_labels:
                for label in labels:
                    observations.append(
                        obs(f'o{idx}', 'A', day(0, offset=idx % 5),
                            label=label))
                    idx += 1
            observations += [
                obs('oA', 'A', day(1)), obs('oB', 'A', day(2))]
            return base_snapshot(
                trigger_periods=2, clear_periods=1,
                observations=observations)

        # Before: window 0 residual 1/4 < trigger; only windows 1,2 cross -> start 1
        before = engine.compute(mcontract.validate_snapshot(
            build([[1, 0, 0, 0]])))
        self.assertEqual(before['episodes'][0]['start'], 1)
        # Correct one past window-0 sample 0 -> 1: residual becomes 1/2 >= trigger,
        # so windows 0,1 form the streak and the episode start backtracks to 0.
        after = engine.compute(mcontract.validate_snapshot(
            build([[1, 1, 0, 0]])))
        self.assertEqual(after['episodes'][0]['start'], 0)

    def test_reference_correction_changes_pava_staircase_and_episode(self):
        # Observations at score 20. With reference r2 label 1 the prediction is 1,
        # labels are 0 -> residual 1 (episode opens). Correcting r2 to label 0 makes
        # the PAVA block predict 0 -> residual 0 (no episode at all).
        observations = [obs('o1', 'A', day(0), score=20, label=0),
                        obs('o2', 'A', day(1), score=20, label=0)]
        ref_before = [
            {'id': 'r1', 'score': 10, 'label': 0, 'weight': 1},
            {'id': 'r2', 'score': 20, 'label': 1, 'weight': 1}]
        ref_after = [
            {'id': 'r1', 'score': 10, 'label': 0, 'weight': 1},
            {'id': 'r2', 'score': 20, 'label': 0, 'weight': 1}]
        before = engine.compute(mcontract.validate_snapshot(
            base_snapshot(reference=ref_before,
                          observations=observations,
                          trigger_periods=1, clear_periods=1)))
        after = engine.compute(mcontract.validate_snapshot(
            base_snapshot(reference=ref_after,
                          observations=observations,
                          trigger_periods=1, clear_periods=1)))
        self.assertEqual(len(before['episodes']), 1)
        self.assertEqual(after['episodes'], [])
        self.assertNotEqual(
            [b['probability'] for b in before['blocks']],
            [b['probability'] for b in after['blocks']])

    def test_no_interpolation_left_closed_staircase(self):
        ref = [{'id': 'r1', 'score': 10, 'label': 0, 'weight': 1},
               {'id': 'r2', 'score': 30, 'label': 1, 'weight': 1}]
        snap = mcontract.validate_snapshot(base_snapshot(
            reference=ref, observations=[
                obs('lo', 'A', day(0), score=1, label=1),    # clamp first block
                obs('mid', 'A', day(0, offset=1), score=29, label=1),
                obs('hi', 'A', day(0, offset=2), score=999, label=0)]))
        result = engine.compute(snap)
        # score 29 still belongs to the block starting at 10 -> p=0; no interpolation.
        # numerator = (1-0) + (1-0) + (0-1) = 1 over total weight 3 -> 1/3
        window = next(w for w in result['windows'] if w['window'] == 0)
        self.assertEqual(window['residual'], '1/3')


# --------------------------------------------------------------------------- #
# Notification state machine
# --------------------------------------------------------------------------- #

class NotificationTests(MonitorDbCase):
    def test_empty_observations_first_sync_is_legal_and_clears_episodes(self):
        # An empty observation set is legal on the first sync: revision 1, no
        # episodes, no notices, no window rows (no cohorts exist yet).
        receipt = self.sync(base_snapshot(snapshot_id='s1'))
        self.assertEqual(receipt['revision'], 1)
        self.assertEqual(receipt['episode_count'], 0)
        status = self.status()
        self.assertEqual(status['episodes'], [])
        self.assertEqual(status['windows'], [])
        self.assertEqual(status['notices'], [])

        # then a populated revision opens an episode; an empty replacement removes it
        populated = base_snapshot(
            snapshot_id='s2',
            observations=[obs('a', 'A', day(0)), obs('b', 'A', day(1))])
        self.sync(populated)
        self.assertEqual(len(self.status()['episodes']), 1)
        self.sync(base_snapshot(snapshot_id='s3'))
        self.assertEqual(self.status()['episodes'], [])

    def test_notice_changes_processed_in_canonical_identity_order(self):
        # two cohorts open episodes in one revision; notice ids follow the sorted
        # (cohort, start window) order, not input order.
        snapshot = base_snapshot(
            snapshot_id='s1', trigger_periods=1, clear_periods=1,
            observations=[
                obs('z1', 'zebra', day(0)),
                obs('a1', 'alpha', day(0))])
        self.sync(snapshot)
        notices = self.notices()
        self.assertEqual([n['cohort'] for n in notices], ['alpha', 'zebra'])
        self.assertEqual([n['notice_id'] for n in notices], [1, 2])

    def _episode_snapshot(self, sid, present=True, payload_change=False):
        if not present:
            return base_snapshot(
                snapshot_id=sid, trigger_periods=1, clear_periods=1,
                observations=[])
        if payload_change:
            # shift the crossing one window later so peak/end move
            return base_snapshot(snapshot_id=sid, trigger_periods=1,
                                 clear_periods=1, observations=[
                                     obs('a', 'A', day(1)),
                                     obs('b', 'A', day(2), label=0)])
        return base_snapshot(snapshot_id=sid, trigger_periods=1,
                             clear_periods=1, observations=[
                                 obs('a', 'A', day(0)),
                                 obs('b', 'A', day(1), label=0)])

    def test_first_appearance_upsert_and_revision_semantics(self):
        r1 = self.sync(base_snapshot(snapshot_id='s1', trigger_periods=1,
                                     clear_periods=1,
                                     observations=[obs('a', 'A', day(0))]))
        self.assertEqual((r1['revision'], r1['notices_created']), (1, 1))
        # identical content, new snapshot id -> revision advances, no new notice
        r2 = self.sync(base_snapshot(snapshot_id='s2', trigger_periods=1,
                                     clear_periods=1,
                                     observations=[obs('a', 'A', day(0))]))
        self.assertEqual(r2['revision'], 2)
        self.assertEqual(r2['notices_created'], 0)
        self.assertEqual(len(self.notices()), 1)

    def test_same_id_different_content_rejected_whole(self):
        self.sync(base_snapshot(snapshot_id='s1'))
        before = self.conn.execute(
            'SELECT COUNT(*) FROM snapshots').fetchone()[0]
        with self.assertRaises(monitor.MonitorError):
            self.sync(base_snapshot(snapshot_id='s1', min_weight=2))
        # no half revision, no new snapshot row
        self.assertEqual(self.conn.execute(
            'SELECT COUNT(*) FROM snapshots').fetchone()[0], before)

    def test_payload_change_cancels_unsent_and_upserts_new(self):
        self.sync(self._episode_snapshot('s1'))
        self.sync(self._episode_snapshot('s2', payload_change=True))
        notices = self.notices()
        self.assertEqual([n['status'] for n in notices],
                         ['cancelled', 'pending'])
        self.assertEqual([n['active'] for n in notices], [0, 1])
        self.assertEqual(notices[0]['cancelled_revision'], 2)
        ids = [n['notice_id'] for n in notices]
        self.assertEqual(ids, sorted(ids))  # monotonic, cancelled ids not reused

    def test_disappear_unsent_cancels_silently_without_retract(self):
        self.sync(self._episode_snapshot('s1'))
        self.sync(self._episode_snapshot('s2', present=False))
        notices = self.notices()
        self.assertEqual(len(notices), 1)
        self.assertEqual(notices[0]['status'], 'cancelled')
        self.assertEqual(
            self.conn.execute('SELECT COUNT(*) FROM delivery').fetchone()[0], 0)

    def test_sent_disappearance_generates_retract_and_revocation(self):
        self.sync(self._episode_snapshot('s1'))
        claimed = self.claim()
        self.ack(claimed['notice_id'], claimed['token'])
        self.sync(self._episode_snapshot('s2', present=False))
        notices = self.notices()
        self.assertEqual([n['action'] for n in notices], ['upsert', 'retract'])
        self.assertEqual(notices[0]['status'], 'sent')
        self.assertEqual(notices[1]['status'], 'pending')
        # history is retained physically
        self.assertEqual(len(notices), 2)
        # retract carries the last sent payload
        self.assertEqual(
            json.loads(notices[1]['payload_json']),
            claimed['payload'])

    def test_retract_revoked_on_reappearance_compared_to_last_sent(self):
        self.sync(self._episode_snapshot('s1'))
        claimed = self.claim()
        sent_payload = claimed['payload']
        self.ack(claimed['notice_id'], claimed['token'])
        self.sync(self._episode_snapshot('s2', present=False))       # retract pending
        # reappearance with the SAME payload as last sent: retract cancelled, no upsert
        self.sync(self._episode_snapshot('s3'))
        notices = self.notices()
        self.assertEqual([n['action'] for n in notices],
                         ['upsert', 'retract'])
        self.assertEqual([n['status'] for n in notices],
                         ['sent', 'cancelled'])
        # reappearance with a changed payload now requires a new upsert
        self.sync(self._episode_snapshot('s4', payload_change=True))
        actions = [(n['action'], n['status']) for n in self.notices()]
        self.assertEqual(actions[-1], ('upsert', 'pending'))

    def test_a_to_b_to_a_unsent_middle_is_cancelled_silently(self):
        self.sync(self._episode_snapshot('s1'))           # A pending
        self.sync(self._episode_snapshot('s2', payload_change=True))  # B pending
        self.sync(self._episode_snapshot('s3'))           # back to A, never sent
        notices = self.notices()
        # B cancels A; returning to A cancels B and re-creates the A upsert because
        # nothing was ever delivered. No retract is ever produced (no delivery row).
        self.assertEqual([n['action'] for n in notices],
                         ['upsert', 'upsert', 'upsert'])
        self.assertEqual([n['status'] for n in notices],
                         ['cancelled', 'cancelled', 'pending'])
        self.assertEqual(
            self.conn.execute('SELECT COUNT(*) FROM delivery').fetchone()[0], 0)
        # only the latest required notice is active
        self.assertEqual(self.conn.execute(
            'SELECT COUNT(*) FROM notices WHERE active = 1').fetchone()[0], 1)

    def test_a_sent_b_pending_back_to_a_cancels_b_without_new_send(self):
        self.sync(self._episode_snapshot('s1'))
        first = self.claim()
        self.ack(first['notice_id'], first['token'])       # A sent
        self.sync(self._episode_snapshot('s2', payload_change=True))  # B pending
        self.sync(self._episode_snapshot('s3'))            # back to A
        notices = self.notices()
        statuses = [(n['action'], n['status'], n['active']) for n in notices]
        # B's upsert was never sent, so disappearing it cancels silently (no retract);
        # returning to the exact last-sent payload A revokes nothing new and leaves no
        # pending/leased notice for the identity.
        self.assertEqual(statuses, [
            ('upsert', 'sent', 0),
            ('upsert', 'cancelled', 0)])
        self.assertEqual(self.claim(), None)
        self.assertEqual(
            self.conn.execute(
                'SELECT COUNT(*) FROM notices WHERE status = "pending" '
                'OR status = "leased"').fetchone()[0], 0)

    def test_sent_retract_then_reappearance_compared_to_last_sent(self):
        self.sync(self._episode_snapshot('s1'))
        claimed = self.claim()
        sent_payload = claimed['payload']
        self.ack(claimed['notice_id'], claimed['token'])
        self.sync(self._episode_snapshot('s2', present=False))   # retract pending
        retract = self.claim()
        self.assertEqual(retract['action'], 'retract')
        self.ack(retract['notice_id'], retract['token'])         # retract sent
        # episode reappears with the exact last-sent payload: retract retired, no upsert
        self.sync(self._episode_snapshot('s3'))
        status = self.status()
        self.assertIsNone(self.claim())                          # nothing pending
        self.assertEqual(
            status['delivery'][0]['last_sent_payload'], sent_payload)
        # reappearance with a different payload now requires a fresh upsert
        self.sync(self._episode_snapshot('s4', payload_change=True))
        self.assertEqual(self.claim()['action'], 'upsert')

    def test_only_latest_required_notice_can_be_pending_or_leased(self):
        self.sync(self._episode_snapshot('s1'))
        self.sync(self._episode_snapshot('s2', payload_change=True))
        active = self.conn.execute(
            'SELECT COUNT(*) FROM notices WHERE active = 1').fetchone()[0]
        self.assertEqual(active, 1)

    def test_reopen_replays_receipt_and_keeps_history(self):
        self.sync(self._episode_snapshot('s1'))
        claimed = self.claim()
        self.ack(claimed['notice_id'], claimed['token'])
        self.reopen()
        replay = monitor.ack(
            self.conn, claimed['notice_id'], claimed['token'], 999_999)
        self.assertEqual(replay['status'], 'sent')
        self.assertEqual(replay['payload'], claimed['payload'])
        status = self.status()
        self.assertEqual(status['revision'], 1)
        self.assertEqual(len(status['ack_receipts']), 1)
        self.assertEqual(len(status['notices']), 1)
        self.assertEqual(status['delivery'][0]['last_sent_payload'],
                         claimed['payload'])


# --------------------------------------------------------------------------- #
# Claims, leases and acks
# --------------------------------------------------------------------------- #

class LeaseTests(MonitorDbCase):
    def _seed(self):
        self.sync(base_snapshot(
            snapshot_id='s1', trigger_periods=1, clear_periods=1,
            observations=[obs('a', 'A', day(0))]))

    def test_empty_queue_returns_null(self):
        self.assertIsNone(self.claim())

    def test_claim_returns_token_and_lease_window(self):
        self._seed()
        claimed = self.claim(worker='w1', now=100, lease_seconds=50)
        self.assertEqual(claimed['lease_until'], 150)
        self.assertEqual(claimed['worker'], 'w1')
        self.assertIsInstance(claimed['token'], str)
        self.assertTrue(claimed['token'])
        # already leased and not expired -> not claimable again
        self.assertIsNone(self.claim(now=149))

    def test_two_workers_race_single_notice(self):
        self._seed()
        barrier = threading.Barrier(2)
        results = []

        def worker(name):
            barrier.wait()
            conn = monitor.connect(self.db)
            try:
                results.append((name, monitor.claim(conn, name, 100, 60)))
            finally:
                conn.close()

        threads = [threading.Thread(target=worker, args=(f'w{i}',))
                   for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        winners = [r for r in results if r[1] is not None]
        self.assertEqual(len(winners), 1)
        self.assertIsNone(next(r[1] for r in results if r != winners[0]))
        self.assertEqual(len({r[1]['token'] for r in winners}), 1)

    def test_expired_lease_reclaimed_old_token_rejected(self):
        self._seed()
        first = self.claim(now=100, lease_seconds=50)        # until 150
        # expiry is now >= lease_until
        second = self.claim(worker='w2', now=150, lease_seconds=50)
        self.assertIsNotNone(second)
        self.assertNotEqual(second['token'], first['token'])
        with self.assertRaises(monitor.MonitorError):
            self.ack(first['notice_id'], first['token'], now=151)
        # the new token works
        receipt = self.ack(second['notice_id'], second['token'], now=151)
        self.assertEqual(receipt['status'], 'sent')
        lease_states = [r['state'] for r in self.conn.execute(
            'SELECT state FROM leases ORDER BY lease_id')]
        self.assertEqual(lease_states, ['expired', 'acked'])

    def test_ack_validates_token_notice_and_lease(self):
        self._seed()
        claimed = self.claim(now=100, lease_seconds=10)
        with self.assertRaises(monitor.MonitorError):
            self.ack(claimed['notice_id'], 'wrong-token', now=101)
        with self.assertRaises(monitor.MonitorError):
            self.ack(claimed['notice_id'] + 999, claimed['token'], now=101)
        with self.assertRaises(monitor.MonitorError):
            self.ack(claimed['notice_id'], claimed['token'], now=110)
        # invalid now / lease arguments
        with self.assertRaises(monitor.MonitorError):
            monitor.claim(self.conn, 'w', -1, 10)
        with self.assertRaises(monitor.MonitorError):
            monitor.claim(self.conn, 'w', 0, 0)
        with self.assertRaises(monitor.MonitorError):
            monitor.claim(self.conn, 'w', 0, 86401)
        with self.assertRaises(monitor.MonitorError):
            monitor.ack(self.conn, claimed['notice_id'], claimed['token'], -1)

    def test_ack_idempotent_replay_ignores_current_state(self):
        self._seed()
        claimed = self.claim(now=100, lease_seconds=10)
        receipt = self.ack(claimed['notice_id'], claimed['token'], now=101)
        # a later sync supersedes the episode and revokes the lease
        self.sync(base_snapshot(
            snapshot_id='s2', trigger_periods=1, clear_periods=1,
            observations=[obs('a', 'A', day(1))]))
        replay = self.ack(claimed['notice_id'], claimed['token'],
                          now=999_999)
        self.assertEqual(replay, receipt)

    def test_sent_notice_is_never_claimed_again(self):
        self._seed()
        claimed = self.claim()
        self.ack(claimed['notice_id'], claimed['token'])
        self.assertIsNone(self.claim())

    def test_tokens_differ_every_claim(self):
        self._seed()
        first = self.claim(now=0, lease_seconds=5)
        second = self.claim(now=5, lease_seconds=5)  # first expired
        self.assertNotEqual(first['token'], second['token'])


# --------------------------------------------------------------------------- #
# Concurrency: ack/sync ordering, stale computation, crashes
# --------------------------------------------------------------------------- #

class ConcurrencyTests(MonitorDbCase):
    def _episode_present(self, sid='s1', observations=None):
        if observations is None:
            observations = [obs('a', 'A', day(0))]
        return base_snapshot(
            snapshot_id=sid, trigger_periods=1, clear_periods=1,
            observations=observations)

    def _episode_gone(self, sid):
        return base_snapshot(
            snapshot_id=sid, trigger_periods=1, clear_periods=1,
            observations=[])

    def test_ack_commits_first_then_sync_must_leave_retract(self):
        self.sync(self._episode_present())
        claimed = self.claim()
        sync_started = threading.Event()
        ack_finished = threading.Event()

        def barrier(stage, snapshot):
            if stage == 'computed':
                sync_started.set()
                ack_finished.wait(5)
                self.assertTrue(ack_finished.is_set())

        def do_sync():
            # SQLite connections are thread-bound: the competing sync gets its own
            # connection to the same database file.
            conn = monitor.connect(self.db)
            try:
                monitor.sync(conn, self._episode_gone('s2'), barrier=barrier)
            finally:
                conn.close()

        t = threading.Thread(target=do_sync)
        t.start()
        self.assertTrue(sync_started.wait(5))
        # competing connection performs the ack while sync waits at the barrier
        other = monitor.connect(self.db)
        try:
            monitor.ack(other, claimed['notice_id'], claimed['token'], 101)
        finally:
            other.close()
        ack_finished.set()
        t.join(5)
        self.assertFalse(t.is_alive())
        notices = self.notices()
        self.assertEqual([n['action'] for n in notices],
                         ['upsert', 'retract'])
        self.assertEqual(notices[1]['status'], 'pending')

    def test_sync_commits_first_then_stale_ack_is_rejected(self):
        self.sync(self._episode_present())
        claimed = self.claim()
        # Serial ordering (one of the two legal orders): the sync commits first and
        # cancels the pending upsert with its lease; the stale ack must then fail.
        self.sync(self._episode_gone('s2'))
        with self.assertRaises(monitor.MonitorError):
            self.ack(claimed['notice_id'], claimed['token'], 102)
        notices = self.notices()
        self.assertEqual([n['status'] for n in notices], ['cancelled'])
        lease = self.conn.execute(
            'SELECT state FROM leases WHERE token = ?',
            (claimed['token'],)).fetchone()
        self.assertEqual(lease['state'], 'revoked')

    def test_stale_out_of_transaction_computation_retries_on_newer_state(self):
        # The parked sync computes against revision 1 (episode present).  While it
        # waits outside the transaction, a competing connection acks the pending
        # notice and then another sync removes the episode.  The ack changes leases/
        # notices/delivery WITHOUT advancing the snapshot revision, which is exactly
        # why the commit must fingerprint the whole notification state, not the
        # revision alone.  On mismatch the parked reconciliation retries and the
        # final state must contain the required retract (never an overwritten state).
        self.sync(self._episode_present('s1'))
        claimed = self.claim()
        parked = threading.Event()
        release = threading.Event()

        def barrier(stage, snapshot):
            if snapshot['snapshot_id'] == 's2':
                parked.set()
                release.wait(5)

        def parked_sync():
            conn = monitor.connect(self.db)
            try:
                monitor.sync(conn, self._episode_gone('s2'), barrier=barrier)
            finally:
                conn.close()

        t = threading.Thread(target=parked_sync)
        t.start()
        self.assertTrue(parked.wait(5))
        other = monitor.connect(self.db)
        try:
            # Competing delivery while s2's computation is already finished.
            monitor.ack(other, claimed['notice_id'], claimed['token'], 101)
        finally:
            other.close()
        release.set()
        t.join(5)
        self.assertFalse(t.is_alive())
        revisions = [r['revision'] for r in self.conn.execute(
            'SELECT revision FROM snapshots ORDER BY revision')]
        self.assertEqual(revisions, [1, 2])
        status = self.status()
        self.assertEqual(status['revision'], 2)
        self.assertEqual(status['episodes'], [])
        # Because the upsert was acked before the disappearance, the retract must
        # survive: a revision-only check would have missed this dependency.
        actions = [(n['action'], n['status']) for n in status['notices']]
        self.assertIn(('retract', 'pending'), actions)
        self.assertEqual(len(status['delivery']), 1)

    def test_competing_sync_moves_revision_and_parked_sync_retries_cleanly(self):
        # Two syncs interleave: the parked one must not overwrite the committed
        # revision 2; after its fingerprint retry both revisions coexist and the
        # latest state is the committed one (no half episode update).
        self.sync(self._episode_present('s1'))
        parked = threading.Event()
        release = threading.Event()

        def barrier(stage, snapshot):
            if snapshot['snapshot_id'] == 's3':
                parked.set()
                release.wait(5)

        def parked_sync():
            conn = monitor.connect(self.db)
            try:
                # parked sync keeps the same episode present but a new id
                monitor.sync(
                    conn,
                    self._episode_present('s3', observations=[obs('a', 'A', day(0))]),
                    barrier=barrier)
            finally:
                conn.close()

        t = threading.Thread(target=parked_sync)
        t.start()
        self.assertTrue(parked.wait(5))
        other = monitor.connect(self.db)
        try:
            monitor.sync(other, self._episode_present(
                's2', observations=[obs('a', 'A', day(0))]))
        finally:
            other.close()
        release.set()
        t.join(5)
        self.assertFalse(t.is_alive())
        revisions = [r['revision'] for r in self.conn.execute(
            'SELECT revision FROM snapshots ORDER BY revision')]
        self.assertEqual(revisions, [1, 2, 3])
        status = self.status()
        self.assertEqual(status['revision'], 3)
        self.assertEqual([e['start'] for e in status['episodes']], [0])

    def test_pre_commit_failure_leaves_no_half_revision(self):
        def blow_up(conn, revision):
            raise RuntimeError('simulated crash before commit')

        with self.assertRaises(RuntimeError):
            self.sync(self._episode_present('s1'),
                      on_before_commit=blow_up)
        self.assertEqual(self.conn.execute(
            'SELECT COUNT(*) FROM snapshots').fetchone()[0], 0)
        self.assertEqual(self.conn.execute(
            'SELECT COUNT(*) FROM episodes').fetchone()[0], 0)
        self.assertEqual(self.conn.execute(
            'SELECT COUNT(*) FROM notices').fetchone()[0], 0)
        # the snapshot id was not consumed: a clean retry succeeds
        receipt = self.sync(self._episode_present('s1'))
        self.assertEqual(receipt['revision'], 1)

    def test_post_commit_stdout_failure_recovers_by_same_id_replay(self):
        receipt = self.sync(self._episode_present('s1'))
        # simulate the CLI dying after commit: rerun the identical snapshot
        replayed = self.sync(self._episode_present('s1'))
        self.assertTrue(replayed['replay'])
        self.assertEqual(replayed['revision'], receipt['revision'])
        self.assertEqual(
            self.conn.execute('SELECT COUNT(*) FROM snapshots').fetchone()[0], 1)
        self.assertEqual(
            self.conn.execute('SELECT COUNT(*) FROM notices').fetchone()[0], 1)


# --------------------------------------------------------------------------- #
# Status publication and CLI
# --------------------------------------------------------------------------- #

class StatusAndCliTests(MonitorDbCase):
    def test_status_consistent_sorted_and_symlink_rejected(self):
        self.sync(base_snapshot(
            snapshot_id='s1', trigger_periods=1, clear_periods=1,
            observations=[obs('a', 'A', day(0)), obs('b', 'B', day(0))]))
        out = self.dir / 'status.json'
        report = self.status()
        raw = out.read_text()
        self.assertEqual(raw, json.dumps(report, sort_keys=True,
                                         ensure_ascii=False, indent=2) + '\n')
        # no runtime-generated timestamps anywhere in the report
        self.assertNotIn('created_at', raw)
        self.assertNotIn('timestamp', raw)
        # symlink output is refused and the link target is untouched
        link = self.dir / 'link.json'
        target = self.dir / 'target.json'
        target.write_text('keep')
        link.symlink_to(target)
        with self.assertRaises(monitor.MonitorError):
            self.status('link.json')
        self.assertEqual(target.read_text(), 'keep')

    def test_status_empty_database(self):
        report = self.status('empty.json')
        self.assertEqual(report['revision'], 0)
        self.assertIsNone(report['snapshot_id'])
        self.assertEqual(report['windows'], [])
        self.assertTrue((self.dir / 'empty.json').exists())

    def _call(self, *args):
        return subprocess.run(
            [*CLI, *map(str, args)], capture_output=True, text=True,
            cwd=ROOT, env=ENV)

    def test_full_cli_lifecycle(self):
        snapshot = self.dir / 'snapshot.json'
        snapshot.write_text(json.dumps(base_snapshot(
            snapshot_id='s1', trigger_periods=1, clear_periods=1,
            observations=[obs('a', 'A', day(0))])))
        sync_run = self._call('monitor-sync', '--db', self.db,
                              '--snapshot', snapshot)
        self.assertEqual(sync_run.returncode, 0, sync_run.stderr)
        self.assertEqual(json.loads(sync_run.stdout)['revision'], 1)

        claim_run = self._call('monitor-claim', '--db', self.db,
                               '--worker', 'w1', '--now', 100,
                               '--lease-seconds', 60)
        self.assertEqual(claim_run.returncode, 0, claim_run.stderr)
        claimed = json.loads(claim_run.stdout)
        self.assertEqual(claimed['lease_until'], 160)

        ack_run = self._call('monitor-ack', '--db', self.db,
                             '--notice', claimed['notice_id'],
                             '--token', claimed['token'], '--now', 101)
        self.assertEqual(ack_run.returncode, 0, ack_run.stderr)
        self.assertEqual(json.loads(ack_run.stdout)['status'], 'sent')

        # invalid token exits 2
        bad = self._call('monitor-ack', '--db', self.db,
                         '--notice', claimed['notice_id'],
                         '--token', 'nope', '--now', 102)
        self.assertEqual(bad.returncode, 2)

        out = self.dir / 'cli-status.json'
        status_run = self._call('monitor-status', '--db', self.db,
                                '--output', out)
        self.assertEqual(status_run.returncode, 0, status_run.stderr)
        report = json.loads(out.read_text())
        self.assertEqual(report['revision'], 1)
        self.assertEqual(len(report['leases']), 1)
        self.assertEqual(len(report['ack_receipts']), 1)

    def test_cli_empty_queue_exit_zero_and_bad_input_exit_two(self):
        empty = self._call('monitor-claim', '--db', self.db,
                           '--worker', 'w1', '--now', 0,
                           '--lease-seconds', 10)
        self.assertEqual(empty.returncode, 0)
        self.assertEqual(empty.stdout.strip(), 'null')

        bad_snapshot = self.dir / 'bad.json'
        bad_snapshot.write_text('{"snapshot_id": "s1"}')
        run = self._call('monitor-sync', '--db', self.db,
                         '--snapshot', bad_snapshot)
        self.assertEqual(run.returncode, 2)
        # status failure preserves an existing output file
        out = self.dir / 'keep.json'
        out.write_text('prior')
        link = self.dir / 'link.json'
        link.symlink_to(out)
        run = self._call('monitor-status', '--db', self.db,
                         '--output', link)
        self.assertEqual(run.returncode, 2)
        self.assertEqual(out.read_text(), 'prior')

    def test_monitor_database_is_independent_of_ledger_tables(self):
        # The monitor creates its own tables and never references ledger objects.
        tables = {r[0] for r in self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertTrue({'snapshots', 'notices', 'leases', 'delivery',
                         'ack_receipts', 'episodes', 'windows'} <= tables)
        self.assertFalse(tables & {'events', 'batches', 'projections',
                                   'effective_observations'})


if __name__ == '__main__':
    unittest.main()
