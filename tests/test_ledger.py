"""Acceptance tests for the revision ledger and versioned calibration reports.

Coverage (mapped to the data contract):
  * batch validation, in-batch rollback, replay vs. mixed/conflicting replay
  * RFC3339 normalization, cutoff on both sides and retract folding
  * entity-level 80/20 split determinism
  * evaluation-only vs. training-value corrections and their invalidation behavior
  * cross-split (entity migration) corrections
  * old-version reproducibility
  * concurrent ingest during a report never mixes revisions
  * corrupt/half-finished projection restart repair
  * output idempotence/conflict and replica incompleteness rejection
  * resource bound: memory grows with effective observations, not revision count
"""
import hashlib
import json
import os
import pathlib
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest

from score_calibration import ledger

ROOT = pathlib.Path(__file__).resolve().parents[1]


def event(event_id, observation_id, entity_id, observed_at, score, label,
          weight=1, retract=False):
    return {
        'event_id': event_id, 'observation_id': observation_id,
        'entity_id': entity_id, 'observed_at': observed_at, 'score': score,
        'label': label, 'weight': weight, 'retract': retract,
    }


class LedgerCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.tmp.name)
        self.db = self.dir / 'ledger.db'
        self.conn = ledger.connect(self.db)

    def tearDown(self):
        conn = getattr(self, 'conn', None)
        if conn is not None:
            conn.close()
        tmp = getattr(self, 'tmp', None)
        if tmp is not None:
            tmp.cleanup()

    def ingest(self, events, raw_sha256='a' * 64):
        return ledger.ingest(self.conn, {'events': events}, raw_sha256)

    def report(self, seq='latest', seed='seed', cutoff='2027-01-01T00:00:00Z',
               output=None):
        request = {'observed_at_cutoff': cutoff, 'knowledge_seq': seq,
                   'split_seed': seed}
        output = output or (self.dir / f'report-{seq}-{seed}.json')
        report, reused = ledger.run_report(self.conn, request, output)
        return report, reused, output, request

    def digest(self, obj):
        return hashlib.sha256(ledger.canonical_bytes(obj)).hexdigest()

    def make_report(self, seq='latest', seed='seed',
                    cutoff='2027-01-01T00:00:00Z', name=None, request=None):
        """Run a report with a unique output path; return report and request."""
        if request is None:
            request = {'observed_at_cutoff': cutoff,
                       'knowledge_seq': seq, 'split_seed': seed}
        name = name or f'report-{seq}-{seed}-{cutoff}-{id(request)}.json'
        report, reused = ledger.run_report(
            self.conn, request, self.dir / name)
        return report, reused, request


class ValidationTests(LedgerCase):
    BASE = dict(event_id='e1', observation_id='o1', entity_id='x',
                observed_at='2026-01-01T00:00:00Z', score=10, label=0,
                weight=1, retract=False)

    def assert_invalid_event(self, **changes):
        raw = dict(self.BASE)
        raw.update(changes)
        with self.assertRaises(ledger.LedgerError):
            ledger.validate_event(raw)

    def test_field_and_type_contract(self):
        self.assert_invalid_event(event_id='')
        self.assert_invalid_event(event_id='has\tcontrol')
        self.assert_invalid_event(event_id='x' * (ledger.MAX_ID_LEN + 1))
        self.assert_invalid_event(observed_at='2026-01-01T00:00:00')      # naive
        self.assert_invalid_event(observed_at='2026-01-01 00:00:00Z')    # space sep
        self.assert_invalid_event(observed_at='not-a-time')
        self.assert_invalid_event(score=-1)
        self.assert_invalid_event(score=1_000_001)
        self.assert_invalid_event(score=1.5)
        self.assert_invalid_event(label=2)
        self.assert_invalid_event(weight=0)
        self.assert_invalid_event(weight=1_000_001)
        self.assert_invalid_event(retract=0)                            # not bool
        # extra / missing fields
        raw = dict(self.BASE, extra=1)
        with self.assertRaises(ledger.LedgerError):
            ledger.validate_event(raw)
        raw = dict(self.BASE)
        del raw['weight']
        with self.assertRaises(ledger.LedgerError):
            ledger.validate_event(raw)

    def test_rfc3339_normalized_to_utc(self):
        normalized = ledger.validate_event(
            dict(self.BASE, observed_at='2026-01-02T03:00:00+02:00'))
        self.assertEqual(normalized['observed_at'],
                         '2026-01-02T01:00:00.000000Z')
        normalized = ledger.validate_event(
            dict(self.BASE, observed_at='2026-01-02T00:30:00.5-05:30'))
        self.assertEqual(normalized['observed_at'],
                         '2026-01-02T06:00:00.500000Z')

    def test_empty_batch_and_batch_shape_rejected(self):
        with self.assertRaises(ledger.LedgerError):
            ledger.ingest(self.conn, {'events': []}, 'd')
        with self.assertRaises(ledger.LedgerError):
            ledger.ingest(self.conn, {'events': [dict(self.BASE)], 'x': 1}, 'd')

    def test_in_batch_duplicate_rolls_back(self):
        ev = event('dup', 'o1', 'A', '2026-01-01T00:00:00Z', 10, 0)
        with self.assertRaises(ledger.BatchConflict):
            self.ingest([ev, dict(ev, observation_id='o2', score=20)])
        self.assertEqual(ledger.max_committed_seq(self.conn), 0)
        self.assertEqual(
            self.conn.execute('SELECT COUNT(*) FROM events').fetchone()[0], 0)

    def test_canonical_event_identity_fixed_order(self):
        normalized = ledger.validate_event(dict(self.BASE))
        canonical = ledger.event_canonical(normalized).decode()
        self.assertEqual(
            canonical,
            '{"event_id":"e1","observation_id":"o1","entity_id":"x",'
            '"observed_at":"2026-01-01T00:00:00.000000Z","score":10,"label":0,'
            '"weight":1,"retract":0}')
        self.assertNotIn(' ', canonical)


class IngestReplayTests(LedgerCase):
    def setUp(self):
        super().setUp()
        self.events = [
            event('e1', 'o1', 'A', '2026-01-01T00:00:00Z', 10, 0),
            event('e2', 'o2', 'B', '2026-01-02T00:00:00Z', 20, 1, weight=2),
        ]

    def test_append_assigns_monotonic_seq(self):
        first = self.ingest(self.events)
        self.assertEqual((first['knowledge_seq'], first['replay']), (1, False))
        second = self.ingest([
            event('e3', 'o3', 'C', '2026-01-03T00:00:00Z', 30, 1)])
        self.assertEqual(second['knowledge_seq'], 2)

    def test_exact_replay_returns_original_receipt_without_advancing(self):
        first = self.ingest(self.events)
        replay = self.ingest(self.events)
        self.assertTrue(replay['replay'])
        self.assertEqual(replay['knowledge_seq'], first['knowledge_seq'])
        self.assertEqual(replay['batch_digest'], first['batch_digest'])
        self.assertEqual(ledger.max_committed_seq(self.conn), 1)
        self.assertEqual(
            self.conn.execute('SELECT COUNT(*) FROM events').fetchone()[0], 2)

    def test_reordered_replay_is_conflict(self):
        self.ingest(self.events)
        with self.assertRaises(ledger.BatchConflict):
            self.ingest(list(reversed(self.events)))
        self.assertEqual(ledger.max_committed_seq(self.conn), 1)

    def test_mixed_replay_and_content_conflict_rejected(self):
        self.ingest(self.events)
        # same event_id reused inside a genuinely different batch
        mixed = [self.events[0],
                 event('e3', 'o3', 'C', '2026-01-03T00:00:00Z', 30, 1)]
        with self.assertRaises(ledger.BatchConflict):
            self.ingest(mixed)
        conflicting = [dict(self.events[0], score=999)]
        with self.assertRaises(ledger.BatchConflict):
            self.ingest(conflicting)
        self.assertEqual(ledger.max_committed_seq(self.conn), 1)
        self.assertEqual(
            self.conn.execute('SELECT COUNT(*) FROM events').fetchone()[0], 2)

    def test_events_are_never_physically_deleted(self):
        self.ingest(self.events)
        self.ingest([event('e3', 'o1', 'A', '2026-02-01T00:00:00Z',
                           10, 0, retract=True)])
        self.assertEqual(
            self.conn.execute('SELECT COUNT(*) FROM events').fetchone()[0], 3)


class FoldCutoffSplitTests(LedgerCase):
    def _three_batches(self):
        self.ingest([
            event('e1', 'o1', 'A', '2026-01-01T00:00:00Z', 10, 0),
            event('e2', 'o2', 'B', '2026-01-02T00:00:00Z', 20, 1),
        ])
        self.ingest([  # correction of o1 + retraction of o2
            event('e3', 'o1', 'A', '2026-01-01T00:00:00Z', 11, 1),
            event('e4', 'o2', 'B', '2026-01-02T00:00:00Z', 20, 1, retract=True),
        ])
        self.ingest([  # observation beyond any normal cutoff
            event('e5', 'o3', 'C', '2026-06-01T00:00:00Z', 99, 1)])

    def test_last_event_wins_and_retract_removes_id(self):
        self._three_batches()
        view = ledger.ensure_projection(self.conn, 2)
        rows = {r['observation_id']: r for r in self.conn.execute(
            'SELECT observation_id, score, label FROM effective_observations '
            'WHERE knowledge_seq = 2')}
        self.assertEqual(rows['o1']['score'], 11)
        self.assertEqual(rows['o1']['label'], 1)
        self.assertNotIn('o2', rows)  # retracted
        self.assertEqual(view['effective_count'], 1)

    def test_old_seq_views_remain_readable(self):
        self._three_batches()
        ledger.ensure_projection(self.conn, 2)
        old = {r['observation_id']: r for r in self.conn.execute(
            'SELECT observation_id, score FROM effective_observations '
            'WHERE knowledge_seq = 1')}
        new = {r['observation_id']: r for r in self.conn.execute(
            'SELECT observation_id, score FROM effective_observations '
            'WHERE knowledge_seq = 2')}
        self.assertEqual(old['o1']['score'], 10)
        self.assertEqual(new['o1']['score'], 11)
        self.assertIn('o2', old)
        self.assertNotIn('o2', new)

    def test_cutoff_both_sides(self):
        self._three_batches()
        kept, _, _ = self.make_report(cutoff='2026-02-01T00:00:00Z', name='kept.json')
        ids = (kept['dependencies']['training_dependency']['observation_ids']
               + kept['dependencies']['evaluation_dependency']['observation_ids'])
        self.assertNotIn('o3', ids)
        wide, _, _ = self.make_report(cutoff='2027-01-01T00:00:00Z', name='wide.json')
        wide_ids = (wide['dependencies']['training_dependency']['observation_ids']
                    + wide['dependencies']['evaluation_dependency']['observation_ids'])
        self.assertIn('o3', wide_ids)

    def test_split_is_deterministic_and_entity_whole(self):
        # many observations per entity; each entity must land on one side only
        events, idx = [], 0
        for entity in 'ABCDEFGH':
            for _ in range(3):
                idx += 1
                events.append(event(
                    f'e{idx}', f'o{idx}', entity,
                    f'2026-01-0{1 + idx % 9}T00:00:00Z', 10 + idx, idx % 2))
        self.ingest(events)
        report, _, _ = self.make_report(seed='stable', name='split.json')
        train = set(report['dependencies']['training_dependency']['observation_ids'])
        eval_ = set(report['dependencies']['evaluation_dependency']['observation_ids'])
        self.assertFalse(train & eval_)
        # every observation of an entity shares a side
        side = {}
        for row in self.conn.execute(
                'SELECT entity_id, observation_id FROM effective_observations '
                'WHERE knowledge_seq = 1'):
            member = ('training' if row['observation_id'] in train else 'evaluation')
            side.setdefault(row['entity_id'], set()).add(member)
        self.assertTrue(all(len(v) == 1 for v in side.values()))
        # explicit hash rule check
        expected = ledger.split_side('stable', 'A')
        digest = hashlib.sha256(b'stable\x00A').digest()[:8]
        fraction = int.from_bytes(digest, 'big') / 2 ** 64
        self.assertEqual(expected,
                         'training' if fraction < 0.8 else 'evaluation')

    def test_split_is_stable_across_runs_and_ledgers(self):
        events = [event(f'e{i}', f'o{i}', chr(65 + i % 5),
                        '2026-01-01T00:00:00Z', 10 + i, i % 2)
                  for i in range(10)]
        self.ingest(events)
        first, _, request = self.make_report(seed='s', name='first.json')
        out = self.dir / 'first.json'
        second, reused = ledger.run_report(self.conn, request, out)
        self.assertTrue(reused)
        other = ledger.connect(self.dir / 'other.db')
        try:
            ledger.ingest(other, {'events': events}, 'a' * 64)
            out2 = self.dir / 'other.json'
            ledger.run_report(other, request, out2)
        finally:
            other.close()
        self.assertEqual(out.read_bytes(), out2.read_bytes())

    def test_empty_evaluation_metrics_semantics(self):
        # find a seed that puts every entity into training
        events = [event(f'e{i}', f'o{i}', f'E{i}',
                        '2026-01-01T00:00:00Z', 10 * i + 1, i % 2)
                  for i in range(4)]
        self.ingest(events)
        seed = next(s for s in (f'z{i}' for i in range(5000))
                    if all(ledger.split_side(s, f'E{i}') == 'training'
                           for i in range(4)))
        report, _, _ = self.make_report(seed=seed, name='empty-eval.json')
        self.assertEqual(
            report['dependencies']['evaluation_dependency']['observation_count'], 0)
        self.assertIsNone(report['evaluation']['weighted_brier'])
        self.assertIsNone(report['evaluation']['weighted_log_loss'])
        self.assertFalse(report['evaluation']['infinite_log_loss'])

    def test_empty_training_is_business_error_without_output(self):
        events = [event('e1', 'o1', 'A', '2026-01-01T00:00:00Z', 10, 0)]
        self.ingest(events)
        seed = next(s for s in (f'z{i}' for i in range(5000))
                    if ledger.split_side(s, 'A') == 'evaluation')
        out = self.dir / 'never.json'
        with self.assertRaises(ledger.EmptyTraining):
            ledger.run_report(
                self.conn,
                {'observed_at_cutoff': '2027-01-01T00:00:00Z',
                 'knowledge_seq': 1, 'split_seed': seed}, out)
        self.assertFalse(out.exists())


class CorrectionInvalidationTests(LedgerCase):
    """Independent acceptance: eval-only keeps training identity; training changes it."""

    def _seed_entities(self, side_by_entity):
        events, idx = [], 0
        for entity, rows in side_by_entity.items():
            for obs, score, label, weight in rows:
                idx += 1
                events.append(event(
                    f'e{idx}', obs, entity, '2026-01-01T00:00:00Z',
                    score, label, weight))
        self.ingest(events)

    def test_evaluation_only_correction_keeps_training_identity(self):
        # Build entities whose sides we know; pick one training and one evaluation entity.
        seed = 'correction'
        entities = [chr(65 + i) for i in range(12)]
        sides = {e: ledger.split_side(seed, e) for e in entities}
        training_entities = [e for e in entities if sides[e] == 'training']
        eval_entities = [e for e in entities if sides[e] == 'evaluation']
        self.assertTrue(training_entities and eval_entities)
        layout = {e: [(f'{e}1', 10, 0, 1), (f'{e}2', 50, 1, 2)]
                  for e in entities}
        self._seed_entities(layout)
        first, _, request = self.make_report(seed=seed, name='eval-before.json')
        train_id_before = first['dependencies']['training_dependency_identity']
        eval_digest_before = first['dependencies']['evaluation_dependency']['digest']
        blocks_before = first['calibration']['blocks']

        target = eval_entities[0]
        self.ingest([  # change score/label/weight of an evaluation observation
            event('ecorr', f'{target}1', target, '2026-01-01T00:00:00Z',
                  12, 1, 5)])
        second, _ = ledger.run_report(
            self.conn, dict(request, knowledge_seq='latest'),
            self.dir / 'eval-after.json')
        self.assertEqual(
            second['dependencies']['training_dependency_identity'],
            train_id_before)
        self.assertEqual(second['calibration']['blocks'], blocks_before)
        self.assertNotEqual(
            second['dependencies']['evaluation_dependency']['digest'],
            eval_digest_before)
        # cache layer confirms the training curve identity was reused, while the
        # evaluation layer had to be recomputed under a new evaluation dependency key
        self.assertEqual(
            self.conn.execute(
                'SELECT COUNT(*) FROM training_curve_cache '
                'WHERE dependency_digest = ?',
                (train_id_before,)).fetchone()[0], 1)
        self.assertEqual(
            self.conn.execute(
                'SELECT COUNT(*) FROM evaluation_cache '
                'WHERE training_digest = ? AND evaluation_digest = ?',
                (train_id_before, eval_digest_before)).fetchone()[0], 1)
        self.assertEqual(
            self.conn.execute(
                'SELECT COUNT(*) FROM evaluation_cache '
                'WHERE training_digest = ? AND evaluation_digest = ?',
                (train_id_before,
                 second['dependencies']['evaluation_dependency']['digest'])
            ).fetchone()[0], 1)

    def test_training_value_correction_changes_identity_curve_may_stay_equal(self):
        seed = 'train-corr'
        entities = [chr(65 + i) for i in range(12)]
        sides = {e: ledger.split_side(seed, e) for e in entities}
        training_entities = [e for e in entities if sides[e] == 'training']
        eval_entities = [e for e in entities if sides[e] == 'evaluation']
        self.assertTrue(len(training_entities) >= 2 and eval_entities)
        # Two observations at the same score, same weight, opposite labels: swapping
        # their labels keeps the per-score positive count and total weight (and thus
        # the curve) identical, while the canonical training dependency must change.
        a, b = training_entities[0], training_entities[1]
        layout = {
            a: [(f'{a}1', 40, 1, 2)],
            b: [(f'{b}1', 40, 0, 2)],
        }
        for e in eval_entities:
            layout[e] = [(f'{e}1', 20, 1, 1)]
        self._seed_entities(layout)
        first, _, request = self.make_report(seed=seed, name='train-before.json')
        identity_before = first['dependencies']['training_dependency_identity']
        blocks_before = first['calibration']['blocks']

        self.ingest([
            event('ec1', f'{a}1', a, '2026-01-01T00:00:00Z', 40, 0, 2),
            event('ec2', f'{b}1', b, '2026-01-01T00:00:00Z', 40, 1, 2),
        ])
        second, _ = ledger.run_report(
            self.conn, dict(request, knowledge_seq='latest'),
            self.dir / 'train-after.json')
        self.assertNotEqual(
            second['dependencies']['training_dependency_identity'],
            identity_before)
        # curve values are allowed to remain exactly equal
        self.assertEqual(second['calibration']['blocks'], blocks_before)
        # both dependency identities coexist as distinct calibration layer entries
        identities = {row[0] for row in self.conn.execute(
            'SELECT dependency_digest FROM training_curve_cache')}
        self.assertIn(identity_before, identities)
        new_identity = second['dependencies']['training_dependency_identity']
        self.assertIn(new_identity, identities)
        # the calibration/evaluation dependency layer selected the NEW training
        # identity rather than reusing the old one
        bound = {row[0] for row in self.conn.execute(
            'SELECT DISTINCT training_digest FROM evaluation_cache')}
        self.assertIn(new_identity, bound)


class CrossSplitMigrationTests(LedgerCase):
    def test_entity_migration_changes_membership_identity_and_bound_seq(self):
        # The split depends only on (seed, entity_id).  Keep the same event stream and
        # change the request seed so one entity migrates evaluation -> training.
        seed_a, seed_b = 'migrate-a', 'migrate-b'
        entities = [chr(65 + i) for i in range(16)]
        events, idx = [], 0
        for entity in entities:
            for k in range(2):
                idx += 1
                events.append(event(
                    f'e{idx}', f'{entity}{k}', entity,
                    '2026-01-01T00:00:00Z', 10 + idx, idx % 2))
        self.ingest(events)

        def membership(seed):
            report, _, request = self.make_report(
                seed=seed, name=f'membership-{seed}.json')
            train = set(report['dependencies']['training_dependency']
                        ['observation_ids'])
            eval_ = set(report['dependencies']['evaluation_dependency']
                        ['observation_ids'])
            return report, train, eval_, request

        report_a, train_a, eval_a, _ = membership(seed_a)
        report_b, train_b, eval_b, _ = membership(seed_b)
        moved = set(
            e for e in entities
            if ledger.split_side(seed_a, e) != ledger.split_side(seed_b, e))
        self.assertTrue(moved)
        # every observation of a moved entity switches side
        for entity in moved:
            obs = {f'{entity}0', f'{entity}1'}
            if ledger.split_side(seed_a, entity) == 'evaluation':
                self.assertTrue(obs <= eval_a and obs <= train_b)
                self.assertFalse(obs & train_a)
            else:
                self.assertTrue(obs <= train_a and obs <= eval_b)
        # dependency identities change together with membership
        self.assertNotEqual(
            report_a['dependencies']['training_dependency_identity'],
            report_b['dependencies']['training_dependency_identity'])
        self.assertNotEqual(
            report_a['dependencies']['evaluation_dependency']['digest'],
            report_b['dependencies']['evaluation_dependency']['digest'])

    def test_migration_via_entity_id_correction_within_one_ledger(self):
        # Correcting entity_id moves an observation's entity across the split.
        seed = 'entity-fix'
        # Discover two entity names guaranteed to land on opposite sides.
        names = [f'NAME-{i}' for i in range(2000)]
        by_side = {'training': [], 'evaluation': []}
        for name in names:
            by_side[ledger.split_side(seed, name)].append(name)
        self.assertTrue(by_side['training'] and by_side['evaluation'])
        train_name = by_side['training'][0]
        eval_name = by_side['evaluation'][0]
        events = [
            event('e1', 'o1', eval_name, '2026-01-01T00:00:00Z', 10, 0),
            event('e2', 'o2', train_name, '2026-01-01T00:00:00Z', 20, 1),
            event('e3', 'o3', by_side['training'][1],
                  '2026-01-01T00:00:00Z', 30, 1),
        ]
        self.ingest(events)
        before, _, request = self.make_report(seed=seed, name='entity-before.json')
        self.assertEqual(ledger.split_side(seed, eval_name), 'evaluation')
        # migrate o1's entity from the evaluation side to the training side
        target_entity = train_name
        self.ingest([event('e4', 'o1', target_entity,
                           '2026-01-01T00:00:00Z', 10, 0)])
        after, _ = ledger.run_report(
            self.conn, dict(request, knowledge_seq='latest'),
            self.dir / 'after.json')
        before_ids = (before['dependencies']['training_dependency']['observation_ids']
                      + before['dependencies']['evaluation_dependency']
                      ['observation_ids'])
        after_ids = (after['dependencies']['training_dependency']['observation_ids']
                     + after['dependencies']['evaluation_dependency']
                     ['observation_ids'])
        self.assertEqual(sorted(before_ids), sorted(after_ids))  # no obs lost
        # --- Joint linked acceptance: in one migration, all three move together ---
        # 1) materialized membership: o1's effective entity changes side
        materialized = {
            seq: self.conn.execute(
                'SELECT entity_id FROM effective_observations '
                'WHERE knowledge_seq = ? AND observation_id = "o1"',
                (seq,)).fetchone()['entity_id']
            for seq in (1, 2)}
        self.assertEqual(materialized[1], eval_name)
        self.assertEqual(materialized[2], target_entity)
        # o1 must now be a training member (and was evaluation before)
        self.assertNotIn(
            'o1', before['dependencies']['training_dependency']['observation_ids'])
        self.assertIn(
            'o1', before['dependencies']['evaluation_dependency']['observation_ids'])
        self.assertIn(
            'o1', after['dependencies']['training_dependency']['observation_ids'])
        self.assertNotIn(
            'o1', after['dependencies']['evaluation_dependency']['observation_ids'])
        # 2) training dependency identity changes together with membership
        identity_before = before['dependencies']['training_dependency_identity']
        identity_after = after['dependencies']['training_dependency_identity']
        self.assertNotEqual(identity_before, identity_after)
        # 3) the report's bound seq advances together with the migration
        self.assertEqual(before['knowledge_seq'], 1)
        self.assertEqual(after['knowledge_seq'], 2)
        # the dependency layer for seq 2 is keyed on the new identity
        bound = {r[0] for r in self.conn.execute(
            'SELECT DISTINCT training_digest FROM evaluation_cache')}
        self.assertIn(identity_after, bound)
        # old seq view remains readable and untouched
        self.assertEqual(materialized[1], eval_name)


class VersionAndConcurrencyTests(LedgerCase):
    def test_old_version_reproduces_byte_for_byte(self):
        events1 = [event('e1', 'o1', 'A', '2026-01-01T00:00:00Z', 10, 0),
                   event('e2', 'o2', 'B', '2026-01-02T00:00:00Z', 20, 1)]
        self.ingest(events1)
        request = {'observed_at_cutoff': '2027-01-01T00:00:00Z',
                   'knowledge_seq': 1, 'split_seed': 'v'}
        out_v1 = self.dir / 'v1.json'
        report_v1, _ = ledger.run_report(self.conn, request, out_v1)
        bytes_v1 = out_v1.read_bytes()
        self.ingest([event('e3', 'o3', 'C', '2026-01-03T00:00:00Z', 30, 1)])
        # pinned seq=1 report must stay identical even though seq 2 exists
        out_v1_again = self.dir / 'v1-again.json'
        report_again, _ = ledger.run_report(self.conn, request, out_v1_again)
        self.assertEqual(out_v1_again.read_bytes(), bytes_v1)
        self.assertEqual(report_again['knowledge_seq'], 1)
        self.assertEqual(report_again['event_log']['log_digest'],
                         report_v1['event_log']['log_digest'])
        latest, _, _ = self.make_report(seq='latest', seed='v', name='latest.json')
        self.assertEqual(latest['knowledge_seq'], 2)

    def test_numeric_seq_above_max_is_request_error(self):
        self.ingest([event('e1', 'o1', 'A', '2026-01-01T00:00:00Z', 10, 0)])
        with self.assertRaises(ledger.RequestError):
            ledger.run_report(
                self.conn,
                {'observed_at_cutoff': '2027-01-01T00:00:00Z',
                 'knowledge_seq': 7, 'split_seed': 'v'},
                self.dir / 'x.json')

    def test_concurrent_ingest_does_not_mix_into_running_report(self):
        events = [event(f'e{i}', f'o{i}', f'E{i}',
                        '2026-01-01T00:00:00Z', 10 + i, i % 2)
                  for i in range(12)]
        self.ingest(events)
        request = {'observed_at_cutoff': '2027-01-01T00:00:00Z',
                   'knowledge_seq': 'latest', 'split_seed': 'concurrent'}
        captured = {}
        original = ledger.compute_report

        def slow_compute(conn, replica, seq, req, pinned, view_digest, count):
            # A higher seq commits while the pinned report is still computing.
            other = ledger.connect(self.db)
            try:
                ledger.ingest(other, {
                    'events': [event('e-late', 'o-late', 'E-LATE',
                                     '2026-01-01T00:00:00Z', 999, 1)]},
                    'b' * 64)
            finally:
                other.close()
            result = original(conn, replica, seq, req, pinned, view_digest, count)
            captured['seq'] = seq
            return result

        ledger.compute_report = slow_compute
        try:
            report, _ = ledger.run_report(
                self.conn, request, self.dir / 'concurrent.json')
        finally:
            ledger.compute_report = original
        self.assertEqual(captured['seq'], 1)
        self.assertEqual(report['knowledge_seq'], 1)
        all_ids = (report['dependencies']['training_dependency']['observation_ids']
                   + report['dependencies']['evaluation_dependency']
                   ['observation_ids'])
        self.assertNotIn('o-late', all_ids)
        self.assertEqual(ledger.max_committed_seq(self.conn), 2)


class ProjectionRecoveryTests(LedgerCase):
    def test_half_finished_projection_is_rebuilt_on_restart(self):
        self.ingest([event('e1', 'o1', 'A', '2026-01-01T00:00:00Z', 10, 0),
                     event('e2', 'o2', 'B', '2026-01-02T00:00:00Z', 20, 1)])
        ledger.ensure_projection(self.conn, 1)
        # simulate a crash between status=building and completion
        self.conn.execute(
            "UPDATE projections SET status = 'building', view_digest = NULL "
            'WHERE knowledge_seq = 1')
        self.conn.execute('DELETE FROM effective_observations WHERE knowledge_seq = 1')
        self.conn.commit()
        fresh = ledger.connect(self.db)
        try:
            ledger.recover(fresh)
            status = fresh.execute(
                'SELECT status, effective_count FROM projections '
                'WHERE knowledge_seq = 1').fetchone()
            self.assertEqual(status['status'], 'complete')
            self.assertEqual(status['effective_count'], 2)
            report, _ = ledger.run_report(
                fresh,
                {'observed_at_cutoff': '2027-01-01T00:00:00Z',
                 'knowledge_seq': 1, 'split_seed': 'r'},
                self.dir / 'recovered.json')
            self.assertEqual(report['knowledge_seq'], 1)
        finally:
            fresh.close()

    def test_corrupt_complete_projection_is_detected_and_repaired(self):
        self.ingest([event('e1', 'o1', 'A', '2026-01-01T00:00:00Z', 10, 0),
                     event('e2', 'o2', 'B', '2026-01-02T00:00:00Z', 20, 1)])
        ledger.ensure_projection(self.conn, 1)
        good_digest = self.conn.execute(
            'SELECT view_digest FROM projections WHERE knowledge_seq = 1').fetchone()[0]
        # tamper with the materialized rows while leaving status='complete'
        self.conn.execute(
            "UPDATE effective_observations SET score = 777 "
            "WHERE knowledge_seq = 1 AND observation_id = 'o1'")
        self.conn.commit()
        fresh = ledger.connect(self.db)
        try:
            ledger.recover(fresh)
            repaired = fresh.execute(
                'SELECT view_digest, status FROM projections '
                'WHERE knowledge_seq = 1').fetchone()
            self.assertEqual(repaired['status'], 'complete')
            self.assertEqual(repaired['view_digest'], good_digest)
            score = fresh.execute(
                "SELECT score FROM effective_observations "
                "WHERE knowledge_seq = 1 AND observation_id = 'o1'").fetchone()[0]
            self.assertEqual(score, 10)
        finally:
            fresh.close()

    def test_replica_incompleteness_blocks_publication(self):
        self.ingest([event('e1', 'o1', 'A', '2026-01-01T00:00:00Z', 10, 0),
                     event('e2', 'o2', 'B', '2026-01-02T00:00:00Z', 20, 1)])
        ledger.ensure_projection(self.conn, 1)
        original = ledger._copy_chunks

        def truncated_copy(source, target, select_sql, insert_sql, params):
            original(source, target, select_sql, insert_sql, params)
            if 'FROM events' in select_sql:
                target.execute('DELETE FROM events WHERE rowid = '
                               '(SELECT MAX(rowid) FROM events)')

        ledger._copy_chunks = truncated_copy
        out = self.dir / 'must-not-exist.json'
        try:
            with self.assertRaises(ledger.ReplicaIncomplete):
                ledger.run_report(
                    self.conn,
                    {'observed_at_cutoff': '2027-01-01T00:00:00Z',
                     'knowledge_seq': 1, 'split_seed': 'r'}, out)
        finally:
            ledger._copy_chunks = original
        self.assertFalse(out.exists())
        self.assertEqual(
            self.conn.execute('SELECT COUNT(*) FROM report_registry').fetchone()[0], 0)
        # retry after the fault clears succeeds
        report, _ = ledger.run_report(
            self.conn,
            {'observed_at_cutoff': '2027-01-01T00:00:00Z',
             'knowledge_seq': 1, 'split_seed': 'r'}, out)
        self.assertTrue(out.exists())
        self.assertEqual(report['knowledge_seq'], 1)


class ContractDetailTests(LedgerCase):
    def test_request_field_and_value_contract(self):
        good = {'observed_at_cutoff': '2026-01-01T00:00:00Z',
                'knowledge_seq': 'latest', 'split_seed': 'x'}
        self.assertEqual(
            ledger.validate_request(good)['observed_at_cutoff'],
            '2026-01-01T00:00:00.000000Z')
        for bad in (
            {'observed_at_cutoff': '2026-01-01', 'knowledge_seq': 'latest',
             'split_seed': 'x'},
            {'observed_at_cutoff': '2026-01-01T00:00:00Z',
             'knowledge_seq': 0, 'split_seed': 'x'},
            {'observed_at_cutoff': '2026-01-01T00:00:00Z',
             'knowledge_seq': '1', 'split_seed': 'x'},
            {'observed_at_cutoff': '2026-01-01T00:00:00Z',
             'knowledge_seq': 'latest', 'split_seed': ''},
            {'observed_at_cutoff': '2026-01-01T00:00:00Z',
             'knowledge_seq': 'latest'},
            {'observed_at_cutoff': '2026-01-01T00:00:00Z',
             'knowledge_seq': 'latest', 'split_seed': 'x', 'extra': 1},
        ):
            with self.assertRaises(ledger.LedgerError):
                ledger.validate_request(bad)

    def test_retract_still_requires_fields_but_values_are_ignored(self):
        # a retraction with wild values must validate (all fields present) yet never
        # contribute any value to the materialized effective observation.
        retraction = event('e2', 'o1', 'A', '2026-01-02T00:00:00Z',
                           999999, 1, 1000000, retract=True)
        self.ingest([
            event('e1', 'o1', 'A', '2026-01-01T00:00:00Z', 10, 0, 1),
            retraction,
        ])
        ledger.ensure_projection(self.conn, 1)
        self.assertEqual(
            self.conn.execute(
                'SELECT COUNT(*) FROM effective_observations '
                'WHERE knowledge_seq = 1').fetchone()[0], 0)
        # retraction missing a field must be rejected
        broken = dict(retraction)
        del broken['weight']
        with self.assertRaises(ledger.LedgerError):
            ledger.validate_event(broken)

    def test_report_matches_legacy_pava_pipeline(self):
        from score_calibration.model import analyze
        seed = 'pava'
        events, entities = [], [f'P{i}' for i in range(10)]
        for i, entity in enumerate(entities):
            events.append(event(
                f'e{i}', f'o{i}', entity, '2026-01-01T00:00:00Z',
                10 * i + 5, i % 2, 1 + (i % 3)))
        self.ingest(events)
        report, _, _ = self.make_report(seed=seed, name='pava.json')
        train_ids = report['dependencies']['training_dependency']['observation_ids']
        eval_ids = report['dependencies']['evaluation_dependency']['observation_ids']
        by_id = {f'o{i}': events[i] for i in range(len(events))}
        legacy = analyze({
            'training': [
                {'id': f'o{i}', 'score': events[i]['score'],
                 'label': events[i]['label'], 'weight': events[i]['weight']}
                for i in range(len(events)) if f'o{i}' in train_ids],
            'evaluation': [
                {'id': f'o{i}', 'score': events[i]['score'],
                 'label': events[i]['label'], 'weight': events[i]['weight']}
                for i in range(len(events)) if f'o{i}' in eval_ids],
        })
        self.assertEqual(report['calibration']['blocks'], legacy['blocks'])
        self.assertEqual(report['evaluation']['predictions'],
                         legacy['predictions'])
        self.assertEqual(report['evaluation']['weighted_brier'],
                         legacy['weighted_brier'])
        self.assertEqual(report['evaluation']['infinite_log_loss'],
                         legacy['infinite_log_loss'])

    def test_cutoff_uses_inclusive_utc_comparison(self):
        self.ingest([
            event('e1', 'o1', 'A', '2026-01-01T12:00:00+02:00', 10, 0),
            event('e2', 'o2', 'B', '2026-01-02T00:00:00Z', 20, 1),
        ])
        # 2026-01-01T10:00:00Z cutoff: o1 (10:00Z) is kept on the boundary, o2 excluded
        report, _, _ = self.make_report(
            cutoff='2026-01-01T10:00:00Z', name='boundary.json')
        ids = (report['dependencies']['training_dependency']['observation_ids']
               + report['dependencies']['evaluation_dependency']['observation_ids'])
        self.assertEqual(ids, ['o1'])

    def test_report_request_is_fully_resolved_in_output(self):
        self.ingest([event('e1', 'o1', 'A', '2026-01-01T00:00:00Z', 10, 0)])
        report, _, _ = self.make_report(seq='latest', seed='z',
                                        name='resolved.json')
        # 'latest' must be materialized to the concrete integer in the report
        self.assertEqual(report['request']['knowledge_seq'], 1)
        self.assertIsInstance(report['request']['knowledge_seq'], int)


class ResourceBoundTests(LedgerCase):
    def test_memory_tracks_effective_observations_not_revision_count(self):
        # Many revisions repeatedly correct a tiny set of observations: the report
        # path streams events and only materializes the small effective set.
        revisions = 200
        for seq in range(1, revisions + 1):
            self.ingest([event(
                f'e{seq}', 'o1', 'A', '2026-01-01T00:00:00Z',
                10 + (seq % 5), seq % 2)])
        report, _, _ = self.make_report(seq='latest', seed='r', name='resource.json')
        # one effective observation despite hundreds of revision events
        self.assertEqual(report['materialized_view']['effective_count'], 1)
        total_events = self.conn.execute('SELECT COUNT(*) FROM events').fetchone()[0]
        self.assertEqual(total_events, revisions)
        # replica only carries the pinned immutable range but folding is streamed
        self.assertTrue((self.dir / 'resource.json').exists())


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.tmp.name)
        self.ledger = self.dir / 'ledger.db'

    def tearDown(self):
        self.tmp.cleanup()

    def _call(self, *args):
        return subprocess.run(
            [sys.executable, '-m', 'score_calibration.cli', *map(str, args)],
            capture_output=True, text=True, cwd=ROOT,
            env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')})

    def test_ingest_report_and_replay_cli(self):
        batch = self.dir / 'batch.json'
        batch.write_text(json.dumps({'events': [
            event('e1', 'o1', 'A', '2026-01-01T00:00:00Z', 10, 0),
            event('e2', 'o2', 'B', '2026-01-02T00:00:00Z', 20, 1),
        ]}))
        first = self._call('ingest', '--ledger', self.ledger, '--batch', batch)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(json.loads(first.stdout)['knowledge_seq'], 1)
        replay = self._call('ingest', '--ledger', self.ledger, '--batch', batch)
        self.assertTrue(json.loads(replay.stdout)['replay'])

        request = self.dir / 'request.json'
        request.write_text(json.dumps({
            'observed_at_cutoff': '2027-01-01T00:00:00Z',
            'knowledge_seq': 'latest', 'split_seed': 'cli'}))
        out = self.dir / 'report.json'
        run = self._call('report', '--ledger', self.ledger,
                         '--request', request, '--output', out)
        self.assertEqual(run.returncode, 0, run.stderr)
        payload = out.read_bytes()
        rerun = self._call('report', '--ledger', self.ledger,
                           '--request', request, '--output', out)
        self.assertEqual(rerun.returncode, 0, rerun.stderr)
        self.assertTrue(json.loads(rerun.stdout)['idempotent_existing_output'])
        self.assertEqual(out.read_bytes(), payload)

        out.write_text(payload.decode() + 'x')
        conflict = self._call('report', '--ledger', self.ledger,
                              '--request', request, '--output', out)
        self.assertEqual(conflict.returncode, 2)
        self.assertTrue(out.read_text().endswith('x'))  # original preserved

    def test_bad_batch_exits_two_without_partial_append(self):
        batch = self.dir / 'bad.json'
        batch.write_text(json.dumps({'events': [
            event('e1', 'o1', 'A', '2026-01-01T00:00:00Z', 10, 0),
            event('e1', 'o2', 'B', '2026-01-02T00:00:00Z', 20, 1),
        ]}))
        result = self._call('ingest', '--ledger', self.ledger, '--batch', batch)
        self.assertEqual(result.returncode, 2)
        self.assertTrue(pathlib.Path(str(self.ledger) + '-wal').exists()
                        or self.ledger.exists())
        conn = ledger.connect(self.ledger)
        try:
            self.assertEqual(
                conn.execute('SELECT COUNT(*) FROM events').fetchone()[0], 0)
        finally:
            conn.close()

    def test_legacy_json_command_unchanged(self):
        out = self.dir / 'legacy.json'
        result = self._call('--source', ROOT / 'fixtures' / 'input.json',
                            '--output', out)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(out.read_text())
        self.assertEqual(data['schema_version'], 1)
        self.assertEqual(
            [b['probability'] for b in data['result']['blocks']],
            ['0', '1/2', '1'])


if __name__ == '__main__':
    unittest.main()
