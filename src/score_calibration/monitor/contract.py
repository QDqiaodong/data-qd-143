"""Strict snapshot contract, fixed configuration and canonical semantic identity.

A snapshot is a full replacement (never an incremental append): it carries the whole
reference training set and the whole observation set for every window.  Validation
reuses the v1 row contract (``id/score/label/weight``, ``common.ident`` and the
existing integer bounds, booleans are not integers) and the monitor-only window
contract.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date, timedelta
from fractions import Fraction

from ..common import ident, number, require, rows
from ..isotonic import fit

SNAPSHOT_FIELDS = (
    'snapshot_id', 'reference', 'observations', 'start_date', 'period_days',
    'period_count', 'min_weight', 'trigger_delta', 'clear_delta',
    'trigger_periods', 'clear_periods',
)
# Configuration is everything except the three replaceable parts.
CONFIG_FIELDS = (
    'start_date', 'period_days', 'period_count', 'min_weight',
    'trigger_delta', 'clear_delta', 'trigger_periods', 'clear_periods',
)
MAX_ROWS = 20000
SCORE_MAX = 1_000_000
WEIGHT_MAX = 1_000_000
# Finite decimal strings in [0, 1]; no exponents, no NaN/Infinity tokens.
_DECIMAL = re.compile(r'^(?:0|1|(?:0\.\d+))$')
_DATE = re.compile(r'^\d{4}-\d{2}-\d{2}$')


class MonitorError(ValueError):
    """Base class for all monitor business errors (validation, leases, sync)."""


def _bounded_int(value, name, lower, upper):
    # bool is a subclass of int; the contract explicitly forbids booleans here.
    if type(value) is not int or not lower <= value <= upper:
        raise MonitorError(f'{name} must be an integer in [{lower}, {upper}]')
    return value


def parse_delta(value, name):
    """Accept only finite decimal strings and return an exact Fraction."""
    if not isinstance(value, str) or not _DECIMAL.match(value):
        raise MonitorError(
            f'{name} must be a finite decimal string in [0, 1], e.g. "0.25"')
    return Fraction(value)


def parse_date(value):
    if not isinstance(value, str) or not _DATE.match(value):
        raise MonitorError('date must be a YYYY-MM-DD string')
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise MonitorError('date must be a valid YYYY-MM-DD calendar date')


def _reference_row(raw):
    require(raw, 'id score label weight')
    rid = ident(raw['id'])
    score = number(raw['score'], 0, SCORE_MAX)
    label = number(raw['label'], 0, 1)
    weight = number(raw['weight'], 1, WEIGHT_MAX)
    return {'id': rid, 'score': score, 'label': label, 'weight': weight}


def _observation_row(raw):
    require(raw, 'id cohort date score label weight')
    rid = ident(raw['id'])
    cohort = ident(raw['cohort'])
    day = parse_date(raw['date'])
    score = number(raw['score'], 0, SCORE_MAX)
    label = number(raw['label'], 0, 1)
    weight = number(raw['weight'], 1, WEIGHT_MAX)
    return {
        'id': rid, 'cohort': cohort, 'date': day, 'score': score,
        'label': label, 'weight': weight,
    }


def window_dates(start, period_days, period_count):
    """Continuous left-closed/right-open calendar windows starting at ``start``."""
    return [
        (start + timedelta(days=period_days * index),
         start + timedelta(days=period_days * (index + 1)))
        for index in range(period_count)
    ]


def validate_snapshot(data):
    """Validate and normalize a snapshot; ordering of the two arrays is erased.

    Returns a normalized dict whose Fraction/date objects are materialized and whose
    arrays are sorted by id.  Raises :class:`MonitorError` on any contract breach; the
    caller rejects the whole sync before the database is touched.
    """
    if not isinstance(data, dict) or set(data) != set(SNAPSHOT_FIELDS):
        raise MonitorError(
            'snapshot must contain exactly the fields: ' + ', '.join(SNAPSHOT_FIELDS))
    snapshot_id = ident(data['snapshot_id'])

    reference_raw = rows(data['reference'], MAX_ROWS)
    reference, ref_ids = [], set()
    for raw in reference_raw:
        row = _reference_row(raw)
        if row['id'] in ref_ids:
            raise MonitorError('duplicate reference id: ' + row['id'])
        ref_ids.add(row['id'])
        reference.append(row)
    if not reference:
        raise MonitorError('reference must be nonempty')
    reference.sort(key=lambda row: row['id'])

    observation_raw = rows(data['observations'], MAX_ROWS)
    observations, obs_ids = [], set()
    for raw in observation_raw:
        row = _observation_row(raw)
        if row['id'] in obs_ids:
            raise MonitorError('duplicate observation id: ' + row['id'])
        obs_ids.add(row['id'])
        observations.append(row)
    if ref_ids & obs_ids:
        raise MonitorError('reference and observation id sets must be disjoint')
    observations.sort(key=lambda row: row['id'])

    start = parse_date(data['start_date'])
    period_days = _bounded_int(data['period_days'], 'period_days', 1, 31)
    period_count = _bounded_int(data['period_count'], 'period_count', 1, 366)
    min_weight = _bounded_int(data['min_weight'], 'min_weight', 1, 10 ** 9)
    trigger_delta = parse_delta(data['trigger_delta'], 'trigger_delta')
    clear_delta = parse_delta(data['clear_delta'], 'clear_delta')
    if not clear_delta < trigger_delta:
        raise MonitorError('clear_delta must be strictly below trigger_delta')
    trigger_periods = _bounded_int(
        data['trigger_periods'], 'trigger_periods', 1, period_count)
    clear_periods = _bounded_int(
        data['clear_periods'], 'clear_periods', 1, period_count)

    window_end = start + timedelta(days=period_days * period_count)
    for row in observations:
        if not start <= row['date'] < window_end:
            raise MonitorError(
                f"observation {row['id']} date {row['date'].isoformat()} falls "
                f"outside the continuous calendar window "
                f"[{start.isoformat()}, {window_end.isoformat()})")

    return {
        'snapshot_id': snapshot_id,
        'reference': reference,
        'observations': observations,
        'start_date': start,
        'period_days': period_days,
        'period_count': period_count,
        'min_weight': min_weight,
        'trigger_delta': trigger_delta,
        'clear_delta': clear_delta,
        'trigger_periods': trigger_periods,
        'clear_periods': clear_periods,
    }


def frozen_config(snapshot):
    """The fields fixed by the first successful sync."""
    config = {field: snapshot[field] for field in CONFIG_FIELDS}
    # Dates are normalized to fixed-width ISO strings; deltas to Fraction strings so
    # the persisted comparison is purely textual and fully deterministic.
    config['start_date'] = config['start_date'].isoformat()
    config['trigger_delta'] = str(config['trigger_delta'])
    config['clear_delta'] = str(config['clear_delta'])
    return config


def _canonical_training_rows(reference):
    # Sort by id, then emit the exact v1 row contract with canonical Fraction-free
    # integer values (scores/labels/weights are always plain integers here).
    return [
        {'id': row['id'], 'score': row['score'], 'label': row['label'],
         'weight': row['weight']}
        for row in sorted(reference, key=lambda row: row['id'])
    ]


def _canonical_observation_rows(observations):
    # Sort by id; dates are emitted as ISO strings; all numeric fields stay integers.
    return [
        {'id': row['id'], 'cohort': row['cohort'],
         'date': row['date'].isoformat(), 'score': row['score'],
         'label': row['label'], 'weight': row['weight']}
        for row in sorted(observations, key=lambda row: row['id'])
    ]


def content_digest(snapshot):
    """Canonical semantic content digest.

    Both sample arrays are sorted by id; values are already exact integers and deltas
    are converted to canonical Fraction strings.  ``snapshot_id`` is deliberately
    excluded so the same content under a new id is a new revision but replaying the
    same id is byte-identical.  All frozen configuration (including ``start_date``)
    is included.  The digest is SHA256 over compact sorted-key UTF-8 JSON.
    """
    body = {
        'reference': _canonical_training_rows(snapshot['reference']),
        'observations': _canonical_observation_rows(snapshot['observations']),
        'config': frozen_config(snapshot),
    }
    raw = json.dumps(body, ensure_ascii=False, separators=(',', ':'),
                     sort_keys=True).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def reference_blocks(snapshot):
    """Run the existing weighted PAVA over the snapshot reference training set."""
    return fit(snapshot['reference'])
