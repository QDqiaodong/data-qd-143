"""Append-only revision ledger, materialized observation views and versioned reports.

The module is built only on the Python standard library and :mod:`sqlite3`.  Events are
stored in an append-only log; materialized observation views are addressed by
``knowledge_seq``; reports are computed against a verified immutable replica so that a
concurrent ingest can never mix a newer revision into a pinned report.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .common import save
from .isotonic import fit, score

SCHEMA_VERSION = 2
EVENT_FIELDS = (
    'event_id', 'observation_id', 'entity_id', 'observed_at',
    'score', 'label', 'weight', 'retract',
)
EFFECTIVE_FIELDS = (
    'observation_id', 'entity_id', 'observed_at', 'score', 'label', 'weight',
)
MAX_ID_LEN = 4096
FETCH_SIZE = 1000
TRAINING_FRACTION = (4, 5)  # 80%
_RFC3339 = re.compile(
    r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$'
)


class LedgerError(ValueError):
    """Base class for ledger business errors."""


class BatchConflict(LedgerError):
    """A non-replay batch collides with committed events."""


class RequestError(LedgerError):
    """A report request violates the request contract."""


class EmptyTraining(LedgerError):
    """The pinned version has no training observations after filtering."""


class OutputConflict(LedgerError):
    """An existing output file differs from the report that would be published."""


class ReplicaIncomplete(LedgerError):
    """The immutable event-range replica failed completeness verification."""


# --------------------------------------------------------------------------- #
# Deterministic encoding and validation
# --------------------------------------------------------------------------- #

def canonical_bytes(obj):
    """UTF-8 JSON bytes with no whitespace and sorted keys (order-independent identity)."""
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':'),
                      sort_keys=True).encode('utf-8')


def _ordered_bytes(obj):
    """UTF-8 JSON bytes with no whitespace preserving insertion order."""
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':')).encode('utf-8')


def event_canonical(event):
    # The event identity pins the exact contract field order in EVENT_FIELDS.
    return _ordered_bytes({key: event[key] for key in EVENT_FIELDS})


def effective_canonical(row):
    return canonical_bytes({key: row[key] for key in EFFECTIVE_FIELDS})


def parse_rfc3339(value):
    """Accept only timezone-aware RFC3339 strings and normalize to fixed-width UTC."""
    if not isinstance(value, str) or not _RFC3339.match(value):
        raise LedgerError('expected timezone-aware RFC3339 timestamp')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:  # defensive, regex already requires an offset
        raise LedgerError('timestamp must carry a timezone offset')
    return parsed.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')


def _identifier(value, name):
    if (not isinstance(value, str) or not value or len(value) > MAX_ID_LEN
            or any(ord(ch) < 32 for ch in value)):
        raise LedgerError(
            f'{name} must be a nonempty string without control characters '
            f'(max {MAX_ID_LEN} characters)')
    return value


def _bounded_int(value, name, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise LedgerError(f'{name} must be an integer in [{lower}, {upper}]')
    return value


def validate_event(raw):
    if not isinstance(raw, dict) or set(raw) != set(EVENT_FIELDS):
        raise LedgerError('event must contain exactly the fields: ' + ', '.join(EVENT_FIELDS))
    if type(raw['retract']) is not bool:
        raise LedgerError('retract must be a boolean')
    normalized = {
        'event_id': _identifier(raw['event_id'], 'event_id'),
        'observation_id': _identifier(raw['observation_id'], 'observation_id'),
        'entity_id': _identifier(raw['entity_id'], 'entity_id'),
        'observed_at': parse_rfc3339(raw['observed_at']),
        'score': _bounded_int(raw['score'], 'score', 0, 1_000_000),
        'label': _bounded_int(raw['label'], 'label', 0, 1),
        'weight': _bounded_int(raw['weight'], 'weight', 1, 1_000_000),
        'retract': int(raw['retract']),
    }
    return normalized


def validate_batch(data):
    """Return (normalized events, canonical byte per event, batch digest)."""
    if not isinstance(data, dict) or set(data) != {'events'}:
        raise LedgerError('batch must be an object with a single "events" array')
    raw_events = data['events']
    if not isinstance(raw_events, list) or not raw_events:
        raise LedgerError('events must be a nonempty array')
    events, seen, pieces = [], set(), []
    for raw in raw_events:
        event = validate_event(raw)
        if event['event_id'] in seen:
            raise BatchConflict('duplicate event_id within batch')
        seen.add(event['event_id'])
        events.append(event)
        pieces.append(event_canonical(event))
    digest = hashlib.sha256(b''.join(pieces)).hexdigest()
    return events, pieces, digest


def validate_request(data):
    if not isinstance(data, dict) or set(data) != {
            'observed_at_cutoff', 'knowledge_seq', 'split_seed'}:
        raise RequestError(
            'request must contain exactly observed_at_cutoff, knowledge_seq, split_seed')
    seq_value = data['knowledge_seq']
    if seq_value != 'latest':
        if type(seq_value) is not int or seq_value < 1:
            raise RequestError('knowledge_seq must be a positive integer or "latest"')
    if not isinstance(data['split_seed'], str) or not data['split_seed']:
        raise RequestError('split_seed must be a nonempty string')
    return {
        'observed_at_cutoff': parse_rfc3339(data['observed_at_cutoff']),
        'knowledge_seq': seq_value,
        'split_seed': data['split_seed'],
    }


def split_side(seed, entity_id):
    """Return 'training' or 'evaluation' for an entity; entities never split."""
    digest = hashlib.sha256(
        seed.encode('utf-8') + b'\x00' + entity_id.encode('utf-8')).digest()[:8]
    value = int.from_bytes(digest, 'big')
    numerator, denominator = TRAINING_FRACTION
    return 'training' if value * denominator < (1 << 64) * numerator else 'evaluation'


# --------------------------------------------------------------------------- #
# Schema and connection management
# --------------------------------------------------------------------------- #

_DDL = (
    """
    CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )""",
    """
    CREATE TABLE IF NOT EXISTS batches (
        knowledge_seq INTEGER PRIMARY KEY CHECK (knowledge_seq > 0),
        event_count INTEGER NOT NULL,
        batch_digest TEXT NOT NULL UNIQUE,
        raw_sha256 TEXT NOT NULL,
        log_digest TEXT NOT NULL,
        received_at TEXT NOT NULL
    )""",
    """
    CREATE TABLE IF NOT EXISTS events (
        knowledge_seq INTEGER NOT NULL CHECK (knowledge_seq > 0),
        position INTEGER NOT NULL CHECK (position >= 0),
        event_id TEXT NOT NULL,
        observation_id TEXT NOT NULL,
        entity_id TEXT NOT NULL,
        observed_at TEXT NOT NULL,
        score INTEGER NOT NULL,
        label INTEGER NOT NULL,
        weight INTEGER NOT NULL,
        retract INTEGER NOT NULL CHECK (retract IN (0, 1)),
        canonical_json TEXT NOT NULL,
        PRIMARY KEY (knowledge_seq, position),
        UNIQUE (knowledge_seq, event_id),
        UNIQUE (event_id)
    )""",
    """
    CREATE TABLE IF NOT EXISTS projections (
        knowledge_seq INTEGER PRIMARY KEY CHECK (knowledge_seq > 0),
        status TEXT NOT NULL CHECK (status IN ('building', 'complete')),
        view_digest TEXT,
        effective_count INTEGER,
        updated_at TEXT NOT NULL
    )""",
    """
    CREATE TABLE IF NOT EXISTS effective_observations (
        knowledge_seq INTEGER NOT NULL,
        observation_id TEXT NOT NULL,
        entity_id TEXT NOT NULL,
        observed_at TEXT NOT NULL,
        score INTEGER NOT NULL,
        label INTEGER NOT NULL,
        weight INTEGER NOT NULL,
        PRIMARY KEY (knowledge_seq, observation_id)
    )""",
    """
    CREATE TABLE IF NOT EXISTS training_curve_cache (
        dependency_digest TEXT PRIMARY KEY,
        blocks_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""",
    """
    CREATE TABLE IF NOT EXISTS evaluation_cache (
        training_digest TEXT NOT NULL,
        evaluation_digest TEXT NOT NULL,
        metrics_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (training_digest, evaluation_digest)
    )""",
    """
    CREATE TABLE IF NOT EXISTS report_registry (
        request_digest TEXT PRIMARY KEY,
        knowledge_seq INTEGER NOT NULL,
        report_sha256 TEXT NOT NULL,
        output_path TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""",
)


def _now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')


def connect(path):
    ledger = Path(path)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(ledger))
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA foreign_keys=ON')
    for statement in _DDL:
        conn.execute(statement)
    conn.execute(
        'INSERT OR IGNORE INTO meta(key, value) VALUES(?, ?)',
        ('schema_version', str(SCHEMA_VERSION)))
    conn.commit()
    return conn


def max_committed_seq(conn):
    return conn.execute('SELECT COALESCE(MAX(knowledge_seq), 0) FROM batches').fetchone()[0]


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #

def _incremental_log_digest(conn, seq, pieces):
    """Prefix log digest for seq = digest(previous prefix digest || this batch bytes).

    Chaining the prior batch's stored digest keeps ingestion work proportional to the
    new batch only, never to the full revision history.  Replica verification recomputes
    the same chained digest independently over the copied immutable event range.
    """
    previous = conn.execute(
        'SELECT log_digest FROM batches WHERE knowledge_seq = ?', (seq - 1,)).fetchone()
    hasher = hashlib.sha256()
    if previous is not None:
        hasher.update(previous['log_digest'].encode('utf-8'))
    for piece in pieces:
        hasher.update(piece)
    return hasher.hexdigest()


def recompute_log_digest(replica_or_conn, seq):
    """Recompute the chained prefix log digest independently from stored event bytes.

    Prefix digest for seq N is sha256(prefix_digest_(N-1)_utf8 || canonical event bytes
    of batch N in position order), with an empty prefix for N=1.  This is the
    verification counterpart of :func:`_incremental_log_digest`.
    """
    cursor = replica_or_conn.execute(
        'SELECT knowledge_seq, canonical_json FROM events '
        'WHERE knowledge_seq <= ? ORDER BY knowledge_seq, position', (seq,))
    digests, current_seq, batch_hasher = {}, 0, None
    while True:
        rows = cursor.fetchmany(FETCH_SIZE)
        if not rows:
            break
        for row in rows:
            if row['knowledge_seq'] != current_seq:
                if batch_hasher is not None:
                    digests[current_seq] = batch_hasher.hexdigest()
                current_seq = row['knowledge_seq']
                batch_hasher = hashlib.sha256()
                prior = digests.get(current_seq - 1)
                if prior is not None:
                    batch_hasher.update(prior.encode('utf-8'))
            batch_hasher.update(row['canonical_json'].encode('utf-8'))
    if batch_hasher is not None:
        digests[current_seq] = batch_hasher.hexdigest()
    return digests.get(seq)


def ingest(conn, data, raw_sha256):
    """Validate and atomically append a batch; exact replays return the old receipt."""
    events, pieces, batch_digest = validate_batch(data)
    conn.execute('BEGIN IMMEDIATE')
    try:
        replay = conn.execute(
            'SELECT knowledge_seq, log_digest FROM batches WHERE batch_digest = ?',
            (batch_digest,)).fetchone()
        if replay is not None:
            conn.rollback()
            return {
                'knowledge_seq': replay['knowledge_seq'],
                'batch_digest': batch_digest,
                'event_count': len(events),
                'replay': True,
                'log_digest': replay['log_digest'],
            }
        ids = [event['event_id'] for event in events]
        placeholders = ','.join('?' * len(ids))
        collision = conn.execute(
            f'SELECT event_id FROM events WHERE event_id IN ({placeholders}) LIMIT 1',
            ids).fetchone()
        if collision is not None:
            raise BatchConflict(
                'event_id already exists in another batch; only an exact full-batch '
                'replay is accepted')
        seq = max_committed_seq(conn) + 1
        conn.execute(
            'INSERT INTO batches(knowledge_seq, event_count, batch_digest, raw_sha256, '
            'log_digest, received_at) VALUES (?, ?, ?, ?, ?, ?)',
            (seq, len(events), batch_digest, raw_sha256, 'pending', _now()))
        conn.executemany(
            'INSERT INTO events(knowledge_seq, position, event_id, observation_id, '
            'entity_id, observed_at, score, label, weight, retract, canonical_json) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            [
                (seq, position, event['event_id'], event['observation_id'],
                 event['entity_id'], event['observed_at'], event['score'],
                 event['label'], event['weight'], event['retract'],
                 pieces[position].decode('utf-8'))
                for position, event in enumerate(events)
            ])
        log_digest = _incremental_log_digest(conn, seq, pieces)
        conn.execute('UPDATE batches SET log_digest = ? WHERE knowledge_seq = ?',
                     (log_digest, seq))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return {
        'knowledge_seq': seq,
        'batch_digest': batch_digest,
        'event_count': len(events),
        'replay': False,
        'log_digest': log_digest,
    }


# --------------------------------------------------------------------------- #
# Materialized observation views
# --------------------------------------------------------------------------- #

_INSERT_EFFECTIVE = """
INSERT INTO effective_observations(
    knowledge_seq, observation_id, entity_id, observed_at, score, label, weight)
SELECT ?, observation_id, entity_id, observed_at, score, label, weight
FROM (
    SELECT observation_id, entity_id, observed_at, score, label, weight, retract,
           ROW_NUMBER() OVER (
               PARTITION BY observation_id
               ORDER BY knowledge_seq DESC, position DESC) AS rn
    FROM events
    WHERE knowledge_seq <= ?
)
WHERE rn = 1 AND retract = 0
"""


def _view_digest(conn, seq):
    hasher, count = hashlib.sha256(), 0
    cursor = conn.execute(
        'SELECT observation_id, entity_id, observed_at, score, label, weight '
        'FROM effective_observations WHERE knowledge_seq = ? ORDER BY observation_id',
        (seq,))
    while True:
        rows = cursor.fetchmany(FETCH_SIZE)
        if not rows:
            break
        for row in rows:
            hasher.update(effective_canonical(row))
            hasher.update(b'\n')
            count += 1
    return hasher.hexdigest(), count


def build_projection(conn, seq):
    """(Re)build the materialized view for seq; only complete at commit time."""
    conn.execute('BEGIN IMMEDIATE')
    try:
        conn.execute(
            'INSERT INTO projections(knowledge_seq, status, updated_at) VALUES (?, ?, ?) '
            'ON CONFLICT(knowledge_seq) DO UPDATE SET status = excluded.status, '
            'view_digest = NULL, effective_count = NULL, updated_at = excluded.updated_at',
            (seq, 'building', _now()))
        conn.execute('DELETE FROM effective_observations WHERE knowledge_seq = ?', (seq,))
        conn.execute(_INSERT_EFFECTIVE, (seq, seq))
        view_digest, count = _view_digest(conn, seq)
        conn.execute(
            'UPDATE projections SET status = ?, view_digest = ?, effective_count = ?, '
            'updated_at = ? WHERE knowledge_seq = ?',
            ('complete', view_digest, count, _now(), seq))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return view_digest, count


def recover(conn, upto_seq=None):
    """Rebuild half-finished projections and repair digest mismatches from the log.

    When ``upto_seq`` is given only projections at or below it are inspected, so a
    report pinned to seq N never pays validation cost proportional to later seqs.
    """
    query = ('SELECT knowledge_seq, status, view_digest FROM projections')
    params = ()
    if upto_seq is not None:
        query += ' WHERE knowledge_seq <= ?'
        params = (upto_seq,)
    query += ' ORDER BY knowledge_seq'
    for row in conn.execute(query, params):
        if row['status'] != 'complete':
            build_projection(conn, row['knowledge_seq'])
            continue
        actual, _ = _view_digest(conn, row['knowledge_seq'])
        if actual != row['view_digest']:
            build_projection(conn, row['knowledge_seq'])


def ensure_projection(conn, seq):
    """Recover within the pinned range, then advance the watermark up to seq."""
    if seq < 1 or seq > max_committed_seq(conn):
        raise RequestError(f'knowledge_seq {seq} is not committed')
    recover(conn, seq)
    existing = {
        row['knowledge_seq'] for row in conn.execute(
            'SELECT knowledge_seq FROM projections WHERE status = "complete"')
    }
    for candidate in range(1, seq + 1):
        if candidate not in existing:
            build_projection(conn, candidate)
    return conn.execute(
        'SELECT view_digest, effective_count FROM projections WHERE knowledge_seq = ?',
        (seq,)).fetchone()


# --------------------------------------------------------------------------- #
# Verified immutable replica (read-view strategy)
# --------------------------------------------------------------------------- #

def _copy_chunks(source, target, select_sql, insert_sql, params):
    cursor = source.execute(select_sql, params)
    while True:
        rows = cursor.fetchmany(FETCH_SIZE)
        if not rows:
            break
        target.executemany(insert_sql, [tuple(row) for row in rows])


def make_replica(conn, seq, expected_view_digest, expected_log_digest,
                 expected_event_count, expected_view_count):
    """Copy the immutable event range for seq into a temp SQLite database and verify it.

    A read snapshot is held only while copying; afterwards the source transaction ends
    (retaining no WAL pages for the long computation).  A replica that cannot be proven
    complete is destroyed and the caller must retry; no report is ever registered from it.
    """
    main_db = next(
        row['file'] for row in conn.execute('PRAGMA database_list')
        if row['name'] == 'main')
    ledger_dir = Path(main_db).parent
    handle = tempfile.NamedTemporaryFile(
        prefix='.score-calibration-replica-', suffix='.db', dir=ledger_dir, delete=False)
    replica_path = Path(handle.name)
    handle.close()
    replica = None
    try:
        replica = sqlite3.connect(str(replica_path))
        replica.row_factory = sqlite3.Row
        replica.execute(
            'CREATE TABLE events (knowledge_seq INTEGER, position INTEGER, event_id TEXT, '
            'observation_id TEXT, entity_id TEXT, observed_at TEXT, score INTEGER, '
            'label INTEGER, weight INTEGER, retract INTEGER, canonical_json TEXT)')
        replica.execute(
            'CREATE TABLE effective_observations (knowledge_seq INTEGER, '
            'observation_id TEXT, entity_id TEXT, observed_at TEXT, score INTEGER, '
            'label INTEGER, weight INTEGER)')
        replica.execute('CREATE INDEX idx_events_order ON events(knowledge_seq, position)')
        replica.execute(
            'CREATE INDEX idx_effective ON effective_observations(knowledge_seq, '
            'observed_at, observation_id)')
        # Snapshot transaction on the source spans the whole copy only.
        conn.execute('BEGIN')
        try:
            source_count, source_max, source_seq_count = conn.execute(
                'SELECT COUNT(*), COALESCE(MAX(knowledge_seq), 0), '
                'COUNT(DISTINCT knowledge_seq) FROM events WHERE knowledge_seq <= ?',
                (seq,)).fetchone()
            _copy_chunks(
                conn, replica,
                'SELECT knowledge_seq, position, event_id, observation_id, entity_id, '
                'observed_at, score, label, weight, retract, canonical_json '
                'FROM events WHERE knowledge_seq <= ? ORDER BY knowledge_seq, position',
                'INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', (seq,))
            _copy_chunks(
                conn, replica,
                'SELECT knowledge_seq, observation_id, entity_id, observed_at, score, '
                'label, weight FROM effective_observations WHERE knowledge_seq = ? '
                'ORDER BY observation_id',
                'INSERT INTO effective_observations VALUES (?, ?, ?, ?, ?, ?, ?)', (seq,))
        finally:
            conn.rollback()  # read-only snapshot; rollback simply ends it
        replica.commit()
        # Independent verification against the replica itself.
        checks = replica.execute(
            'SELECT COUNT(*), COALESCE(MAX(knowledge_seq), 0), '
            'COUNT(DISTINCT knowledge_seq) FROM events').fetchone()
        copied_count, copied_max, copied_seq_count = checks
        if (copied_count != source_count or copied_count != expected_event_count
                or copied_max != seq or copied_seq_count != source_seq_count
                or copied_seq_count != seq):
            raise ReplicaIncomplete('event range replica is incomplete')
        replica_log_digest = recompute_log_digest(replica, seq)
        if replica_log_digest != expected_log_digest:
            raise ReplicaIncomplete('replica event log digest mismatch')
        view = replica.execute(
            'SELECT COUNT(*) FROM effective_observations WHERE knowledge_seq = ?',
            (seq,)).fetchone()[0]
        replica_view = hashlib.sha256()
        cursor = replica.execute(
            'SELECT observation_id, entity_id, observed_at, score, label, weight '
            'FROM effective_observations WHERE knowledge_seq = ? ORDER BY observation_id',
            (seq,))
        while True:
            rows = cursor.fetchmany(FETCH_SIZE)
            if not rows:
                break
            for row in rows:
                replica_view.update(effective_canonical(row))
                replica_view.update(b'\n')
        if replica_view.hexdigest() != expected_view_digest or view != expected_view_count:
            raise ReplicaIncomplete('replica materialized view mismatch')
        return replica, replica_path, view
    except Exception:
        if replica is not None:
            replica.close()
        for suffix in ('', '-wal', '-shm'):
            stale = Path(str(replica_path) + suffix)
            if stale.exists():
                stale.unlink()
        raise


# --------------------------------------------------------------------------- #
# Report computation
# --------------------------------------------------------------------------- #

def _dependency_digest(rows):
    hasher = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item['observation_id']):
        hasher.update(effective_canonical(row))
        hasher.update(b'\n')
    return hasher.hexdigest()


def _eligible_observations(replica, seq, cutoff):
    """Stream the pinned effective observations that satisfy the cutoff."""
    cursor = replica.execute(
        'SELECT observation_id, entity_id, observed_at, score, label, weight '
        'FROM effective_observations WHERE knowledge_seq = ? AND observed_at <= ? '
        'ORDER BY observation_id', (seq, cutoff))
    while True:
        rows = cursor.fetchmany(FETCH_SIZE)
        if not rows:
            break
        for row in rows:
            yield dict(row)


def _model_rows(rows):
    return [
        {'id': row['observation_id'], 'score': row['score'],
         'label': row['label'], 'weight': row['weight']}
        for row in rows
    ]


def compute_report(conn, replica, seq, request, pinned, view_digest, effective_count):
    cutoff, seed = request['observed_at_cutoff'], request['split_seed']
    sides, eligible = {}, []
    for row in _eligible_observations(replica, seq, cutoff):
        side = sides.get(row['entity_id'])
        if side is None:
            side = split_side(seed, row['entity_id'])
            sides[row['entity_id']] = side
        row['_side'] = side
        eligible.append(row)
    training = [row for row in eligible if row['_side'] == 'training']
    evaluation = [row for row in eligible if row['_side'] == 'evaluation']
    if not training:
        raise EmptyTraining('training split is empty after cutoff filtering')

    training_digest = _dependency_digest(training)
    evaluation_digest = _dependency_digest(evaluation)
    training_ids = sorted(row['observation_id'] for row in training)
    evaluation_ids = sorted(row['observation_id'] for row in evaluation)

    # Calibration layer: keyed strictly by the training dependency identity.
    curve = conn.execute(
        'SELECT blocks_json FROM training_curve_cache WHERE dependency_digest = ?',
        (training_digest,)).fetchone()
    if curve is None:
        blocks = fit(_model_rows(training))
        conn.execute('BEGIN IMMEDIATE')
        try:
            conn.execute(
                'INSERT OR IGNORE INTO training_curve_cache(dependency_digest, '
                'blocks_json, created_at) VALUES (?, ?, ?)',
                (training_digest, json.dumps(blocks, sort_keys=True), _now()))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    else:
        blocks = json.loads(curve['blocks_json'])
    # Evaluation layer: reuse the training identity and key on the evaluation digest.
    cached_eval = conn.execute(
        'SELECT metrics_json FROM evaluation_cache WHERE training_digest = ? '
        'AND evaluation_digest = ?', (training_digest, evaluation_digest)).fetchone()
    if cached_eval is None:
        metrics = score(blocks, _model_rows(evaluation))
        conn.execute('BEGIN IMMEDIATE')
        try:
            conn.execute(
                'INSERT OR IGNORE INTO evaluation_cache(training_digest, '
                'evaluation_digest, metrics_json, created_at) VALUES (?, ?, ?, ?)',
                (training_digest, evaluation_digest,
                 json.dumps(metrics, sort_keys=True), _now()))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    else:
        metrics = json.loads(cached_eval['metrics_json'])
        # JSON round-trip keeps booleans; normalize defensively for deterministic bytes.
        metrics['infinite_log_loss'] = bool(metrics['infinite_log_loss'])

    resolved_request = {
        'observed_at_cutoff': cutoff,
        'knowledge_seq': seq,
        'split_seed': seed,
    }
    # The report content excludes the two hash fields; its canonical digest is the
    # report identity.  External readers reproduce it by dropping those two keys.
    content = {
        'schema_version': SCHEMA_VERSION,
        'request': resolved_request,
        'knowledge_seq': seq,
        'event_log': {
            'knowledge_seq': seq,
            'event_count': pinned['event_count'],
            'log_digest': pinned['log_digest'],
        },
        'materialized_view': {
            'knowledge_seq': seq,
            'effective_count': effective_count,
            'view_digest': view_digest,
        },
        'split': {
            'rule': 'sha256(split_seed || NUL || entity_id) first 8 bytes unsigned '
                    'big-endian divided by 2**64; < 4/5 trains, the rest evaluate',
            'training_fraction': '4/5',
            'entity_count': len(sides),
        },
        'dependencies': {
            'training_dependency_identity': training_digest,
            'training_dependency': {
                'observation_count': len(training),
                'digest': training_digest,
                'observation_ids': training_ids,
            },
            'evaluation_dependency': {
                'observation_count': len(evaluation),
                'digest': evaluation_digest,
                'observation_ids': evaluation_ids,
            },
        },
        'calibration': {'blocks': blocks},
        'evaluation': metrics,
    }
    request_digest = hashlib.sha256(canonical_bytes(resolved_request)).hexdigest()
    report_sha256 = hashlib.sha256(canonical_bytes(content)).hexdigest()
    report = dict(content)
    report['request_sha256'] = request_digest
    report['report_sha256'] = report_sha256
    return report, request_digest, training_digest, evaluation_digest


def _report_payload(report):
    return (json.dumps(report, sort_keys=True, ensure_ascii=False,
                       allow_nan=False, indent=2) + '\n').encode('utf-8')


def publish_report(conn, output, report, request_digest):
    """Atomically publish; identical existing output is idempotent, mismatch conflicts."""
    dest = Path(output)
    if dest.is_symlink():
        raise OutputConflict('output path cannot be a symlink')
    payload = _report_payload(report)
    if dest.exists():
        existing = dest.read_bytes()
        if existing != payload:
            raise OutputConflict(f'existing output {dest} differs; refusing to overwrite')
        reused = True
    else:
        save(dest, report)  # same deterministic serialization, atomic replace
        reused = False
    conn.execute('BEGIN IMMEDIATE')
    try:
        conn.execute(
            'INSERT OR IGNORE INTO report_registry(request_digest, knowledge_seq, '
            'report_sha256, output_path, created_at) VALUES (?, ?, ?, ?, ?)',
            (request_digest, report['knowledge_seq'], report['report_sha256'],
             str(dest), _now()))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return reused


def run_report(conn, request_data, output):
    """Pin the version, build a verified replica, compute and publish the report."""
    request = validate_request(request_data)
    # Pin the committed version and its immutable log summary first.
    conn.execute('BEGIN')
    try:
        current_max = max_committed_seq(conn)
        if request['knowledge_seq'] == 'latest':
            seq = current_max
        else:
            seq = request['knowledge_seq']
        if seq < 1 or seq > current_max:
            raise RequestError(
                f'knowledge_seq {seq} exceeds the latest committed seq {current_max}')
        batch = conn.execute(
            'SELECT log_digest FROM batches WHERE knowledge_seq = ?', (seq,)).fetchone()
        event_count = conn.execute(
            'SELECT COALESCE(SUM(event_count), 0) FROM batches WHERE knowledge_seq <= ?',
            (seq,)).fetchone()[0]
        pinned = {'log_digest': batch['log_digest'], 'event_count': event_count}
    finally:
        conn.rollback()
    view = ensure_projection(conn, seq)
    replica, replica_path, _ = make_replica(
        conn, seq, view['view_digest'], pinned['log_digest'], pinned['event_count'],
        view['effective_count'])
    try:
        report, request_digest, _, _ = compute_report(
            conn, replica, seq, request, pinned,
            view['view_digest'], view['effective_count'])
    finally:
        replica.close()
        for suffix in ('', '-wal', '-shm'):
            stale = Path(str(replica_path) + suffix)
            if stale.exists():
                stale.unlink()
    reused = publish_report(conn, output, report, request_digest)
    return report, reused
