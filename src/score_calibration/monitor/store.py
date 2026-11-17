"""SQLite persistence, atomic sync reconciliation, leases and acks.

Concurrency model
-----------------
Episode computation is pure and runs outside any transaction.  The commit, however,
runs inside ``BEGIN IMMEDIATE`` and re-reads the *entire* notification/delivery state
under the write lock, then verifies a fingerprint that binds every table a competing
claim/ack/sync could touch.  Checking only the snapshot revision is insufficient: an
ack changes leases/notices/delivery (and therefore which notifications are required)
without advancing the snapshot revision.  On fingerprint mismatch the transaction is
rolled back and the cheap reconciliation retries, so stale out-of-transaction
computation can never overwrite a newer committed state.

Notifications are never physically deleted: supersession flips ``active`` to 0 and
records ``cancelled_revision``; leases are append-only and retain their final state.
"""
from __future__ import annotations

import json
import secrets
import sqlite3
from pathlib import Path

from ..common import ident, save
from .contract import MonitorError, content_digest, frozen_config, validate_snapshot
from .engine import compute

SCHEMA_VERSION = 1
MAX_SYNC_ATTEMPTS = 50

_DDL = (
    """
    CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )""",
    """
    CREATE TABLE IF NOT EXISTS snapshots (
        revision INTEGER PRIMARY KEY CHECK (revision > 0),
        snapshot_id TEXT NOT NULL UNIQUE,
        content_digest TEXT NOT NULL,
        config_json TEXT NOT NULL,
        receipt_json TEXT NOT NULL
    )""",
    """
    CREATE TABLE IF NOT EXISTS windows (
        revision INTEGER NOT NULL,
        cohort TEXT NOT NULL,
        window INTEGER NOT NULL CHECK (window >= 0),
        start_date TEXT NOT NULL,
        end_date TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('eligible', 'insufficient')),
        total_weight INTEGER NOT NULL,
        residual TEXT,
        PRIMARY KEY (revision, cohort, window)
    )""",
    """
    CREATE TABLE IF NOT EXISTS episodes (
        revision INTEGER NOT NULL,
        cohort TEXT NOT NULL,
        start_window INTEGER NOT NULL CHECK (start_window >= 0),
        end_window INTEGER,
        open INTEGER NOT NULL CHECK (open IN (0, 1)),
        peak_window INTEGER NOT NULL,
        peak_value TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        PRIMARY KEY (revision, cohort, start_window)
    )""",
    """
    CREATE TABLE IF NOT EXISTS notices (
        notice_id INTEGER PRIMARY KEY AUTOINCREMENT,
        cohort TEXT NOT NULL,
        start_window INTEGER NOT NULL,
        action TEXT NOT NULL CHECK (action IN ('upsert', 'retract')),
        payload_json TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN
            ('pending', 'leased', 'sent', 'cancelled')),
        active INTEGER NOT NULL CHECK (active IN (0, 1)),
        created_revision INTEGER NOT NULL,
        cancelled_revision INTEGER,
        lease_until INTEGER
    )""",
    """
    CREATE TABLE IF NOT EXISTS leases (
        lease_id INTEGER PRIMARY KEY AUTOINCREMENT,
        notice_id INTEGER NOT NULL REFERENCES notices(notice_id),
        worker TEXT NOT NULL,
        token TEXT NOT NULL UNIQUE,
        leased_at INTEGER NOT NULL,
        lease_until INTEGER NOT NULL,
        state TEXT NOT NULL CHECK (state IN
            ('active', 'expired', 'revoked', 'acked'))
    )""",
    """
    CREATE TABLE IF NOT EXISTS delivery (
        cohort TEXT NOT NULL,
        start_window INTEGER NOT NULL,
        last_sent_notice_id INTEGER NOT NULL,
        last_sent_payload_json TEXT NOT NULL,
        acked_at INTEGER NOT NULL,
        token TEXT NOT NULL,
        PRIMARY KEY (cohort, start_window)
    )""",
    """
    CREATE TABLE IF NOT EXISTS ack_receipts (
        token TEXT PRIMARY KEY,
        receipt_json TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_notices_claim ON notices(active, notice_id)",
    "CREATE INDEX IF NOT EXISTS idx_leases_notice ON leases(notice_id, lease_id)",
)


# --------------------------------------------------------------------------- #
# Connection
# --------------------------------------------------------------------------- #

def connect(path):
    db = Path(path)
    if db.is_symlink():
        raise MonitorError('database path cannot be a symlink')
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    conn.isolation_level = None  # explicit BEGIN/COMMIT only
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA foreign_keys=ON')
    conn.execute('PRAGMA busy_timeout=10000')
    for statement in _DDL:
        conn.execute(statement)
    conn.execute(
        'INSERT OR IGNORE INTO meta(key, value) VALUES(?, ?)',
        ('schema_version', str(SCHEMA_VERSION)))
    return conn


def _non_negative_int(value, name):
    if type(value) is not int or value < 0:
        raise MonitorError(f'{name} must be a nonnegative integer')
    return value


def _bounded_lease_seconds(value):
    if type(value) is not int or not 1 <= value <= 86400:
        raise MonitorError('lease_seconds must be an integer in [1, 86400]')
    return value


# --------------------------------------------------------------------------- #
# Fingerprint over every table that can change required notifications
# --------------------------------------------------------------------------- #

def _state_fingerprint(conn):
    # An ack moves leased -> sent and upserts delivery without touching the snapshot
    # revision; a claim appends a lease; a sync appends notices.  All of them must be
    # visible here, which is why revision alone is never enough.
    snapshot_count = conn.execute(
        'SELECT COUNT(*) FROM snapshots').fetchone()[0]
    max_notice = conn.execute(
        'SELECT COALESCE(MAX(notice_id), 0) FROM notices').fetchone()[0]
    max_lease = conn.execute(
        'SELECT COALESCE(MAX(lease_id), 0) FROM leases').fetchone()[0]
    delivery_count, max_delivery_notice = conn.execute(
        'SELECT COUNT(*), COALESCE(MAX(last_sent_notice_id), 0) FROM delivery'
    ).fetchone()
    active_count = conn.execute(
        'SELECT COUNT(*) FROM notices WHERE active = 1').fetchone()[0]
    return (snapshot_count, max_notice, max_lease, delivery_count,
            max_delivery_notice, active_count)


# --------------------------------------------------------------------------- #
# Sync: atomic episode publication and notification reconciliation
# --------------------------------------------------------------------------- #

def sync(conn, data, barrier=None, on_before_commit=None):
    """Validate, compute and atomically publish a snapshot revision.

    Exact replays (same ``snapshot_id`` and identical canonical content) return the
    original persisted receipt without advancing the revision or touching notices.
    The same id with different content is rejected outright.  ``barrier`` (if given)
    fires after the out-of-transaction computation but before the write transaction,
    letting tests interleave a competing connection deterministically.
    """
    snapshot = validate_snapshot(data)
    digest = content_digest(snapshot)
    result = compute(snapshot)
    episodes = result['episodes']
    windows = result['windows']

    config = frozen_config(snapshot)
    for _ in range(MAX_SYNC_ATTEMPTS):
        # The barrier deliberately runs before the fingerprint and BEGIN IMMEDIATE:
        # a competing connection must be able to commit while the computed revision
        # waits outside any lock.
        if barrier is not None and _ == 0:
            barrier('computed', snapshot)
        binding = _state_fingerprint(conn)
        conn.execute('BEGIN IMMEDIATE')
        try:
            existing = conn.execute(
                'SELECT revision, content_digest, receipt_json FROM snapshots '
                'WHERE snapshot_id = ?', (snapshot['snapshot_id'],)).fetchone()
            if existing is not None:
                if existing['content_digest'] != digest:
                    raise MonitorError(
                        'snapshot_id already exists with different canonical content; '
                        'the whole request is rejected')
                conn.rollback()
                receipt = json.loads(existing['receipt_json'])
                receipt['replay'] = True
                return receipt

            stored_config = conn.execute(
                "SELECT value FROM meta WHERE key = 'config'").fetchone()
            if stored_config is not None \
                    and json.loads(stored_config['value']) != config:
                raise MonitorError(
                    'monitor configuration is fixed after the first successful sync; '
                    'only snapshot_id/reference/observations may change')

            if _state_fingerprint(conn) != binding:
                # A competing claim/ack/sync committed while we were outside the
                # transaction: discard this reconciliation and retry on fresh state.
                conn.rollback()
                continue

            revision = conn.execute(
                'SELECT COALESCE(MAX(revision), 0) + 1 FROM snapshots').fetchone()[0]
            changes = _publish_revision(
                conn, snapshot, digest, config, revision, windows, episodes,
                on_before_commit)
            receipt = {
                'snapshot_id': snapshot['snapshot_id'],
                'revision': revision,
                'content_digest': digest,
                'replay': False,
                'episode_count': len(episodes),
                'notices_created': changes['created'],
                'notices_cancelled': changes['cancelled'],
            }
            conn.execute(
                'UPDATE snapshots SET receipt_json = ? WHERE revision = ?',
                (json.dumps(receipt, sort_keys=True, ensure_ascii=False), revision))
            conn.commit()
            return receipt
        except Exception:
            conn.rollback()
            raise
    raise MonitorError('sync retried too many times against competing writers')


def _publish_revision(conn, snapshot, digest, config, revision, windows,
                      episodes, on_before_commit):
    if revision == 1:
        conn.execute(
            'INSERT INTO meta(key, value) VALUES(?, ?)',
            ('config', json.dumps(config, sort_keys=True, ensure_ascii=False)))
    conn.execute(
        'INSERT INTO snapshots(revision, snapshot_id, content_digest, config_json, '
        'receipt_json) VALUES(?, ?, ?, ?, ?)',
        (revision, snapshot['snapshot_id'], digest,
         json.dumps(config, sort_keys=True, ensure_ascii=False), '{}'))
    conn.executemany(
        'INSERT INTO windows(revision, cohort, window, start_date, end_date, status, '
        'total_weight, residual) VALUES(?, ?, ?, ?, ?, ?, ?, ?)',
        [(revision, w['cohort'], w['window'], w['start_date'], w['end_date'],
          w['status'], w['total_weight'], w['residual']) for w in windows])
    conn.executemany(
        'INSERT INTO episodes(revision, cohort, start_window, end_window, open, '
        'peak_window, peak_value, payload_json) VALUES(?, ?, ?, ?, ?, ?, ?, ?)',
        [(revision, e['cohort'], e['start'],
          e['end'], 1 if e['open'] else 0, e['peak_window'], e['peak_value'],
          json.dumps(e['payload'], sort_keys=True, ensure_ascii=False))
         for e in episodes])
    changes = _reconcile_notices(conn, revision, episodes)
    if on_before_commit is not None:
        on_before_commit(conn, revision)
    return changes


def _latest_notices(conn):
    rows = conn.execute(
        'SELECT n.* FROM notices n JOIN ('
        'SELECT cohort, start_window, MAX(notice_id) AS max_id FROM notices '
        'GROUP BY cohort, start_window) m '
        'ON n.notice_id = m.max_id').fetchall()
    return {(r['cohort'], r['start_window']): r for r in rows}


def _delivery_map(conn):
    return {
        (r['cohort'], r['start_window']): r
        for r in conn.execute(
            'SELECT * FROM delivery ORDER BY cohort, start_window')}


def _cancel_notice(conn, notice, revision):
    """Cancel a pending/leased notice; its lease (if any) is immediately invalid."""
    conn.execute(
        'UPDATE notices SET active = 0, status = ?, cancelled_revision = ? '
        'WHERE notice_id = ?', ('cancelled', revision, notice['notice_id']))
    conn.execute(
        "UPDATE leases SET state = 'revoked' WHERE notice_id = ? AND state = 'active'",
        (notice['notice_id'],))


def _create_notice(conn, revision, identity, action, payload):
    conn.execute(
        'INSERT INTO notices(notice_id, cohort, start_window, action, payload_json, '
        'status, active, created_revision, cancelled_revision, lease_until) '
        'VALUES (NULL, ?, ?, ?, ?, ?, 1, ?, NULL, NULL)',
        (identity[0], identity[1], action,
         json.dumps(payload, sort_keys=True, ensure_ascii=False), 'pending',
         revision))
    return conn.execute('SELECT last_insert_rowid()').fetchone()[0]


def _reconcile_notices(conn, revision, episodes):
    """Drive notice upsert/retract/cancel from the freshly computed episode set.

    Processed in canonical (cohort, start window) order so notice ids are monotonic
    with a deterministic identity order.  Decision inputs are the latest notice of
    each identity and its actual last-delivered payload (the ``delivery`` table,
    updated only by successful acks) -- never the most recently computed state.
    """
    latest = _latest_notices(conn)
    delivered = _delivery_map(conn)
    current = {
        (e['cohort'], e['start']): e for e in episodes}
    # A cohort that still has any current episode is present even when its start
    # window moved (payload correction).  Such a replacement never owes a retract for
    # the old identity; only the cohort's genuine disappearance does.
    cohorts_present = {e['cohort'] for e in episodes}
    identities = sorted(set(current) | set(latest) | set(delivered))

    # Phase 1: retire stale *active* notices for identities that no longer have an
    # episode, without touching immutable sent history incorrectly:
    #  * pending/leased notices were never delivered -> cancel silently, no retract;
    #  * a sent upsert whose cohort still has another episode -> keep 'sent' history,
    #    just mark inactive (the replacement episode reconciles on its own identity);
    #  * a sent upsert whose whole cohort disappeared -> leave active so phase 2
    #    emits the required retract.
    for identity in identities:
        if identity in current:
            continue
        notice = latest.get(identity)
        if notice is None or not notice['active']:
            continue
        if notice['action'] == 'retract':
            continue  # required retract stays current until acked or revoked
        cohort, _start = identity
        if notice['status'] in ('pending', 'leased'):
            _cancel_notice(conn, notice, revision)
        elif cohort in cohorts_present:
            conn.execute(
                'UPDATE notices SET active = 0 WHERE notice_id = ?',
                (notice['notice_id'],))

    # Refresh the view after phase-1 deactivation.
    latest = _latest_notices(conn)
    created = cancelled = 0

    for identity in identities:
        episode = current.get(identity)
        notice = latest.get(identity)
        sent = delivered.get(identity)
        if episode is not None:
            created_now, cancelled_now = _reconcile_present(
                conn, revision, identity, episode['payload'], notice, sent)
        elif notice is not None and notice['active']:
            created_now, cancelled_now = _reconcile_absent(
                conn, revision, identity, notice, sent)
        else:
            created_now, cancelled_now = 0, 0
        created += created_now
        cancelled += cancelled_now
    return {'created': created, 'cancelled': cancelled}


def _reconcile_present(conn, revision, identity, payload, notice, sent):
    if notice is None or not notice['active']:
        # First appearance, or only cancelled history exists.  A prior delivery with
        # an identical payload means the episode came back exactly as last sent: the
        # consumer needs nothing new.
        if sent is not None and sent['last_sent_payload_json'] == \
                json.dumps(payload, sort_keys=True, ensure_ascii=False):
            return 0, 0
        _create_notice(conn, revision, identity, 'upsert', payload)
        return 1, 0

    if notice['action'] == 'upsert' and notice['status'] in ('pending', 'leased'):
        if notice['payload_json'] == json.dumps(
                payload, sort_keys=True, ensure_ascii=False):
            return 0, 0  # same required notification already pending/leased
        # Unsent replacement: cancel the old lease/notice and publish the new payload.
        _cancel_notice(conn, notice, revision)
        _create_notice(conn, revision, identity, 'upsert', payload)
        return 1, 1

    if notice['action'] == 'upsert' and notice['status'] == 'sent':
        if notice['payload_json'] == json.dumps(
                payload, sort_keys=True, ensure_ascii=False):
            return 0, 0
        # The old payload was delivered and stays in history; the correction is a new
        # upsert -- no retract, because the episode is still present.
        conn.execute(
            'UPDATE notices SET active = 0 WHERE notice_id = ?',
            (notice['notice_id'],))
        _create_notice(conn, revision, identity, 'upsert', payload)
        return 1, 0

    # A retract (pending/leased or already sent) is outstanding while the episode is
    # back: revoke the retract and compare against the last actually sent payload.
    if notice['status'] in ('pending', 'leased'):
        _cancel_notice(conn, notice, revision)
        cancelled_now = 1
    else:
        conn.execute(
            'UPDATE notices SET active = 0 WHERE notice_id = ?',
            (notice['notice_id'],))
        cancelled_now = 0
    if sent is not None and sent['last_sent_payload_json'] == json.dumps(
            payload, sort_keys=True, ensure_ascii=False):
        return 0, cancelled_now
    _create_notice(conn, revision, identity, 'upsert', payload)
    return 1, cancelled_now


def _reconcile_absent(conn, revision, identity, notice, sent):
    if notice is None or not notice['active']:
        return 0, 0

    if notice['action'] == 'retract':
        # The required retract already exists: pending/leased keeps waiting, a sent
        # retract stays as the delivered terminal state.
        return 0, 0

    # notice.action == 'upsert'
    if notice['status'] in ('pending', 'leased'):
        # The notice's own status decides delivery: a pending/leased upsert was never
        # acknowledged even if an older payload of the same episode was sent. Such an
        # unsent replacement (B after A was sent) therefore disappears silently -- the
        # consumer still holds the last sent payload and a later reappearance is
        # reconciled against the delivery record, not by emitting a retract here.
        _cancel_notice(conn, notice, revision)
        return 0, 1

    # status == 'sent': the consumer has the payload and must now be told it is gone.
    conn.execute(
        'UPDATE notices SET active = 0 WHERE notice_id = ?',
        (notice['notice_id'],))
    _create_notice(
        conn, revision, identity, 'retract',
        json.loads(notice['payload_json']))
    return 1, 0


# --------------------------------------------------------------------------- #
# Claim: first available current notice by notice id
# --------------------------------------------------------------------------- #

def claim(conn, worker, now, lease_seconds):
    worker = ident(worker)
    _non_negative_int(now, 'now')
    _bounded_lease_seconds(lease_seconds)
    conn.execute('BEGIN IMMEDIATE')
    try:
        row = conn.execute(
            'SELECT * FROM notices WHERE active = 1 AND ('
            "status = 'pending' OR (status = 'leased' AND lease_until <= ?)) "
            'ORDER BY notice_id LIMIT 1', (now,)).fetchone()
        if row is None:
            conn.commit()
            return None
        lease_until = now + lease_seconds
        if row['status'] == 'leased':
            # The previous lease has expired; retire it before re-leasing.
            conn.execute(
                "UPDATE leases SET state = 'expired' WHERE notice_id = ? "
                "AND state = 'active'", (row['notice_id'],))
        token = _unique_token(conn)
        conn.execute(
            'INSERT INTO leases(lease_id, notice_id, worker, token, leased_at, '
            'lease_until, state) VALUES (NULL, ?, ?, ?, ?, ?, ?)',
            (row['notice_id'], worker, token, now, lease_until, 'active'))
        conn.execute(
            "UPDATE notices SET status = 'leased', lease_until = ? "
            'WHERE notice_id = ?', (lease_until, row['notice_id']))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return {
        'notice_id': row['notice_id'],
        'cohort': row['cohort'],
        'start_window': row['start_window'],
        'action': row['action'],
        'payload': json.loads(row['payload_json']),
        'worker': worker,
        'token': token,
        'lease_until': lease_until,
    }


def _unique_token(conn):
    for _ in range(8):
        token = secrets.token_urlsafe(32)
        if conn.execute(
                'SELECT 1 FROM leases WHERE token = ?', (token,)).fetchone() is None:
            return token
    raise MonitorError('could not allocate a unique lease token')


# --------------------------------------------------------------------------- #
# Ack: idempotent delivery confirmation bound to a live lease
# --------------------------------------------------------------------------- #

def ack(conn, notice_id, token, now):
    if type(notice_id) is not int or notice_id < 1:
        raise MonitorError('notice must be a positive integer id')
    if not isinstance(token, str) or not token:
        raise MonitorError('token must be a nonempty string')
    _non_negative_int(now, 'now')

    # Replay is checked before any transaction: a successful ack always returns its
    # original receipt, regardless of the current lease state or the supplied now.
    stored = conn.execute(
        'SELECT receipt_json FROM ack_receipts WHERE token = ?',
        (token,)).fetchone()
    if stored is not None:
        return json.loads(stored['receipt_json'])

    conn.execute('BEGIN IMMEDIATE')
    try:
        lease = conn.execute(
            'SELECT * FROM leases WHERE token = ?', (token,)).fetchone()
        if lease is None:
            raise MonitorError('invalid token: no such lease')
        notice = conn.execute(
            'SELECT * FROM notices WHERE notice_id = ?',
            (lease['notice_id'],)).fetchone()
        if notice is None or notice['notice_id'] != notice_id:
            raise MonitorError('token does not lease the requested notice')
        if lease['state'] != 'active':
            raise MonitorError('lease is no longer valid')
        if not notice['active'] or notice['status'] != 'leased':
            raise MonitorError('notice is no longer a valid leased notification')
        if now >= lease['lease_until']:
            raise MonitorError('lease has expired')

        conn.execute(
            "UPDATE leases SET state = 'acked' WHERE lease_id = ?",
            (lease['lease_id'],))
        conn.execute(
            "UPDATE notices SET status = 'sent' WHERE notice_id = ?",
            (notice_id,))
        if notice['action'] == 'upsert':
            conn.execute(
                'INSERT INTO delivery(cohort, start_window, last_sent_notice_id, '
                'last_sent_payload_json, acked_at, token) VALUES(?, ?, ?, ?, ?, ?) '
                'ON CONFLICT(cohort, start_window) DO UPDATE SET '
                'last_sent_notice_id = excluded.last_sent_notice_id, '
                'last_sent_payload_json = excluded.last_sent_payload_json, '
                'acked_at = excluded.acked_at, token = excluded.token',
                (notice['cohort'], notice['start_window'], notice_id,
                 notice['payload_json'], now, token))
        else:
            # A retract confirms removal: remember the retract delivery itself but
            # preserve the last payload so a later identical reappearance is silent.
            conn.execute(
                'UPDATE delivery SET last_sent_notice_id = ?, acked_at = ?, '
                'token = ? WHERE cohort = ? AND start_window = ?',
                (notice_id, now, token, notice['cohort'],
                 notice['start_window']))
        receipt = {
            'notice_id': notice_id,
            'token': token,
            'action': notice['action'],
            'cohort': notice['cohort'],
            'start_window': notice['start_window'],
            'payload': json.loads(notice['payload_json']),
            'acked_at': now,
            'status': 'sent',
        }
        conn.execute(
            'INSERT INTO ack_receipts(token, receipt_json) VALUES(?, ?)',
            (token, json.dumps(receipt, sort_keys=True, ensure_ascii=False)))
        conn.commit()
        return receipt
    except Exception:
        conn.rollback()
        raise


# --------------------------------------------------------------------------- #
# Status: one consistent read snapshot, atomically published as sorted JSON
# --------------------------------------------------------------------------- #

def _build_status(conn):
    # A single deferred read transaction gives one WAL snapshot; episodes and
    # notifications can never be read from different revisions.
    conn.execute('BEGIN')
    try:
        current = conn.execute(
            'SELECT COALESCE(MAX(revision), 0) FROM snapshots').fetchone()[0]
        snapshot_row = conn.execute(
            'SELECT snapshot_id, config_json FROM snapshots WHERE revision = ?',
            (current,)).fetchone() if current else None
        windows = [dict(r) for r in conn.execute(
            'SELECT cohort, window, start_date, end_date, status, total_weight, '
            'residual FROM windows WHERE revision = ? '
            'ORDER BY cohort, window', (current,))]
        episodes = [{
            'cohort': r['cohort'], 'start': r['start_window'],
            'end': r['end_window'], 'open': bool(r['open']),
            'peak_window': r['peak_window'], 'peak_value': r['peak_value'],
            'payload': json.loads(r['payload_json']),
        } for r in conn.execute(
            'SELECT * FROM episodes WHERE revision = ? '
            'ORDER BY cohort, start_window', (current,))]
        notices = [{
            'notice_id': r['notice_id'], 'cohort': r['cohort'],
            'start_window': r['start_window'], 'action': r['action'],
            'payload': json.loads(r['payload_json']), 'status': r['status'],
            'active': bool(r['active']), 'created_revision': r['created_revision'],
            'cancelled_revision': r['cancelled_revision'],
            'lease_until': r['lease_until'],
        } for r in conn.execute(
            'SELECT * FROM notices ORDER BY notice_id')]
        leases = [dict(r) for r in conn.execute(
            'SELECT lease_id, notice_id, worker, token, leased_at, lease_until, '
            'state FROM leases ORDER BY lease_id')]
        delivery = [{
            'cohort': r['cohort'], 'start_window': r['start_window'],
            'last_sent_notice_id': r['last_sent_notice_id'],
            'last_sent_payload': json.loads(r['last_sent_payload_json']),
            'acked_at': r['acked_at'], 'token': r['token'],
        } for r in conn.execute(
            'SELECT * FROM delivery ORDER BY cohort, start_window')]
        ack_receipts = [{
            'token': r['token'], 'receipt': json.loads(r['receipt_json']),
        } for r in conn.execute(
            'SELECT * FROM ack_receipts ORDER BY token')]
        receipts = [{
            'revision': r['revision'], 'snapshot_id': r['snapshot_id'],
            'content_digest': r['content_digest'],
            'receipt': json.loads(r['receipt_json']),
        } for r in conn.execute(
            'SELECT * FROM snapshots ORDER BY revision')]
    finally:
        conn.rollback()  # read-only snapshot; rollback simply ends it
    return {
        'revision': current,
        'snapshot_id': snapshot_row['snapshot_id'] if snapshot_row else None,
        'config': json.loads(snapshot_row['config_json'])
        if snapshot_row else None,
        'windows': windows,
        'episodes': episodes,
        'notices': notices,
        'leases': leases,
        'delivery': delivery,
        'ack_receipts': ack_receipts,
        'receipts': receipts,
    }


def status_report(conn, output):
    report = _build_status(conn)
    if Path(output).is_symlink():
        raise MonitorError('status output path cannot be a symlink')
    save(output, report)  # temp file atomic replace
    return report
