DESCRIPTION = 'Weighted isotonic score calibration with independent evaluation'
import argparse
import json
import sys
from pathlib import Path

from . import annotation, ledger, monitor, policy
from .common import read, save
from .model import analyze

# Ledger batches/requests are not bound by the legacy 8 MB single-shot analysis cap.
LEDGER_JSON_LIMIT = 256 * 1024 * 1024
# Annotation packages/answers are small bounded contracts.
ANNOTATION_JSON_LIMIT = 8 * 1024 * 1024
# Monitor snapshots carry two arrays of at most 20000 rows each.
MONITOR_JSON_LIMIT = 64 * 1024 * 1024
# Policy requests are single-shot JSON like the legacy analysis.
POLICY_JSON_LIMIT = 8 * 1024 * 1024

_BUSINESS_ERRORS = (
    ValueError, TypeError, KeyError, OSError, OverflowError, RecursionError,
    ledger.LedgerError, annotation.AnnotationError,
    monitor.MonitorError, policy.PolicyError,
)


def run(source):
    data, digest = read(source)
    return {'schema_version': 1, 'source_sha256': digest, 'result': analyze(data)}


def _load_json(path):
    return read(path, byte_limit=LEDGER_JSON_LIMIT)


def _ingest(args):
    data, raw_sha256 = _load_json(args.batch)
    conn = ledger.connect(args.ledger)
    try:
        receipt = ledger.ingest(conn, data, raw_sha256)
    finally:
        conn.close()
    print(json.dumps(receipt, sort_keys=True, ensure_ascii=False))
    return 0


def _report(args):
    data, _ = _load_json(args.request)
    conn = ledger.connect(args.ledger)
    try:
        report, reused = ledger.run_report(conn, data, args.output)
    finally:
        conn.close()
    print(json.dumps(
        {'knowledge_seq': report['knowledge_seq'],
         'report_sha256': report['report_sha256'],
         'idempotent_existing_output': reused},
        sort_keys=True, ensure_ascii=False))
    return 0


def _annotation_plan(args):
    data, _ = read(args.source, byte_limit=ANNOTATION_JSON_LIMIT)
    package = annotation.build_plan_package(data)
    save(args.output, package)
    print(json.dumps({
        'status': package['status'],
        'package_sha256': package[annotation.SHA_FIELD],
        'task_count': len(package['tasks']),
        'budget_used': package['usage']['budget_used'],
        'group_coverage': package['group_coverage'],
        'gaps': package['gaps'],
    }, sort_keys=True, ensure_ascii=False))
    return 0


def _annotation_resolve(args):
    # Read and parse both inputs completely before any output is touched so that an
    # invalid batch or tampered package always preserves the previous output file,
    # including when --output is the same path as --package.
    package_data, _ = read(args.package, byte_limit=ANNOTATION_JSON_LIMIT)
    answers_data, _ = read(args.answers, byte_limit=ANNOTATION_JSON_LIMIT)
    package, accepted = annotation.resolve_package(package_data, answers_data)
    save(args.output, package)
    print(json.dumps({
        'status': package['status'],
        'package_sha256': package[annotation.SHA_FIELD],
        'accepted_answers': accepted,
        'task_count': len(package['tasks']),
        'truth_count': len(package['truth']),
        'budget_used': package['usage']['budget_used'],
        'group_coverage': package['group_coverage'],
        'gaps': package['gaps'],
    }, sort_keys=True, ensure_ascii=False))
    # A legal but infeasible refill is a successful resolution (exit 0); the package
    # stays usable for collecting the remaining answers.
    return 0


def _monitor_sync(args):
    data, _ = read(args.snapshot, byte_limit=MONITOR_JSON_LIMIT)
    conn = monitor.connect(args.db)
    try:
        receipt = monitor.sync(conn, data)
    finally:
        conn.close()
    print(json.dumps(receipt, sort_keys=True, ensure_ascii=False))
    return 0


def _monitor_claim(args):
    conn = monitor.connect(args.db)
    try:
        result = monitor.claim(conn, args.worker, args.now, args.lease_seconds)
    finally:
        conn.close()
    # An empty valid queue exits 0 and prints JSON null.
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0


def _monitor_ack(args):
    conn = monitor.connect(args.db)
    try:
        receipt = monitor.ack(conn, args.notice, args.token, args.now)
    finally:
        conn.close()
    print(json.dumps(receipt, sort_keys=True, ensure_ascii=False))
    return 0


def _monitor_status(args):
    conn = monitor.connect(args.db)
    try:
        report = monitor.status_report(conn, args.output)
    finally:
        conn.close()
    print(json.dumps({
        'revision': report['revision'],
        'output': str(args.output),
        'notice_count': len(report['notices']),
        'episode_count': len(report['episodes']),
    }, sort_keys=True, ensure_ascii=False))
    return 0


def _policy(args):
    data, _ = read(args.source, byte_limit=POLICY_JSON_LIMIT)
    outcome = policy.build_policy(data)
    if outcome['status'] == 'infeasible':
        # A legal but infeasible request is a successful run (exit 0): no
        # policy is published and any previous output file is preserved.
        print(json.dumps(outcome, sort_keys=True, ensure_ascii=False))
        return 0
    reused = policy.publish_policy(args.output, outcome)
    print(json.dumps({
        'status': 'feasible',
        'source_sha256': outcome['source_sha256'],
        'policy_sha256': outcome[policy.SHA_FIELD],
        'cohort_count': len(outcome['cohorts']),
        'total_positive_coverage': outcome['totals']['positive_coverage'],
        'total_false_accepts': outcome['totals']['false_accepts'],
        'idempotent_existing_output': reused,
    }, sort_keys=True, ensure_ascii=False))
    return 0


def _legacy(args):
    if args.source.resolve() == args.output.resolve():
        raise ValueError('input and output must differ')
    report = run(args.source)
    save(args.output, report)
    print(json.dumps(report, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    sub = parser.add_subparsers(dest='command')

    ingest_parser = sub.add_parser(
        'ingest', help='append a revision batch to the SQLite ledger')
    ingest_parser.add_argument('--ledger', type=Path, required=True)
    ingest_parser.add_argument('--batch', type=Path, required=True)
    ingest_parser.set_defaults(func=_ingest)

    report_parser = sub.add_parser(
        'report', help='build a versioned calibration report from the ledger')
    report_parser.add_argument('--ledger', type=Path, required=True)
    report_parser.add_argument('--request', type=Path, required=True)
    report_parser.add_argument('--output', type=Path, required=True)
    report_parser.set_defaults(func=_report)

    plan_parser = sub.add_parser(
        'annotation-plan',
        help='build a reusable double-blind human annotation package')
    plan_parser.add_argument('--source', type=Path, required=True)
    plan_parser.add_argument('--output', type=Path, required=True)
    plan_parser.set_defaults(func=_annotation_plan)

    resolve_parser = sub.add_parser(
        'annotation-resolve',
        help='collect annotation answers and run constrained refill')
    resolve_parser.add_argument('--package', type=Path, required=True)
    resolve_parser.add_argument('--answers', type=Path, required=True)
    resolve_parser.add_argument('--output', type=Path, required=True)
    resolve_parser.set_defaults(func=_annotation_resolve)

    sync_parser = sub.add_parser(
        'monitor-sync', help='atomically publish a drift snapshot revision')
    sync_parser.add_argument('--db', type=Path, required=True)
    sync_parser.add_argument('--snapshot', type=Path, required=True)
    sync_parser.set_defaults(func=_monitor_sync)

    claim_parser = sub.add_parser(
        'monitor-claim', help='lease the first available current notification')
    claim_parser.add_argument('--db', type=Path, required=True)
    claim_parser.add_argument('--worker', required=True)
    claim_parser.add_argument('--now', type=int, required=True)
    claim_parser.add_argument('--lease-seconds', type=int, required=True)
    claim_parser.set_defaults(func=_monitor_claim)

    ack_parser = sub.add_parser(
        'monitor-ack', help='confirm delivery of a leased notification')
    ack_parser.add_argument('--db', type=Path, required=True)
    ack_parser.add_argument('--notice', type=int, required=True)
    ack_parser.add_argument('--token', required=True)
    ack_parser.add_argument('--now', type=int, required=True)
    ack_parser.set_defaults(func=_monitor_ack)

    status_parser = sub.add_parser(
        'monitor-status', help='publish a consistent status JSON report')
    status_parser.add_argument('--db', type=Path, required=True)
    status_parser.add_argument('--output', type=Path, required=True)
    status_parser.set_defaults(func=_monitor_status)

    policy_parser = sub.add_parser(
        'policy',
        help='build a grouped decision policy under a shared false-positive '
             'budget')
    policy_parser.add_argument('--source', type=Path, required=True)
    policy_parser.add_argument('--output', type=Path, required=True)
    policy_parser.set_defaults(func=_policy)

    # Legacy single-shot JSON analysis (unchanged contract).
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path)

    args = parser.parse_args(argv)
    try:
        if args.command in ('ingest', 'report', 'annotation-plan',
                            'annotation-resolve', 'monitor-sync', 'monitor-claim',
                            'monitor-ack', 'monitor-status', 'policy'):
            return args.func(args)
        if args.source and args.output:
            return _legacy(args)
        parser.error('use ingest, report, annotation-plan, annotation-resolve, '
                     'monitor-sync, monitor-claim, monitor-ack, monitor-status, '
                     'policy, or --source/--output')
    except _BUSINESS_ERRORS as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
