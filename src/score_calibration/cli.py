DESCRIPTION = 'Weighted isotonic score calibration with independent evaluation'
import argparse
import json
import sys
from pathlib import Path

from . import annotation, ledger
from .common import read, save
from .model import analyze

# Ledger batches/requests are not bound by the legacy 8 MB single-shot analysis cap.
LEDGER_JSON_LIMIT = 256 * 1024 * 1024
# Annotation packages/answers are small bounded contracts.
ANNOTATION_JSON_LIMIT = 8 * 1024 * 1024

_BUSINESS_ERRORS = (
    ValueError, TypeError, KeyError, OSError, OverflowError, RecursionError,
    ledger.LedgerError, annotation.AnnotationError,
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

    # Legacy single-shot JSON analysis (unchanged contract).
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path)

    args = parser.parse_args(argv)
    try:
        if args.command in ('ingest', 'report', 'annotation-plan',
                            'annotation-resolve'):
            return args.func(args)
        if args.source and args.output:
            return _legacy(args)
        parser.error('use ingest, report, annotation-plan, annotation-resolve, '
                     'or --source/--output')
    except _BUSINESS_ERRORS as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
