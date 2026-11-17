"""Grouped decision policies under a shared false-positive budget.

Pure standard library.  This link never reads or writes the revision ledger or
the monitor database; it consumes one complete JSON request and publishes one
deterministic, self-verifying grouped decision policy for downstream scoring
services.

Two separated responsibilities live here:

* :func:`generate_candidates` -- candidate generation: for one cohort, every
  distinct exact :class:`~fractions.Fraction` predicted probability plus the
  reject-all sentinel (strictly greater than 1), each with exact positive
  coverage / false accepts and the local qualified mark.
* :func:`select_combination` -- combination selection: an exact dynamic
  program over cohorts (states are merged only when the used false-accept
  weight is identical) that picks exactly one locally qualified candidate per
  cohort under the shared budget: maximum total positive coverage, then
  minimum total false accepts, then the lexicographically smallest threshold
  vector (cohorts in lexicographic order, thresholds ascending, sentinel
  largest).
"""
from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from fractions import Fraction
from pathlib import Path

from .common import ident, number, require, rows, save
from .isotonic import fit
from .model import observations

MAX_ROWS = 20000
SCORE_MAX = 1_000_000
WEIGHT_MAX = 1_000_000
MIN_POSITIVE_WEIGHT_MAX = 10 ** 9
BUDGET_MAX = 10 ** 9
SCHEMA_VERSION = 5
SHA_FIELD = 'policy_sha256'
# Serialization of the reject-all sentinel: strictly greater than 1, so it
# accepts nothing (every predicted probability is a Fraction in [0, 1]).
SENTINEL = '>1'
# Sentinel sort key: the flag dominates, placing it above every real threshold.
_SENTINEL_KEY = (1, Fraction(2))


class PolicyError(ValueError):
    """A policy request violates the policy contract or cannot be published."""


# --------------------------------------------------------------------------- #
# Deterministic encoding
# --------------------------------------------------------------------------- #

def canonical_bytes(obj):
    """Compact UTF-8 JSON with sorted keys and no whitespace."""
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':'),
                      sort_keys=True).encode('utf-8')


def policy_digest(body):
    """SHA256 over the policy body with the self-hash field removed."""
    stripped = {key: value for key, value in body.items() if key != SHA_FIELD}
    return hashlib.sha256(canonical_bytes(stripped)).hexdigest()


# --------------------------------------------------------------------------- #
# Strict validation
# --------------------------------------------------------------------------- #

def _bounded_int(value, name, lower, upper):
    # bool is a subclass of int; the contract explicitly forbids booleans here.
    if type(value) is not int or not lower <= value <= upper:
        raise PolicyError(f'{name} must be an integer in [{lower}, {upper}]')
    return value


def _evaluation_rows(value):
    result, seen = [], set()
    for raw in rows(value, MAX_ROWS):
        require(raw, 'id cohort score label weight')
        rid = ident(raw['id'])
        cohort = ident(raw['cohort'])
        score = number(raw['score'], 0, SCORE_MAX)
        label = number(raw['label'], 0, 1)
        weight = number(raw['weight'], 1, WEIGHT_MAX)
        if rid in seen:
            raise PolicyError('duplicate evaluation id: ' + rid)
        seen.add(rid)
        result.append({'id': rid, 'cohort': cohort, 'score': score,
                       'label': label, 'weight': weight})
    return result, seen


def validate_request(data):
    """Validate and normalize a policy request; input array order is erased."""
    require(data, 'training evaluation cohort_rules false_positive_budget')

    training, training_ids = observations(data['training'])
    if not training:
        raise PolicyError('training must be nonempty')
    training.sort(key=lambda row: row['id'])

    evaluation, evaluation_ids = _evaluation_rows(data['evaluation'])
    if training_ids & evaluation_ids:
        raise PolicyError('training/evaluation identity overlap')
    evaluation.sort(key=lambda row: row['id'])

    rules_raw = data['cohort_rules']
    if not isinstance(rules_raw, dict):
        raise PolicyError(
            'cohort_rules must be a cohort -> min_positive_weight mapping')
    rules = {}
    for cohort, minimum in rules_raw.items():
        rules[ident(cohort)] = _bounded_int(
            minimum, 'min_positive_weight', 1, MIN_POSITIVE_WEIGHT_MAX)

    # Every evaluation cohort must be governed by exactly one rule; rules may
    # also cover cohorts without evaluation rows (those only have the sentinel
    # candidate and can never meet a positive minimum).
    for row in evaluation:
        if row['cohort'] not in rules:
            raise PolicyError(
                f"evaluation cohort {row['cohort']} has no cohort rule")

    budget = _bounded_int(
        data['false_positive_budget'], 'false_positive_budget', 0, BUDGET_MAX)
    return {
        'training': training,
        'evaluation': evaluation,
        'cohort_rules': {name: rules[name] for name in sorted(rules)},
        'false_positive_budget': budget,
    }


def request_digest(request):
    """Canonical semantic digest of the normalized request.

    Both sample arrays are already sorted by id and every value is an exact
    integer or string, so reordering the input arrays cannot change the digest
    (and therefore cannot change the published policy bytes).
    """
    return hashlib.sha256(canonical_bytes(request)).hexdigest()


# --------------------------------------------------------------------------- #
# Candidate generation (exact, per cohort)
# --------------------------------------------------------------------------- #

def _probability_at(blocks, score):
    """Left-closed staircase prediction, identical to the v1/annotation rule."""
    starts = [block['lower'] for block in blocks]
    index = max(0, bisect_right(starts, score) - 1)
    return Fraction(blocks[index]['probability'])


def _candidate(key, threshold, accepted, min_positive_weight):
    coverage = sum(row['weight'] for row in accepted if row['label'] == 1)
    false_accepts = sum(row['weight'] for row in accepted if row['label'] == 0)
    return {
        'threshold': threshold,
        'positive_coverage': coverage,
        'false_accepts': false_accepts,
        'accepted_ids': sorted(row['id'] for row in accepted),
        'qualified': coverage >= min_positive_weight,
        '_key': key,
    }


def generate_candidates(rows, min_positive_weight):
    """Every distinct exact threshold of one cohort plus the reject-all sentinel.

    ``rows`` are the cohort's evaluation rows with an exact ``_probability``
    Fraction attached.  Acceptance is left-closed: a row is accepted exactly
    when its probability is >= the threshold, so the sentinel (strictly
    greater than 1) accepts nothing.  Each candidate carries its exact
    positive coverage, false accepts, sorted accepted ids and the local
    qualified mark (coverage >= min_positive_weight).  No floats are involved
    anywhere.
    """
    candidates = []
    for threshold in sorted({row['_probability'] for row in rows}):
        accepted = [row for row in rows if row['_probability'] >= threshold]
        candidates.append(_candidate(
            (0, threshold), str(threshold), accepted, min_positive_weight))
    candidates.append(
        _candidate(_SENTINEL_KEY, SENTINEL, [], min_positive_weight))
    return candidates


# --------------------------------------------------------------------------- #
# Combination selection (exact dynamic program over the shared budget)
# --------------------------------------------------------------------------- #

def select_combination(cohorts, budget):
    """Pick exactly one locally qualified candidate per cohort, exactly.

    ``cohorts`` is a list of ``{'cohort', 'min_positive_weight', 'candidates'}``
    in lexicographic cohort order.  Exact dynamic program over cohorts: a state
    is the exact used false-accept weight, and states with different used
    weight are never merged.  Within one state only the best partial is kept:
    higher coverage always wins (coverage dominates the objective), and equal
    coverage keeps the lexicographically smaller threshold vector -- a prefix
    comparison, so appending any identical suffix preserves the winner.  States
    above the shared budget are dropped (false accepts never decrease, so they
    can never become feasible again); no coverage-bound pruning is applied, so
    a lexicographically better tie is never discarded.  Returns ``(chosen,
    total_coverage, total_false_accepts)`` or None when no combination of
    locally qualified candidates fits the budget.
    """
    states = {0: (0, ())}  # used false accepts -> (coverage, threshold keys)
    for cohort in cohorts:
        qualified = [c for c in cohort['candidates'] if c['qualified']]
        nxt = {}
        for used, (coverage, vector) in states.items():
            for cand in qualified:
                total_f = used + cand['false_accepts']
                if total_f > budget:
                    continue
                total_c = coverage + cand['positive_coverage']
                new_vector = vector + (cand['_key'],)
                current = nxt.get(total_f)
                if current is None or total_c > current[0] or (
                        total_c == current[0] and new_vector < current[1]):
                    nxt[total_f] = (total_c, new_vector)
        states = nxt
        if not states:
            return None
    best = None  # (coverage, used false accepts, threshold key vector)
    for used, (coverage, vector) in states.items():
        if best is None or coverage > best[0] or (
                coverage == best[0] and used < best[1]) or (
                coverage == best[0] and used == best[1] and vector < best[2]):
            best = (coverage, used, vector)
    total_coverage, total_false_accepts, vector = best
    chosen = []
    for cohort, key in zip(cohorts, vector):
        chosen.append(next(
            cand for cand in cohort['candidates']
            if cand['qualified'] and cand['_key'] == key))
    return chosen, total_coverage, total_false_accepts


# --------------------------------------------------------------------------- #
# Policy assembly
# --------------------------------------------------------------------------- #

def _infeasible(request, digest, kind, cohort_gaps=None, budget_gap=None):
    return {
        'schema_version': SCHEMA_VERSION,
        'status': 'infeasible',
        'source_sha256': digest,
        'false_positive_budget': request['false_positive_budget'],
        'infeasibility': {
            'kind': kind,
            'cohort_gaps': cohort_gaps or {},
            'budget_gap': budget_gap,
        },
    }


def _assemble_policy(request, digest, cohorts, chosen, total_coverage,
                     total_false_accepts):
    entries = []
    for cohort, cand in zip(cohorts, chosen):
        entries.append({
            'cohort': cohort['cohort'],
            'min_positive_weight': cohort['min_positive_weight'],
            'threshold': cand['threshold'],
            'positive_coverage': cand['positive_coverage'],
            'false_accepts': cand['false_accepts'],
            'accepted_ids': list(cand['accepted_ids']),
        })
    body = {
        'schema_version': SCHEMA_VERSION,
        'status': 'feasible',
        'source_sha256': digest,
        'false_positive_budget': request['false_positive_budget'],
        'cohorts': entries,
        'totals': {
            'positive_coverage': total_coverage,
            'false_accepts': total_false_accepts,
        },
        'objective': {
            'positive_coverage': total_coverage,
            'false_accepts': total_false_accepts,
            'threshold_vector': [entry['threshold'] for entry in entries],
        },
    }
    body[SHA_FIELD] = policy_digest(body)
    return body


def build_policy(data):
    """Validate the request and return a feasible policy or infeasible receipt.

    A feasible result is a complete, self-verifying policy; an infeasible
    result names the honest cohort gaps or budget gap and never stitches
    per-cohort local optima into a fake success.
    """
    request = validate_request(data)
    digest = request_digest(request)
    budget = request['false_positive_budget']
    blocks = fit(request['training'])
    by_cohort = {name: [] for name in request['cohort_rules']}
    for row in request['evaluation']:
        probability = _probability_at(blocks, row['score'])
        by_cohort[row['cohort']].append({**row, '_probability': probability})
    cohorts = []
    for name in sorted(request['cohort_rules']):
        minimum = request['cohort_rules'][name]
        cohorts.append({
            'cohort': name,
            'min_positive_weight': minimum,
            'candidates': generate_candidates(by_cohort[name], minimum),
        })

    # Local feasibility first: every cohort must reach its lower bound.
    gaps = {}
    for cohort in cohorts:
        best = max(c['positive_coverage'] for c in cohort['candidates'])
        if best < cohort['min_positive_weight']:
            gaps[cohort['cohort']] = cohort['min_positive_weight'] - best
    if gaps:
        return _infeasible(request, digest, 'cohort_gap', cohort_gaps=gaps)

    selected = select_combination(cohorts, budget)
    if selected is None:
        # The cheapest locally qualified combination already exceeds the
        # budget; the gap is exact because cohort minima are independent.
        minimum_false_accepts = sum(
            min(c['false_accepts'] for c in cohort['candidates']
                if c['qualified'])
            for cohort in cohorts)
        return _infeasible(request, digest, 'budget_gap',
                           budget_gap=minimum_false_accepts - budget)

    chosen, total_coverage, total_false_accepts = selected
    return _assemble_policy(request, digest, cohorts, chosen,
                            total_coverage, total_false_accepts)


# --------------------------------------------------------------------------- #
# Publication (idempotent, conflicting, atomic)
# --------------------------------------------------------------------------- #

def _policy_payload(policy):
    return (json.dumps(policy, sort_keys=True, ensure_ascii=False,
                       allow_nan=False, indent=2) + '\n').encode('utf-8')


def publish_policy(output, policy):
    """Atomically publish; identical existing output is idempotent, mismatch conflicts."""
    dest = Path(output)
    if dest.is_symlink():
        raise PolicyError('output path cannot be a symlink')
    if dest.exists():
        if dest.read_bytes() != _policy_payload(policy):
            raise PolicyError(
                f'existing output {dest} differs; refusing to overwrite')
        return True
    save(dest, policy)  # same deterministic serialization, atomic replace
    return False
