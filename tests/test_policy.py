"""Acceptance tests for the grouped decision policy link (v5).

Coverage (mapped to the policy contract):
  * strict request validation (unknown fields, duplicate keys, bool-as-int,
    cohort rules, budget bounds, id uniqueness/overlap, nonempty training)
  * candidate generation: left-closed equality acceptance, PAVA plateau tied
    thresholds, reject-all sentinel, exact coverage/false-accept sums, local
    qualified marks, boundary probabilities 0 and 1 (no clipping)
  * combination selection: shared budget, global objective order, lexicographic
    threshold-vector tie-break, cohort gaps and budget gaps
  * exhaustive independent oracle over random instances (both feasible and
    infeasible outcomes, compared in full)
  * linked workflow: per-cohort max-coverage policies exceed the shared budget;
    a one-row label/weight correction removes a cohort's original optimal
    candidate and forces the other cohort to re-select its threshold; local
    per-cohort optima are infeasible while a suboptimal combination is feasible
  * determinism (input reordering), self-check digest, idempotent/conflicting
    publication, symlink rejection, infeasible runs preserving previous output,
    no ledger/monitor imports
"""
import copy
import itertools
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from bisect import bisect_right
from fractions import Fraction
from pathlib import Path

from score_calibration import policy
from score_calibration.isotonic import fit
from score_calibration.policy import (
    PolicyError, build_policy, generate_candidates, select_combination,
    validate_request,
)

ROOT = Path(__file__).resolve().parents[1]
CLI = [sys.executable, '-m', 'score_calibration.cli']
ENV = {**os.environ, 'PYTHONPATH': str(ROOT / 'src')}

# PAVA blocks for this training set: 10 -> 0, 20..30 -> 1/2, 40 -> 1.
TRAINING = [
    {'id': 't1', 'score': 10, 'label': 0, 'weight': 1},
    {'id': 't2', 'score': 20, 'label': 1, 'weight': 1},
    {'id': 't3', 'score': 30, 'label': 0, 'weight': 1},
    {'id': 't4', 'score': 40, 'label': 1, 'weight': 1},
]


def make_request(evaluation, rules, budget, training=None):
    return {
        'training': copy.deepcopy(TRAINING if training is None else training),
        'evaluation': evaluation,
        'cohort_rules': rules,
        'false_positive_budget': budget,
    }


def row(rid, cohort, score, label, weight):
    return {'id': rid, 'cohort': cohort, 'score': score,
            'label': label, 'weight': weight}


# Scenario A: per-cohort max-coverage picks (a:'1/2' f=3, b:'1/2' f=4) sum to
# f=7 and exceed the shared budget 4; the exact optimum tightens cohort a.
SCENARIO_A_EVAL = [
    row('a1', 'a', 25, 1, 4),   # p = 1/2
    row('a2', 'a', 25, 0, 3),   # p = 1/2
    row('a3', 'a', 45, 1, 2),   # p = 1
    row('b1', 'b', 25, 1, 5),   # p = 1/2
    row('b2', 'b', 25, 0, 4),   # p = 1/2
    row('b3', 'b', 45, 1, 1),   # p = 1
]
SCENARIO_A_RULES = {'a': 2, 'b': 1}
SCENARIO_A_BUDGET = 4


def scenario_a():
    return make_request(SCENARIO_A_EVAL, SCENARIO_A_RULES, SCENARIO_A_BUDGET)


def scenario_b():
    """Scenario A with one weight correction: a3 weight 2 -> 1."""
    evaluation = copy.deepcopy(SCENARIO_A_EVAL)
    evaluation[2]['weight'] = 1
    return make_request(evaluation, SCENARIO_A_RULES, SCENARIO_A_BUDGET)


def cohort_entry(result, name):
    return next(c for c in result['cohorts'] if c['cohort'] == name)


# --------------------------------------------------------------------------- #
# Independent exhaustive oracle
# --------------------------------------------------------------------------- #

def oracle(request):
    """Enumerate every combination of locally qualified candidates.

    Independent of the module's dynamic program: itertools.product over the
    per-cohort qualified lists, then the plain lexicographic objective.
    """
    req = validate_request(request)
    blocks = fit(req['training'])
    starts = [block['lower'] for block in blocks]

    def predict(score):
        index = max(0, bisect_right(starts, score) - 1)
        return Fraction(blocks[index]['probability'])

    names = sorted(req['cohort_rules'])
    per_cohort = {}
    for name in names:
        rows = [r for r in req['evaluation'] if r['cohort'] == name]
        probs = {r['id']: predict(r['score']) for r in rows}
        candidates = []
        for threshold in sorted(set(probs.values())):
            accepted = [r for r in rows if probs[r['id']] >= threshold]
            candidates.append({
                'threshold': str(threshold),
                'key': (0, threshold),
                'cov': sum(r['weight'] for r in accepted if r['label'] == 1),
                'fa': sum(r['weight'] for r in accepted if r['label'] == 0),
                'ids': tuple(sorted(r['id'] for r in accepted)),
            })
        candidates.append({'threshold': '>1', 'key': (1, Fraction(2)),
                           'cov': 0, 'fa': 0, 'ids': ()})
        minimum = req['cohort_rules'][name]
        per_cohort[name] = (
            [c for c in candidates if c['cov'] >= minimum], candidates)

    gaps = {}
    for name in names:
        qualified, all_candidates = per_cohort[name]
        if not qualified:
            best = max(c['cov'] for c in all_candidates)
            gaps[name] = req['cohort_rules'][name] - best
    if gaps:
        return ('cohort_gap', gaps)

    budget = req['false_positive_budget']
    lists = [per_cohort[name][0] for name in names]
    best = None
    for combo in itertools.product(*lists):
        total_f = sum(c['fa'] for c in combo)
        if total_f > budget:
            continue
        total_c = sum(c['cov'] for c in combo)
        vector = tuple(c['key'] for c in combo)
        if best is None or (total_c, -total_f) > (best[0], -best[1]) or (
                total_c == best[0] and total_f == best[1]
                and vector < best[2]):
            best = (total_c, total_f, vector, combo)
    if best is None:
        cheapest = sum(min(c['fa'] for c in per_cohort[name][0])
                       for name in names)
        return ('budget_gap', cheapest - budget)
    return ('feasible', best)


def assert_matches_oracle(testcase, request, result):
    expected = oracle(request)
    if expected[0] == 'cohort_gap':
        testcase.assertEqual(result['status'], 'infeasible')
        testcase.assertEqual(result['infeasibility']['kind'], 'cohort_gap')
        testcase.assertEqual(
            result['infeasibility']['cohort_gaps'], expected[1])
        testcase.assertIsNone(result['infeasibility']['budget_gap'])
        return
    if expected[0] == 'budget_gap':
        testcase.assertEqual(result['status'], 'infeasible')
        testcase.assertEqual(result['infeasibility']['kind'], 'budget_gap')
        testcase.assertEqual(
            result['infeasibility']['budget_gap'], expected[1])
        testcase.assertEqual(result['infeasibility']['cohort_gaps'], {})
        return
    _, (total_c, total_f, _, combo) = expected
    testcase.assertEqual(result['status'], 'feasible')
    testcase.assertEqual(result['totals'], {
        'positive_coverage': total_c, 'false_accepts': total_f})
    testcase.assertEqual(
        [entry['threshold'] for entry in result['cohorts']],
        [cand['threshold'] for cand in combo])
    for entry, cand in zip(result['cohorts'], combo):
        testcase.assertEqual(entry['positive_coverage'], cand['cov'])
        testcase.assertEqual(entry['false_accepts'], cand['fa'])
        testcase.assertEqual(entry['accepted_ids'], list(cand['ids']))


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

class ValidationTests(unittest.TestCase):
    def bad(self, request):
        with self.assertRaises((ValueError, TypeError)):
            build_policy(request)

    def test_unknown_and_missing_fields(self):
        req = scenario_a()
        req['extra'] = 1
        self.bad(req)
        req = scenario_a()
        del req['cohort_rules']
        self.bad(req)

    def test_training_contract(self):
        req = make_request([], {}, 0, training=[])
        self.bad(req)  # training must be nonempty
        req = scenario_a()
        req['training'][0]['label'] = True  # bool is not an integer
        self.bad(req)
        req = scenario_a()
        req['training'][1]['id'] = 't1'  # duplicate training id
        self.bad(req)
        req = scenario_a()
        req['training'][0]['weight'] = 0  # weights are positive
        self.bad(req)

    def test_evaluation_contract(self):
        req = scenario_a()
        req['evaluation'][0]['extra'] = 1
        self.bad(req)
        req = scenario_a()
        del req['evaluation'][0]['cohort']
        self.bad(req)
        req = scenario_a()
        req['evaluation'][1]['id'] = 'a1'  # duplicate evaluation id
        self.bad(req)
        req = scenario_a()
        req['evaluation'][0]['id'] = 't1'  # training/evaluation overlap
        self.bad(req)
        req = scenario_a()
        req['evaluation'][0]['weight'] = -2
        self.bad(req)
        req = scenario_a()
        req['evaluation'][0]['label'] = 2
        self.bad(req)

    def test_cohort_rules_contract(self):
        req = scenario_a()
        req['cohort_rules'] = ['a']
        self.bad(req)
        for bad_value in (0, -1, True, '2', 2.5, 10 ** 9 + 1):
            req = scenario_a()
            req['cohort_rules']['a'] = bad_value
            self.bad(req)
        req = scenario_a()
        req['evaluation'][0]['cohort'] = 'zzz'  # cohort without a rule
        self.bad(req)

    def test_budget_contract(self):
        for bad_value in (-1, True, '4', 4.5, 10 ** 9 + 1):
            req = scenario_a()
            req['false_positive_budget'] = bad_value
            self.bad(req)

    def test_rule_without_rows_is_cohort_gap(self):
        # A rule covering a cohort with no evaluation rows only has the
        # sentinel candidate (coverage 0), so its positive minimum is missed.
        result = build_policy(make_request([], {'g': 3}, 0))
        self.assertEqual(result['status'], 'infeasible')
        self.assertEqual(result['infeasibility'], {
            'kind': 'cohort_gap', 'cohort_gaps': {'g': 3},
            'budget_gap': None})

    def test_empty_request_is_feasible(self):
        result = build_policy(make_request([], {}, 0))
        self.assertEqual(result['status'], 'feasible')
        self.assertEqual(result['cohorts'], [])
        self.assertEqual(result['totals'], {
            'positive_coverage': 0, 'false_accepts': 0})
        self.assertEqual(result['objective']['threshold_vector'], [])


# --------------------------------------------------------------------------- #
# Candidate generation
# --------------------------------------------------------------------------- #

class CandidateTests(unittest.TestCase):
    def test_left_closed_equality_acceptance(self):
        rows = [
            {'id': 'e1', 'cohort': 'g', 'score': 30, 'label': 1, 'weight': 2,
             '_probability': Fraction(1, 2)},
            {'id': 'e2', 'cohort': 'g', 'score': 40, 'label': 0, 'weight': 3,
             '_probability': Fraction(1)},
        ]
        candidates = generate_candidates(rows, 1)
        self.assertEqual([c['threshold'] for c in candidates],
                         ['1/2', '1', '>1'])
        # A row whose probability equals the threshold is accepted (left-closed).
        self.assertEqual(candidates[0]['accepted_ids'], ['e1', 'e2'])
        self.assertEqual(candidates[1]['accepted_ids'], ['e2'])
        self.assertEqual(candidates[2]['accepted_ids'], [])
        # Exact integer sums over accepted rows.
        self.assertEqual(
            [(c['positive_coverage'], c['false_accepts']) for c in candidates],
            [(2, 3), (0, 3), (0, 0)])
        self.assertEqual([c['qualified'] for c in candidates],
                         [True, False, False])

    def test_plateau_rows_share_one_threshold(self):
        # Scores 20, 25 and 30 all land on the same PAVA block (p = 1/2):
        # one candidate threshold, not three.
        rows = [
            {'id': f'e{i}', 'cohort': 'g', 'score': score, 'label': label,
             'weight': 1, '_probability': Fraction(1, 2)}
            for i, (score, label) in enumerate([(20, 1), (25, 0), (30, 1)])]
        candidates = generate_candidates(rows, 2)
        self.assertEqual([c['threshold'] for c in candidates], ['1/2', '>1'])
        self.assertEqual(candidates[0]['accepted_ids'], ['e0', 'e1', 'e2'])
        self.assertEqual(candidates[0]['positive_coverage'], 2)
        self.assertEqual(candidates[0]['false_accepts'], 1)
        self.assertTrue(candidates[0]['qualified'])

    def test_sentinel_rejects_all(self):
        rows = [{'id': 'e1', 'cohort': 'g', 'score': 45, 'label': 1,
                 'weight': 5, '_probability': Fraction(1)}]
        sentinel = generate_candidates(rows, 1)[-1]
        self.assertEqual(sentinel['threshold'], '>1')
        self.assertEqual(sentinel['positive_coverage'], 0)
        self.assertEqual(sentinel['false_accepts'], 0)
        self.assertEqual(sentinel['accepted_ids'], [])
        # min_positive_weight is a positive integer, so the sentinel never
        # qualifies.
        self.assertFalse(sentinel['qualified'])

    def test_boundary_probabilities_are_not_clipped(self):
        # Training labels 0 then 1 give exact probabilities 0 and 1; both are
        # real thresholds and neither is clipped.
        training = [
            {'id': 't1', 'score': 10, 'label': 0, 'weight': 1},
            {'id': 't2', 'score': 20, 'label': 1, 'weight': 1},
        ]
        evaluation = [row('e1', 'g', 10, 1, 1), row('e2', 'g', 20, 0, 1)]
        result = build_policy(make_request(evaluation, {'g': 1}, 0))
        # Only t=0 qualifies (coverage 1, false accepts 1) but it busts the
        # zero budget: exact budget gap 1.
        self.assertEqual(result['status'], 'infeasible')
        self.assertEqual(result['infeasibility'], {
            'kind': 'budget_gap', 'cohort_gaps': {}, 'budget_gap': 1})
        result = build_policy(make_request(evaluation, {'g': 1}, 1))
        entry = cohort_entry(result, 'g')
        self.assertEqual(entry['threshold'], '0')
        self.assertEqual(entry['accepted_ids'], ['e1', 'e2'])

    def test_qualified_boundary_is_inclusive(self):
        rows = [{'id': 'e1', 'cohort': 'g', 'score': 25, 'label': 1,
                 'weight': 3, '_probability': Fraction(1, 2)}]
        # coverage == min_positive_weight qualifies (>=, not >).
        self.assertTrue(generate_candidates(rows, 3)[0]['qualified'])
        self.assertFalse(generate_candidates(rows, 4)[0]['qualified'])


# --------------------------------------------------------------------------- #
# Combination selection (hand-computed)
# --------------------------------------------------------------------------- #

class SelectionTests(unittest.TestCase):
    def test_scenario_a_exact_optimum(self):
        result = build_policy(scenario_a())
        self.assertEqual(result['status'], 'feasible')
        entry_a = cohort_entry(result, 'a')
        entry_b = cohort_entry(result, 'b')
        # Hand-computed: a tightens to t=1 (cov 2, f 0) so that b can keep
        # its coverage-maximizing t=1/2 (cov 6, f 4) within budget 4.
        self.assertEqual(entry_a, {
            'cohort': 'a', 'min_positive_weight': 2, 'threshold': '1',
            'positive_coverage': 2, 'false_accepts': 0,
            'accepted_ids': ['a3']})
        self.assertEqual(entry_b, {
            'cohort': 'b', 'min_positive_weight': 1, 'threshold': '1/2',
            'positive_coverage': 6, 'false_accepts': 4,
            'accepted_ids': ['b1', 'b2', 'b3']})
        self.assertEqual(result['totals'], {
            'positive_coverage': 8, 'false_accepts': 4})
        self.assertEqual(result['objective'], {
            'positive_coverage': 8, 'false_accepts': 4,
            'threshold_vector': ['1', '1/2']})

    def test_local_optima_infeasible_but_suboptimal_combination_feasible(self):
        # Each cohort's own max-coverage candidate: a:'1/2' (f=3), b:'1/2'
        # (f=4); together f=7 > budget 4.  Stitching local optima is
        # infeasible, yet a suboptimal combination exists and must be found
        # (neither truncated local optima nor a false infeasible).
        request = scenario_a()
        req = validate_request(request)
        blocks = fit(req['training'])
        local_f_sum = 0
        for name, minimum in req['cohort_rules'].items():
            rows = [{**r, '_probability': policy._probability_at(
                blocks, r['score'])}
                for r in req['evaluation'] if r['cohort'] == name]
            candidates = generate_candidates(rows, minimum)
            best_local = max(candidates, key=lambda c: c['positive_coverage'])
            local_f_sum += best_local['false_accepts']
        self.assertGreater(local_f_sum, req['false_positive_budget'])
        result = build_policy(request)
        self.assertEqual(result['status'], 'feasible')
        self.assertLessEqual(result['totals']['false_accepts'],
                             req['false_positive_budget'])
        # The feasible optimum is not any cohort's local coverage maximum.
        self.assertEqual(
            [c['threshold'] for c in result['cohorts']], ['1', '1/2'])

    def test_lexicographic_tie_break(self):
        # Two combinations tie on (coverage 4, false accepts 1); the
        # threshold vector ('1/2','1') beats ('1','1/2').
        evaluation = [
            row('x1', 'a', 25, 1, 2), row('x2', 'a', 25, 0, 1),
            row('x3', 'a', 45, 1, 1),
            row('y1', 'b', 25, 1, 2), row('y2', 'b', 25, 0, 1),
            row('y3', 'b', 45, 1, 1),
        ]
        result = build_policy(make_request(evaluation, {'a': 1, 'b': 1}, 1))
        self.assertEqual(result['status'], 'feasible')
        self.assertEqual(result['totals'], {
            'positive_coverage': 4, 'false_accepts': 1})
        self.assertEqual(result['objective']['threshold_vector'], ['1/2', '1'])
        self.assertEqual(cohort_entry(result, 'a')['accepted_ids'],
                         ['x1', 'x2', 'x3'])
        self.assertEqual(cohort_entry(result, 'b')['accepted_ids'], ['y3'])

    def test_coverage_beats_false_accepts_in_objective(self):
        # (cov 8, f 4) must win over (cov 7, f 3): coverage dominates.
        result = build_policy(scenario_a())
        self.assertEqual(result['totals']['positive_coverage'], 8)
        request = make_request(SCENARIO_A_EVAL, {'a': 2, 'b': 2}, 4)
        result = build_policy(request)
        # b's minimum 2 excludes b:'1' (cov 1); only a:'1/2'+b:'1/2' (f=7) and
        # a:'1'+b:'1/2' (f=4, cov 8) remain locally qualified under budget 4.
        self.assertEqual(result['totals'], {
            'positive_coverage': 8, 'false_accepts': 4})

    def test_budget_gap_is_exact(self):
        # Cheapest locally qualified combination: a:'1/2' (f=3) + b:'1/2'
        # (f=4) = 7 > budget 5 -> exact gap 2.
        result = build_policy(
            make_request(SCENARIO_A_EVAL, {'a': 3, 'b': 6}, 5))
        self.assertEqual(result['status'], 'infeasible')
        self.assertEqual(result['infeasibility'], {
            'kind': 'budget_gap', 'cohort_gaps': {}, 'budget_gap': 2})

    def test_cohort_gap_is_exact(self):
        # Cohort a reaches at most coverage 6 < min 7 -> gap 1.
        result = build_policy(
            make_request(SCENARIO_A_EVAL, {'a': 7, 'b': 1}, 9))
        self.assertEqual(result['status'], 'infeasible')
        self.assertEqual(result['infeasibility'], {
            'kind': 'cohort_gap', 'cohort_gaps': {'a': 1},
            'budget_gap': None})

    def test_select_combination_picks_one_qualified_per_cohort(self):
        request = validate_request(scenario_a())
        blocks = fit(request['training'])
        cohorts = []
        for name in sorted(request['cohort_rules']):
            rows = [{**r, '_probability': policy._probability_at(
                blocks, r['score'])}
                for r in request['evaluation'] if r['cohort'] == name]
            cohorts.append({
                'cohort': name,
                'min_positive_weight': request['cohort_rules'][name],
                'candidates': generate_candidates(
                    rows, request['cohort_rules'][name])})
        chosen, total_c, total_f = select_combination(cohorts, 4)
        self.assertEqual(len(chosen), 2)
        self.assertTrue(all(c['qualified'] for c in chosen))
        self.assertEqual((total_c, total_f), (8, 4))
        # Budget 2: cheapest qualified combination costs 0 (a:'1' + b:'1'),
        # so it stays feasible with coverage 3.
        chosen, total_c, total_f = select_combination(cohorts, 2)
        self.assertEqual((total_c, total_f), (3, 0))


# --------------------------------------------------------------------------- #
# Exhaustive oracle over random instances
# --------------------------------------------------------------------------- #

class OracleTests(unittest.TestCase):
    def test_against_exhaustive_oracle(self):
        rng = random.Random(20261117)
        for _ in range(400):
            training = [
                {'id': f't{i}', 'score': score, 'label': rng.randrange(2),
                 'weight': rng.randrange(1, 4)}
                for i, score in enumerate(
                    sorted(rng.sample([10, 20, 30, 40], rng.randrange(1, 5))))]
            names = sorted(rng.sample(['a', 'b', 'c'], rng.randrange(0, 4)))
            rules = {name: rng.randrange(1, 6) for name in names}
            evaluation = []
            for name in names:
                for i in range(rng.randrange(0, 4)):
                    evaluation.append(row(
                        f'{name}{i}', name,
                        rng.choice([10, 20, 25, 30, 40, 45]),
                        rng.randrange(2), rng.randrange(1, 5)))
            request = make_request(
                evaluation, rules, rng.randrange(0, 9), training=training)
            result = build_policy(request)
            assert_matches_oracle(self, request, result)
            if result['status'] != 'feasible':
                continue
            # Self-check digest and hard invariants on every feasible result.
            self.assertEqual(
                result[policy.SHA_FIELD], policy.policy_digest(result))
            for entry in result['cohorts']:
                self.assertGreaterEqual(
                    entry['positive_coverage'], entry['min_positive_weight'])
            self.assertLessEqual(result['totals']['false_accepts'],
                                 request['false_positive_budget'])


# --------------------------------------------------------------------------- #
# Linked workflow: one-row correction re-solves the other cohort
# --------------------------------------------------------------------------- #

class LinkedWorkflowTests(unittest.TestCase):
    def test_correction_forces_joint_reselection(self):
        result_a = build_policy(scenario_a())
        self.assertEqual(result_a['status'], 'feasible')
        self.assertEqual(
            [c['threshold'] for c in result_a['cohorts']], ['1', '1/2'])
        self.assertEqual(result_a['totals'], {
            'positive_coverage': 8, 'false_accepts': 4})

        # The weight correction removes cohort a's original optimal candidate:
        # t='1' drops to coverage 1 < min 2 and leaves the qualified set.
        request_b = scenario_b()
        req_b = validate_request(request_b)
        blocks = fit(req_b['training'])
        rows_a = [{**r, '_probability': policy._probability_at(
            blocks, r['score'])}
            for r in req_b['evaluation'] if r['cohort'] == 'a']
        candidates_a = generate_candidates(rows_a, req_b['cohort_rules']['a'])
        removed = next(c for c in candidates_a if c['threshold'] == '1')
        self.assertFalse(removed['qualified'])
        self.assertEqual(removed['positive_coverage'], 1)

        result_b = build_policy(request_b)
        self.assertEqual(result_b['status'], 'feasible')
        # Both cohorts re-select: a falls back to '1/2' (its only qualified
        # candidate, f=3), which squeezes b off the shared budget down to '1'.
        self.assertEqual(
            [c['threshold'] for c in result_b['cohorts']], ['1/2', '1'])
        self.assertEqual(cohort_entry(result_b, 'a')['accepted_ids'],
                         ['a1', 'a2', 'a3'])
        self.assertEqual(cohort_entry(result_b, 'b')['accepted_ids'], ['b3'])
        # Budget feasibility and the objective order survive the correction.
        self.assertEqual(result_b['totals'], {
            'positive_coverage': 6, 'false_accepts': 3})
        self.assertLessEqual(result_b['totals']['false_accepts'],
                             SCENARIO_A_BUDGET)
        assert_matches_oracle(self, request_b, result_b)
        # Policy, totals and input digest all move together.
        self.assertNotEqual(result_a['totals'], result_b['totals'])
        self.assertNotEqual(result_a['source_sha256'],
                            result_b['source_sha256'])
        self.assertNotEqual(result_a[policy.SHA_FIELD],
                            result_b[policy.SHA_FIELD])


# --------------------------------------------------------------------------- #
# Determinism and self-check
# --------------------------------------------------------------------------- #

class DeterminismTests(unittest.TestCase):
    def test_input_reordering_keeps_policy_bytes(self):
        shuffled = scenario_a()
        shuffled['training'] = shuffled['training'][::-1]
        shuffled['evaluation'] = shuffled['evaluation'][::-1]
        shuffled['cohort_rules'] = {'b': 1, 'a': 2}
        base = build_policy(scenario_a())
        moved = build_policy(shuffled)
        self.assertEqual(base, moved)
        self.assertEqual(base['source_sha256'], moved['source_sha256'])

    def test_self_check_digest_is_externally_recomputable(self):
        result = build_policy(scenario_a())
        stripped = {k: v for k, v in result.items() if k != policy.SHA_FIELD}
        import hashlib
        digest = hashlib.sha256(
            policy.canonical_bytes(stripped)).hexdigest()
        self.assertEqual(result[policy.SHA_FIELD], digest)
        # Any tampering breaks the digest.
        tampered = copy.deepcopy(result)
        tampered['cohorts'][0]['threshold'] = '0'
        self.assertNotEqual(policy.policy_digest(tampered),
                            result[policy.SHA_FIELD])


# --------------------------------------------------------------------------- #
# CLI: publication, idempotency, conflicts, infeasible handling
# --------------------------------------------------------------------------- #

class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(CLI + list(args), capture_output=True,
                              text=True, env=ENV)

    @staticmethod
    def write(path, obj):
        path.write_text(json.dumps(obj, indent=2) + '\n')

    def test_success_then_idempotent_replay(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / 'in.json'
            out = Path(td) / 'policy.json'
            self.write(src, scenario_a())
            first = self.run_cli('policy', '--source', str(src),
                                 '--output', str(out))
            self.assertEqual(first.returncode, 0, first.stderr)
            receipt = json.loads(first.stdout)
            self.assertEqual(receipt['status'], 'feasible')
            self.assertFalse(receipt['idempotent_existing_output'])
            self.assertEqual(receipt['cohort_count'], 2)
            self.assertEqual(receipt['total_positive_coverage'], 8)
            self.assertEqual(receipt['total_false_accepts'], 4)
            published = json.loads(out.read_text())
            self.assertEqual(receipt['policy_sha256'],
                             published[policy.SHA_FIELD])
            self.assertEqual(receipt['source_sha256'],
                             published['source_sha256'])
            payload = out.read_bytes()
            self.assertEqual(payload, policy._policy_payload(published))

            replay = self.run_cli('policy', '--source', str(src),
                                  '--output', str(out))
            self.assertEqual(replay.returncode, 0, replay.stderr)
            self.assertTrue(json.loads(replay.stdout)[
                'idempotent_existing_output'])
            self.assertEqual(out.read_bytes(), payload)

    def test_fixture_matches_hand_computation(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / 'policy.json'
            run = self.run_cli('policy', '--source',
                               str(ROOT / 'fixtures' / 'policy.json'),
                               '--output', str(out))
            self.assertEqual(run.returncode, 0, run.stderr)
            published = json.loads(out.read_text())
            self.assertEqual(
                [c['threshold'] for c in published['cohorts']], ['1', '1/2'])
            self.assertEqual(published['totals'], {
                'positive_coverage': 8, 'false_accepts': 4})

    def test_conflict_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / 'in.json'
            out = Path(td) / 'policy.json'
            self.write(src, scenario_a())
            out.write_bytes(b'{"different": true}\n')
            run = self.run_cli('policy', '--source', str(src),
                               '--output', str(out))
            self.assertEqual(run.returncode, 2)
            self.assertEqual(out.read_bytes(), b'{"different": true}\n')

    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / 'in.json'
            out = Path(td) / 'link.json'
            self.write(src, scenario_a())
            out.symlink_to(src)
            run = self.run_cli('policy', '--source', str(src),
                               '--output', str(out))
            self.assertEqual(run.returncode, 2)
            self.assertTrue(out.is_symlink())

    def test_output_equal_to_source_is_a_conflict_not_an_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / 'in.json'
            self.write(src, scenario_a())
            before = src.read_bytes()
            run = self.run_cli('policy', '--source', str(src),
                               '--output', str(src))
            self.assertEqual(run.returncode, 2)
            self.assertEqual(src.read_bytes(), before)

    def test_infeasible_prints_gaps_and_never_publishes(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / 'in.json'
            out = Path(td) / 'policy.json'
            self.write(src, make_request(SCENARIO_A_EVAL, {'a': 7, 'b': 1}, 9))
            run = self.run_cli('policy', '--source', str(src),
                               '--output', str(out))
            self.assertEqual(run.returncode, 0, run.stderr)
            receipt = json.loads(run.stdout)
            self.assertEqual(receipt['status'], 'infeasible')
            self.assertEqual(receipt['infeasibility']['cohort_gaps'],
                             {'a': 1})
            self.assertFalse(out.exists())

            # A previous successful policy survives a later infeasible run.
            self.write(src, scenario_a())
            run = self.run_cli('policy', '--source', str(src),
                               '--output', str(out))
            self.assertEqual(run.returncode, 0, run.stderr)
            payload = out.read_bytes()
            self.write(src, make_request(SCENARIO_A_EVAL, {'a': 3, 'b': 6}, 5))
            run = self.run_cli('policy', '--source', str(src),
                               '--output', str(out))
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(run.stdout)[
                'infeasibility']['budget_gap'], 2)
            self.assertEqual(out.read_bytes(), payload)

    def test_invalid_input_preserves_output(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / 'in.json'
            out = Path(td) / 'policy.json'
            out.write_text('prior')
            bad_bodies = [
                '{}',
                '{"training": [], "training": []}',
                json.dumps({**scenario_a(), 'unknown': 1}),
                json.dumps(make_request(SCENARIO_A_EVAL, {'a': 0}, 4)),
                json.dumps(make_request(SCENARIO_A_EVAL,
                                        {'a': 2, 'b': 1}, -1)),
                json.dumps(make_request(SCENARIO_A_EVAL, {'a': True}, 4)),
            ]
            for body in bad_bodies:
                src.write_text(body)
                run = self.run_cli('policy', '--source', str(src),
                                   '--output', str(out))
                self.assertEqual(run.returncode, 2, body)
                self.assertEqual(out.read_text(), 'prior')

    def test_reordered_input_publishes_identical_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            src_a = Path(td) / 'a.json'
            src_b = Path(td) / 'b.json'
            out_a = Path(td) / 'a.out.json'
            out_b = Path(td) / 'b.out.json'
            shuffled = scenario_a()
            shuffled['training'] = shuffled['training'][::-1]
            shuffled['evaluation'] = shuffled['evaluation'][::-1]
            self.write(src_a, scenario_a())
            self.write(src_b, shuffled)
            self.assertEqual(self.run_cli(
                'policy', '--source', str(src_a), '--output', str(out_a)
            ).returncode, 0)
            self.assertEqual(self.run_cli(
                'policy', '--source', str(src_b), '--output', str(out_b)
            ).returncode, 0)
            self.assertEqual(out_a.read_bytes(), out_b.read_bytes())

    def test_module_never_imports_ledger_or_monitor(self):
        code = (
            'import sys\n'
            'sys.path.insert(0, {!r})\n'
            'import score_calibration.policy\n'
            'leaked = [m for m in sys.modules\n'
            '          if m.startswith(("score_calibration.ledger",\n'
            '                           "score_calibration.monitor"))]\n'
            'sys.exit(1 if leaked else 0)\n'
        ).format(str(ROOT / 'src'))
        run = subprocess.run([sys.executable, '-c', code],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)


if __name__ == '__main__':
    unittest.main()
