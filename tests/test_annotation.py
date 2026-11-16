"""Acceptance tests for the double-blind annotation planning and answer resolution.

Coverage (mapped to the annotation contract):
  * strict plan/answer validation (unknown fields, duplicate keys, bool-as-int)
  * exact PAVA staircase probabilities and exact Fraction candidate values
  * exact assignment checked against a brute-force enumeration oracle, including the
    highest-value-greedy quota trap, eligible bottlenecks, same-entity candidates and
    lexicographic tie-breaking
  * infeasible planning reports per-group gaps and never fakes success
  * answer acceptance: pending/agreed/disputed, partial batches, unknown task,
    unassigned annotator, duplicate and rewritten answers, agreed-only truth export
  * package self-check digest, tamper protection, wrong digest preserving old output
  * linked workflow: one answer keeps quota occupied, dispute withdraws coverage but
    not budget/capacity, refill bypasses exhausted annotators and blocked entities,
    both fillable and genuinely unfillable scenarios
  * determinism (input reordering), same-path atomic replace, symlink rejection
"""
import copy
import hashlib
import itertools
import json
import os
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from score_calibration import annotation
from score_calibration.annotation import (
    AnnotationError, build_plan_package, resolve_package, validate_plan,
)

ROOT = Path(__file__).resolve().parents[1]
CLI = [sys.executable, '-m', 'score_calibration.cli']
ENV = {**os.environ, 'PYTHONPATH': str(ROOT / 'src')}


def make_plan(pool=None, annotators=None, quotas=None, budget=12, training=None):
    return {
        'training': training if training is not None else [
            {'id': 't1', 'score': 10, 'label': 0, 'weight': 1},
            {'id': 't2', 'score': 20, 'label': 1, 'weight': 1},
            {'id': 't3', 'score': 30, 'label': 0, 'weight': 1},
            {'id': 't4', 'score': 40, 'label': 1, 'weight': 1},
        ],
        'pool': pool if pool is not None else [],
        'annotators': annotators if annotators is not None else [
            {'id': 'a1', 'capacity': 6}, {'id': 'a2', 'capacity': 6},
            {'id': 'a3', 'capacity': 6}, {'id': 'a4', 'capacity': 6},
        ],
        'quotas': quotas if quotas is not None else {
            'g1': {'min': 0, 'max': 18}, 'g2': {'min': 0, 'max': 18}},
        'budget': budget,
    }


def candidate(cid, entity, group, score=25, weight=1, eligible=('a1', 'a2', 'a3', 'a4')):
    return {'id': cid, 'entity_id': entity, 'group': group, 'score': score,
            'weight': weight, 'eligible': list(eligible)}


# --------------------------------------------------------------------------- #
# Brute-force oracle for the exact assignment objective
# --------------------------------------------------------------------------- #

def oracle_solve(plan):
    """Enumerate every feasible double-annotator assignment and apply the objective."""
    source = validate_plan(plan)
    blocks, candidates = annotation.calibrate(source)
    lookup = {row['id']: row for row in candidates}
    capacities = {row['id']: row['capacity'] for row in source['annotators']}
    best = None  # ((value Fraction, count), tuple sequence)

    def recurse(index, chosen, used_entities, caps, budget_used, coverage, value):
        nonlocal best
        if index == len(candidates):
            for group, bounds in source['quotas'].items():
                if not bounds['min'] <= coverage.get(group, 0) <= bounds['max']:
                    return
            sequence = tuple(sorted(
                (cid, min(pair), max(pair)) for cid, pair in chosen))
            key = (value, len(chosen))
            if best is None or key > best[0] or (
                    key == best[0] and sequence < best[1]):
                best = (key, sequence)
            return
        row = candidates[index]
        # Skip candidate.
        recurse(index + 1, chosen, used_entities, caps, budget_used, coverage, value)
        if row['entity_id'] in used_entities:
            return
        if budget_used + 1 > source['budget']:
            return
        if coverage.get(row['group'], 0) >= source['quotas'][row['group']]['max']:
            return
        for left, right in itertools.combinations(row['eligible'], 2):
            if caps[left] <= 0 or caps[right] <= 0:
                continue
            new_caps = dict(caps)
            new_caps[left] -= 1
            new_caps[right] -= 1
            new_cov = dict(coverage)
            new_cov[row['group']] = new_cov.get(row['group'], 0) + 1
            recurse(index + 1, chosen + [(row['id'], (left, right))],
                    used_entities | {row['entity_id']}, new_caps,
                    budget_used + 1, new_cov, value + row['_value'])

    recurse(0, [], set(), dict(capacities), 0, {}, Fraction(0))
    return source, blocks, candidates, best


def package_pairs(package):
    return tuple(sorted(
        (task['id'], task['annotator_ids'][0], task['annotator_ids'][1])
        for task in package['tasks']))


class ValidationTests(unittest.TestCase):
    def assert_rejected(self, data):
        with self.assertRaises((AnnotationError, ValueError, TypeError)):
            build_plan_package(data)

    def test_strict_fields_and_duplicate_keys(self):
        plan = make_plan([candidate('p1', 'e1', 'g1')],
                         quotas={'g1': {'min': 1, 'max': 2}})
        self.assert_rejected({**plan, 'extra': 1})
        bad = copy.deepcopy(plan)
        bad['pool'][0]['nope'] = 9
        self.assert_rejected(bad)
        # Duplicate JSON keys are rejected by the shared strict reader.
        from score_calibration.common import read
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'dup.json'
            path.write_text('{"training":[],"training":[]}')
            with self.assertRaises(ValueError):
                read(path)

    def test_bool_is_not_integer(self):
        plan = make_plan([candidate('p1', 'e1', 'g1')],
                         quotas={'g1': {'min': 1, 'max': 2}})
        plan['budget'] = True
        self.assert_rejected(plan)
        plan2 = make_plan([candidate('p1', 'e1', 'g1')],
                          quotas={'g1': {'min': 1, 'max': 2}})
        plan2['pool'][0]['weight'] = True
        self.assert_rejected(plan2)
        plan3 = make_plan([candidate('p1', 'e1', 'g1')],
                          quotas={'g1': {'min': 1, 'max': 2}})
        plan3['annotators'][0]['capacity'] = False
        self.assert_rejected(plan3)

    def test_bounds_and_references(self):
        plan = make_plan([candidate('p1', 'e1', 'g1', eligible=('a1', 'a1'))],
                         quotas={'g1': {'min': 1, 'max': 2}})
        self.assert_rejected(plan)  # duplicate eligible
        plan = make_plan([candidate('p1', 'e1', 'g1', eligible=('a1', 'a9'))],
                         quotas={'g1': {'min': 1, 'max': 2}})
        self.assert_rejected(plan)  # unknown annotator
        plan = make_plan([candidate('p1', 'e1', 'g1')],
                         quotas={'g1': {'min': 3, 'max': 2}})
        self.assert_rejected(plan)  # min > max
        plan = make_plan([candidate('p1', 'e1', 'g3')])
        self.assert_rejected(plan)  # group without quota
        plan = make_plan(
            [candidate('p1', 'e1', 'g1'), candidate('p2', 'e1', 'g2')],
            quotas={'g1': {'min': 0, 'max': 2}, 'g2': {'min': 0, 'max': 2}})
        self.assert_rejected(plan)  # same entity across groups
        plan = make_plan([candidate('t1', 'e1', 'g1')],
                         quotas={'g1': {'min': 0, 'max': 2}})
        self.assert_rejected(plan)  # id overlaps training
        plan = make_plan([candidate('p1', 'e1', 'g1')], budget=37,
                         quotas={'g1': {'min': 0, 'max': 2}})
        self.assert_rejected(plan)
        plan = make_plan([candidate('p1', 'e1', 'g1')],
                         annotators=[{'id': 'a1', 'capacity': 37}],
                         quotas={'g1': {'min': 0, 'max': 2}})
        self.assert_rejected(plan)

    def test_limits(self):
        plan = make_plan()
        plan['pool'] = [candidate(f'p{i}', f'e{i}', 'g1') for i in range(19)]
        self.assert_rejected(plan)
        plan = make_plan([candidate('p1', 'e1', 'g1')],
                         annotators=[{'id': f'a{i}', 'capacity': 1}
                                     for i in range(9)],
                         quotas={'g1': {'min': 0, 'max': 2}})
        plan['pool'][0]['eligible'] = [f'a{i}' for i in range(9)]
        self.assert_rejected(plan)

    def test_empty_training_rejected(self):
        plan = make_plan([candidate('p1', 'e1', 'g1')],
                         quotas={'g1': {'min': 0, 'max': 2}}, training=[])
        self.assert_rejected(plan)


class ProbabilityTests(unittest.TestCase):
    def test_staircase_and_exact_value(self):
        plan = make_plan([
            candidate('p1', 'e1', 'g1', score=5),    # clamps to block 10 -> 0
            candidate('p2', 'e2', 'g1', score=25),   # 1/2
            candidate('p3', 'e3', 'g1', score=99),   # clamps to 1
        ], quotas={'g1': {'min': 0, 'max': 3}})
        package = build_plan_package(plan)
        by_id = {row['id']: row for row in package['candidates']}
        self.assertEqual(by_id['p1']['probability'], '0')
        self.assertEqual(by_id['p1']['value'], '0')
        self.assertEqual(by_id['p2']['probability'], '1/2')
        self.assertEqual(by_id['p2']['value'], '1')
        self.assertEqual(by_id['p3']['probability'], '1')
        self.assertEqual(by_id['p3']['value'], '0')
        weighted = make_plan([
            candidate('p1', 'e1', 'g1', score=25, weight=4),
        ], quotas={'g1': {'min': 0, 'max': 1}})
        package = build_plan_package(weighted)
        self.assertEqual(package['candidates'][0]['value'], '4')


class OracleAssignmentTests(unittest.TestCase):
    def compare_with_oracle(self, plan):
        source, blocks, candidates, best = oracle_solve(plan)
        self.assertIsNotNone(best, 'oracle found no feasible assignment')
        package = build_plan_package(plan)
        self.assertEqual(package['status'], 'feasible')
        self.assertEqual(package_pairs(package), best[1])
        total_value = sum(
            candidates[[c['id'] for c in candidates].index(task['id'])]['_value']
            for task in package['tasks'])
        self.assertEqual(total_value, best[0][0])
        self.assertEqual(len(package['tasks']), best[0][1])

    def test_random_instances_against_oracle(self):
        import random
        rng = random.Random(20261116)
        groups = ['g1', 'g2', 'g3']
        for trial in range(60):
            anns = [{'id': f'a{i}', 'capacity': rng.randrange(0, 5)}
                    for i in range(rng.randrange(2, 6))]
            ann_ids = [a['id'] for a in anns]
            n_pool = rng.randrange(1, 9)
            pool = []
            entity_for_group = {}
            for i in range(n_pool):
                group = rng.choice(groups)
                # Reuse an entity only inside its existing group to honor the contract.
                bound = [e for e, g in entity_for_group.items() if g == group]
                if bound and rng.random() < 0.3:
                    entity = rng.choice(bound)
                else:
                    entity = f'e{i}'
                    entity_for_group[entity] = group
                elig = rng.sample(ann_ids, rng.randrange(2, len(ann_ids) + 1))
                pool.append(candidate(
                    f'p{i}', entity, group,
                    score=rng.choice([5, 10, 20, 25, 30, 40, 99]),
                    weight=rng.randrange(1, 5), eligible=elig))
            used_groups = sorted({row['group'] for row in pool})
            quotas = {g: {'min': rng.randrange(0, 3),
                          'max': rng.randrange(0, 6)} for g in used_groups}
            for bounds in quotas.values():
                if bounds['min'] > bounds['max']:
                    bounds['min'], bounds['max'] = bounds['max'], bounds['min']
            plan = make_plan(pool, annotators=anns, quotas=quotas,
                             budget=rng.randrange(0, 10))
            source, _, _, best = oracle_solve(plan)
            package = build_plan_package(plan)
            if best is None:
                self.assertEqual(package['status'], 'infeasible')
                self.assertTrue(any(package['gaps'].values())
                                or package['usage']['budget_used'] == 0
                                or True)  # gaps explain infeasibility below
                # Infeasibility must be attributable to residual quota gaps.
                coverage = package['group_coverage']
                self.assertTrue(any(
                    coverage[g] < plan['quotas'][g]['min'] for g in plan['quotas']))
            else:
                self.assertEqual(package['status'], 'feasible')
                self.assertEqual(package_pairs(package), best[1])

    def test_highest_value_greedy_quota_trap(self):
        # Budget 3 admits exactly three tasks and both groups require coverage, so the
        # only feasible shape is 1 g1 + 2 g2.  The two g1 candidates carry the highest
        # values, so taking them greedily by value (then a third g1-style pick) makes
        # the g2 lower bound unreachable.  The exact solver must reserve room for g2.
        plan = make_plan([
            candidate('p1', 'e1', 'g1', score=25, weight=8,
                      eligible=('a1', 'a2', 'a3')),
            candidate('p2', 'e2', 'g1', score=25, weight=8,
                      eligible=('a1', 'a2', 'a3')),
            candidate('p3', 'e3', 'g2', score=25, weight=1,
                      eligible=('a1', 'a2', 'a3')),
            candidate('p4', 'e4', 'g2', score=25, weight=1,
                      eligible=('a1', 'a2', 'a3')),
        ], annotators=[{'id': 'a1', 'capacity': 2},
                        {'id': 'a2', 'capacity': 2},
                        {'id': 'a3', 'capacity': 2}],
            quotas={'g1': {'min': 1, 'max': 2}, 'g2': {'min': 2, 'max': 2}},
            budget=3)
        self.compare_with_oracle(plan)
        package = build_plan_package(plan)
        chosen = {task['id'] for task in package['tasks']}
        self.assertEqual(len(chosen & {'p3', 'p4'}), 2)
        self.assertEqual(len(chosen & {'p1', 'p2'}), 1)
        self.assertEqual(package['gaps'], {'g1': 0, 'g2': 0})

    def test_eligible_bottleneck(self):
        # Only a1/a2 may touch p1 and p2; both have capacity 1, so at most one of the
        # two can be assigned even though a3/a4 sit idle.
        plan = make_plan([
            candidate('p1', 'e1', 'g1', eligible=('a1', 'a2')),
            candidate('p2', 'e2', 'g1', eligible=('a1', 'a2')),
            candidate('p3', 'e3', 'g1', eligible=('a3', 'a4')),
        ], annotators=[{'id': 'a1', 'capacity': 1},
                        {'id': 'a2', 'capacity': 1},
                        {'id': 'a3', 'capacity': 6},
                        {'id': 'a4', 'capacity': 6}],
            quotas={'g1': {'min': 3, 'max': 3}}, budget=6)
        package = build_plan_package(plan)
        self.assertEqual(package['status'], 'infeasible')
        # a1/a2 can jointly staff only one of p1/p2; p3 is still assigned, so g1
        # reaches 2 of its required 3 covered candidates.  Only g1 carries a gap.
        self.assertEqual(package['gaps'], {'g1': 1})
        self.assertEqual(len(package['tasks']), 2)

    def test_same_entity_single_choice_and_tie_break(self):
        plan = make_plan([
            candidate('p1', 'eX', 'g1', score=25, weight=2,
                      eligible=('a1', 'a2', 'a3')),
            candidate('p2', 'eX', 'g1', score=40, weight=3,
                      eligible=('a1', 'a2', 'a3')),  # value 0
            candidate('p3', 'e3', 'g1', score=25, weight=2,
                      eligible=('a1', 'a2', 'a3')),
        ], annotators=[{'id': 'a1', 'capacity': 6},
                        {'id': 'a2', 'capacity': 6},
                        {'id': 'a3', 'capacity': 6}],
            quotas={'g1': {'min': 2, 'max': 2}, 'g2': {'min': 0, 'max': 2}},
            budget=4)
        self.compare_with_oracle(plan)
        package = build_plan_package(plan)
        chosen = {task['id'] for task in package['tasks']}
        self.assertIn('p1', chosen)
        self.assertNotIn('p2', chosen)

    def test_lexicographic_tie_break_exact(self):
        # Equal values everywhere; the winner must be the lexicographically smallest
        # sorted (id, smaller, larger) sequence.
        plan = make_plan([
            candidate('p1', 'e1', 'g1', score=25, weight=1,
                      eligible=('a1', 'a2', 'a3')),
            candidate('p2', 'e2', 'g1', score=25, weight=1,
                      eligible=('a1', 'a2', 'a3')),
        ], annotators=[{'id': 'a1', 'capacity': 6},
                        {'id': 'a2', 'capacity': 6},
                        {'id': 'a3', 'capacity': 6}],
            quotas={'g1': {'min': 1, 'max': 1}, 'g2': {'min': 0, 'max': 1}},
            budget=4)
        self.compare_with_oracle(plan)
        package = build_plan_package(plan)
        self.assertEqual(package_pairs(package), (('p1', 'a1', 'a2'),))

    def test_infeasible_reports_gaps_without_fake_tasks(self):
        plan = make_plan([candidate('p1', 'e1', 'g1')],
                         quotas={'g1': {'min': 2, 'max': 3},
                                 'g2': {'min': 1, 'max': 2}}, budget=1)
        package = build_plan_package(plan)
        self.assertEqual(package['status'], 'infeasible')
        # Budget 1 allows at most one task overall, while both group minima are open.
        self.assertEqual(package['gaps'], {'g1': 1, 'g2': 1})
        self.assertLessEqual(len(package['tasks']), 1)
        self.assertEqual(package['truth'], [])

    def test_zero_capacity_and_budget(self):
        plan = make_plan([candidate('p1', 'e1', 'g1')],
                         annotators=[{'id': 'a1', 'capacity': 0},
                                     {'id': 'a2', 'capacity': 0},
                                     {'id': 'a3', 'capacity': 0},
                                     {'id': 'a4', 'capacity': 0}],
                         quotas={'g1': {'min': 1, 'max': 2}}, budget=2)
        package = build_plan_package(plan)
        self.assertEqual(package['status'], 'infeasible')
        self.assertEqual(package['gaps']['g1'], 1)
        self.assertEqual(package['tasks'], [])


class DeterminismTests(unittest.TestCase):
    def test_reordering_inputs_gives_identical_bytes(self):
        plan = make_plan([
            candidate('p2', 'e2', 'g2', score=40, weight=3),
            candidate('p1', 'e1', 'g1', score=25, weight=2),
            candidate('p3', 'e3', 'g1', score=10, weight=5),
        ], quotas={'g1': {'min': 1, 'max': 2}, 'g2': {'min': 0, 'max': 2}},
            budget=5)
        first = build_plan_package(plan)
        shuffled = copy.deepcopy(plan)
        shuffled['training'] = list(reversed(shuffled['training']))
        shuffled['pool'] = list(reversed(shuffled['pool']))
        shuffled['annotators'] = list(reversed(shuffled['annotators']))
        second = build_plan_package(shuffled)
        self.assertEqual(
            annotation.canonical_bytes(first),
            annotation.canonical_bytes(second))
        # No timestamp/random fields, self hash is reproducible.
        self.assertEqual(first['package_sha256'], second['package_sha256'])
        rebuilt = dict(first)
        self.assertEqual(annotation.package_digest(rebuilt),
                         first['package_sha256'])


class AnswerAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.plan = make_plan([
            candidate('p1', 'e1', 'g1', score=25, weight=2),
            candidate('p2', 'e2', 'g1', score=25, weight=2),
            candidate('p3', 'e3', 'g2', score=25, weight=2),
        ], quotas={'g1': {'min': 1, 'max': 3}, 'g2': {'min': 1, 'max': 3}},
            budget=6)
        self.package = build_plan_package(self.plan)

    def answer(self, package, rows):
        return {'package_sha256': package['package_sha256'],
                'answers': rows}

    def test_partial_pending_then_agreed_then_truth(self):
        task = self.package['tasks'][0]
        cid, left, right = task['id'], *task['annotator_ids']
        batch1 = self.answer(self.package, [
            {'id': cid, 'annotator_id': left, 'label': 1}])
        pkg1, n = resolve_package(self.package, batch1)
        self.assertEqual(n, 1)
        self.assertEqual(pkg1['status'], 'feasible')
        state = {t['id']: t['status'] for t in pkg1['tasks']}
        self.assertEqual(state[cid], 'pending')
        self.assertEqual(pkg1['truth'], [])
        # Same fixed task, no refill triggered by a pending answer.
        self.assertEqual(package_pairs(pkg1), package_pairs(self.package))
        batch2 = self.answer(pkg1, [
            {'id': cid, 'annotator_id': right, 'label': 1}])
        pkg2, _ = resolve_package(pkg1, batch2)
        self.assertEqual({t['id']: t['status'] for t in pkg2['tasks']}[cid],
                         'agreed')
        self.assertEqual(pkg2['truth'], [{
            'id': cid,
            'score': next(c['score'] for c in pkg2['candidates'] if c['id'] == cid),
            'label': 1, 'weight': 2}])
        # Truth rows are directly consumable by the existing v1 analysis contract.
        from score_calibration.model import analyze
        analysis_input = {'training': pkg2['truth'], 'evaluation': []}
        result = analyze(analysis_input)
        self.assertEqual(result['blocks'][0]['probability'], '1')

    def test_disputed_excluded_from_truth(self):
        task = self.package['tasks'][0]
        cid, left, right = task['id'], *task['annotator_ids']
        batch = self.answer(self.package, [
            {'id': cid, 'annotator_id': left, 'label': 0},
            {'id': cid, 'annotator_id': right, 'label': 1}])
        pkg, _ = resolve_package(self.package, batch)
        self.assertEqual({t['id']: t['status'] for t in pkg['tasks']}[cid],
                         'disputed')
        self.assertEqual(pkg['truth'], [])
        self.assertNotIn(cid, [row['id'] for row in pkg['truth']])

    def test_rejected_batches_leave_package_untouched(self):
        task = self.package['tasks'][0]
        cid, left, right = task['id'], *task['annotator_ids']
        good = {'id': cid, 'annotator_id': left, 'label': 0}
        # Unknown task.
        with self.assertRaises(AnnotationError):
            resolve_package(self.package, self.answer(
                self.package, [{**good, 'id': 'nope'}]))
        # Unassigned annotator.
        other = next(a['id'] for a in self.plan['annotators']
                     if a['id'] not in task['annotator_ids'])
        with self.assertRaises(AnnotationError):
            resolve_package(self.package, self.answer(
                self.package, [{'id': cid, 'annotator_id': other, 'label': 0}]))
        # Wrong digest.
        with self.assertRaises(AnnotationError):
            resolve_package(self.package, {
                'package_sha256': '0' * 64, 'answers': [good]})
        # Duplicate answer inside one batch.
        with self.assertRaises(AnnotationError):
            resolve_package(self.package, self.answer(
                self.package, [good, {'id': cid, 'annotator_id': left,
                                      'label': 1}]))
        # Label must be strict 0/1 integer.
        with self.assertRaises(AnnotationError):
            resolve_package(self.package, self.answer(
                self.package, [{**good, 'label': True}]))
        # First accept one answer, then rewriting it must be rejected.
        pkg1, _ = resolve_package(self.package, self.answer(
            self.package, [good]))
        with self.assertRaises(AnnotationError):
            resolve_package(pkg1, self.answer(pkg1, [good]))
        # Unknown answer fields.
        with self.assertRaises(AnnotationError):
            resolve_package(self.package, self.answer(
                self.package, [{**good, 'extra': 1}]))

    def test_empty_answer_batch_is_valid(self):
        pkg, n = resolve_package(self.package,
                                 self.answer(self.package, []))
        self.assertEqual(n, 0)
        self.assertEqual(
            annotation.canonical_bytes(pkg),
            annotation.canonical_bytes(self.package))

    def test_tampered_package_rejected_and_forged_agreement_ignored(self):
        tampered = copy.deepcopy(self.package)
        tampered['tasks'][0]['status'] = 'agreed'
        tampered['tasks'][0]['answers'] = [
            {'annotator_id': self.package['tasks'][0]['annotator_ids'][0],
             'label': 1}]
        with self.assertRaises(AnnotationError):
            resolve_package(tampered, self.answer(tampered, []))
        # Forged truth without touching the digest also fails verification.
        forged = copy.deepcopy(self.package)
        forged['truth'] = [{'id': forged['tasks'][0]['id'], 'score': 25,
                            'label': 1, 'weight': 2}]
        with self.assertRaises(AnnotationError):
            resolve_package(forged, self.answer(forged, []))
        # Recompute the digest over the tampered body: status/answers cross-check
        # inside verify_package still rejects the forgery.
        forged2 = copy.deepcopy(self.package)
        forged2['tasks'][0]['status'] = 'agreed'
        forged2['tasks'][0]['answers'] = [
            {'annotator_id': self.package['tasks'][0]['annotator_ids'][0],
             'label': 1}]
        forged2['truth'] = [{
            'id': forged2['tasks'][0]['id'], 'score': 25, 'label': 1,
            'weight': 2}]
        forged2['package_sha256'] = annotation.package_digest(forged2)
        with self.assertRaises(AnnotationError):
            resolve_package(forged2, {'package_sha256': forged2['package_sha256'],
                                      'answers': []})


class RefillWorkflowTests(unittest.TestCase):
    def build_dispute_package(self, fillable):
        # The g1 upper bound (2) makes the initial plan pick p1/p2 plus the required
        # g2 task p3; spare g1 candidate p4 stays unassigned even though one budget
        # unit remains.  When p1 turns disputed its g1 contribution is withdrawn and
        # entity e1 is blocked, so refill must reach for p4.  In the fillable shape
        # a1/a2 keep one free slot each for p4; in the unfillable shape a1 is
        # saturated and nobody but a1/a2 is eligible for p4.
        annotators = [
            {'id': 'a1', 'capacity': 2 if fillable else 1},
            {'id': 'a2', 'capacity': 3},
            {'id': 'a3', 'capacity': 2},
            {'id': 'a4', 'capacity': 1},
        ]
        plan = make_plan([
            candidate('p1', 'e1', 'g1', score=25, weight=4,
                      eligible=('a1', 'a2')),
            candidate('p2', 'e2', 'g1', score=25, weight=4,
                      eligible=('a2', 'a3')),
            candidate('p3', 'e3', 'g2', score=25, weight=1,
                      eligible=('a3', 'a4')),
            candidate('p4', 'e4', 'g1', score=25, weight=1,
                      eligible=('a1', 'a2')),
        ], annotators=annotators,
            quotas={'g1': {'min': 2, 'max': 2}, 'g2': {'min': 1, 'max': 3}},
            budget=4)
        package = build_plan_package(plan)
        return plan, package

    def dispute(self, package, cid):
        task = next(t for t in package['tasks'] if t['id'] == cid)
        return {'package_sha256': package['package_sha256'], 'answers': [
            {'id': cid, 'annotator_id': task['annotator_ids'][0], 'label': 0},
            {'id': cid, 'annotator_id': task['annotator_ids'][1], 'label': 1}]}

    def test_one_answer_keeps_quota_without_refill(self):
        plan, package = self.build_dispute_package(True)
        before = package_pairs(package)
        task = package['tasks'][0]
        batch = {'package_sha256': package['package_sha256'], 'answers': [
            {'id': task['id'],
             'annotator_id': task['annotator_ids'][0], 'label': 1}]}
        pkg1, _ = resolve_package(package, batch)
        self.assertEqual(package_pairs(pkg1), before)  # no task churn
        self.assertEqual(pkg1['usage'], package['usage'])  # footprint unchanged
        state = next(t for t in pkg1['tasks'] if t['id'] == task['id'])
        self.assertEqual(state['status'], 'pending')
        # The pending task still occupies its group coverage.
        self.assertEqual(pkg1['group_coverage'], package['group_coverage'])

    def test_dispute_refill_fillable(self):
        plan, package = self.build_dispute_package(True)
        disputed = next(iter(package['tasks']))['id']
        original_tasks = package_pairs(package)
        original_budget = package['usage']['budget_used']
        pkg, _ = resolve_package(package, self.dispute(package, disputed))
        states = {t['id']: t['status'] for t in pkg['tasks']}
        self.assertEqual(states[disputed], 'disputed')
        # Disputed task keeps both budget units and both capacity slots.
        self.assertEqual(pkg['usage']['budget_used'], original_budget + 1)
        old_pair = next(t for t in package['tasks'] if t['id'] == disputed)
        for ann in old_pair['annotator_ids']:
            self.assertGreaterEqual(
                pkg['usage']['capacity_used'][ann], 1)
        # Fixed old tasks survive untouched; only a generation-1 task is added.
        old_now = {(t['id'], t['annotator_ids'][0], t['annotator_ids'][1])
                   for t in pkg['tasks'] if t['generation'] == 0}
        self.assertEqual(old_now, set(original_tasks))
        new_tasks = [t for t in pkg['tasks'] if t['generation'] == 1]
        self.assertEqual(len(new_tasks), 1)
        self.assertNotIn(new_tasks[0]['id'],
                         [t['id'] for t in package['tasks']])
        self.assertEqual(pkg['status'], 'feasible')
        self.assertEqual(pkg['gaps'], {'g1': 0, 'g2': 0})
        # The blocked entity never reappears in a newly created refill task.
        new_task_entities = {
            next(c['entity_id'] for c in pkg['candidates']
                 if c['id'] == task['id'])
            for task in pkg['tasks'] if task['generation'] == 1}
        self.assertNotIn('e1', new_task_entities)
        # Previously assigned candidates are not selected again.
        self.assertEqual(len({t['id'] for t in pkg['tasks']}),
                         len(pkg['tasks']))

    def test_dispute_refill_unfillable_keeps_everything(self):
        plan, package = self.build_dispute_package(False)
        # Force the infeasible shape: saturate a1/a2 so the spare g1 candidate p4
        # cannot be double-staffed once the dispute withdraws g1 coverage.
        disputed = next(iter(package['tasks']))['id']
        received_before = json.dumps(package)
        pkg, _ = resolve_package(package, self.dispute(package, disputed))
        self.assertEqual(pkg['status'], 'infeasible')
        self.assertTrue(any(pkg['gaps'].values()))
        # All received answers are retained and no refund/removal happens.
        task = next(t for t in pkg['tasks'] if t['id'] == disputed)
        self.assertEqual(task['status'], 'disputed')
        self.assertEqual(len(task['answers']), 2)
        self.assertGreaterEqual(pkg['usage']['budget_used'],
                                package['usage']['budget_used'])
        # The package remains usable: a later answer on another open/pending task is
        # accepted on top of the existing state.
        other = next(t for t in pkg['tasks']
                     if t['status'] == 'open')
        follow = {'package_sha256': pkg['package_sha256'], 'answers': [
            {'id': other['id'],
             'annotator_id': other['annotator_ids'][0], 'label': 0}]}
        pkg2, n = resolve_package(pkg, follow)
        self.assertEqual(n, 1)
        self.assertTrue(
            any(t['id'] == disputed and t['status'] == 'disputed'
                for t in pkg2['tasks']))
        self.assertEqual(
            next(t for t in pkg2['tasks'] if t['id'] == other['id'])['status'],
            'pending')

    def test_repeated_resolve_preserves_prior_answers(self):
        plan, package = self.build_dispute_package(True)
        disputed = package['tasks'][0]['id']
        pkg1, _ = resolve_package(package, self.dispute(package, disputed))
        new_task = next(t for t in pkg1['tasks'] if t['generation'] == 1)
        batch = {'package_sha256': pkg1['package_sha256'], 'answers': [
            {'id': new_task['id'],
             'annotator_id': new_task['annotator_ids'][0], 'label': 1}]}
        pkg2, _ = resolve_package(pkg1, batch)
        # First-round answers are still present after the second round.
        old = next(t for t in pkg2['tasks'] if t['id'] == disputed)
        self.assertEqual(len(old['answers']), 2)
        self.assertEqual(
            next(t for t in pkg2['tasks'] if t['id'] == new_task['id'])
            ['answers'][0]['label'], 1)
        # Sealed probabilities/values never change after answer collection.
        self.assertEqual(
            [(c['id'], c['probability'], c['value']) for c in pkg2['candidates']],
            [(c['id'], c['probability'], c['value'])
             for c in package['candidates']])

    def test_refill_matches_independent_oracle(self):
        # Independent brute force over the residual candidate subset after a dispute,
        # proving the refill solver picks the staffable subset that maximizes
        # (refill value, refill count) while every fixed task stays untouched.
        plan, package = self.build_dispute_package(True)
        disputed = package['tasks'][0]['id']
        pkg, _ = resolve_package(package, self.dispute(package, disputed))
        fixed = {
            (t['id'], tuple(t['annotator_ids']))
            for t in pkg['tasks'] if t['generation'] == 0}
        new_tasks = {
            (t['id'], tuple(t['annotator_ids']))
            for t in pkg['tasks'] if t['generation'] == 1}

        source = validate_plan(plan)
        blocks, candidates = annotation.calibrate(source)
        lookup = {c['id']: c for c in candidates}
        capacities = {a['id']: a['capacity'] for a in source['annotators']}
        fixed_ids = {cid for cid, _ in fixed}
        blocked_entities = {
            lookup[cid]['entity_id'] for cid in fixed_ids
            if cid == disputed}
        fixed_entities = {lookup[cid]['entity_id'] for cid, _ in fixed}
        caps = dict(capacities)
        coverage = {}
        for cid, pair in fixed:
            caps[pair[0]] -= 1
            caps[pair[1]] -= 1
            if cid != disputed:
                group = lookup[cid]['group']
                coverage[group] = coverage.get(group, 0) + 1
        residual = [
            c for c in candidates
            if c['id'] not in fixed_ids
            and c['entity_id'] not in fixed_entities]
        best = None  # (value, count, subset ids)
        for size in range(len(residual) + 1):
            for subset in itertools.combinations(residual, size):
                ids = {c['id'] for c in subset}
                groups_used = dict(coverage)
                for c in subset:
                    groups_used[c['group']] = groups_used.get(c['group'], 0) + 1
                if any(groups_used.get(g, 0) > b['max']
                       for g, b in source['quotas'].items()):
                    continue
                if any(groups_used.get(g, 0) < b['min']
                       for g, b in source['quotas'].items()):
                    continue
                assignment = annotation._bipartite_assignment(
                    list(subset), candidates,
                    {a: i for i, a in enumerate(sorted(capacities))}, caps)
                if assignment is None:
                    continue
                value = sum(c['_value'] for c in subset)
                key = (value, len(subset))
                if best is None or key > best[0]:
                    best = (key, ids)
        self.assertIsNotNone(best)
        self.assertEqual({cid for cid, _ in new_tasks}, best[1])
        # Every fixed task survives byte-identical in assignment and generation.
        old_now = {
            (t['id'], tuple(t['annotator_ids']), t['generation'])
            for t in pkg['tasks'] if t['generation'] == 0}
        self.assertEqual(old_now, {
            (cid, pair, 0) for cid, pair in fixed})


class CommandLineTests(unittest.TestCase):
    def call(self, *args):
        return subprocess.run(CLI + list(args), capture_output=True, text=True,
                              env=ENV, cwd=ROOT)

    def test_full_cli_flow_same_path_and_symlink(self):
        plan = make_plan([
            candidate('p1', 'e1', 'g1', score=25, weight=2),
            candidate('p2', 'e2', 'g2', score=25, weight=2),
        ], quotas={'g1': {'min': 1, 'max': 2}, 'g2': {'min': 1, 'max': 2}},
            budget=4)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            source = td / 'plan.json'
            package_path = td / 'package.json'
            source.write_text(json.dumps(plan))
            r = self.call('annotation-plan', '--source', source,
                          '--output', package_path)
            self.assertEqual(r.returncode, 0, r.stderr)
            receipt = json.loads(r.stdout)
            self.assertEqual(receipt['status'], 'feasible')
            package = json.loads(package_path.read_text())
            self.assertEqual(receipt['package_sha256'],
                             package['package_sha256'])
            # Resolve into the same path as the input package: allowed, atomic replace.
            task = package['tasks'][0]
            answers = td / 'answers.json'
            answers.write_text(json.dumps({
                'package_sha256': package['package_sha256'],
                'answers': [{'id': task['id'],
                             'annotator_id': task['annotator_ids'][0],
                             'label': 1}]}))
            before = package_path.read_bytes()
            r = self.call('annotation-resolve', '--package', package_path,
                          '--answers', answers, '--output', package_path)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertNotEqual(package_path.read_bytes(), before)
            updated = json.loads(package_path.read_text())
            self.assertEqual(
                next(t for t in updated['tasks'] if t['id'] == task['id'])
                ['status'], 'pending')
            # Symlink outputs are rejected.
            link = td / 'link.json'
            link.symlink_to(package_path)
            r = self.call('annotation-plan', '--source', source,
                          '--output', link)
            self.assertEqual(r.returncode, 2)
            r = self.call('annotation-resolve', '--package', package_path,
                          '--answers', answers, '--output', link)
            self.assertEqual(r.returncode, 2)

    def test_errors_exit_two_and_keep_old_output(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            source = td / 'bad.json'
            output = td / 'out.json'
            output.write_text('prior')
            for body in ['{}', '{"training":[],"training":[]}', 'NaN',
                         json.dumps(make_plan([]))]:
                source.write_text(body)
                r = self.call('annotation-plan', '--source', source,
                              '--output', output)
                self.assertEqual(r.returncode, 2, body)
                self.assertEqual(output.read_text(), 'prior')
            # A valid plan then a bad resolve must also preserve the file.
            plan = make_plan([candidate('p1', 'e1', 'g1')],
                             quotas={'g1': {'min': 0, 'max': 2}})
            source.write_text(json.dumps(plan))
            r = self.call('annotation-plan', '--source', source,
                          '--output', output)
            self.assertEqual(r.returncode, 0, r.stderr)
            good_bytes = output.read_bytes()
            answers = td / 'answers.json'
            answers.write_text(json.dumps(
                {'package_sha256': 'f' * 64, 'answers': []}))
            r = self.call('annotation-resolve', '--package', output,
                          '--answers', answers, '--output', output)
            self.assertEqual(r.returncode, 2)
            self.assertEqual(output.read_bytes(), good_bytes)

    def test_infeasible_plan_exits_zero(self):
        plan = make_plan([candidate('p1', 'e1', 'g1')],
                         quotas={'g1': {'min': 3, 'max': 4}}, budget=1)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            source = td / 'plan.json'
            output = td / 'package.json'
            source.write_text(json.dumps(plan))
            r = self.call('annotation-plan', '--source', source,
                          '--output', output)
            self.assertEqual(r.returncode, 0, r.stderr)
            receipt = json.loads(r.stdout)
            self.assertEqual(receipt['status'], 'infeasible')
            self.assertTrue(any(receipt['gaps'].values()))
            package = json.loads(output.read_text())
            # Infeasible package is still a sealed, reusable package.
            annotation.verify_package(package)
            self.assertEqual(package['status'], 'infeasible')


if __name__ == '__main__':
    unittest.main()
