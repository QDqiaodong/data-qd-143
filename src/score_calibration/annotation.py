"""Double-blind human annotation planning and answer resolution.

The module is pure standard library.  It produces deterministic, self-verifying JSON
annotation packages that carry their own todo tasks and the growing ground-truth
dataset.  The revision ledger is never touched.

Two responsibilities live here:

* :func:`build_plan_package` -- validate the plan contract, fit the shared PAVA curve,
  compute exact staircase probabilities and candidate values, and solve the exact
  double-annotator assignment (max value, then max count, then lexicographic minimum).
* :func:`resolve_package` -- verify a package digest and all internal invariants,
  accept a partial answer batch, recompute task states, rebuild the truth dataset from
  agreed labels only, and run a constrained refill round whenever a task turns
  disputed.
"""
from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from fractions import Fraction
from itertools import combinations

from .common import ident, save
from .isotonic import fit
from .model import observations

MAX_POOL = 18
MAX_ANNOTATORS = 8
MAX_GROUPS = 4
MAX_CAPACITY = 36
MAX_BUDGET = 36
MAX_QUOTA = 18
SCORE_MAX = 1_000_000
WEIGHT_MAX = 1_000_000
SCHEMA_VERSION = 3
SHA_FIELD = 'package_sha256'


class AnnotationError(ValueError):
    """A plan, package or answer batch violates the annotation contract."""


# --------------------------------------------------------------------------- #
# Deterministic encoding
# --------------------------------------------------------------------------- #

def canonical_bytes(obj):
    """Compact UTF-8 JSON with sorted keys and no whitespace."""
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':'),
                      sort_keys=True).encode('utf-8')


def package_digest(body):
    """SHA256 over the package body with the self-hash field removed."""
    stripped = {key: value for key, value in body.items() if key != SHA_FIELD}
    return hashlib.sha256(canonical_bytes(stripped)).hexdigest()


# --------------------------------------------------------------------------- #
# Strict validation
# --------------------------------------------------------------------------- #

def _strict_fields(obj, fields, name):
    if not isinstance(obj, dict) or set(obj) != set(fields.split()):
        raise AnnotationError(f'{name} must contain exactly the fields: {fields}')
    return obj


def _bounded_int(value, name, lower, upper):
    # bool is a subclass of int; the contract explicitly forbids booleans as integers.
    if type(value) is not int or not lower <= value <= upper:
        raise AnnotationError(f'{name} must be an integer in [{lower}, {upper}]')
    return value


def validate_plan(data):
    """Validate and normalize a plan request; ordering of input arrays is erased."""
    _strict_fields(data, 'training pool annotators quotas budget', 'plan request')

    training, training_ids = observations(data['training'])
    if not training:
        raise AnnotationError('training must be nonempty')
    training.sort(key=lambda row: row['id'])

    annotators_raw = data['annotators']
    if not isinstance(annotators_raw, list) or not annotators_raw \
            or len(annotators_raw) > MAX_ANNOTATORS:
        raise AnnotationError(f'annotators must be 1..{MAX_ANNOTATORS} rows')
    annotators, annotator_ids = [], set()
    for row in annotators_raw:
        _strict_fields(row, 'id capacity', 'annotator row')
        ann_id = ident(row['id'])
        if ann_id in annotator_ids:
            raise AnnotationError('duplicate annotator id: ' + ann_id)
        annotator_ids.add(ann_id)
        annotators.append({
            'id': ann_id,
            'capacity': _bounded_int(row['capacity'], 'capacity', 0, MAX_CAPACITY),
        })
    annotators.sort(key=lambda row: row['id'])

    quotas_raw = data['quotas']
    if not isinstance(quotas_raw, dict) or not quotas_raw:
        raise AnnotationError('quotas must be a nonempty group -> {min,max} mapping')
    quotas = {}
    for group, bounds in quotas_raw.items():
        group = ident(group)
        _strict_fields(bounds, 'min max', 'quota bounds')
        lower = _bounded_int(bounds['min'], 'quota min', 0, MAX_QUOTA)
        upper = _bounded_int(bounds['max'], 'quota max', 0, MAX_QUOTA)
        if lower > upper:
            raise AnnotationError('quota min must be <= max')
        quotas[group] = {'min': lower, 'max': upper}
    if len(quotas) > MAX_GROUPS:
        raise AnnotationError(f'at most {MAX_GROUPS} groups are allowed')

    pool_raw = data['pool']
    if not isinstance(pool_raw, list) or not pool_raw or len(pool_raw) > MAX_POOL:
        raise AnnotationError(f'pool must be 1..{MAX_POOL} rows')
    pool, pool_ids = [], set()
    entity_groups = {}
    for row in pool_raw:
        _strict_fields(row, 'id entity_id group score weight eligible', 'pool row')
        candidate_id = ident(row['id'])
        if candidate_id in pool_ids:
            raise AnnotationError('duplicate pool id: ' + candidate_id)
        if candidate_id in training_ids:
            raise AnnotationError('pool id overlaps training id: ' + candidate_id)
        pool_ids.add(candidate_id)
        entity_id = ident(row['entity_id'])
        group = ident(row['group'])
        if group not in quotas:
            raise AnnotationError(f'pool group {group} has no quota')
        if entity_id in entity_groups and entity_groups[entity_id] != group:
            raise AnnotationError(
                f'candidates of entity {entity_id} must share one group')
        entity_groups[entity_id] = group
        eligible_raw = row['eligible']
        if not isinstance(eligible_raw, list) or not eligible_raw:
            raise AnnotationError('eligible must be a nonempty annotator id array')
        eligible, eligible_seen = [], set()
        for ann_id in eligible_raw:
            ann_id = ident(ann_id)
            if ann_id in eligible_seen:
                raise AnnotationError('duplicate eligible annotator: ' + ann_id)
            if ann_id not in annotator_ids:
                raise AnnotationError('eligible references unknown annotator: ' + ann_id)
            eligible_seen.add(ann_id)
            eligible.append(ann_id)
        pool.append({
            'id': candidate_id,
            'entity_id': entity_id,
            'group': group,
            'score': _bounded_int(row['score'], 'score', 0, SCORE_MAX),
            'weight': _bounded_int(row['weight'], 'weight', 1, WEIGHT_MAX),
            'eligible': sorted(eligible),
        })
    pool.sort(key=lambda row: row['id'])
    pool_groups = {row['group'] for row in pool}
    if len(pool_groups) > MAX_GROUPS:
        raise AnnotationError(f'at most {MAX_GROUPS} groups are allowed')

    budget = _bounded_int(data['budget'], 'budget', 0, MAX_BUDGET)
    return {
        'training': training,
        'pool': pool,
        'annotators': annotators,
        'quotas': {group: quotas[group] for group in sorted(quotas)},
        'budget': budget,
    }


def validate_answers(data):
    _strict_fields(data, 'package_sha256 answers', 'answer batch')
    if not isinstance(data['package_sha256'], str) or not data['package_sha256']:
        raise AnnotationError('package_sha256 must be a nonempty string')
    raw_answers = data['answers']
    if not isinstance(raw_answers, list):
        raise AnnotationError('answers must be an array')
    answers, seen = [], set()
    for row in raw_answers:
        _strict_fields(row, 'id annotator_id label', 'answer row')
        candidate_id = ident(row['id'])
        annotator_id = ident(row['annotator_id'])
        label = _bounded_int(row['label'], 'label', 0, 1)
        key = (candidate_id, annotator_id)
        if key in seen:
            raise AnnotationError(
                f'duplicate answer for task {candidate_id} / {annotator_id}')
        seen.add(key)
        answers.append({'id': candidate_id, 'annotator_id': annotator_id,
                        'label': label})
    return data['package_sha256'], answers


# --------------------------------------------------------------------------- #
# PAVA probabilities and exact candidate values
# --------------------------------------------------------------------------- #

def probability_at(blocks, score):
    """Staircase prediction: last block whose lower bound is <= score (clamped)."""
    starts = [block['lower'] for block in blocks]
    index = max(0, bisect_right(starts, score) - 1)
    return Fraction(blocks[index]['probability'])


def candidate_value(weight, probability):
    """weight * (1 - |2p - 1|), kept as an exact rational Fraction."""
    return weight * (1 - abs(2 * probability - 1))


def calibrate(source):
    """Return (pava blocks, candidate records with exact probability/value strings)."""
    blocks = fit(source['training'])
    candidates = []
    for row in source['pool']:
        probability = probability_at(blocks, row['score'])
        value = candidate_value(row['weight'], probability)
        candidates.append({
            **row,
            'probability': str(probability),
            'value': str(value),
            '_probability': probability,
            '_value': value,
        })
    candidates.sort(key=lambda row: row['id'])
    return blocks, candidates


# --------------------------------------------------------------------------- #
# Exact assignment solver
# --------------------------------------------------------------------------- #

def _max_flow_feasible(requirements, ann_ids, ann_index, capacity_left):
    """Bipartite b-matching feasibility via Dinic max flow.

    ``requirements`` maps candidate id to its eligible annotator id list; each
    candidate demands two distinct annotators and each annotator supplies up to its
    residual capacity.  Returns True when every candidate can be fully staffed.
    """
    candidate_ids = sorted(requirements)
    n_cand = len(candidate_ids)
    n_ann = len(ann_ids)
    source_node = 0
    cand_base = 1
    ann_base = cand_base + n_cand
    sink_node = ann_base + n_ann
    graph = [[] for _ in range(sink_node + 1)]

    def add_edge(u, v, cap):
        graph[u].append([v, cap, len(graph[v])])
        graph[v].append([u, 0, len(graph[u]) - 1])

    for i, cid in enumerate(candidate_ids):
        add_edge(source_node, cand_base + i, 2)
        for ann in requirements[cid]:
            if ann in ann_index:
                add_edge(cand_base + i, ann_base + ann_index[ann], 1)
    for j, ann in enumerate(ann_ids):
        add_edge(ann_base + j, sink_node, capacity_left[ann])

    def bfs():
        level = [-1] * len(graph)
        level[source_node] = 0
        queue = [source_node]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v, cap, rev in graph[u]:
                if cap > 0 and level[v] < 0:
                    level[v] = level[u] + 1
                    queue.append(v)
        return level

    def dfs_flow(u, pushed, level, it):
        if u == sink_node:
            return pushed
        while it[u] < len(graph[u]):
            edge = graph[u][it[u]]
            v, cap, rev = edge
            if cap > 0 and level[v] == level[u] + 1:
                flow = dfs_flow(v, min(pushed, cap), level, it)
                if flow:
                    edge[1] -= flow
                    graph[v][rev][1] += flow
                    return flow
            it[u] += 1
        return 0

    total = 0
    while True:
        level = bfs()
        if level[sink_node] < 0:
            break
        it = [0] * len(graph)
        while True:
            pushed = dfs_flow(source_node, 10 ** 9, level, it)
            if not pushed:
                break
            total += pushed
    return total == 2 * n_cand


def _bipartite_assignment(selected, universe, ann_index, capacity_left):
    """Lexicographically smallest feasible double-annotator assignment.

    Candidates are processed in id order and each candidate's annotator pair is fixed
    to the smallest feasible ``(left, right)`` under the remaining capacity, testing
    pair completion with a max-flow feasibility check at every step.  Greedily taking
    the smallest pair that preserves global feasibility yields the lexicographically
    smallest sorted ``(candidate_id, smaller, larger)`` tuple sequence.  Returns the
    per-candidate pair mapping or None when no complete staffing exists.
    """
    if not selected:
        return {}
    ann_ids = sorted(capacity_left)
    ordered = sorted(selected, key=lambda candidate: candidate['id'])
    requirements = {candidate['id']: list(candidate['eligible'])
                    for candidate in ordered}
    remaining_capacity = dict(capacity_left)
    assignment = {}
    for candidate in ordered:
        cid = candidate['id']
        eligible = sorted(
            ann for ann in candidate['eligible'] if ann in ann_index)
        chosen_pair = None
        for left, right in combinations(eligible, 2):
            if remaining_capacity[left] <= 0 or remaining_capacity[right] <= 0:
                continue
            trial_requirements = dict(requirements)
            trial_capacity = dict(remaining_capacity)
            trial_capacity[left] -= 1
            trial_capacity[right] -= 1
            # Already-fixed candidates are removed from the pending requirement set;
            # their capacity consumption is reflected in trial_capacity.
            pending = {
                other: trial_requirements[other]
                for other in trial_requirements if other not in assignment
                and other != cid}
            if _max_flow_feasible(pending, ann_ids, ann_index,
                                  trial_capacity):
                chosen_pair = (left, right)
                break
        if chosen_pair is None:
            return None
        assignment[cid] = chosen_pair
        remaining_capacity[chosen_pair[0]] -= 1
        remaining_capacity[chosen_pair[1]] -= 1
    return assignment


def optimize(universe, residual_budget, capacity_left, residual_quotas,
             blocked_entities, relax_min=False):
    """Maximize the exact assignment objective over residual candidates.

    Subset choice and annotator pairing are solved separately.  A memoized DFS
    enumerates candidate subsets (one candidate per entity, group and budget bounds),
    while feasibility of staffing a subset with two distinct eligible annotators per
    candidate within residual capacity is decided by a bipartite b-matching max flow
    at each leaf.  This keeps the search at O(2**18) subset branches with polynomial
    matching checks, instead of multiplying in every annotator pair.  Returns
    ``(Fraction value, count, sorted tuple sequence)`` where each sequence element is
    ``(candidate_id, smaller_annotator_id, larger_annotator_id)``, or None when the
    residual group lower bounds cannot be met; ``relax_min`` drops lower bounds for
    best-effort infeasible packages.
    """
    ann_ids = sorted(capacity_left)
    ann_index = {ann: pos for pos, ann in enumerate(ann_ids)}
    groups = sorted(residual_quotas)
    group_index = {group: pos for pos, group in enumerate(groups)}
    mins = [residual_quotas[group]['min'] for group in groups]
    maxs = [residual_quotas[group]['max'] for group in groups]
    if relax_min:
        mins = [0] * len(groups)

    n = len(universe)
    candidate_groups = [group_index[row['group']] for row in universe]
    candidate_entities = [row['entity_id'] for row in universe]
    distinct_entities = sorted(set(candidate_entities))
    entity_bit_index = {entity: i for i, entity in enumerate(distinct_entities)}
    candidate_entity_bits = [
        1 << entity_bit_index[entity] for entity in candidate_entities]
    blocked_mask = 0
    for entity in blocked_entities:
        if entity in entity_bit_index:
            blocked_mask |= 1 << entity_bit_index[entity]
    selectable = [
        not (candidate_entity_bits[pos] & blocked_mask) for pos in range(n)]

    # Staffing-aware subset search.  Leaf b-matching feasibility depends on the
    # concrete selected subset, which the (pos, cov, blocked, budget) state does not
    # uniquely encode, so this search is not memoized; with at most 18 candidates and
    # polynomial max-flow checks the exhaustive 2**n tree stays bounded.  A leaf is
    # feasible only when its exact selected subset can be staffed with two distinct
    # eligible annotators per candidate within residual capacity.
    def tail_optimum(pos, budget_left, cov, blocked, selected):
        for k in range(len(groups)):
            if cov[k] >= mins[k]:
                continue
            reachable = 0
            for j in range(pos, n):
                if candidate_groups[j] == k and not (
                        blocked & candidate_entity_bits[j]):
                    reachable += 1
            if cov[k] + reachable < mins[k]:
                return None
        if pos == n:
            if any(cov[k] < mins[k] for k in range(len(groups))):
                return None
            if _bipartite_assignment(selected, universe, ann_index,
                                     capacity_left) is None:
                return None
            return (Fraction(0), 0)
        best = None
        bit = candidate_entity_bits[pos]
        group = candidate_groups[pos]
        if (selectable[pos] and not (blocked & bit) and budget_left > 0
                and cov[group] < maxs[group]):
            new_cov = list(cov)
            new_cov[group] += 1
            child = tail_optimum(pos + 1, budget_left - 1, tuple(new_cov),
                                 blocked | bit,
                                 selected + [universe[pos]])
            if child is not None:
                best = (universe[pos]['_value'] + child[0], 1 + child[1])
        child = tail_optimum(pos + 1, budget_left, cov, blocked, selected)
        if child is not None and (best is None or child > best):
            best = child
        return best

    # The staffing-aware optimum (b-matching enforced at every leaf) is the true
    # objective.  Compute it once, then greedily include candidates in id order
    # whenever a staffable completion still reaches that optimum, producing the
    # lexicographically smallest optimal id tuple.
    staffing_target = tail_optimum(
        0, residual_budget, tuple(0 for _ in groups), blocked_mask, [])
    if staffing_target is None:
        return None
    target_value, target_count = staffing_target
    chosen = []
    cov = [0] * len(groups)
    blocked = blocked_mask
    budget_left = residual_budget
    for pos in range(n):
        bit = candidate_entity_bits[pos]
        group = candidate_groups[pos]
        if not (selectable[pos]) or (blocked & bit) or budget_left <= 0 \
                or cov[group] >= maxs[group]:
            continue
        trial_cov = list(cov)
        trial_cov[group] += 1
        feasible_tail = tail_optimum(
            pos + 1, budget_left - 1, tuple(trial_cov), blocked | bit,
            chosen + [universe[pos]])
        if feasible_tail is None:
            continue
        projected_value = sum(c['_value'] for c in chosen) \
            + universe[pos]['_value'] + feasible_tail[0]
        projected_count = len(chosen) + 1 + feasible_tail[1]
        if (projected_value, projected_count) == (target_value, target_count):
            chosen.append(universe[pos])
            cov = trial_cov
            blocked |= bit
            budget_left -= 1
    assignment = _bipartite_assignment(chosen, universe, ann_index,
                                       capacity_left)
    if assignment is None:  # pragma: no cover - subset was proven staffable
        return None
    sequence = tuple(sorted(
        (candidate['id'], assignment[candidate['id']][0],
         assignment[candidate['id']][1]) for candidate in chosen))
    return (target_value, target_count, sequence)


def _residual_context(universe, candidate_lookup, source, fixed_pairs,
                      blocked_entities, disputed_ids):
    """Freeze every fixed-task footprint and return the residual solver context.

    Coverage is rebuilt from scratch out of the fixed pairs so it can never be
    double-counted: disputed fixed tasks forfeit their group contribution while
    retaining their budget and capacity footprint.
    """
    fixed_set = {pair[0] for pair in fixed_pairs}
    capacity_left = {row['id']: row['capacity'] for row in source['annotators']}
    coverage = {group: 0 for group in source['quotas']}
    fixed_entities = set()
    for cid, left, right in fixed_pairs:
        candidate = candidate_lookup[cid]
        if capacity_left[left] <= 0 or capacity_left[right] <= 0:
            return None
        capacity_left[left] -= 1
        capacity_left[right] -= 1
        if cid not in disputed_ids:
            coverage[candidate['group']] += 1
        fixed_entities.add(candidate['entity_id'])
    if len(fixed_pairs) > source['budget']:
        return None
    residual_quotas = {}
    for group, bounds in source['quotas'].items():
        covered = coverage[group]
        residual_min = max(0, bounds['min'] - covered)
        residual_max = bounds['max'] - covered
        if residual_max < 0:
            return None
        residual_quotas[group] = {'min': residual_min, 'max': residual_max}
    residual_universe = [
        row for row in universe if row['id'] not in fixed_set]
    return {
        'universe': residual_universe,
        'capacity_left': capacity_left,
        'residual_budget': source['budget'] - len(fixed_pairs),
        'residual_quotas': residual_quotas,
        # Entities owned by fixed tasks join the externally blocked entities so the
        # optimizer can never pick a second candidate sharing any of those entities.
        'blocked_entities': set(blocked_entities) | fixed_entities,
    }


def solve_assignment(universe, source, candidate_lookup, initial_fixed=None,
                     initial_blocked=None, disputed_ids=None):
    """Exact assignment under the full value -> count -> lexicographic objective.

    Returns ``(status, new_pairs)``; ``new_pairs`` lists only tasks added beyond
    ``initial_fixed`` as ``(candidate_id, smaller_annotator_id, larger_annotator_id)``.
    Fixed tasks keep their budget/capacity footprint; disputed fixed tasks keep the
    footprint but forfeit group coverage and block their entity.  The residual search
    (see :func:`optimize`) enumerates candidate subsets and proves staffing with a
    bipartite b-matching, so it never reassigns fixed people or swaps fixed samples.
    When no completion satisfies every group lower bound, the lower bounds are
    relaxed and the best feasible assignment is still returned with status
    ``infeasible``; the caller reports the honest residual gaps.
    """
    initial_fixed = list(initial_fixed or ())
    initial_blocked = set(initial_blocked or ())
    disputed_ids = set(disputed_ids or ())
    context = _residual_context(
        universe, candidate_lookup, source, initial_fixed, initial_blocked,
        disputed_ids)
    if context is None:
        return 'infeasible', []
    result = optimize(
        context['universe'], context['residual_budget'],
        context['capacity_left'], context['residual_quotas'],
        context['blocked_entities'])
    relaxed = False
    if result is None:
        relaxed_quotas = {group: {'min': 0, 'max': bounds['max']}
                          for group, bounds in
                          context['residual_quotas'].items()}
        result = optimize(
            context['universe'], context['residual_budget'],
            context['capacity_left'], relaxed_quotas,
            context['blocked_entities'])
        relaxed = True
        if result is None:
            return 'infeasible', []
    # The DFS already embedded the lexicographic tie-break into its third element.
    new_pairs = list(result[2])
    return ('infeasible' if relaxed else 'feasible'), new_pairs


# --------------------------------------------------------------------------- #
# Package construction, verification and resolution
# --------------------------------------------------------------------------- #

def _task(candidate, pair, generation, answers=None):
    return {
        'id': candidate['id'],
        'annotator_ids': [pair[0], pair[1]],
        'status': 'open',
        'answers': list(answers or ()),
        'generation': generation,
    }


def _status_from_answers(answers):
    if len(answers) == 0:
        return 'open'
    if len(answers) == 1:
        return 'pending'
    labels = {row['label'] for row in answers}
    return 'agreed' if len(labels) == 1 else 'disputed'


def _public_candidate(row):
    return {key: row[key] for key in
            ('id', 'entity_id', 'group', 'score', 'weight', 'eligible',
             'probability', 'value')}


def _capacity_used(tasks, annotators):
    used = {row['id']: 0 for row in annotators}
    for task in tasks:
        for ann_id in task['annotator_ids']:
            used[ann_id] += 1
    return used


def _coverage(tasks, candidate_map):
    coverage = {}
    for task in tasks:
        if task['status'] == 'disputed':
            continue
        group = candidate_map[task['id']]['group']
        coverage[group] = coverage.get(group, 0) + 1
    return coverage


def _truth(tasks, candidate_map):
    truth = []
    for task in tasks:
        if task['status'] != 'agreed':
            continue
        candidate = candidate_map[task['id']]
        label = task['answers'][0]['label']
        truth.append({'id': candidate['id'], 'score': candidate['score'],
                      'label': label, 'weight': candidate['weight']})
    truth.sort(key=lambda row: row['id'])
    return truth


def _gaps(coverage, quotas):
    return {group: max(0, bounds['min'] - coverage.get(group, 0))
            for group, bounds in quotas.items()}


def build_plan_package(data):
    """Validate plan input and return a sealed, deterministic annotation package."""
    source = validate_plan(data)
    blocks, candidates = calibrate(source)
    candidate_lookup = candidate_map(candidates)
    status, pairs = solve_assignment(candidates, source, candidate_lookup)
    tasks = [_task(candidate_lookup[cid], (left, right), 0)
             for cid, left, right in pairs]
    coverage = _coverage(tasks, candidate_lookup)
    gaps = _gaps(coverage, source['quotas'])
    package = _assemble_package(source, blocks, candidates, tasks, gaps, status)
    verify_package(package)
    return package


def candidate_map(candidates, cid=None):
    mapping = {row['id']: row for row in candidates}
    return mapping if cid is None else mapping[cid]


def _assemble_package(source, blocks, candidates, tasks, gaps, status):
    candidate_mapping = candidate_map(candidates)
    coverage = _coverage(tasks, candidate_mapping)
    usage = {
        'budget_used': len(tasks),
        'capacity_used': _capacity_used(tasks, source['annotators']),
    }
    body = {
        'schema_version': SCHEMA_VERSION,
        'status': status,
        'budget': source['budget'],
        'source': source,
        'training_blocks': blocks,
        'candidates': [_public_candidate(row) for row in candidates],
        'tasks': sorted(tasks, key=lambda task: task['id']),
        'usage': usage,
        'group_coverage': {group: coverage.get(group, 0)
                           for group in sorted(source['quotas'])},
        'gaps': {group: gaps[group] for group in sorted(gaps)},
        'truth': _truth(tasks, candidate_mapping),
    }
    body[SHA_FIELD] = package_digest(body)
    return body


def verify_package(package):
    """Recompute the digest and every internal invariant; raise on tampering."""
    if not isinstance(package, dict):
        raise AnnotationError('package must be a JSON object')
    if SHA_FIELD not in package or not isinstance(package[SHA_FIELD], str):
        raise AnnotationError('package is missing its self-check digest')
    if package_digest(package) != package[SHA_FIELD]:
        raise AnnotationError('package digest mismatch; refusing tampered package')
    if package.get('schema_version') != SCHEMA_VERSION:
        raise AnnotationError('unsupported package schema_version')
    if package.get('status') not in ('feasible', 'infeasible'):
        raise AnnotationError('package status must be feasible or infeasible')
    for field in ('source', 'training_blocks', 'candidates', 'tasks', 'usage',
                  'group_coverage', 'gaps', 'truth', 'budget'):
        if field not in package:
            raise AnnotationError('package missing field: ' + field)

    source = validate_plan(package['source'])
    if source['budget'] != package['budget']:
        raise AnnotationError('package budget disagrees with source')
    blocks, expected_candidates = calibrate(source)
    if blocks != package['training_blocks']:
        raise AnnotationError('training blocks disagree with the sealed training data')
    if [_public_candidate(row) for row in expected_candidates] != package['candidates']:
        raise AnnotationError('candidate probabilities/values disagree with training')
    candidate_mapping = candidate_map(expected_candidates)

    tasks = package['tasks']
    if not isinstance(tasks, list):
        raise AnnotationError('tasks must be an array')
    seen_tasks, seen_entities, generations = set(), set(), set()
    for task in tasks:
        _strict_fields(task, 'id annotator_ids status answers generation', 'task')
        cid = ident(task['id'])
        if cid not in candidate_mapping:
            raise AnnotationError('task references unknown candidate: ' + cid)
        if cid in seen_tasks:
            raise AnnotationError('duplicate task: ' + cid)
        seen_tasks.add(cid)
        candidate = candidate_mapping[cid]
        entity = candidate['entity_id']
        if entity in seen_entities:
            raise AnnotationError(f'two tasks share entity {entity}')
        seen_entities.add(entity)
        pair = task['annotator_ids']
        if (not isinstance(pair, list) or len(pair) != 2
                or pair[0] == pair[1]
                or any(ann not in candidate['eligible'] for ann in pair)):
            raise AnnotationError('task annotator pair must be two distinct eligible ids')
        if type(task['generation']) is not int or task['generation'] < 0:
            raise AnnotationError('task generation must be a nonnegative integer')
        generations.add(task['generation'])
        answers = task['answers']
        if not isinstance(answers, list) or len(answers) > 2:
            raise AnnotationError('task answers must be an array of at most two rows')
        answer_anns = set()
        for row in answers:
            _strict_fields(row, 'annotator_id label', 'stored answer')
            ann_id = ident(row['annotator_id'])
            if ann_id not in pair:
                raise AnnotationError('answer from an unassigned annotator')
            if ann_id in answer_anns:
                raise AnnotationError('duplicate stored answer slot')
            answer_anns.add(ann_id)
            if type(row['label']) is not int or row['label'] not in (0, 1) \
                    or isinstance(row['label'], bool):
                raise AnnotationError('stored label must be integer 0 or 1')
        ordered = sorted(answers, key=lambda row: row['annotator_id'])
        if ordered != answers:
            raise AnnotationError('task answers must be sorted by annotator id')
        expected_status = _status_from_answers(answers)
        if task['status'] != expected_status:
            raise AnnotationError(
                f'task {cid} status {task["status"]} disagrees with answers')

    # Usage, coverage, gaps and truth must all be reproducible from the tasks.
    expected_usage = {
        'budget_used': len(tasks),
        'capacity_used': _capacity_used(tasks, source['annotators']),
    }
    if package['usage'] != expected_usage:
        raise AnnotationError('package usage disagrees with tasks')
    declared_capacity = {row['id']: row['capacity'] for row in source['annotators']}
    for ann_id, used in expected_usage['capacity_used'].items():
        if used > declared_capacity[ann_id]:
            raise AnnotationError('annotator capacity exceeded in package')
    if len(tasks) > source['budget']:
        raise AnnotationError('package exceeds its sealed budget')

    coverage = _coverage(tasks, candidate_mapping)
    expected_coverage = {group: coverage.get(group, 0)
                         for group in sorted(source['quotas'])}
    if package['group_coverage'] != expected_coverage:
        raise AnnotationError('package group coverage disagrees with tasks')
    expected_gaps = _gaps(coverage, source['quotas'])
    if package['gaps'] != {group: expected_gaps[group] for group in sorted(expected_gaps)}:
        raise AnnotationError('package gaps disagree with tasks')
    if package['truth'] != _truth(tasks, candidate_mapping):
        raise AnnotationError(
            'truth dataset must be exactly the agreed task labels; forged agreed '
            'status is not accepted')
    expected_status = 'infeasible' if any(expected_gaps.values()) else 'feasible'
    if package['status'] != expected_status:
        raise AnnotationError('package status disagrees with quota gaps')
    return source, blocks, expected_candidates, tasks


def resolve_package(package, answers_data):
    """Accept an answer batch, update states, refill after disputes, reseal."""
    source, blocks, candidates, old_tasks = verify_package(package)
    claimed_sha, incoming = validate_answers(answers_data)
    if claimed_sha != package[SHA_FIELD]:
        raise AnnotationError('package_sha256 does not match the input package')
    candidate_mapping = candidate_map(candidates)
    task_by_id = {task['id']: task for task in old_tasks}

    # Reject the whole batch before mutating anything.
    for row in incoming:
        task = task_by_id.get(row['id'])
        if task is None:
            raise AnnotationError('unknown task: ' + row['id'])
        if row['annotator_id'] not in task['annotator_ids']:
            raise AnnotationError(
                f'annotator {row["annotator_id"]} was not assigned to task {row["id"]}')
        if any(stored['annotator_id'] == row['annotator_id']
               for stored in task['answers']):
            raise AnnotationError(
                f'task {row["id"]} already has an answer from {row["annotator_id"]}')

    tasks = [dict(task, answers=list(task['answers'])) for task in old_tasks]
    task_by_id = {task['id']: task for task in tasks}
    for row in incoming:
        task_by_id[row['id']]['answers'].append(
            {'annotator_id': row['annotator_id'], 'label': row['label']})
    for task in tasks:
        task['answers'].sort(key=lambda row: row['annotator_id'])
        task['status'] = _status_from_answers(task['answers'])

    blocked = {candidate_mapping[task['id']]['entity_id']
               for task in tasks if task['status'] == 'disputed'}
    disputed_ids = {task['id'] for task in tasks
                    if task['status'] == 'disputed'}
    assigned_ids = {task['id'] for task in tasks}
    # Refill only runs while a dispute opened coverage; pending tasks keep their quota.
    if blocked:
        fixed_pairs = [
            (task['id'], task['annotator_ids'][0], task['annotator_ids'][1])
            for task in tasks]
        # Refill candidates exclude already assigned ids only; the solver rebuilds
        # fixed coverage/footprint from the pairs and enforces entity blocking.
        refill_universe = [
            row for row in candidates if row['id'] not in assigned_ids]
        status, new_pairs = solve_assignment(
            refill_universe, source, candidate_mapping,
            initial_fixed=fixed_pairs, initial_blocked=blocked,
            disputed_ids=disputed_ids)
        generation = max((task['generation'] for task in tasks), default=-1) + 1
        for cid, left, right in new_pairs:
            tasks.append(_task(candidate_mapping[cid], (left, right), generation))
    else:
        status = 'feasible'

    # Refill optimizes only the refill portion: recompute the residual gaps from the
    # final task set and derive the package status from them (a refill that cannot
    # close every gap is still a complete, sealed resolution, just marked infeasible).
    tasks.sort(key=lambda task: task['id'])
    coverage = _coverage(tasks, candidate_mapping)
    gaps = _gaps(coverage, source['quotas'])
    status = 'infeasible' if any(gaps.values()) else 'feasible'

    resealed = _assemble_package(source, blocks, candidates, tasks, gaps, status)
    verify_package(resealed)
    return resealed, len(incoming)


def save_package(path, package):
    save(path, package)
