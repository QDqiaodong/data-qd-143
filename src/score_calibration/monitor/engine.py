"""Pure exact drift model core.

No I/O and no SQLite: given a validated snapshot the engine returns every calendar
window metric (exact :class:`~fractions.Fraction` residuals) and the deterministic
episode list per cohort.  Predictions reuse the existing weighted PAVA blocks and the
left-closed staircase rule (last block whose lower bound is <= the score, clamped at
both ends) -- there is no interpolation and no probability clipping.

Episode state machine, per cohort, consuming the *complete* calendar (empty windows are
emitted and reset streaks; they are never skipped to glue streaks together):

* an episode opens only after ``trigger_periods`` consecutive *eligible* windows whose
  residual is ``>= trigger_delta``; ``start`` is the first threshold-crossing window of
  that streak (the streak is backtracked, not the firing window);
* while open, it closes only after ``clear_periods`` consecutive eligible windows with
  residual ``<= clear_delta``; ``end`` is the window that completes the clearing;
* any in-between eligible window resets the pending count; an ``insufficient`` window
  resets both pending counts but never closes an already open episode;
* after closing, the next window may start a new episode;
* peak residual is the maximum over every eligible window in [start, end] (or the
  current horizon while open); ties resolve to the earliest window.
"""
from __future__ import annotations

from bisect import bisect_right
from fractions import Fraction

from .contract import reference_blocks, window_dates


def _predict(blocks, score):
    """Left-closed staircase prediction, identical to the v1/annotation rule."""
    starts = [block['lower'] for block in blocks]
    index = max(0, bisect_right(starts, score) - 1)
    return Fraction(blocks[index]['probability'])


def _window_index(day, start, period_days):
    return (day - start).days // period_days


def compute(snapshot):
    """Return ``{'windows', 'episodes', 'blocks'}`` for a validated snapshot."""
    blocks = reference_blocks(snapshot)
    start = snapshot['start_date']
    period_days = snapshot['period_days']
    period_count = snapshot['period_count']
    min_weight = snapshot['min_weight']
    trigger_delta = snapshot['trigger_delta']
    clear_delta = snapshot['clear_delta']
    trigger_periods = snapshot['trigger_periods']
    clear_periods = snapshot['clear_periods']

    spans = window_dates(start, period_days, period_count)

    # cohort -> window index -> aggregate accumulators
    cohorts = {}
    for row in snapshot['observations']:
        cohort = cohorts.setdefault(
            row['cohort'],
            [{'weight': 0, 'numerator': Fraction(0)} for _ in range(period_count)])
        index = _window_index(row['date'], start, period_days)
        bucket = cohort[index]
        probability = _predict(blocks, row['score'])
        bucket['weight'] += row['weight']
        bucket['numerator'] += row['weight'] * (row['label'] - probability)

    windows, episodes = [], []
    for cohort in sorted(cohorts):
        buckets = cohorts[cohort]
        cohort_windows = []
        for index, bucket in enumerate(buckets):
            span_start, span_end = spans[index]
            if bucket['weight'] < min_weight:  # covers empty windows (weight 0)
                record = {
                    'cohort': cohort, 'window': index,
                    'start_date': span_start.isoformat(),
                    'end_date': span_end.isoformat(),
                    'total_weight': bucket['weight'],
                    'status': 'insufficient', 'residual': None,
                    '_residual': None,
                }
            else:
                residual = abs(bucket['numerator']) / bucket['weight']
                record = {
                    'cohort': cohort, 'window': index,
                    'start_date': span_start.isoformat(),
                    'end_date': span_end.isoformat(),
                    'total_weight': bucket['weight'],
                    'status': 'eligible', 'residual': str(residual),
                    '_residual': residual,
                }
            cohort_windows.append(record)
        windows.extend(cohort_windows)
        episodes.extend(
            _cohort_episodes(cohort, cohort_windows, trigger_delta, clear_delta,
                             trigger_periods, clear_periods))
    episodes.sort(key=lambda episode: (episode['cohort'], episode['start']))
    return {'windows': windows, 'episodes': episodes, 'blocks': blocks}


def _cohort_episodes(cohort, cohort_windows, trigger_delta, clear_delta,
                     trigger_periods, clear_periods):
    episodes = []
    open_episode = None
    trigger_pending = 0
    clear_pending = 0

    def note_peak(episode, record):
        # Only eligible windows participate; ties keep the earliest window.
        residual = record['_residual']
        if residual is None:
            return
        if episode['peak_value'] is None or residual > episode['peak_value']:
            episode['peak_value'] = residual
            episode['peak_window'] = record['window']

    for record in cohort_windows:
        index = record['window']
        if record['status'] == 'insufficient':
            # Insufficient windows reset pending counters but never close an episode.
            trigger_pending = 0
            clear_pending = 0
            if open_episode is not None:
                note_peak(open_episode, record)
            continue
        residual = record['_residual']
        if open_episode is None:
            if residual >= trigger_delta:
                trigger_pending += 1
                if trigger_pending >= trigger_periods:
                    # Backtrack the streak: start is the first crossing window.
                    open_episode = {
                        'cohort': cohort,
                        'start': index - trigger_periods + 1,
                        'end': None, 'open': True,
                        'peak_value': None, 'peak_window': None,
                    }
                    for streak_window in cohort_windows[open_episode['start']:index + 1]:
                        note_peak(open_episode, streak_window)
                    trigger_pending = 0
                    clear_pending = 0
            else:
                trigger_pending = 0
        else:
            note_peak(open_episode, record)
            if residual <= clear_delta:
                clear_pending += 1
                if clear_pending >= clear_periods:
                    open_episode['end'] = index
                    open_episode['open'] = False
                    episodes.append(_finalize(open_episode))
                    open_episode = None
                    trigger_pending = 0
                    clear_pending = 0
            else:
                # An in-between eligible window resets the clearing streak.
                clear_pending = 0

    if open_episode is not None:
        episodes.append(_finalize(open_episode))
    return episodes


def _finalize(episode):
    if episode['peak_value'] is None:  # defensive: trigger windows are eligible
        raise RuntimeError('episode opened without any eligible peak window')
    return {
        'cohort': episode['cohort'],
        'start': episode['start'],
        'end': episode['end'],
        'open': episode['open'],
        'peak_window': episode['peak_window'],
        'peak_value': str(episode['peak_value']),
        'payload': {
            'start': episode['start'],
            'end': episode['end'],
            'open': episode['open'],
            'peak_window': episode['peak_window'],
            'peak_value': str(episode['peak_value']),
        },
    }
