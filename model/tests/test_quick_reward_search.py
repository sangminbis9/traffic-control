"""Selection and holdout safeguards for a bounded search, without simulation."""

import pytest

from model.experiments.quick_reward_search import (
    confirm_locked_winner, select_validation, summarize_episodes,
)


def episode(waiting, throughput=100, collisions=0):
    return {
        'avg_waiting_time': waiting, 'max_waiting_time': 60,
        'avg_queue': 12, 'phase_changes': 30, 'throughput': throughput,
        'max_queue': 24, 'departed': 120, 'unfinished': 120 - throughput,
        'pending': 3, 'max_pending': 7, 'forced_changes': 2, 'duration': 300,
        'episode_reward': -50, 'collisions': collisions, 'teleports': 0,
    }


def results(values):
    rows = {candidate: [episode(*value)] for candidate, value in values.items()}
    summaries = [
        {'experiment': candidate, **summarize_episodes(episodes)}
        for candidate, episodes in rows.items()
    ]
    return summaries, rows


def test_selection_rejects_unsafe_and_low_throughput_and_keeps_baseline():
    summaries, rows = results({1: (20,), 2: (18,), 3: (19,), 4: (5, 94), 5: (4, 100, 1)})
    selected = select_validation(summaries, rows)
    assert selected['validation_winner_candidate_id'] == 2
    assert selected['shortlist_candidate_ids'] == [2, 3]
    assert selected['holdout_candidate_ids'] == [1, 2, 3]
    assert not selected['eligibility'][4]['eligible']
    assert not selected['eligibility'][5]['eligible']


def test_holdout_cannot_promote_runner_up_when_locked_winner_gets_worse():
    validation, validation_rows = results({1: (20,), 2: (18,), 3: (19,)})
    locked = select_validation(validation, validation_rows)
    holdout, holdout_rows = results({1: (20,), 2: (22,), 3: (15,)})
    confirmation = confirm_locked_winner(locked, holdout, holdout_rows)
    assert confirmation['candidate_id'] == 2
    assert not confirmation['confirmed']
    assert not confirmation['holdout_reselection_performed']
    assert confirmation['mean_waiting_change_pct'] == pytest.approx(10)


def test_holdout_confirms_direction_but_does_not_claim_significance():
    summaries, rows = results({1: (20,), 2: (18, 95)})
    confirmation = confirm_locked_winner(select_validation(summaries, rows), summaries, rows)
    assert confirmation['confirmed']
    assert confirmation['throughput_retention'] == pytest.approx(.95)
    assert confirmation['mean_paired_waiting_difference_seconds'] == pytest.approx(-2)
    assert not confirmation['statistical_significance_established']


def test_missing_holdout_baseline_prevents_confirmation():
    summaries, rows = results({1: (20,), 2: (18,)})
    locked = select_validation(summaries, rows)
    confirmation = confirm_locked_winner(locked, summaries[1:], {2: rows[2]})
    assert not confirmation['confirmed']
    assert confirmation['reason'] == 'holdout_evaluation_failed'


def test_pending_and_unfinished_diagnostics_are_preserved():
    summary = summarize_episodes([episode(20), episode(10, 110)])
    assert summary['pending'] == 3
    assert summary['maximum_pending_worst_episode'] == 7
    assert summary['unfinished'] == 15
    assert summary['avg_waiting_time_std'] == pytest.approx(50 ** .5)
