import pytest

from agent_inspect.exception import EvaluationError
from agent_inspect.metrics.utils.metrics_utils import get_majority_voted_score, get_config_or_default, match_to_int, \
    map_subgoal_validations_to_binary_matrix


def test_majority_voted_score_returns_correct_score():
    scores = {"A": 3, "B": 2, "C": 1}
    result = get_majority_voted_score(scores)
    assert result == "A"

def test_majority_voted_score_handles_tie_correctly():
    scores = {"A": 2, "B": 2}
    result = get_majority_voted_score(scores)
    assert result in ["A", "B"]

def test_config_or_default_returns_config_value_when_key_exists():
    config = {"key1": "value1", "key2": "value2"}
    result = get_config_or_default(config, "key1", "default_value")
    assert result == "value1"

def test_config_or_default_returns_default_when_key_does_not_exist():
    config = {"key1": "value1"}
    result = get_config_or_default(config, "key2", "default_value")
    assert result == "default_value"

def test_match_to_int_returns_correct_int_for_valid_completion():
    completion = "Grade: C"
    result = match_to_int(completion)
    assert result == 1

    completion = "Grade: I"
    result = match_to_int(completion)
    assert result == 0


def test_match_to_int_raises_error_for_invalid_completion():
    completion = "This is a just dummy judge explanation.\n\nGrade: X"
    with pytest.raises(EvaluationError, match="Internal Code: 050003, Error Message: Could not find the judge grade from the completion: This is a just dummy judge explanation.\n\nGrade: X"):
        match_to_int(completion)

def test_match_to_int_raises_error_when_no_match_found():
    completion = "No grade here"
    with pytest.raises(EvaluationError, match="Internal Code: 050003, Error Message: Could not find the judge grade from the completion: No grade here"):
        match_to_int(completion)


def test_map_subgoal_validations_handles_valid_completions():
    completions = ["Grade: C", "Grade: I", "Grade: C"]
    result = map_subgoal_validations_to_binary_matrix(completions)
    assert result == [1, 0, 1]

def test_map_subgoal_validations_skips_invalid_completions():
    completions = ["Grade: C", "Invalid Grade", "Grade: I"]
    result = map_subgoal_validations_to_binary_matrix(completions)
    assert result == [1, 0]

def test_map_subgoal_validations_returns_empty_for_all_invalid_completions():
    completions = ["Invalid Grade", "Another Invalid Grade"]
    result = map_subgoal_validations_to_binary_matrix(completions)
    assert result == []

def test_map_subgoal_validations_handles_empty_input():
    completions = []
    result = map_subgoal_validations_to_binary_matrix(completions)
    assert result == []
