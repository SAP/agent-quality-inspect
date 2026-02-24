import pytest

from agent_inspect.exception import InvalidInputValueError
from agent_inspect.models.tools import ErrorAnalysisDataSample, StatisticAnalysisResult
from agent_inspect.models.metrics import SubGoal, SubGoalValidationResult
from agent_inspect.tools import StatisticAnalysis


@pytest.fixture
def error_analysis_data_sample_1() -> ErrorAnalysisDataSample:
    # Judge score for subgoal 1: [0,0,0,0,1]
    dummy_subgoal1_validations = SubGoalValidationResult(
        is_completed=False,
        explanations=['DUMMY STRING', '\n\nGRADE: I', '\n\nTherefore, the task is INCOMPLETE.\n\nGRADE: I',
                      '\n\nGRADE: I', '\n\nGRADE: I',
                      ' the agent has validly terminated their portion of the dialogue per instructions.\n\nGRADE: C'],
        sub_goal=SubGoal(
            details='Agent books one way economy flight from DTW to SEA on 2024-05-17 with flights HAT097 and HAT251 for passenger Ivan Smith, no baggage, no insurance.')
    )
    # Judge score for subgoal 2: [0,0,0,0,0]
    dummy_subgoal2_validations = SubGoalValidationResult(
        is_completed=False,
        explanations=['DUMMY STRING', '\n\nGRADE: I', '\n\nTherefore, the task is INCOMPLETE.\n\nGRADE: I',
                      '\n\nGRADE: I', '\n\nGRADE: I',
                      ' the agent has validly terminated their portion of the dialogue per instructions.\n\nGRADE: I'],
        sub_goal=SubGoal(details='Agent charges $128 to gift_card_8516878 and $247 to credit_card_3563913.')
    )
    dummy_sample_data = ErrorAnalysisDataSample(
        data_sample_id=0,
        subgoal_validations=[dummy_subgoal1_validations, dummy_subgoal2_validations]
    )

    return dummy_sample_data


@pytest.fixture
def error_analysis_data_sample_2() -> ErrorAnalysisDataSample:
    # Judge score for subgoal 1: [1,1,1,1,1]
    dummy_subgoal1_validations = SubGoalValidationResult(
        is_completed=False,
        explanations=['DUMMY STRING', '\n\nGRADE: C', '\n\nTherefore, the task is INCOMPLETE.\n\nGRADE: C',
                      '\n\nGRADE: C', '\n\nGRADE: C',
                      ' the agent has validly terminated their portion of the dialogue per instructions.\n\nGRADE: C'],
        sub_goal=SubGoal(
            details='Agent books one way economy flight from DTW to SEA on 2024-05-17 with flights HAT097 and HAT251 for passenger Ivan Smith, no baggage, no insurance.')
    )
    # Judge score for subgoal 2: [0,0,0,0,0]
    dummy_subgoal2_validations = SubGoalValidationResult(
        is_completed=False,
        explanations=['DUMMY STRING', '\n\nGRADE: I', '\n\nTherefore, the task is INCOMPLETE.\n\nGRADE: I',
                      '\n\nGRADE: I', '\n\nGRADE: I',
                      ' the agent has validly terminated their portion of the dialogue per instructions.\n\nGRADE: I'],
        sub_goal=SubGoal(details='Agent charges $128 to gift_card_8516878 and $247 to credit_card_3563913.')
    )
    dummy_sample_data = ErrorAnalysisDataSample(
        data_sample_id=0,
        subgoal_validations=[dummy_subgoal1_validations, dummy_subgoal2_validations]
    )

    return dummy_sample_data

@pytest.fixture
def error_analysis_data_sample_no_explanation() -> ErrorAnalysisDataSample:
    # Subgoal with only one explanation
    dummy_subgoal_validations = SubGoalValidationResult(
        is_completed=False,
        explanations=['DUMMY STRING'],  # Only one explanation
        sub_goal=SubGoal(
            details='Agent books one way economy flight from DTW to SEA on 2024-05-17 with flights HAT097 and HAT251 for passenger Ivan Smith, no baggage, no insurance.')
    )
    dummy_sample_data = ErrorAnalysisDataSample(
        data_sample_id=0,
        subgoal_validations=[dummy_subgoal_validations]
    )

    return dummy_sample_data

@pytest.fixture
def error_analysis_data_sample_no_validation() -> ErrorAnalysisDataSample:
    dummy_sample_data = ErrorAnalysisDataSample(
        data_sample_id=0,
        subgoal_validations=[]
    )

    return dummy_sample_data


def test_statistic_analysis_compute1(error_analysis_data_sample_1: ErrorAnalysisDataSample):
    """Test the compute method of StatisticAnalysis."""
    result = StatisticAnalysis.compute_statistic_analysis_result(error_analysis_data_sample_1)

    # Check that the result is an instance of StatisticAnalysisResult
    assert isinstance(result, StatisticAnalysisResult)

    # Check that judge_expectation and judge_variance are floats
    assert isinstance(result.judge_expectation, float)
    assert isinstance(result.judge_std, float)

    # Check that the expectation and variance values are within valid ranges
    assert result.judge_expectation == 0.1
    assert result.judge_std == 0.2


def test_statistic_analysis_compute2(error_analysis_data_sample_2: ErrorAnalysisDataSample):
    """Test the compute method of StatisticAnalysis."""
    result = StatisticAnalysis.compute_statistic_analysis_result(error_analysis_data_sample_2)

    # Check that the result is an instance of StatisticAnalysisResult
    assert isinstance(result, StatisticAnalysisResult)

    # Check that judge_expectation and judge_variance are floats
    assert isinstance(result.judge_expectation, float)
    assert isinstance(result.judge_std, float)

    # Check that the expectation and variance values are within valid ranges
    assert result.judge_expectation == 0.5
    assert result.judge_std == 0.0

def test_statistic_analysis_no_explanation(error_analysis_data_sample_no_explanation: ErrorAnalysisDataSample):
    """Test the compute method of StatisticAnalysis with missing explanations."""
    with pytest.raises(InvalidInputValueError, match="Internal Code: 080008, Error Message: Each SubGoalValidationResult must contain at least one judge explanation besides the summarized one."):
        StatisticAnalysis.compute_statistic_analysis_result(error_analysis_data_sample_no_explanation)


def test_statistic_analysis_no_validation_result(error_analysis_data_sample_no_validation: ErrorAnalysisDataSample):
    """Test the compute method of StatisticAnalysis with missing explanations."""
    result = StatisticAnalysis.compute_statistic_analysis_result(error_analysis_data_sample_no_validation)
    result.judge_std = None
    result.judge_expectation = None
