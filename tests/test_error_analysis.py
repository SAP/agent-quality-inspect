import json
from unittest.mock import AsyncMock, MagicMock

import pytest
import asyncio

from agent_inspect.clients import LLMClient
from agent_inspect.metrics.constants import STATUS_200
from agent_inspect.exception import ToolError
from agent_inspect.models.metrics import SubGoal, SubGoalValidationResult
from agent_inspect.models import LLMResponse, LLMPayload
from agent_inspect.tools import ErrorAnalysis
from agent_inspect.tools.error_analysis.llm_constants import CLUSTERING_OUTPUT_SCHEMA, CLUSTERING_PROMPT_TEMPLATE
from agent_inspect.models.tools import AnalyzedSubgoalValidation, ErrorAnalysisResult, ErrorAnalysisDataSample

def mock_asyncio_run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

@pytest.fixture
def mock_llm_client():
    client = MagicMock(spec=LLMClient)
    client.make_request_with_payload = AsyncMock()
    return client

@pytest.fixture
def error_analysis(mock_llm_client):
    return ErrorAnalysis(mock_llm_client)

@pytest.fixture
def mock_subgoal_1():
    return SubGoal(details="Ensure calendar tool execution")

@pytest.fixture
def mock_subgoal_2():
    return SubGoal(details="Draft an email to the client")

@pytest.fixture
def mock_subgoal_validation_result_all_incomplete(mock_subgoal_1):
    return SubGoalValidationResult(
        is_completed=False,
        explanations=[
            "Overall explanation of failure.",
            "Judge Trial 1 Explanation. The agent failed to use the calendar tool correctly.\n\nGrade: I",
            "Judge Trial 2 Explanation. The agent did not schedule the meeting as required.\n\nGrade: I"
        ],
        sub_goal=mock_subgoal_1
    )

@pytest.fixture
def mock_subgoal_validation_result_some_complete(mock_subgoal_2):
    return SubGoalValidationResult(
        is_completed=False,
        explanations=[
            "Overall explanation of failure.",
            "Judge Trial 1 Explanation. The agent drafted the email but missed key details.\n\nGrade: C",
            "Judge Trial 2 Explanation. The agent failed to draft the email entirely.\n\nGrade: I"
        ],
        sub_goal=mock_subgoal_2
    )

@pytest.fixture
def mock_subgoal_validation_result_is_complete(mock_subgoal_2):
    return SubGoalValidationResult(
        is_completed=True,
        explanations=[
            "Overall explanation of success.",
            "Judge Trial 1 Explanation. The agent drafted the email correctly.\n\nGrade: C",
            "Judge Trial 2 Explanation. The agent drafted the email correctly.\n\nGrade: C"
        ],
        sub_goal=mock_subgoal_2
    )

@pytest.fixture
def mock_analyzed_subgoal_validation_1(mock_subgoal_validation_result_all_incomplete):
    return AnalyzedSubgoalValidation(
        subgoal_validation=mock_subgoal_validation_result_all_incomplete,
        data_sample_id=1,
        base_error="Tool misuse"
    )

@pytest.fixture
def mock_analyzed_subgoal_validation_2(mock_subgoal_validation_result_some_complete):
    return AnalyzedSubgoalValidation(
        subgoal_validation=mock_subgoal_validation_result_some_complete,
        data_sample_id=2,
        base_error="Incomplete task execution"
    )

@pytest.fixture
def mock_analyzed_subgoal_validation_3(mock_subgoal_validation_result_some_complete):
    return AnalyzedSubgoalValidation(
        subgoal_validation=mock_subgoal_validation_result_some_complete,
        data_sample_id=3,
        base_error="Task not fully completed"
    )

@pytest.fixture
def mock_analyzed_subgoal_validation_4(mock_subgoal_validation_result_is_complete):
    return AnalyzedSubgoalValidation(
        subgoal_validation=mock_subgoal_validation_result_is_complete,
        data_sample_id=4,
        base_error=None
    )

@pytest.fixture
def sample_llm_clusterings():
    return {
        "clusters": [
            {
                "cluster_label": "Calendar tool issues",
                "error_types": ["Tool misuse"],
                "error_ids": ["0"]
            },
            {
                "cluster_label": "Communication gaps",
                "error_types": ["Incomplete task execution"],
                "error_ids": ["1", "2"]
            }
        ]
    }


def test_get_judge_trial_explanations_returns_trials(error_analysis, mock_subgoal_validation_result_all_incomplete):
    judge_trials = error_analysis._get_judge_trial_explanations_from_subgoal_validation(mock_subgoal_validation_result_all_incomplete)

    assert judge_trials == [
        "Judge Trial 1 Explanation. The agent failed to use the calendar tool correctly.\n\nGrade: I",
        "Judge Trial 2 Explanation. The agent did not schedule the meeting as required.\n\nGrade: I"
    ]

@pytest.mark.parametrize(
    "explanations",
    [
        ["Only overall explanation present"],
        []
    ],
)
def test_get_judge_trial_explanations_raises_for_invalid_format(error_analysis, mock_subgoal_1, explanations):
    subgoal_validation = SubGoalValidationResult(
        is_completed=False,
        explanations=explanations,
        sub_goal=mock_subgoal_1
    )

    with pytest.raises(ValueError, match="Invalid SubGoalValidationResult.explanation format"):
        error_analysis._get_judge_trial_explanations_from_subgoal_validation(subgoal_validation)


def test_has_failed_consistently_returns_true_with_all_incomplete(error_analysis, mock_subgoal_validation_result_all_incomplete):
    assert error_analysis._has_failed_consistently(mock_subgoal_validation_result_all_incomplete) is True

def test_has_failed_consistently_returns_false_when_any_complete_string(error_analysis, mock_subgoal_validation_result_some_complete):
    assert error_analysis._has_failed_consistently(mock_subgoal_validation_result_some_complete) is False


@pytest.mark.asyncio
async def test_summarize_error_returns_error_type(error_analysis, mock_llm_client):
    mock_llm_client.make_request_with_payload.return_value = LLMResponse(
        status=STATUS_200,
        completion=json.dumps({"error_type": "Tool misuse", "explanation": "desc"})
    )

    result = await error_analysis._summarize_error("Judge explanation", "Use calendar tool")

    assert result == "Tool misuse"
    mock_llm_client.make_request_with_payload.assert_awaited_once()

@pytest.mark.asyncio
async def test_summarize_error_raises_when_llm_fails(error_analysis, mock_llm_client):
    mock_llm_client.make_request_with_payload.return_value = LLMResponse(
        status=500,
        completion=None,
        error_message="Boom"
    )

    with pytest.raises(ToolError, match="LLM request failed with status 500 and error: Boom"):
        await error_analysis._summarize_error("Judge explanation", "Use calendar tool")


@pytest.mark.asyncio
async def test_perform_majority_voting_returns_most_probable_error(error_analysis, mock_llm_client):
    mock_llm_client.make_request_with_payload.return_value = LLMResponse(
        status=STATUS_200,
        completion=json.dumps({"most_probable_error_type": "Tool misuse"})
    )

    result = await error_analysis._perform_majority_voting(["Err A", "Err B"])

    assert result == "Tool misuse"
    mock_llm_client.make_request_with_payload.assert_awaited_once()

@pytest.mark.asyncio
async def test_perform_majority_voting_raises_when_llm_fails(error_analysis, mock_llm_client):
    mock_llm_client.make_request_with_payload.return_value = LLMResponse(
        status=500,
        completion=None,
        error_message="Failure"
    )

    with pytest.raises(ToolError, match="LLM request failed with status 500 and error: Failure"):
        await error_analysis._perform_majority_voting(["Err A", "Err B"])


@pytest.mark.asyncio
async def test_cluster_errors_successfully_returns_clusters(
    error_analysis,
    mock_llm_client,
    mock_analyzed_subgoal_validation_1,
    mock_analyzed_subgoal_validation_2,
    mock_analyzed_subgoal_validation_3,
    sample_llm_clusterings
):
    mock_llm_client.make_request_with_payload.return_value = LLMResponse(
        status=STATUS_200,
        completion=json.dumps(sample_llm_clusterings)
    )
    analyzed = [
        mock_analyzed_subgoal_validation_1,
        mock_analyzed_subgoal_validation_2,
        mock_analyzed_subgoal_validation_3
    ]

    result = await error_analysis._cluster_errors(analyzed)

    expected_error_types = json.dumps(
        {
            "0": "Tool misuse",
            "1": "Incomplete task execution",
            "2": "Task not fully completed"
        },
        indent=2
    )
    expected_subgoals = json.dumps(
        [
            "Ensure calendar tool execution",
            "Draft an email to the client"
        ],
        indent=2
    )
    expected_payload = LLMPayload(
        user_prompt=CLUSTERING_PROMPT_TEMPLATE.format(
            error_types=expected_error_types,
            subgoals=expected_subgoals
        ),
        structured_output=CLUSTERING_OUTPUT_SCHEMA
    )

    mock_llm_client.make_request_with_payload.assert_called_with(expected_payload)
    assert result == sample_llm_clusterings

@pytest.mark.asyncio
async def test_cluster_errors_raises_when_llm_fails(error_analysis, mock_llm_client, mock_analyzed_subgoal_validation_1):
    analyzed = [mock_analyzed_subgoal_validation_1]
    mock_llm_client.make_request_with_payload.return_value = LLMResponse(
        status=500,
        completion=None,
        error_message="Failure"
    )

    with pytest.raises(ToolError, match="LLM request failed with status 500 and error: Failure"):
        _ = await error_analysis._cluster_errors(analyzed)


def test_build_clustered_result_maps_indices(
        error_analysis,
        mock_analyzed_subgoal_validation_1,
        mock_analyzed_subgoal_validation_2,
        mock_analyzed_subgoal_validation_3,
        sample_llm_clusterings
):
    result = error_analysis._build_clustered_result(
        sample_llm_clusterings,
        [
            mock_analyzed_subgoal_validation_1,
            mock_analyzed_subgoal_validation_2,
            mock_analyzed_subgoal_validation_3
        ]
    )

    assert isinstance(result, dict)
    assert result["Calendar tool issues"] == [mock_analyzed_subgoal_validation_1]
    assert result["Communication gaps"] == [mock_analyzed_subgoal_validation_2,
                                            mock_analyzed_subgoal_validation_3]


@pytest.mark.asyncio
async def test_summarize_errors_into_base_error_returns_none_when_completed(
    error_analysis,
    mock_subgoal_1
):
    completed_validation = SubGoalValidationResult(
        is_completed=True,
        explanations=["Overall: Completed", "Trial 1: C"],
        sub_goal=mock_subgoal_1
    )

    result = await error_analysis._summarize_errors_into_base_error(completed_validation)

    assert result is None

@pytest.mark.asyncio
async def test_summarize_errors_into_base_error_returns_base_error_when_consistent(
    error_analysis,
    monkeypatch,
    mock_subgoal_validation_result_all_incomplete
):
    monkeypatch.setattr(error_analysis, "_has_failed_consistently", lambda _: True)
    summarize_mock = AsyncMock(return_value="Base error")
    monkeypatch.setattr(error_analysis, "_summarize_error", summarize_mock)

    result = await error_analysis._summarize_errors_into_base_error(mock_subgoal_validation_result_all_incomplete)

    assert result == "Base error"
    summarize_mock.assert_awaited_once_with(
        mock_subgoal_validation_result_all_incomplete.explanations[1],
        mock_subgoal_validation_result_all_incomplete.sub_goal.details
    )

@pytest.mark.asyncio
async def test_summarize_errors_into_base_error_performs_majority_vote_when_inconsistent(
    error_analysis,
    monkeypatch,
    mock_subgoal_validation_result_some_complete
):
    monkeypatch.setattr(error_analysis, "_has_failed_consistently", lambda _: False)
    summarize_mock = AsyncMock(side_effect=["Error one", "Error two"])
    majority_mock = AsyncMock(return_value="Majority error")
    monkeypatch.setattr(error_analysis, "_summarize_error", summarize_mock)
    monkeypatch.setattr(error_analysis, "_perform_majority_voting", majority_mock)

    result = await error_analysis._summarize_errors_into_base_error(mock_subgoal_validation_result_some_complete)

    assert result == "Majority error"
    assert summarize_mock.await_count == 2
    majority_mock.assert_awaited_once_with(["Error one", "Error two"])


@pytest.mark.asyncio
async def test_analyze_returns_analyzed_subgoal_validations(
    error_analysis,
    monkeypatch,
    mock_subgoal_validation_result_all_incomplete,
    mock_subgoal_validation_result_some_complete
):
    # Mock _summarize_errors_into_base_error
    summarize_mock = AsyncMock(side_effect=["Tool misuse", "Incomplete task execution"])
    monkeypatch.setattr(error_analysis, "_summarize_errors_into_base_error", summarize_mock)
    
    # Create a data sample with two subgoal validations
    data_sample = ErrorAnalysisDataSample(
        data_sample_id=1,
        subgoal_validations=[
            mock_subgoal_validation_result_all_incomplete,
            mock_subgoal_validation_result_some_complete
        ],
        agent_run_id=1
    )
    
    result = await error_analysis._analyze(data_sample)
    
    # Verify the result
    assert len(result) == 2
    assert all(isinstance(asv, AnalyzedSubgoalValidation) for asv in result)
    assert result[0].data_sample_id == 1
    assert result[0].base_error == "Tool misuse"
    assert result[0].subgoal_validation == mock_subgoal_validation_result_all_incomplete
    assert result[0].agent_run_id == 1
    assert result[1].data_sample_id == 1
    assert result[1].base_error == "Incomplete task execution"
    assert result[1].subgoal_validation == mock_subgoal_validation_result_some_complete
    assert result[1].agent_run_id == 1
    assert summarize_mock.await_count == 2

@pytest.mark.asyncio
async def test_analyze_handles_empty_subgoal_validations(error_analysis):
    """Test that analyze() handles data samples with no subgoal validations."""
    data_sample = ErrorAnalysisDataSample(
        data_sample_id=1,
        subgoal_validations=[]
    )
    
    result = await error_analysis._analyze(data_sample)
    
    assert result == []


def test_split_asv_by_completeness_separates_correctly(
    error_analysis,
    mock_analyzed_subgoal_validation_1,
    mock_analyzed_subgoal_validation_2,
    mock_analyzed_subgoal_validation_4
):
    """Test that _split_asv_by_completeness correctly separates completed and incomplete validations."""
    complete_asv = mock_analyzed_subgoal_validation_4
    incomplete_asv_1 = mock_analyzed_subgoal_validation_1
    incomplete_asv_2 = mock_analyzed_subgoal_validation_2
    all_asvs = [
        [complete_asv, incomplete_asv_1],
        [incomplete_asv_2]
    ]
    
    completed, incomplete = error_analysis._split_analysed_subgoal_validations_by_completeness(all_asvs)
    
    assert len(completed) == 1
    assert len(incomplete) == 2
    assert completed[0] == complete_asv
    assert incomplete[0] == incomplete_asv_1
    assert incomplete[1] == incomplete_asv_2


def test_analyze_batch_returns_error_analysis_result(
    error_analysis,
    monkeypatch,
    mock_subgoal_validation_result_all_incomplete,
    mock_subgoal_validation_result_some_complete,
    mock_subgoal_validation_result_is_complete,
    mock_analyzed_subgoal_validation_1,
    mock_analyzed_subgoal_validation_2,
    mock_analyzed_subgoal_validation_3,
    mock_analyzed_subgoal_validation_4,
    sample_llm_clusterings
):
    """Test that analyze_batch() processes multiple data samples and returns clustered results."""
    # Create test data samples
    data_sample_1 = ErrorAnalysisDataSample(
        data_sample_id=1,
        subgoal_validations=[mock_subgoal_validation_result_all_incomplete,
                             mock_subgoal_validation_result_is_complete]
    )
    data_sample_2 = ErrorAnalysisDataSample(
        data_sample_id=2,
        subgoal_validations=[mock_subgoal_validation_result_some_complete,
                             mock_subgoal_validation_result_some_complete]
    )
    
    incomplete_asv_1 = mock_analyzed_subgoal_validation_1
    incomplete_asv_2 = mock_analyzed_subgoal_validation_2
    incomplete_asv_3 = mock_analyzed_subgoal_validation_3
    completed_asv = mock_analyzed_subgoal_validation_4

    # Mock analyze to return analyzed subgoal validations
    async def mock_analyze(data_sample):
        if data_sample.data_sample_id == 1:
            return [incomplete_asv_1, completed_asv]
        else:
            return [incomplete_asv_2, incomplete_asv_3]
    monkeypatch.setattr(error_analysis, "_analyze", mock_analyze)

    # Mock asyncio.run to execute the coroutine without creating a new event loop
    monkeypatch.setattr("asyncio.run", mock_asyncio_run)
    
    # Mock cluster_errors
    async def mock_cluster_errors(asv_list):
        return sample_llm_clusterings
    cluster_errors_mock = AsyncMock(side_effect=mock_cluster_errors)
    monkeypatch.setattr(error_analysis, "_cluster_errors", cluster_errors_mock)
    
    # Mock build_clustered_result
    expected_clustered = {
        "Calendar tool issues": [mock_analyzed_subgoal_validation_1],
        "Communication gaps": [mock_analyzed_subgoal_validation_2, mock_analyzed_subgoal_validation_3]
    }
    monkeypatch.setattr(error_analysis, "_build_clustered_result", lambda llm_cluster, asv_list: expected_clustered)

    result = error_analysis.analyze_batch([data_sample_1, data_sample_2])
    
    # Verify only incomplete validations were passed to clustering
    cluster_errors_mock.assert_called_once_with([
        mock_analyzed_subgoal_validation_1,
        mock_analyzed_subgoal_validation_2,
        mock_analyzed_subgoal_validation_3
    ])

    # Verify the result structure
    assert isinstance(result, ErrorAnalysisResult)
    assert result.analyzed_validations_clustered_by_errors == expected_clustered
    assert len(result.completed_subgoal_validations) == 1
    assert result.completed_subgoal_validations[0] == completed_asv

def test_analyze_batch_handles_empty_data_samples(error_analysis, monkeypatch):
    monkeypatch.setattr("asyncio.run", mock_asyncio_run)
    async def mock_cluster_errors(_):
        return {"clusters": []}
    monkeypatch.setattr(error_analysis, "_cluster_errors", mock_cluster_errors)
    monkeypatch.setattr(error_analysis, "_build_clustered_result", lambda llm_cluster, asv_list: {})
    
    result = error_analysis.analyze_batch([])
    
    assert isinstance(result, ErrorAnalysisResult)
    assert result.analyzed_validations_clustered_by_errors == {}
    assert result.completed_subgoal_validations == []

@pytest.mark.asyncio
async def test_retry_if_json_decode_error_returns_parsed_response_on_success(error_analysis, mock_llm_client):
    payload = LLMPayload(user_prompt="prompt", structured_output={})
    mock_response = LLMResponse(status=STATUS_200, completion=json.dumps({"key": "value"}))

    mock_llm_client.make_request_with_payload.return_value = mock_response
    result = await error_analysis._retry_if_json_decode_error(payload)

    assert result == {"key": "value"}


@pytest.mark.asyncio
async def test_retry_if_json_decode_error_raises_tool_error_on_non_200_status(monkeypatch, error_analysis, mock_llm_client):
    payload = LLMPayload(user_prompt="prompt", structured_output={})
    mock_response = LLMResponse(status=500, completion=None, error_message="Error")

    mock_llm_client.make_request_with_payload.return_value = mock_response

    with pytest.raises(ToolError, match="LLM request failed with status 500 and error: Error"):
        await error_analysis._retry_if_json_decode_error(payload)


@pytest.mark.asyncio
async def test_retry_if_json_decode_error_retries_on_json_decode_error(monkeypatch, error_analysis, mock_llm_client):
    payload = LLMPayload(user_prompt="prompt", structured_output={})
    mock_response = LLMResponse(status=STATUS_200, completion="invalid json")
    mock_llm_client.make_request_with_payload.return_value = mock_response

    with pytest.raises(ToolError, match="Maximum retry attempts exceeded for JSON decode error."):
        await error_analysis._retry_if_json_decode_error(payload)


@pytest.mark.asyncio
async def test_retry_if_json_decode_error_logs_warning_on_decode_error(monkeypatch, error_analysis, mock_llm_client, caplog):
    payload = LLMPayload(user_prompt="prompt", structured_output={})
    mock_response = LLMResponse(status=STATUS_200, completion="invalid json")
    mock_llm_client.make_request_with_payload.return_value = mock_response

    with pytest.raises(ToolError):
        await error_analysis._retry_if_json_decode_error(payload)

    assert "JSON decode error on attempt" in caplog.text


@pytest.mark.asyncio
async def test_summarize_error_raises_when_error_type_key_missing(error_analysis, mock_llm_client):
    mock_llm_client.make_request_with_payload.return_value = LLMResponse(
        status=STATUS_200,
        completion=json.dumps({"wrong_key": "some value", "explanation": "some explanation"})
    )

    with pytest.raises(Exception, match="LLM error summarization request failed as no error_type found in response"):
        await error_analysis._summarize_error("Judge explanation", "Use calendar tool")


@pytest.mark.asyncio
async def test_perform_majority_voting_raises_when_key_missing(error_analysis, mock_llm_client):
    mock_llm_client.make_request_with_payload.return_value = LLMResponse(
        status=STATUS_200,
        completion=json.dumps({"wrong_key": "some value", "explanation": "some explanation"})
    )

    with pytest.raises(Exception, match="LLM majority voting request failed as no most_probable_error_type found in response"):
        await error_analysis._perform_majority_voting(["Error A", "Error B"])


@pytest.mark.asyncio
async def test_retry_if_json_decode_error_raises_when_completion_is_none(error_analysis, mock_llm_client):
    payload = LLMPayload(user_prompt="prompt", structured_output={})
    mock_response = LLMResponse(status=STATUS_200, completion=None)
    
    mock_llm_client.make_request_with_payload.return_value = mock_response

    with pytest.raises(ToolError, match="Internal Code: 080014, Error Message: Maximum retry attempts exceeded for JSON decode error."):
        await error_analysis._retry_if_json_decode_error(payload)


@pytest.mark.asyncio
async def test_retry_if_json_decode_error_succeeds_on_third_attempt(error_analysis, mock_llm_client):
    """Test that retry succeeds after 2 failed attempts with JSON decode errors on the 3rd attempt."""
    payload = LLMPayload(user_prompt="prompt", structured_output={})
    
    # First two attempts return invalid JSON, third attempt succeeds
    invalid_response = LLMResponse(status=STATUS_200, completion="invalid json")
    valid_response = LLMResponse(status=STATUS_200, completion=json.dumps({"key": "value"}))
    
    mock_llm_client.make_request_with_payload.side_effect = [
        invalid_response,
        invalid_response,
        valid_response
    ]
    
    result = await error_analysis._retry_if_json_decode_error(payload)
    
    assert result == {"key": "value"}
    assert mock_llm_client.make_request_with_payload.call_count == 3


@pytest.mark.asyncio
async def test_retry_if_json_decode_error_succeeds_on_fourth_attempt(error_analysis, mock_llm_client):
    """Test that retry succeeds after 3 failed attempts with JSON decode errors on the 4th attempt."""
    payload = LLMPayload(user_prompt="prompt", structured_output={})
    
    # First three attempts return invalid JSON, fourth attempt succeeds
    invalid_response = LLMResponse(status=STATUS_200, completion="invalid json")
    valid_response = LLMResponse(status=STATUS_200, completion=json.dumps({"result": "success"}))
    
    mock_llm_client.make_request_with_payload.side_effect = [
        invalid_response,
        invalid_response,
        invalid_response,
        valid_response
    ]
    
    result = await error_analysis._retry_if_json_decode_error(payload)
    
    assert result == {"result": "success"}
    assert mock_llm_client.make_request_with_payload.call_count == 4


@pytest.mark.asyncio
async def test_retry_if_json_decode_error_exhausts_all_retries_and_fails(error_analysis, mock_llm_client, caplog):
    """Test that retry exhausts all attempts and raises ToolError with 'Maximum retry attempts exceeded' message."""
    payload = LLMPayload(user_prompt="prompt", structured_output={})
    
    # All attempts return invalid JSON
    invalid_response = LLMResponse(status=STATUS_200, completion="invalid json")
    mock_llm_client.make_request_with_payload.return_value = invalid_response
    
    with pytest.raises(ToolError, match="Maximum retry attempts exceeded for JSON decode error."):
        await error_analysis._retry_if_json_decode_error(payload)
    
    # Verify that all retry attempts were made (default is 5 from MAX_RETRY_JSON_DECODE_ERROR constant)
    assert mock_llm_client.make_request_with_payload.call_count == 5
    
    # Verify that warnings were logged for each failed attempt
    assert "JSON decode error on attempt 1/5" in caplog.text
    assert "JSON decode error on attempt 2/5" in caplog.text
    assert "JSON decode error on attempt 3/5" in caplog.text
    assert "JSON decode error on attempt 4/5" in caplog.text
    assert "JSON decode error on attempt 5/5" in caplog.text


