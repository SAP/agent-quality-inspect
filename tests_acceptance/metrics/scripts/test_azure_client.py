import pytest

from dotenv import load_dotenv

from agent_inspect.clients import AzureOpenAIClient
from agent_inspect.metrics.constants import INCLUDE_JUDGE_EXPLANATION, MAX_TURNS
from agent_inspect.metrics.scorer import AUC, SuccessScore

from tests_acceptance.metrics.scripts.utils import load_agent_trace, load_data_sample_dynamic, load_data_sample_static


@pytest.fixture
def multiturn_agent_trace_1():
    return load_agent_trace("tests_acceptance/metrics/sample_data/example_multiturn_PAB_traces.json")

@pytest.fixture
def static_multiturn_data_sample_1(): 
    return load_data_sample_static("tests_acceptance/metrics/sample_data/example_static_multi_turn_data_sample_turn_specific.json")  

@pytest.fixture
def dynamic_sample_1(): 
    return load_data_sample_dynamic("tests_acceptance/metrics/sample_data/example_dynamic_multi_turn_user_proxy_data_sample.json")  

@pytest.fixture
def azure_openai_client():
    load_dotenv()
    client = AzureOpenAIClient(model="gpt-4.1", max_tokens=4096, temperature=0)
    return client


@pytest.mark.asyncio
async def test_client_api_call(azure_openai_client):
    prompt = "This is a test prompt. Reply with 'Test Success.'"
    response = await azure_openai_client.make_llm_request(prompt)

    assert response.status == 200
    assert response.completion is not None
    assert "Test Success" in response.completion

def test_auc_metric(azure_openai_client, dynamic_sample_1, multiturn_agent_trace_1):
    metric = AUC(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True, MAX_TURNS: 20})
    metric_result = metric.evaluate(agent_trace=multiturn_agent_trace_1, evaluation_data_sample=dynamic_sample_1)
    assert round(metric_result.score, 4) == 0.9868

def test_success_score(azure_openai_client, static_multiturn_data_sample_1, multiturn_agent_trace_1):
    metric = SuccessScore(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    metric_result = metric.evaluate(agent_trace=multiturn_agent_trace_1, evaluation_data_sample=static_multiturn_data_sample_1)
    assert metric_result.score == 1.0
