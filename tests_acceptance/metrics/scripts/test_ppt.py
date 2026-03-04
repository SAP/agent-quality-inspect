import pytest

from tests_acceptance.azure_openai_client import AzureOpenAIClient
from agent_inspect.metrics.scorer import PPT
from agent_inspect.metrics.constants import INCLUDE_JUDGE_EXPLANATION, MAX_TURNS, OPTIMIZE_JUDGE_TRIALS
from agent_inspect.metrics.adapters import Tau2BenchAdapter

from .utils import load_agent_trace, load_data_sample_dynamic


@pytest.fixture
def multiturn_agent_trace_1():
    return load_agent_trace("tests_acceptance/metrics/sample_data/example_multiturn_PAB_traces.json")

@pytest.fixture
def multiturn_agent_trace_2():
    tau2bench_adapter = Tau2BenchAdapter()
    conversation_data = tau2bench_adapter.load_json("tests_acceptance/metrics/sample_data/single_trace_tau2bench.json")
    tau2bench_trace = tau2bench_adapter.convert_to_agent_trace(conversation_data)
    return tau2bench_trace

@pytest.fixture
def static_multiturn_data_sample_1():
    return load_data_sample_dynamic("tests_acceptance/metrics/sample_data/example_dynamic_multi_turn_user_proxy_data_sample.json")

@pytest.fixture
def static_tau2bench_data_sample_2():
    return load_data_sample_dynamic("tests_acceptance/metrics/sample_data/example_dynamic_tau2bench_data_sample.json")

@pytest.fixture
def azure_openai_client():
    return AzureOpenAIClient(model="gpt-4.1", max_tokens=4096, temperature=0)


def test_ppt_score_with_multiturn_agent_trace_1(azure_openai_client, static_multiturn_data_sample_1, multiturn_agent_trace_1):
    metric = PPT(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    metric_result = metric.evaluate(agent_trace=multiturn_agent_trace_1, evaluation_data_sample=static_multiturn_data_sample_1)

    assert metric_result.score == 0.3333


def test_ppt_score_with_multiturn_agent_trace_2(azure_openai_client, static_tau2bench_data_sample_2, multiturn_agent_trace_2):
    metric = PPT(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True, MAX_TURNS: 15, OPTIMIZE_JUDGE_TRIALS: True})
    metric_result = metric.evaluate(agent_trace=multiturn_agent_trace_2, evaluation_data_sample=static_tau2bench_data_sample_2)

    assert round(metric_result.score, 4) == 0.3333
