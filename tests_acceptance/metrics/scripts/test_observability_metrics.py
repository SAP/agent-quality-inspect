import pytest

from agent_inspect.metrics.observed import (
    TotalLatency, AverageLatency, InputTotalTokenCount, 
    OutputTotalTokenCount, ReasoningTotalTokenCount, 
    TotalTokenConsumption, ToolCallCount
)
from tests_acceptance.metrics.scripts.utils import load_agent_trace

@pytest.fixture
def multiturn_agent_trace_1():
    return load_agent_trace("tests_acceptance/metrics/sample_data/example_multiturn_PAB_traces.json")   

@pytest.fixture
def singleturn_agent_trace_1():
    return load_agent_trace("tests_acceptance/metrics/sample_data/example_singleturn_PAB_traces.json")   

def test_total_latency_with_singleturn_agent_trace_1(singleturn_agent_trace_1):
    metric = TotalLatency()
    metric_result = metric.evaluate(agent_turn_traces=singleturn_agent_trace_1.turns)
    assert metric_result.score == 24805.0

def test_avg_latency_with_singleturn_agent_trace_1(singleturn_agent_trace_1):
    metric = AverageLatency()
    metric_result = metric.evaluate(agent_turn_traces=singleturn_agent_trace_1.turns)
    assert metric_result.score == 24805.0

def test_total_latency_with_multiturn_agent_trace_1(multiturn_agent_trace_1):
    metric = TotalLatency()
    metric_result = metric.evaluate(agent_turn_traces=multiturn_agent_trace_1.turns)
    assert metric_result.score == 33005.0

def test_avg_latency_with_multiturn_agent_trace_1(multiturn_agent_trace_1):
    metric = AverageLatency()
    metric_result = metric.evaluate(agent_turn_traces=multiturn_agent_trace_1.turns)
    assert metric_result.score == 11001.6667


def test_total_input_tokens_with_multiturn_agent_trace_1(multiturn_agent_trace_1):
    metric = InputTotalTokenCount()
    metric_result = metric.evaluate(agent_turn_traces=multiturn_agent_trace_1.turns)
    assert metric_result.score == 63158

def test_total_output_tokens_with_multiturn_agent_trace_1(multiturn_agent_trace_1):
    metric = OutputTotalTokenCount()
    metric_result = metric.evaluate(agent_turn_traces=multiturn_agent_trace_1.turns)
    assert metric_result.score == 2085

def test_total_reasoning_tokens_with_multiturn_agent_trace_1(multiturn_agent_trace_1):
    metric = ReasoningTotalTokenCount()
    metric_result = metric.evaluate(agent_turn_traces=multiturn_agent_trace_1.turns)
    assert metric_result.score == 0

def test_total_tokens_with_multiturn_agent_trace_1(multiturn_agent_trace_1):
    metric = TotalTokenConsumption()
    metric_result = metric.evaluate(agent_turn_traces=multiturn_agent_trace_1.turns)
    assert metric_result.score == 65243


def test_tool_call_count_with_multiturn_agent_trace_1(multiturn_agent_trace_1):
    metric = ToolCallCount()
    metric_result = metric.evaluate(agent_turn_traces=multiturn_agent_trace_1.turns)
    assert metric_result.score == 5