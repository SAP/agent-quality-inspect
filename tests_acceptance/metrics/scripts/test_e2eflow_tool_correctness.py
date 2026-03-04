import pytest
import asyncio

from agent_inspect.metrics.constants import INCLUDE_JUDGE_EXPLANATION
from tests_acceptance.azure_openai_client import AzureOpenAIClient
from agent_inspect.models.metrics import (
    AgentDialogueTrace,
    TurnTrace,
    Step,
    AgentResponse,
    ExpectedToolCall,
    ToolInputParameter,
    ToolOutput,
)
from agent_inspect.metrics.scorer import ToolCorrectnessMetric
from agent_inspect.metrics.validator import ToolCallCompletionValidator
from agent_inspect.metrics.adapters import Tau2BenchAdapter

from tests_acceptance.metrics.scripts.utils import load_agent_trace, load_data_sample_static


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
def singleturn_agent_trace_1():
    return load_agent_trace("tests_acceptance/metrics/sample_data/example_singleturn_PAB_traces.json")   


@pytest.fixture
def static_multiturn_data_sample_1(): 
    return load_data_sample_static("tests_acceptance/metrics/sample_data/example_static_multi_turn_data_sample_turn_specific.json") 
 
@pytest.fixture
def static_multiturn_data_sample_2(): 
    return load_data_sample_static("tests_acceptance/metrics/sample_data/example_static_multi_turn_data_sample_turn_specific_2.json")  

@pytest.fixture
def static_multiturn_data_sample_3(): 
    return load_data_sample_static("tests_acceptance/metrics/sample_data/example_static_multi_turn_tau2bench_data_sample.json")  

@pytest.fixture
def static_singleturn_data_sample_1(): 
    return load_data_sample_static("tests_acceptance/metrics/sample_data/example_static_single_turn_data_sample.json")  

@pytest.fixture
def static_singleturn_data_sample_2(): 
    return load_data_sample_static("tests_acceptance/metrics/sample_data/example_static_single_turn_data_sample_2.json")  


@pytest.fixture
def custom_trace_tool_1():
    tool_input_args = [ToolInputParameter(name='reminder_message', value='Purchase a loaf of bread'),
                       ToolInputParameter(name='time', value='10:00 AM')]
    step_obj = Step(
        id="custom_tool_trace_1",
        parent_ids=["0"],
        tool="reminder_tool",
        tool_input_args=tool_input_args,
    )
    turn_obj = TurnTrace(
        id="turn_custom_tool_trace_1",
        agent_input="Mock agent input",
        agent_response= AgentResponse(response="Mock agent response", status_code="200"),
        steps=[step_obj]
    )
    agent_trace = AgentDialogueTrace(turns=[turn_obj])
    return agent_trace

@pytest.fixture
def custom_trace_tool_2():
    tool_input_args = [ToolInputParameter(name='reminder_message', value='Purchase a loaf of bread at 10:00 AM'),
                       ToolInputParameter(name='time', value='10:00 AM')]
    step_obj = Step(
        id="custom_tool_trace_2",
        parent_ids=["0"],
        tool="reminder_tool",
        tool_input_args=tool_input_args,
    )
    turn_obj = TurnTrace(
        id="turn_custom_tool_trace_2",
        agent_input="Mock agent input",
        agent_response= AgentResponse(response="Mock agent response", status_code="200"),
        steps=[step_obj]
    )
    agent_trace = AgentDialogueTrace(turns=[turn_obj])
    return agent_trace

@pytest.fixture
def custom_trace_tool_3():
    tool_input_args = [ToolInputParameter(name='reminder_message', value='Purchase a loaf of bread at 11:00 AM'),
                       ToolInputParameter(name='time', value='10:00 AM')]
    step_obj = Step(
        id="custom_tool_trace_3",
        parent_ids=["0"],
        tool="reminder_tool",
        tool_input_args=tool_input_args,
    )
    turn_obj = TurnTrace(
        id="turn_custom_tool_trace_3",
        agent_input="Mock agent input",
        agent_response= AgentResponse(response="Mock agent response", status_code="200"),
        steps=[step_obj]
    )
    agent_trace = AgentDialogueTrace(turns=[turn_obj])
    return agent_trace


@pytest.fixture
def custom_trace_tool_4():
    step_obj = Step(
        id="custom_tool_trace_4",
        parent_ids=["0"],
        tool="calculator",
        tool_output="234.00"
    )
    turn_obj = TurnTrace(
        id="turn_custom_tool_trace_4",
        agent_input="Mock agent input",
        agent_response= AgentResponse(response="Mock agent response", status_code="200"),
        steps=[step_obj]
    )
    agent_trace = AgentDialogueTrace(turns=[turn_obj])
    return agent_trace

@pytest.fixture
def custom_trace_tool_5():
    step_obj = Step(
        id="custom_tool_trace_5",
        parent_ids=["0"],
        tool="book_flight",
        tool_output="Booked flight to New York on July 10th at 3 PM"
    )
    turn_obj = TurnTrace(
        id="turn_custom_tool_trace_5",
        agent_input="Mock agent input",
        agent_response= AgentResponse(response="Mock agent response", status_code="200"),
        steps=[step_obj]
    )
    agent_trace = AgentDialogueTrace(turns=[turn_obj])
    return agent_trace

@pytest.fixture
def custom_trace_tool_6():
    step_obj = Step(
        id="custom_tool_trace_6",
        parent_ids=["0"],
        tool="calculator",
        tool_output="0.33333"
    )
    turn_obj = TurnTrace(
        id="turn_custom_tool_trace_6",
        agent_input="Mock agent input",
        agent_response= AgentResponse(response="Mock agent response", status_code="200"),
        steps=[step_obj]
    )
    agent_trace = AgentDialogueTrace(turns=[turn_obj])
    return agent_trace


@pytest.fixture
def custom_expected_tool_sample_1():
    expected_parameters= [ToolInputParameter(name='reminder_message', check='Buy a loaf of bread at 10:00 AM tmr'),
                          ToolInputParameter(name='time', value='10:00 AM')]
    expected_tool = ExpectedToolCall(tool='reminder_tool', expected_parameters=expected_parameters, turn=0)
    return expected_tool

@pytest.fixture
def custom_expected_tool_sample_2():
    expected_parameters= [ToolInputParameter(name='reminder_message', check='Buy a loaf of bread'),
                          ToolInputParameter(name='time', value='10:00 AM')]
    expected_tool = ExpectedToolCall(tool='reminder_tool', expected_parameters=expected_parameters, turn=0)
    return expected_tool

@pytest.fixture
def custom_expected_tool_sample_3():
    expected_parameters= [ToolInputParameter(name='reminder_message', check='Buy a loaf of bread'),
                          ToolInputParameter(name='time', value='10:00 AM')]
    expected_tool = ExpectedToolCall(tool='reminder_tool', expected_parameters=expected_parameters, turn=0)
    return expected_tool

@pytest.fixture
def custom_expected_tool_sample_4():
    expected_output=ToolOutput(check='234')  
    expected_tool = ExpectedToolCall(tool='calculator', expected_output=expected_output, turn=0)
    return expected_tool

@pytest.fixture
def custom_expected_tool_sample_5():
    expected_output=ToolOutput(check='Reserved a seat on flight to New York on July 10th at 4 PM')  
    expected_tool = ExpectedToolCall(tool='book_flight', expected_output=expected_output, turn=0)
    return expected_tool

@pytest.fixture
def custom_expected_tool_sample_6():
    expected_output=ToolOutput(check='1/3')  
    expected_tool = ExpectedToolCall(tool='calculator', expected_output=expected_output, turn=0)
    return expected_tool

@pytest.fixture
def azure_openai_client():
    return AzureOpenAIClient(model="gpt-4.1", max_tokens=4096, temperature=0)

def test_custom_expected_tool_1_complete(azure_openai_client, custom_expected_tool_sample_1, custom_trace_tool_1):
    agent_trace =  custom_trace_tool_1.turns
    expected_tool = custom_expected_tool_sample_1
    tool_validator = ToolCallCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    res = asyncio.run(tool_validator.validate(agent_trace_turns = agent_trace, expected_tool_call = expected_tool))
    assert res.is_completed == True

def test_custom_expected_tool_2_complete(azure_openai_client, custom_expected_tool_sample_2, custom_trace_tool_2):
    agent_trace =  custom_trace_tool_2.turns
    expected_tool = custom_expected_tool_sample_2
    tool_validator = ToolCallCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    res = asyncio.run(tool_validator.validate(agent_trace_turns = agent_trace, expected_tool_call = expected_tool))
    assert res.is_completed == True

def test_custom_expected_tool_3_incomplete(azure_openai_client, custom_expected_tool_sample_3, custom_trace_tool_3):
    agent_trace =  custom_trace_tool_3.turns
    expected_tool = custom_expected_tool_sample_3
    tool_validator = ToolCallCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    res = asyncio.run(tool_validator.validate(agent_trace_turns = agent_trace, expected_tool_call = expected_tool))
    assert res.is_completed == False


def test_custom_expected_tool_4_complete(azure_openai_client, custom_expected_tool_sample_4, custom_trace_tool_4):
    agent_trace =  custom_trace_tool_4.turns
    expected_tool = custom_expected_tool_sample_4
    tool_validator = ToolCallCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    res = asyncio.run(tool_validator.validate(agent_trace_turns = agent_trace, expected_tool_call = expected_tool))
    assert res.is_completed == True


def test_custom_expected_tool_5_incomplete(azure_openai_client, custom_expected_tool_sample_5, custom_trace_tool_5):
    agent_trace =  custom_trace_tool_5.turns
    expected_tool = custom_expected_tool_sample_5
    tool_validator = ToolCallCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    res = asyncio.run(tool_validator.validate(agent_trace_turns = agent_trace, expected_tool_call = expected_tool))
    assert res.is_completed == False

def test_custom_expected_tool_6_complete(azure_openai_client, custom_expected_tool_sample_6, custom_trace_tool_6):
    agent_trace =  custom_trace_tool_6.turns
    expected_tool = custom_expected_tool_sample_6
    tool_validator = ToolCallCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    res = asyncio.run(tool_validator.validate(agent_trace_turns = agent_trace, expected_tool_call = expected_tool))
    assert res.is_completed == True


def test_expected_tool_0_complete_with_static_multiturn_sample_1_trace_1(azure_openai_client, static_multiturn_data_sample_1, multiturn_agent_trace_1):
    turn_idx = 0
    tool_idx = 0
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    expected_tool = static_multiturn_data_sample_1.expected_tool_calls[tool_idx]
    tool_validator = ToolCallCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    res = asyncio.run(tool_validator.validate(agent_trace_turns = agent_trace, expected_tool_call = expected_tool))
    assert res.is_completed == True
    
def test_expected_tool_1_complete_with_static_multiturn_sample_1_trace_1(azure_openai_client, static_multiturn_data_sample_1, multiturn_agent_trace_1):
    turn_idx = 0
    tool_idx = 1
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    expected_tool = static_multiturn_data_sample_1.expected_tool_calls[tool_idx]
    tool_validator = ToolCallCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    res = asyncio.run(tool_validator.validate(agent_trace_turns = agent_trace, expected_tool_call = expected_tool))
    assert res.is_completed == True

def test_expected_tool_2_complete_with_static_multiturn_sample_1_trace_1(azure_openai_client, static_multiturn_data_sample_1, multiturn_agent_trace_1):
    turn_idx = 0
    tool_idx = 2
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    expected_tool = static_multiturn_data_sample_1.expected_tool_calls[tool_idx]
    tool_validator = ToolCallCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    res = asyncio.run(tool_validator.validate(agent_trace_turns = agent_trace, expected_tool_call = expected_tool))
    assert res.is_completed == True

def test_expected_tool_3_complete_with_static_multiturn_sample_1_trace_1(azure_openai_client, static_multiturn_data_sample_1, multiturn_agent_trace_1):
    turn_idx = 2
    tool_idx = 3
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    expected_tool = static_multiturn_data_sample_1.expected_tool_calls[tool_idx]
    tool_validator = ToolCallCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    res = asyncio.run(tool_validator.validate(agent_trace_turns = agent_trace, expected_tool_call = expected_tool))
    assert res.is_completed == True


def test_expected_tool_0_complete_with_static_multiturn_sample_2_trace_1(azure_openai_client, static_multiturn_data_sample_2, multiturn_agent_trace_1):
    turn_idx = 0
    tool_idx = 0
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    expected_tool = static_multiturn_data_sample_2.expected_tool_calls[tool_idx]
    tool_validator = ToolCallCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    res = asyncio.run(tool_validator.validate(agent_trace_turns = agent_trace, expected_tool_call = expected_tool))
    assert res.is_completed == True

def test_expected_tool_1_incomplete_with_static_multiturn_sample_2_trace_1(azure_openai_client, static_multiturn_data_sample_2, multiturn_agent_trace_1):
    turn_idx = 0
    tool_idx = 1
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    expected_tool = static_multiturn_data_sample_2.expected_tool_calls[tool_idx]
    tool_validator = ToolCallCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    res = asyncio.run(tool_validator.validate(agent_trace_turns = agent_trace, expected_tool_call = expected_tool))
    assert res.is_completed == False


def test_expected_tool_2_complete_with_static_multiturn_sample_2_trace_1(azure_openai_client, static_multiturn_data_sample_2, multiturn_agent_trace_1):
    turn_idx = 0
    tool_idx = 2
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    expected_tool = static_multiturn_data_sample_2.expected_tool_calls[tool_idx]
    tool_validator = ToolCallCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    res = asyncio.run(tool_validator.validate(agent_trace_turns = agent_trace, expected_tool_call = expected_tool))
    assert res.is_completed == True

def test_expected_tool_3_complete_with_static_multiturn_sample_2_trace_1(azure_openai_client, static_multiturn_data_sample_2, multiturn_agent_trace_1):
    turn_idx = 0
    tool_idx = 3
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    expected_tool = static_multiturn_data_sample_2.expected_tool_calls[tool_idx]
    tool_validator = ToolCallCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    res = asyncio.run(tool_validator.validate(agent_trace_turns = agent_trace, expected_tool_call = expected_tool))
    assert res.is_completed == True

def test_expected_tool_4_incomplete_with_static_multiturn_sample_2_trace_1(azure_openai_client, static_multiturn_data_sample_2, multiturn_agent_trace_1):
    turn_idx = 2
    tool_idx = 4
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    expected_tool = static_multiturn_data_sample_2.expected_tool_calls[tool_idx]
    tool_validator = ToolCallCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    res = asyncio.run(tool_validator.validate(agent_trace_turns = agent_trace, expected_tool_call = expected_tool))
    assert res.is_completed == False


def test_expected_tool_0_complete_with_static_multiturn_sample_3_trace_2(azure_openai_client, static_multiturn_data_sample_3, multiturn_agent_trace_2):
    turn_idx = 1
    tool_idx = 0
    agent_trace =  multiturn_agent_trace_2.turns[:turn_idx+1]
    expected_tool = static_multiturn_data_sample_3.expected_tool_calls[tool_idx]
    tool_validator = ToolCallCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    res = asyncio.run(tool_validator.validate(agent_trace_turns = agent_trace, expected_tool_call = expected_tool))
    assert res.is_completed == True

def test_expected_tool_1_complete_with_static_multiturn_sample_3_trace_2(azure_openai_client, static_multiturn_data_sample_3, multiturn_agent_trace_2):
    turn_idx = 1
    tool_idx = 1
    agent_trace =  multiturn_agent_trace_2.turns[:turn_idx+1]
    expected_tool = static_multiturn_data_sample_3.expected_tool_calls[tool_idx]
    tool_validator = ToolCallCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    res = asyncio.run(tool_validator.validate(agent_trace_turns = agent_trace, expected_tool_call = expected_tool))
    assert res.is_completed == True

def test_expected_tool_2_complete_with_static_multiturn_sample_3_trace_2(azure_openai_client, static_multiturn_data_sample_3, multiturn_agent_trace_2):
    turn_idx = 1
    tool_idx = 2
    agent_trace =  multiturn_agent_trace_2.turns[:turn_idx+1]
    expected_tool = static_multiturn_data_sample_3.expected_tool_calls[tool_idx]
    tool_validator = ToolCallCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    res = asyncio.run(tool_validator.validate(agent_trace_turns = agent_trace, expected_tool_call = expected_tool))
    assert res.is_completed == True

def test_expected_tool_3_complete_with_static_multiturn_sample_3_trace_2(azure_openai_client, static_multiturn_data_sample_3, multiturn_agent_trace_2):
    turn_idx = 2
    tool_idx = 3
    agent_trace =  multiturn_agent_trace_2.turns[:turn_idx+1]
    expected_tool = static_multiturn_data_sample_3.expected_tool_calls[tool_idx]
    tool_validator = ToolCallCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    res = asyncio.run(tool_validator.validate(agent_trace_turns = agent_trace, expected_tool_call = expected_tool))
    assert res.is_completed == True


def test_expected_tool_0_complete_with_static_singleturn_sample_1_trace_1(azure_openai_client, static_singleturn_data_sample_1, singleturn_agent_trace_1):
    turn_idx = 0
    tool_idx = 0
    agent_trace =  singleturn_agent_trace_1.turns[:turn_idx+1]
    expected_tool = static_singleturn_data_sample_1.expected_tool_calls[tool_idx]
    tool_validator = ToolCallCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    res = asyncio.run(tool_validator.validate(agent_trace_turns = agent_trace, expected_tool_call = expected_tool))
    assert res.is_completed == True

def test_expected_tool_1_complete_with_static_singleturn_sample_1_trace_1(azure_openai_client, static_singleturn_data_sample_1, singleturn_agent_trace_1):
    turn_idx = 0
    tool_idx = 1
    agent_trace =  singleturn_agent_trace_1.turns[:turn_idx+1]
    expected_tool = static_singleturn_data_sample_1.expected_tool_calls[tool_idx]
    tool_validator = ToolCallCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    res = asyncio.run(tool_validator.validate(agent_trace_turns = agent_trace, expected_tool_call = expected_tool))
    assert res.is_completed == True

def test_expected_tool_2_complete_with_static_singleturn_sample_1_trace_1(azure_openai_client, static_singleturn_data_sample_1, singleturn_agent_trace_1):
    turn_idx = 0
    tool_idx = 2
    agent_trace =  singleturn_agent_trace_1.turns[:turn_idx+1]
    expected_tool = static_singleturn_data_sample_1.expected_tool_calls[tool_idx]
    tool_validator = ToolCallCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    res = asyncio.run(tool_validator.validate(agent_trace_turns = agent_trace, expected_tool_call = expected_tool))
    assert res.is_completed == True


def test_expected_tool_0_complete_with_static_singleturn_sample_2_trace_1(azure_openai_client, static_singleturn_data_sample_2, singleturn_agent_trace_1):
    turn_idx = 0
    tool_idx = 0
    agent_trace =  singleturn_agent_trace_1.turns[:turn_idx+1]
    expected_tool = static_singleturn_data_sample_2.expected_tool_calls[tool_idx]
    tool_validator = ToolCallCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    res = asyncio.run(tool_validator.validate(agent_trace_turns = agent_trace, expected_tool_call = expected_tool))
    assert res.is_completed == True

def test_expected_tool_1_incomplete_with_static_singleturn_sample_2_trace_1(azure_openai_client, static_singleturn_data_sample_2, singleturn_agent_trace_1):
    turn_idx = 0
    tool_idx = 1
    agent_trace =  singleturn_agent_trace_1.turns[:turn_idx+1]
    expected_tool = static_singleturn_data_sample_2.expected_tool_calls[tool_idx]
    tool_validator = ToolCallCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    res = asyncio.run(tool_validator.validate(agent_trace_turns = agent_trace, expected_tool_call = expected_tool))
    assert res.is_completed == False

def test_expected_tool_2_complete_with_static_singleturn_sample_2_trace_1(azure_openai_client, static_singleturn_data_sample_2, singleturn_agent_trace_1):
    turn_idx = 0
    tool_idx = 2
    agent_trace =  singleturn_agent_trace_1.turns[:turn_idx+1]
    expected_tool = static_singleturn_data_sample_2.expected_tool_calls[tool_idx]
    tool_validator = ToolCallCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    res = asyncio.run(tool_validator.validate(agent_trace_turns = agent_trace, expected_tool_call = expected_tool))
    assert res.is_completed == True

def test_expected_tool_3_complete_with_static_singleturn_sample_2_trace_1(azure_openai_client, static_singleturn_data_sample_2, singleturn_agent_trace_1):
    turn_idx = 0
    tool_idx = 3
    agent_trace =  singleturn_agent_trace_1.turns[:turn_idx+1]
    expected_tool = static_singleturn_data_sample_2.expected_tool_calls[tool_idx]
    tool_validator = ToolCallCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    res = asyncio.run(tool_validator.validate(agent_trace_turns = agent_trace, expected_tool_call = expected_tool))
    assert res.is_completed == True

def test_tool_correctness_score_full_completion_with_static_multiturn_sample1_trace1(azure_openai_client, static_multiturn_data_sample_1, multiturn_agent_trace_1):
    metric = ToolCorrectnessMetric(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    metric_result = metric.evaluate(agent_trace=multiturn_agent_trace_1, evaluation_data_sample=static_multiturn_data_sample_1)
    assert metric_result.score == 1.0

def test_tool_correctness_score_partial_completion_with_static_multiturn_sample1_trace1(azure_openai_client, static_multiturn_data_sample_2, multiturn_agent_trace_1):
    metric = ToolCorrectnessMetric(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    metric_result = metric.evaluate(agent_trace=multiturn_agent_trace_1, evaluation_data_sample=static_multiturn_data_sample_2)
    assert metric_result.score == 0.6

def test_tool_correctness_score_full_completion_with_static_multiturn_sample3_trace2(azure_openai_client, static_multiturn_data_sample_3, multiturn_agent_trace_2):
    metric = ToolCorrectnessMetric(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    metric_result = metric.evaluate(agent_trace=multiturn_agent_trace_2, evaluation_data_sample=static_multiturn_data_sample_3)
    assert metric_result.score == 1.0

def test_tool_correctness_score_full_completion_with_static_singleturn_sample1_trace1(azure_openai_client, static_singleturn_data_sample_1, singleturn_agent_trace_1):
    metric = ToolCorrectnessMetric(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    metric_result = metric.evaluate(agent_trace=singleturn_agent_trace_1, evaluation_data_sample=static_singleturn_data_sample_1)
    assert metric_result.score == 1.0

def test_tool_correctness_score_partial_completion_with_static_singleturn_sample2_trace1(azure_openai_client, static_singleturn_data_sample_2, singleturn_agent_trace_1):
    metric = ToolCorrectnessMetric(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    metric_result = metric.evaluate(agent_trace=singleturn_agent_trace_1, evaluation_data_sample=static_singleturn_data_sample_2)
    assert metric_result.score == 0.75
