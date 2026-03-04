import pytest
import asyncio

from agent_inspect.metrics.constants import INCLUDE_JUDGE_EXPLANATION, MAX_TURNS, OPTIMIZE_JUDGE_TRIALS
from tests_acceptance.azure_openai_client import AzureOpenAIClient
from agent_inspect.metrics.scorer import ProgressScore, ProgressScoresThroughTurns, SuccessScore, SuccessScoreFinalTurn
from agent_inspect.metrics.validator import SubGoalCompletionValidator
from agent_inspect.metrics.scorer import AUC
from tests_acceptance.metrics.scripts.utils import load_agent_trace, load_data_sample_static, load_data_sample_dynamic, remove_previous_turn_tool_calls, remove_current_turn_tool_calls


@pytest.fixture
def multiturn_agent_trace_1():
    return load_agent_trace("tests_acceptance/metrics/sample_data/example_multiturn_PAB_traces.json")   

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
def static_singleturn_data_sample_1(): 
    return load_data_sample_static("tests_acceptance/metrics/sample_data/example_static_single_turn_data_sample.json")  

@pytest.fixture
def static_singleturn_data_sample_2(): 
    return load_data_sample_static("tests_acceptance/metrics/sample_data/example_static_single_turn_data_sample_2.json")  

@pytest.fixture
def dynamic_sample_1(): 
    return load_data_sample_dynamic("tests_acceptance/metrics/sample_data/example_dynamic_multi_turn_user_proxy_data_sample.json")  

@pytest.fixture
def dynamic_sample_2(): 
    return load_data_sample_dynamic("tests_acceptance/metrics/sample_data/example_dynamic_multi_turn_user_proxy_data_sample_toolGN.json")  

@pytest.fixture
def azure_openai_client():
    return AzureOpenAIClient(model="gpt-4.1", max_tokens=4096, temperature=0)

# AR agent
def test_subgoal_0_complete_with_static_multiturn_sample_1_trace_1(azure_openai_client, static_multiturn_data_sample_1, multiturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 0
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = static_multiturn_data_sample_1.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate(turn_traces = agent_trace, sub_goal = subgoal))
    assert res.is_completed == True

def test_subgoal_0_complete_with_no_tool_static_multiturn_sample_1_trace_1(azure_openai_client, static_multiturn_data_sample_1, multiturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 0
    agent_trace =  remove_current_turn_tool_calls(multiturn_agent_trace_1.turns[:turn_idx+1])
    subgoal = static_multiturn_data_sample_1.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate(turn_traces = agent_trace, sub_goal = subgoal))
    assert res.is_completed == True

def test_subgoal_1_complete_with_static_multiturn_sample_1_trace_1(azure_openai_client, static_multiturn_data_sample_1, multiturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 1
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = static_multiturn_data_sample_1.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate(turn_traces = agent_trace, sub_goal = subgoal))
    assert res.is_completed == True

def test_subgoal_1_complete_with_no_tool_static_multiturn_sample_1_trace_1(azure_openai_client, static_multiturn_data_sample_1, multiturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 1
    agent_trace =  remove_current_turn_tool_calls(multiturn_agent_trace_1.turns[:turn_idx+1])
    subgoal = static_multiturn_data_sample_1.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate(turn_traces = agent_trace, sub_goal = subgoal))
    assert res.is_completed == True

def test_subgoal_2_complete_with_static_multiturn_sample_1_trace_1(azure_openai_client, static_multiturn_data_sample_1, multiturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 2
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = static_multiturn_data_sample_1.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate(turn_traces = agent_trace, sub_goal = subgoal))
    assert res.is_completed == True

def test_subgoal_2_complete_with_no_tool_static_multiturn_sample_1_trace_1(azure_openai_client, static_multiturn_data_sample_1, multiturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 2
    agent_trace =  remove_current_turn_tool_calls(multiturn_agent_trace_1.turns[:turn_idx+1])
    subgoal = static_multiturn_data_sample_1.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate(turn_traces = agent_trace, sub_goal = subgoal))
    assert res.is_completed == True

def test_subgoal_3_complete_with_static_multiturn_sample_1_trace_1(azure_openai_client, static_multiturn_data_sample_1, multiturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 3
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = static_multiturn_data_sample_1.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate(turn_traces = agent_trace, sub_goal = subgoal))
    assert res.is_completed == True

def test_subgoal_3_complete_with_no_tool_static_multiturn_sample_1_trace_1(azure_openai_client, static_multiturn_data_sample_1, multiturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 3
    agent_trace =  remove_current_turn_tool_calls(multiturn_agent_trace_1.turns[:turn_idx+1])
    subgoal = static_multiturn_data_sample_1.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate(turn_traces = agent_trace, sub_goal = subgoal))
    assert res.is_completed == True

def test_subgoal_4_complete_with_static_multiturn_sample_1_trace_1(azure_openai_client, static_multiturn_data_sample_1, multiturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 4
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = static_multiturn_data_sample_1.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate(turn_traces = agent_trace, sub_goal = subgoal))
    assert res.is_completed == True

def test_subgoal_4_complete_with_no_tool_static_multiturn_sample_1_trace_1(azure_openai_client, static_multiturn_data_sample_1, multiturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 4
    agent_trace =  remove_current_turn_tool_calls(multiturn_agent_trace_1.turns[:turn_idx+1])
    subgoal = static_multiturn_data_sample_1.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate(turn_traces = agent_trace, sub_goal = subgoal))
    assert res.is_completed == True


def test_subgoal_5_complete_with_static_multiturn_sample_1_trace_1(azure_openai_client, static_multiturn_data_sample_1, multiturn_agent_trace_1):
    turn_idx = 2
    subgoal_idx = 5
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = static_multiturn_data_sample_1.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate(turn_traces = agent_trace, sub_goal = subgoal))
    assert res.is_completed == True    



def test_subgoal_5_complete_with_no_tool_static_multiturn_sample_1_trace_1(azure_openai_client, static_multiturn_data_sample_1, multiturn_agent_trace_1):
    turn_idx = 2
    subgoal_idx = 5
    agent_trace =  remove_current_turn_tool_calls(multiturn_agent_trace_1.turns[:turn_idx+1])
    subgoal = static_multiturn_data_sample_1.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate(turn_traces = agent_trace, sub_goal = subgoal))
    assert res.is_completed == True    

def test_subgoal_5_complete_with_specific_turn_static_multiturn_sample_1_trace_1(azure_openai_client, static_multiturn_data_sample_1, multiturn_agent_trace_1):
    turn_idx = 2
    subgoal_idx = 5
    agent_trace = [multiturn_agent_trace_1.turns[turn_idx]]
    subgoal = static_multiturn_data_sample_1.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate(turn_traces = agent_trace, sub_goal = subgoal))
    assert res.is_completed == True


def test_subgoal_5_complete_with_specific_turn_no_tool_static_multiturn_sample_1_trace_1(azure_openai_client, static_multiturn_data_sample_1, multiturn_agent_trace_1):
    turn_idx = 2
    subgoal_idx = 5
    agent_trace = remove_current_turn_tool_calls([multiturn_agent_trace_1.turns[turn_idx]])
    subgoal = static_multiturn_data_sample_1.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate(turn_traces = agent_trace, sub_goal = subgoal))
    assert res.is_completed == True


def test_subgoal_5_complete_with_past_chat_static_multiturn_sample_1_trace_1(azure_openai_client, static_multiturn_data_sample_1, multiturn_agent_trace_1):
    turn_idx = 2
    subgoal_idx = 5
    agent_trace =  remove_previous_turn_tool_calls(multiturn_agent_trace_1.turns[:turn_idx+1])
    subgoal = static_multiturn_data_sample_1.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate(turn_traces = agent_trace, sub_goal = subgoal))
    assert res.is_completed == True

def test_subgoal_0_incomplete_with_static_multiturn_sample_2_trace_1(azure_openai_client, static_multiturn_data_sample_2, multiturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 0
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = static_multiturn_data_sample_2.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate(turn_traces = agent_trace, sub_goal = subgoal))
    assert res.is_completed == False

def test_subgoal_0_incomplete_with_no_tool_static_multiturn_sample_2_trace_1(azure_openai_client, static_multiturn_data_sample_2, multiturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 0
    agent_trace =  remove_current_turn_tool_calls(multiturn_agent_trace_1.turns[:turn_idx+1])
    subgoal = static_multiturn_data_sample_2.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate(turn_traces = agent_trace, sub_goal = subgoal))
    assert res.is_completed == False

def test_subgoal_1_incomplete_with_static_multiturn_sample_2_trace_1(azure_openai_client, static_multiturn_data_sample_2, multiturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 1
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = static_multiturn_data_sample_2.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate(turn_traces = agent_trace, sub_goal = subgoal))
    assert res.is_completed == False

def test_subgoal_1_incomplete_with_no_tool_static_multiturn_sample_2_trace_1(azure_openai_client, static_multiturn_data_sample_2, multiturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 1
    agent_trace =  remove_current_turn_tool_calls(multiturn_agent_trace_1.turns[:turn_idx+1])
    subgoal = static_multiturn_data_sample_2.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate(turn_traces = agent_trace, sub_goal = subgoal))
    assert res.is_completed == False

def test_subgoal_2_incomplete_with_static_multiturn_sample_2_trace_1(azure_openai_client, static_multiturn_data_sample_2, multiturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 2
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = static_multiturn_data_sample_2.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate(turn_traces = agent_trace, sub_goal = subgoal))
    assert res.is_completed == False

def test_subgoal_2_incomplete_with_no_tool_static_multiturn_sample_2_trace_1(azure_openai_client, static_multiturn_data_sample_2, multiturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 2
    agent_trace =  remove_current_turn_tool_calls(multiturn_agent_trace_1.turns[:turn_idx+1])
    subgoal = static_multiturn_data_sample_2.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate(turn_traces = agent_trace, sub_goal = subgoal))
    assert res.is_completed == False

def test_subgoal_5_incomplete_with_static_multiturn_sample_2_trace_1(azure_openai_client, static_multiturn_data_sample_2, multiturn_agent_trace_1):
    turn_idx = 2
    subgoal_idx = 5
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = static_multiturn_data_sample_2.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate(turn_traces = agent_trace, sub_goal = subgoal))
    assert res.is_completed == False

def test_subgoal_5_incomplete_with_specific_turn_static_multiturn_sample_2_trace_1(azure_openai_client, static_multiturn_data_sample_2, multiturn_agent_trace_1):
    turn_idx = 2
    subgoal_idx = 5
    agent_trace =  [multiturn_agent_trace_1.turns[turn_idx]]
    subgoal = static_multiturn_data_sample_2.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate(turn_traces = agent_trace, sub_goal = subgoal))
    assert res.is_completed == False

def test_subgoal_5_incomplete_with_specific_turn_no_tool_static_multiturn_sample_2_trace_1(azure_openai_client, static_multiturn_data_sample_2, multiturn_agent_trace_1):
    turn_idx = 2
    subgoal_idx = 5
    agent_trace =  remove_current_turn_tool_calls([multiturn_agent_trace_1.turns[turn_idx]])
    subgoal = static_multiturn_data_sample_2.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate(turn_traces = agent_trace, sub_goal = subgoal))
    assert res.is_completed == False    

def test_subgoal_5_incomplete_with_past_chat_static_multiturn_sample_2_trace_1(azure_openai_client, static_multiturn_data_sample_2, multiturn_agent_trace_1):
    turn_idx = 2
    subgoal_idx = 5
    agent_trace =  remove_previous_turn_tool_calls(multiturn_agent_trace_1.turns[:turn_idx+1])
    subgoal = static_multiturn_data_sample_2.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate(turn_traces = agent_trace, sub_goal = subgoal))
    assert res.is_completed == False

def test_subgoal_0_complete_with_static_singleturn_sample_1_trace_1(azure_openai_client, static_singleturn_data_sample_1, singleturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 0
    agent_trace =  singleturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = static_singleturn_data_sample_1.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate(turn_traces = agent_trace, sub_goal = subgoal))
    assert res.is_completed == True

def test_subgoal_3_complete_with_static_singleturn_sample_1_trace_1(azure_openai_client, static_singleturn_data_sample_1, singleturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 3
    agent_trace =  singleturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = static_singleturn_data_sample_1.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate(turn_traces = agent_trace, sub_goal = subgoal))
    assert res.is_completed == True

def test_subgoal_3_complete_with_no_tool_static_singleturn_sample_1_trace_1(azure_openai_client, static_singleturn_data_sample_1, singleturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 3
    agent_trace =  remove_current_turn_tool_calls(singleturn_agent_trace_1.turns[:turn_idx+1])
    subgoal = static_singleturn_data_sample_1.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate(turn_traces = agent_trace, sub_goal = subgoal))
    assert res.is_completed == True

def test_subgoal_0_incomplete_with_static_singleturn_sample_2_trace_1(azure_openai_client, static_singleturn_data_sample_2, singleturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 0
    agent_trace =  singleturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = static_singleturn_data_sample_2.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate(turn_traces = agent_trace, sub_goal = subgoal))
    assert res.is_completed == False

def test_subgoal_0_incomplete_with_no_tool_static_singleturn_sample_2_trace_1(azure_openai_client, static_singleturn_data_sample_2, singleturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 0
    agent_trace =  remove_current_turn_tool_calls(singleturn_agent_trace_1.turns[:turn_idx+1])
    subgoal = static_singleturn_data_sample_2.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate(turn_traces = agent_trace, sub_goal = subgoal))
    assert res.is_completed == False

def test_subgoal_1_incomplete_with_static_singleturn_sample_2_trace_1(azure_openai_client, static_singleturn_data_sample_2, singleturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 1
    agent_trace =  singleturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = static_singleturn_data_sample_2.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate(turn_traces = agent_trace, sub_goal = subgoal))
    assert res.is_completed == False

def test_subgoal_1_incomplete_with_no_tool_static_singleturn_sample_2_trace_1(azure_openai_client, static_singleturn_data_sample_2, singleturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 1
    agent_trace =  remove_current_turn_tool_calls(singleturn_agent_trace_1.turns[:turn_idx+1])
    subgoal = static_singleturn_data_sample_2.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate(turn_traces = agent_trace, sub_goal = subgoal))
    assert res.is_completed == False

def test_subgoal_0_turn_0_complete_with_dynamic_sample_1_trace_1(azure_openai_client, dynamic_sample_1, multiturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 0
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = dynamic_sample_1.sub_goals[subgoal_idx]
    user_task = dynamic_sample_1.user_instruction
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction = user_task))
    assert res.is_completed == True

def test_subgoal_0_turn_0_complete_with_dynamic_sample_1_trace_1_without_instruction(azure_openai_client, dynamic_sample_1, multiturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 0
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = dynamic_sample_1.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction=""))
    assert res.is_completed == True


def test_subgoal_1_turn_0_complete_with_dynamic_sample_1_trace_1(azure_openai_client, dynamic_sample_1, multiturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 1
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = dynamic_sample_1.sub_goals[subgoal_idx]
    user_task = dynamic_sample_1.user_instruction
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction = user_task))
    assert res.is_completed == True

def test_subgoal_1_turn_0_complete_with_dynamic_sample_1_trace_1_without_instruction(azure_openai_client, dynamic_sample_1, multiturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 1
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = dynamic_sample_1.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction=""))
    assert res.is_completed == True

def test_subgoal_2_turn_0_complete_with_dynamic_sample_1_trace_1(azure_openai_client, dynamic_sample_1, multiturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 2
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = dynamic_sample_1.sub_goals[subgoal_idx]
    user_task = dynamic_sample_1.user_instruction
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction = user_task))
    assert res.is_completed == True

def test_subgoal_2_turn_0_complete_with_dynamic_sample_1_trace_1_without_instruction(azure_openai_client, dynamic_sample_1, multiturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 2
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = dynamic_sample_1.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction=""))
    assert res.is_completed == True

def test_subgoal_3_turn_0_complete_with_dynamic_sample_1_trace_1(azure_openai_client, dynamic_sample_1, multiturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 3
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = dynamic_sample_1.sub_goals[subgoal_idx]
    user_task = dynamic_sample_1.user_instruction
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction = user_task))
    assert res.is_completed == True

def test_subgoal_3_turn_0_complete_with_dynamic_sample_1_trace_1_without_instruction(azure_openai_client, dynamic_sample_1, multiturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 3
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = dynamic_sample_1.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction=""))
    assert res.is_completed == True


def test_subgoal_4_turn_0_complete_with_dynamic_sample_1_trace_1(azure_openai_client, dynamic_sample_1, multiturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 4
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = dynamic_sample_1.sub_goals[subgoal_idx]
    user_task = dynamic_sample_1.user_instruction
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction = user_task))
    assert res.is_completed == True

def test_subgoal_4_turn_0_complete_with_dynamic_sample_1_trace_1_without_instruction(azure_openai_client, dynamic_sample_1, multiturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 4
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = dynamic_sample_1.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction=""))
    assert res.is_completed == True

def test_subgoal_5_turn_0_incomplete_with_dynamic_sample_1_trace_1(azure_openai_client, dynamic_sample_1, multiturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 5
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = dynamic_sample_1.sub_goals[subgoal_idx]
    user_task = dynamic_sample_1.user_instruction
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction = user_task))
    assert res.is_completed == False

def test_subgoal_5_turn_0_incomplete_with_dynamic_sample_1_trace_1_without_instruction(azure_openai_client, dynamic_sample_1, multiturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 5
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = dynamic_sample_1.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction=""))
    assert res.is_completed == False

def test_subgoal_5_turn_1_incomplete_with_dynamic_sample_1_trace_1(azure_openai_client, dynamic_sample_1, multiturn_agent_trace_1):
    turn_idx = 1
    subgoal_idx = 5
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = dynamic_sample_1.sub_goals[subgoal_idx]
    user_task = dynamic_sample_1.user_instruction
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction = user_task))
    assert res.is_completed == False

def test_subgoal_5_turn_1_incomplete_with_dynamic_sample_1_trace_1_without_instruction(azure_openai_client, dynamic_sample_1, multiturn_agent_trace_1):
    turn_idx = 1
    subgoal_idx = 5
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = dynamic_sample_1.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction=""))
    assert res.is_completed == False


def test_subgoal_5_turn_2_complete_with_dynamic_sample_1_trace_1(azure_openai_client, dynamic_sample_1, multiturn_agent_trace_1):
    turn_idx = 2
    subgoal_idx = 5
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = dynamic_sample_1.sub_goals[subgoal_idx]
    user_task = dynamic_sample_1.user_instruction
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction = user_task))
    assert res.is_completed == True

def test_subgoal_5_turn_2_complete_with_dynamic_sample_1_trace_1_without_instruction(azure_openai_client, dynamic_sample_1, multiturn_agent_trace_1):
    turn_idx = 2
    subgoal_idx = 5
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = dynamic_sample_1.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction=""))
    assert res.is_completed == True


def test_subgoal_0_turn_0_complete_with_dynamic_sample_2_trace_1(azure_openai_client, dynamic_sample_2, multiturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 0
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = dynamic_sample_2.sub_goals[subgoal_idx]
    user_task = dynamic_sample_2.user_instruction
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction = user_task))
    assert res.is_completed == True

def test_subgoal_0_turn_0_complete_with_dynamic_sample_2_trace_1_without_instruction(azure_openai_client, dynamic_sample_2, multiturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 0
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = dynamic_sample_2.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction=""))
    assert res.is_completed == True

def test_subgoal_3_turn_0_incomplete_with_dynamic_sample_2_trace_1(azure_openai_client, dynamic_sample_2, multiturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 3
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = dynamic_sample_2.sub_goals[subgoal_idx]
    user_task = dynamic_sample_2.user_instruction
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction = user_task))
    assert res.is_completed == False

def test_subgoal_3_turn_0_incomplete_with_dynamic_sample_2_trace_1_without_instruction(azure_openai_client, dynamic_sample_2, multiturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 3
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = dynamic_sample_2.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction=""))
    assert res.is_completed == False

def test_subgoal_3_turn_1_incomplete_with_dynamic_sample_2_trace_1(azure_openai_client, dynamic_sample_2, multiturn_agent_trace_1):
    turn_idx = 1
    subgoal_idx = 3
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = dynamic_sample_2.sub_goals[subgoal_idx]
    user_task = dynamic_sample_2.user_instruction
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction = user_task))
    assert res.is_completed == False

def test_subgoal_3_turn_1_incomplete_with_dynamic_sample_2_trace_1_without_instruction(azure_openai_client, dynamic_sample_2, multiturn_agent_trace_1):
    turn_idx = 1
    subgoal_idx = 3
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = dynamic_sample_2.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction=""))
    assert res.is_completed == False

def test_subgoal_3_turn_2_complete_with_dynamic_sample_2_trace_1(azure_openai_client, dynamic_sample_2, multiturn_agent_trace_1):
    turn_idx = 2
    subgoal_idx = 3
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = dynamic_sample_2.sub_goals[subgoal_idx]
    user_task = dynamic_sample_2.user_instruction
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction = user_task))
    assert res.is_completed == True

def test_subgoal_3_turn_2_complete_with_dynamic_sample_2_trace_1_without_instruction(azure_openai_client, dynamic_sample_2, multiturn_agent_trace_1):
    turn_idx = 2
    subgoal_idx = 3
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = dynamic_sample_2.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction=""))
    assert res.is_completed == True

def test_subgoal_4_turn_0_incomplete_with_dynamic_sample_2_trace_1(azure_openai_client, dynamic_sample_2, multiturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 4
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = dynamic_sample_2.sub_goals[subgoal_idx]
    user_task = dynamic_sample_2.user_instruction
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction = user_task))
    assert res.is_completed == False

def test_subgoal_4_turn_0_incomplete_with_dynamic_sample_2_trace_1_without_instruction(azure_openai_client, dynamic_sample_2, multiturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 4
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = dynamic_sample_2.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction=""))
    assert res.is_completed == False

def test_subgoal_4_turn_1_incomplete_with_dynamic_sample_2_trace_1(azure_openai_client, dynamic_sample_2, multiturn_agent_trace_1):
    turn_idx = 1
    subgoal_idx = 4
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = dynamic_sample_2.sub_goals[subgoal_idx]
    user_task = dynamic_sample_2.user_instruction
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction = user_task))
    assert res.is_completed == False

def test_subgoal_4_turn_1_incomplete_with_dynamic_sample_2_trace_1_without_instruction(azure_openai_client, dynamic_sample_2, multiturn_agent_trace_1):
    turn_idx = 1
    subgoal_idx = 4
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = dynamic_sample_2.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction=""))
    assert res.is_completed == False

def test_subgoal_4_turn_2_incomplete_with_dynamic_sample_2_trace_1(azure_openai_client, dynamic_sample_2, multiturn_agent_trace_1):
    turn_idx = 2
    subgoal_idx = 4
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = dynamic_sample_2.sub_goals[subgoal_idx]
    user_task = dynamic_sample_2.user_instruction
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction = user_task))
    assert res.is_completed == False

def test_subgoal_4_turn_2_incomplete_with_dynamic_sample_2_trace_1_without_instruction(azure_openai_client, dynamic_sample_2, multiturn_agent_trace_1):
    turn_idx = 2
    subgoal_idx = 4
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = dynamic_sample_2.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction=""))
    assert res.is_completed == False

def test_subgoal_5_turn_0_complete_with_dynamic_sample_2_trace_1(azure_openai_client, dynamic_sample_2, multiturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 5
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = dynamic_sample_2.sub_goals[subgoal_idx]
    user_task = dynamic_sample_2.user_instruction
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction = user_task))
    assert res.is_completed == True

def test_subgoal_5_turn_0_complete_with_dynamic_sample_2_trace_1_without_instruction(azure_openai_client, dynamic_sample_2, multiturn_agent_trace_1):
    turn_idx = 0
    subgoal_idx = 5
    agent_trace =  multiturn_agent_trace_1.turns[:turn_idx+1]
    subgoal = dynamic_sample_2.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction=""))
    assert res.is_completed == True

def test_progress_score_full_completion_with_static_multiturn_sample_1_trace_1(azure_openai_client, static_multiturn_data_sample_1, multiturn_agent_trace_1):
    metric = ProgressScore(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    metric_result = metric.evaluate(agent_trace=multiturn_agent_trace_1, evaluation_data_sample=static_multiturn_data_sample_1)
    assert metric_result.score == 1.0

def test_progress_score_partial_completion_with_static_multiturn_sample_2_trace_1(azure_openai_client, static_multiturn_data_sample_2, multiturn_agent_trace_1):
    metric = ProgressScore(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    metric_result = metric.evaluate(agent_trace=multiturn_agent_trace_1, evaluation_data_sample=static_multiturn_data_sample_2)
    assert metric_result.score == 0.3333
   

def test_progress_score_full_completion_with_static_singleturn_sample_1_trace_1(azure_openai_client, static_singleturn_data_sample_1, singleturn_agent_trace_1):
    metric = ProgressScore(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    metric_result = metric.evaluate(agent_trace=singleturn_agent_trace_1, evaluation_data_sample=static_singleturn_data_sample_1)
    assert metric_result.score == 1.0

def test_progress_score_partial_completion_with_static_singleturn_sample_2_trace_1(azure_openai_client, static_singleturn_data_sample_2, singleturn_agent_trace_1):
    metric = ProgressScore(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    metric_result = metric.evaluate(agent_trace=singleturn_agent_trace_1, evaluation_data_sample=static_singleturn_data_sample_2)
    assert metric_result.score == 0.4

def test_progress_turns_full_completion_with_dynamic_sample_1_trace_1(azure_openai_client, dynamic_sample_1, multiturn_agent_trace_1):
    metric = ProgressScoresThroughTurns(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True, MAX_TURNS: 20, OPTIMIZE_JUDGE_TRIALS: True})
    metric_results = metric.evaluate(agent_trace=multiturn_agent_trace_1, evaluation_data_sample=dynamic_sample_1)
    gt_values = [0.8333, 0.8333, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert [r.score for r in metric_results] == gt_values

def test_progress_turns_partial_completion_with_dynamic_sample_2_trace_1(azure_openai_client, dynamic_sample_2, multiturn_agent_trace_1):
    metric = ProgressScoresThroughTurns(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True, MAX_TURNS: 20})
    metric_results = metric.evaluate(agent_trace=multiturn_agent_trace_1, evaluation_data_sample=dynamic_sample_2)
    gt_values = [0.7143, 0.7143, 0.8571, 0.8571, 0.8571, 0.8571, 0.8571, 0.8571, 0.8571, 0.8571, 0.8571, 0.8571, 0.8571, 0.8571, 0.8571, 0.8571, 0.8571, 0.8571, 0.8571, 0.8571]
    assert [r.score for r in metric_results] == gt_values


def test_progress_turns_full_completion_max_turns5_with_dynamic_sample_1_trace_1(azure_openai_client, dynamic_sample_1, multiturn_agent_trace_1):
    metric = ProgressScoresThroughTurns(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True, MAX_TURNS: 5})
    metric_results = metric.evaluate(agent_trace=multiturn_agent_trace_1, evaluation_data_sample=dynamic_sample_1)
    gt_values = [0.8333, 0.8333, 1.0, 1.0, 1.0]
    assert [r.score for r in metric_results] == gt_values

def test_progress_turns_partial_completion_max_turns5_with_dynamic_sample_2_trace_1(azure_openai_client, dynamic_sample_2, multiturn_agent_trace_1):
    metric = ProgressScoresThroughTurns(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True, MAX_TURNS: 5, OPTIMIZE_JUDGE_TRIALS: True})
    metric_results = metric.evaluate(agent_trace=multiturn_agent_trace_1, evaluation_data_sample=dynamic_sample_2)
    gt_values = [0.7143, 0.7143, 0.8571, 0.8571, 0.8571] 
    assert [r.score for r in metric_results] == gt_values


def test_success_score_successful_with_static_multiturn_sample_1_trace_1(azure_openai_client, static_multiturn_data_sample_1, multiturn_agent_trace_1):
    metric = SuccessScore(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    metric_result = metric.evaluate(agent_trace=multiturn_agent_trace_1, evaluation_data_sample=static_multiturn_data_sample_1)
    assert metric_result.score == 1.0

def test_success_score_unsuccessful_with_static_multiturn_sample_2_trace_1(azure_openai_client, static_multiturn_data_sample_2, multiturn_agent_trace_1):
    metric = SuccessScore(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    metric_result = metric.evaluate(agent_trace=multiturn_agent_trace_1, evaluation_data_sample=static_multiturn_data_sample_2)
    assert metric_result.score == 0.0

def test_success_score_successful_with_static_singleturn_sample_1_trace_1(azure_openai_client, static_singleturn_data_sample_1, singleturn_agent_trace_1):
    metric = SuccessScore(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    metric_result = metric.evaluate(agent_trace=singleturn_agent_trace_1, evaluation_data_sample=static_singleturn_data_sample_1)
    assert metric_result.score == 1.0

def test_success_score_unsuccessful_with_static_singleturn_sample_2_trace_1(azure_openai_client, static_singleturn_data_sample_2, singleturn_agent_trace_1):
    metric = SuccessScore(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    metric_result = metric.evaluate(agent_trace=singleturn_agent_trace_1, evaluation_data_sample=static_singleturn_data_sample_2)
    assert metric_result.score == 0.0


def test_success_final_turn_successful_with_dynamic_sample_1_trace_1(azure_openai_client, dynamic_sample_1, multiturn_agent_trace_1):
    metric = SuccessScoreFinalTurn(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True, MAX_TURNS: 20, OPTIMIZE_JUDGE_TRIALS: True})
    metric_result = metric.evaluate(agent_trace=multiturn_agent_trace_1, evaluation_data_sample=dynamic_sample_1)
    assert metric_result.score == 1.0

def test_success_final_turn_unsuccessful_with_dynamic_sample_2_trace_1(azure_openai_client, dynamic_sample_2, multiturn_agent_trace_1):
    metric = SuccessScoreFinalTurn(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True, MAX_TURNS: 20, OPTIMIZE_JUDGE_TRIALS: True})
    metric_result = metric.evaluate(agent_trace=multiturn_agent_trace_1, evaluation_data_sample=dynamic_sample_2)
    assert metric_result.score == 0.0

def test_auc_with_dynamic_sample_1_trace_1(azure_openai_client, dynamic_sample_1, multiturn_agent_trace_1):
    metric = AUC(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True, MAX_TURNS: 20, OPTIMIZE_JUDGE_TRIALS: True})
    metric_result = metric.evaluate(agent_trace=multiturn_agent_trace_1, evaluation_data_sample=dynamic_sample_1)
    assert round(metric_result.score, 4) == 0.9868

def test_auc_max_turns5_with_dynamic_sample_1_trace_1(azure_openai_client, dynamic_sample_1, multiturn_agent_trace_1):
    metric = AUC(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True, MAX_TURNS: 5})
    metric_result = metric.evaluate(agent_trace=multiturn_agent_trace_1, evaluation_data_sample=dynamic_sample_1)
    assert round(metric_result.score, 4) == 0.9375

def test_auc_with_dynamic_sample_2_trace_1(azure_openai_client, dynamic_sample_2, multiturn_agent_trace_1):
    metric = AUC(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True, MAX_TURNS: 20, OPTIMIZE_JUDGE_TRIALS: True})
    metric_result = metric.evaluate(agent_trace=multiturn_agent_trace_1, evaluation_data_sample=dynamic_sample_2)
    assert round(metric_result.score, 4) == 0.8458

def test_auc_max_turns5_with_dynamic_sample_2_trace_1(azure_openai_client, dynamic_sample_2, multiturn_agent_trace_1):
    metric = AUC(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True, MAX_TURNS: 5})
    metric_result = metric.evaluate(agent_trace=multiturn_agent_trace_1, evaluation_data_sample=dynamic_sample_2)
    assert round(metric_result.score, 4) == 0.8035
