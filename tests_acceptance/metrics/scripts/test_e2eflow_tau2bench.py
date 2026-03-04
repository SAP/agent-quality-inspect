import pytest
import asyncio
from agent_inspect.metrics.constants import INCLUDE_JUDGE_EXPLANATION, MAX_TURNS
from tests_acceptance.azure_openai_client import AzureOpenAIClient
from agent_inspect.metrics.scorer import ProgressScore, ProgressScoresThroughTurns, SuccessScore, SuccessScoreFinalTurn
from agent_inspect.metrics.validator import SubGoalCompletionValidator
from agent_inspect.metrics.scorer import AUC
from agent_inspect.metrics.adapters import Tau2BenchAdapter
from tests_acceptance.metrics.scripts.utils import load_data_sample_static, load_data_sample_dynamic


@pytest.fixture
def multiturn_agent_trace_2():
    tau2bench_adapter = Tau2BenchAdapter()
    conversation_data = tau2bench_adapter.load_json("tests_acceptance/metrics/sample_data/single_trace_tau2bench.json")   
    tau2bench_trace = tau2bench_adapter.convert_to_agent_trace(conversation_data)
    return tau2bench_trace


@pytest.fixture
def static_multiturn_data_sample_3(): 
    return load_data_sample_static("tests_acceptance/metrics/sample_data/example_static_multi_turn_tau2bench_data_sample.json")  


@pytest.fixture
def dynamic_sample_3(): 
    return load_data_sample_dynamic("tests_acceptance/metrics/sample_data/example_dynamic_tau2bench_data_sample.json")  

@pytest.fixture
def azure_openai_client():
    return AzureOpenAIClient(model="gpt-4.1", max_tokens=4096, temperature=0)


# Tau2bench Agent
def test_subgoal_0_complete_with_static_multiturn_sample_3_trace_2(azure_openai_client, static_multiturn_data_sample_3, multiturn_agent_trace_2):
    turn_idx = 1
    subgoal_idx = 0
    agent_trace =  multiturn_agent_trace_2.turns[:turn_idx+1]
    subgoal = static_multiturn_data_sample_3.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate(turn_traces = agent_trace, sub_goal = subgoal))
    assert res.is_completed == True

def test_subgoal_1_complete_with_static_multiturn_sample_3_trace_2(azure_openai_client, static_multiturn_data_sample_3, multiturn_agent_trace_2):
    turn_idx = 1
    subgoal_idx = 1
    agent_trace =  multiturn_agent_trace_2.turns[:turn_idx+1]
    subgoal = static_multiturn_data_sample_3.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate(turn_traces = agent_trace, sub_goal = subgoal))
    assert res.is_completed == True

def test_subgoal_2_complete_with_static_multiturn_sample_3_trace_2(azure_openai_client, static_multiturn_data_sample_3, multiturn_agent_trace_2):
    turn_idx = 2
    subgoal_idx = 2
    agent_trace =  multiturn_agent_trace_2.turns[:turn_idx+1]
    subgoal = static_multiturn_data_sample_3.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate(turn_traces = agent_trace, sub_goal = subgoal))
    assert res.is_completed == True

def test_subgoal_3_complete_with_static_multiturn_sample_3_trace_2(azure_openai_client, static_multiturn_data_sample_3, multiturn_agent_trace_2):
    turn_idx = 2
    subgoal_idx = 3
    agent_trace =  multiturn_agent_trace_2.turns[:turn_idx+1]
    subgoal = static_multiturn_data_sample_3.sub_goals[subgoal_idx]
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate(turn_traces = agent_trace, sub_goal = subgoal))
    assert res.is_completed == True

def test_subgoal_0_turn_0_incomplete_with_dynamic_sample_3_trace_2(azure_openai_client, dynamic_sample_3, multiturn_agent_trace_2):
    turn_idx = 0
    subgoal_idx = 0
    agent_trace =  multiturn_agent_trace_2.turns[:turn_idx+1]
    subgoal = dynamic_sample_3.sub_goals[subgoal_idx]
    user_task = dynamic_sample_3.user_instruction
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction = user_task))
    assert res.is_completed == False


def test_subgoal_1_turn_0_incomplete_with_dynamic_sample_3_trace_2(azure_openai_client, dynamic_sample_3, multiturn_agent_trace_2):
    turn_idx = 0
    subgoal_idx = 1
    agent_trace =  multiturn_agent_trace_2.turns[:turn_idx+1]
    subgoal = dynamic_sample_3.sub_goals[subgoal_idx]
    user_task = dynamic_sample_3.user_instruction
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction = user_task))
    assert res.is_completed == False


def test_subgoal_2_turn_0_incomplete_with_dynamic_sample_3_trace_2(azure_openai_client, dynamic_sample_3, multiturn_agent_trace_2):
    turn_idx = 0
    subgoal_idx = 2
    agent_trace =  multiturn_agent_trace_2.turns[:turn_idx+1]
    subgoal = dynamic_sample_3.sub_goals[subgoal_idx]
    user_task = dynamic_sample_3.user_instruction
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction = user_task))
    assert res.is_completed == False


def test_subgoal_3_turn_0_incomplete_with_dynamic_sample_3_trace_2(azure_openai_client, dynamic_sample_3, multiturn_agent_trace_2):
    turn_idx = 0
    subgoal_idx = 3
    agent_trace =  multiturn_agent_trace_2.turns[:turn_idx+1]
    subgoal = dynamic_sample_3.sub_goals[subgoal_idx]
    user_task = dynamic_sample_3.user_instruction
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction = user_task))
    assert res.is_completed == False

def test_subgoal_0_turn_1_complete_with_dynamic_sample_3_trace_2(azure_openai_client, dynamic_sample_3, multiturn_agent_trace_2):
    turn_idx = 1
    subgoal_idx = 0
    agent_trace =  multiturn_agent_trace_2.turns[:turn_idx+1]
    subgoal = dynamic_sample_3.sub_goals[subgoal_idx]
    user_task = dynamic_sample_3.user_instruction
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction = user_task))
    assert res.is_completed == True

def test_subgoal_1_turn_1_complete_with_dynamic_sample_3_trace_2(azure_openai_client, dynamic_sample_3, multiturn_agent_trace_2):
    turn_idx = 1
    subgoal_idx = 1
    agent_trace =  multiturn_agent_trace_2.turns[:turn_idx+1]
    subgoal = dynamic_sample_3.sub_goals[subgoal_idx]
    user_task = dynamic_sample_3.user_instruction
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction = user_task))
    assert res.is_completed == True

def test_subgoal_2_turn_1_incomplete_with_dynamic_sample_3_trace_2(azure_openai_client, dynamic_sample_3, multiturn_agent_trace_2):
    turn_idx = 1
    subgoal_idx = 2
    agent_trace =  multiturn_agent_trace_2.turns[:turn_idx+1]
    subgoal = dynamic_sample_3.sub_goals[subgoal_idx]
    user_task = dynamic_sample_3.user_instruction
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction = user_task))
    assert res.is_completed == False

def test_subgoal_3_turn_1_incomplete_with_dynamic_sample_3_trace_2(azure_openai_client, dynamic_sample_3, multiturn_agent_trace_2):
    turn_idx = 1
    subgoal_idx = 3
    agent_trace =  multiturn_agent_trace_2.turns[:turn_idx+1]
    subgoal = dynamic_sample_3.sub_goals[subgoal_idx]
    user_task = dynamic_sample_3.user_instruction
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction = user_task))
    assert res.is_completed == False

def test_subgoal_2_turn_2_complete_with_dynamic_sample_3_trace_2(azure_openai_client, dynamic_sample_3, multiturn_agent_trace_2):
    turn_idx = 2
    subgoal_idx = 2
    agent_trace =  multiturn_agent_trace_2.turns[:turn_idx+1]
    subgoal = dynamic_sample_3.sub_goals[subgoal_idx]
    user_task = dynamic_sample_3.user_instruction
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction = user_task))
    assert res.is_completed == True 

def test_subgoal_3_turn_2_complete_with_dynamic_sample_3_trace_2(azure_openai_client, dynamic_sample_3, multiturn_agent_trace_2):
    turn_idx = 2
    subgoal_idx = 3
    agent_trace =  multiturn_agent_trace_2.turns[:turn_idx+1]
    subgoal = dynamic_sample_3.sub_goals[subgoal_idx]
    user_task = dynamic_sample_3.user_instruction
    subgoal_validator = SubGoalCompletionValidator(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})    
    res = asyncio.run(subgoal_validator.validate_dynamic(turn_traces = agent_trace, sub_goal = subgoal, user_instruction = user_task))
    assert res.is_completed == True 



def test_progress_score_full_completion_with_static_multiturn_sample_3_trace_2(azure_openai_client, static_multiturn_data_sample_3, multiturn_agent_trace_2):
    metric = ProgressScore(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    metric_result = metric.evaluate(agent_trace=multiturn_agent_trace_2, evaluation_data_sample=static_multiturn_data_sample_3)
    assert  metric_result.score == 1.0

def test_progress_turns_full_completion_max_turns15_with_dynamic_sample_3_trace_2(azure_openai_client, dynamic_sample_3, multiturn_agent_trace_2):
    metric = ProgressScoresThroughTurns(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True, MAX_TURNS: 15 })
    metric_results = metric.evaluate(agent_trace=multiturn_agent_trace_2, evaluation_data_sample=dynamic_sample_3)
    gt_values = [0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert [r.score for r in metric_results] == gt_values

def test_success_score_successful_with_static_multiturn_sample_3_trace_2(azure_openai_client, static_multiturn_data_sample_3, multiturn_agent_trace_2):
    metric = SuccessScore(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True})
    metric_result = metric.evaluate(agent_trace=multiturn_agent_trace_2, evaluation_data_sample=static_multiturn_data_sample_3)
    assert metric_result.score == 1.0

def test_success_final_turn_successful_max_turns15_with_dynamic_sample_3_trace_2(azure_openai_client, dynamic_sample_3, multiturn_agent_trace_2):
    metric = SuccessScoreFinalTurn(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True, MAX_TURNS: 15})
    metric_result = metric.evaluate(agent_trace=multiturn_agent_trace_2, evaluation_data_sample=dynamic_sample_3)
    assert metric_result.score == 1.0


def test_auc_max_turns15_with_dynamic_sample3_trace_2(azure_openai_client, dynamic_sample_3, multiturn_agent_trace_2):
    metric = AUC(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True, MAX_TURNS: 15})
    metric_result = metric.evaluate(agent_trace=multiturn_agent_trace_2, evaluation_data_sample=dynamic_sample_3)
    assert round(metric_result.score, 4) == 0.9286

def test_auc_max_turns5_with_dynamic_sample3_trace_2(azure_openai_client, dynamic_sample_3, multiturn_agent_trace_2):
    metric = AUC(llm_client=azure_openai_client, config={INCLUDE_JUDGE_EXPLANATION: True, MAX_TURNS: 5})
    metric_result = metric.evaluate(agent_trace=multiturn_agent_trace_2, evaluation_data_sample=dynamic_sample_3)
    print(metric_result.score)
    assert round(metric_result.score, 4) == 0.75
