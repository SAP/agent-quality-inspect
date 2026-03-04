import json
import uuid
from datetime import datetime
from typing import List
from agent_inspect.models.metrics import (
    AgentDialogueTrace,
    TurnTrace,
    Step,
    AgentResponse,
    SubGoal,
    Conversation,
    ExpectedToolCall,
    ToolInputParameter,
    EvaluationSample,
    ToolOutput,
)

import copy

def load_json(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def load_agent_trace(trace_file_path: str):
    trace_json_obj = load_json(trace_file_path)
    tools_to_exclude = [
        "Direct Instructions",
        "Decomposition",
        "Plan",
        "Refinement",
        "Intermediate Answer",
    ]

    turns = []

    for turn_json_obj in trace_json_obj:
        to_id_to_step_json_obj_map = {}
        from_id_to_step_json_obj_map = {}

        steps = []

        for step_json_obj in turn_json_obj:
            if step_json_obj["fromId"] and step_json_obj["data"] and step_json_obj["data"][
                "description"] == "Tool result":
                from_id_to_step_json_obj_map[step_json_obj["fromId"]] = step_json_obj
            if step_json_obj["toId"] and step_json_obj["data"] and step_json_obj["data"]["description"] == "Tool input":
                to_id_to_step_json_obj_map[step_json_obj["toId"]] = step_json_obj

        for step_json_obj in turn_json_obj:
            agent_thought = None
            tool_name = None
            tool_input_args = None
            tool_output = None

            if step_json_obj["type"] == "tool" and step_json_obj["data"]["tool"] not in tools_to_exclude:
                tool_name = step_json_obj["data"]["tool"]
                input_step = to_id_to_step_json_obj_map[step_json_obj["ID"]]
                output_step = from_id_to_step_json_obj_map[step_json_obj["ID"]]
                tool_input_args = []
                for key, value in input_step["data"]["data"].items():
                    tool_input_args.append(
                        ToolInputParameter(name=key, value=str(value))
                    )
                tool_output = output_step["data"]["data"]
            if step_json_obj["type"] == "agent":
                agent_thought = step_json_obj["data"]["thought"]

            if len(step_json_obj["tokenConsumption"]) > 0:
                input_token_consumption = sum(
                    token_consumption.get("inputTokens", 0)for token_consumption in step_json_obj["tokenConsumption"])
                output_token_consumption = sum(
                    token_consumption.get("outputTokens", 0) for token_consumption in step_json_obj["tokenConsumption"])
                reasoning_token_consumption = sum(
                    token_consumption.get("reasoningTokens", 0) for token_consumption in step_json_obj["tokenConsumption"])
            else:
                input_token_consumption = 0
                output_token_consumption = 0
                reasoning_token_consumption = 0

            step_obj = Step(
                id=step_json_obj["ID"],
                parent_ids=[step_json_obj["fromId"]],
                tool=tool_name,
                tool_input_args=tool_input_args,
                tool_output=tool_output,
                agent_thought=agent_thought,
                input_token_consumption=input_token_consumption,
                output_token_consumption=output_token_consumption,
                reasoning_token_consumption=reasoning_token_consumption
            )
            steps.append(step_obj)

        agent_response = None
        for step_json_obj in turn_json_obj:
            if step_json_obj["data"] and "tool" in step_json_obj["data"] and step_json_obj["data"][
                "tool"] == "Refinement":
                output_step = from_id_to_step_json_obj_map[step_json_obj["ID"]]
                agent_response = AgentResponse(
                    status_code="200",
                    response=output_step["data"]["data"]
                )
                break

        agent_input = ""
        for step_json_obj in turn_json_obj:
            if step_json_obj["data"] and "description" in step_json_obj["data"] and (step_json_obj["data"][
                "description"] == "Input prompt" or step_json_obj["data"][
                "description"] == "Input value"):
                agent_input = step_json_obj["data"]["data"]
                break

        # Calculate turn latency includeing interstep delays by sorting start and end times
        chronological_steps = sorted(turn_json_obj, key=lambda x: x["createdAt"])
        start_time = datetime.fromisoformat(chronological_steps[0]["createdAt"])
        end_time = datetime.fromisoformat(chronological_steps[-1]["modifiedAt"])
        turn_latency = (end_time - start_time).total_seconds() * 1000
                

        turn_obj = TurnTrace(
            id=str(uuid.uuid4()),
            agent_input=agent_input,
            from_id=None,
            steps=steps,
            agent_response=agent_response,
            latency_in_ms=turn_latency
        )

        turns.append(turn_obj)

    agent_trace = AgentDialogueTrace(turns=turns)

    return agent_trace

def load_data_sample_static(sample_file_path: str):
    dataset_json_obj = load_json(sample_file_path)

    sub_goals = []

    for sub_goal_json in dataset_json_obj["metadata"]["subgoals"]:
        sub_goals.append(
            SubGoal(
                type=sub_goal_json["type"],
                details=sub_goal_json["details"],
                turn=sub_goal_json["turn"]
            )
        )
    i = 0
    conversations = []
    if len(dataset_json_obj["target"]) > 0:
        for input, target in zip(dataset_json_obj["input"], dataset_json_obj["target"]):
            conversations.append(Conversation(turn_id=i, message=input["content"], expected_response=target))
            i += 1
    else:
        for input in dataset_json_obj["input"]:
            conversations.append(Conversation(turn_id=i, message=input["content"]))
            i += 1

    expected_tool_calls = []

    for expected_tool in dataset_json_obj["metadata"]["expectedTools"]:
        expected_parameters = [] 
        if "tool_inputs" in expected_tool:
            parameter_ls = expected_tool["tool_inputs"]
            for param in parameter_ls:
                if param["mode"] == "exact":
                    expected_parameter = ToolInputParameter(name=param["name"], value=param["content"])
                elif param["mode"] == "judge": 
                    expected_parameter = ToolInputParameter(name=param["name"], check=param["content"])
                else:
                    raise NotImplementedError("Method not implemented.")
                expected_parameters.append(expected_parameter)
        
        if "tool_output" in expected_tool:
            output_tool = expected_tool["tool_output"]
            if output_tool["mode"] == "judge":
                expected_output = ToolOutput(check=output_tool["content"])
            elif output_tool["mode"] == "exact":
                expected_output = ToolOutput(value=output_tool["content"])
            else:
                raise NotImplementedError("Method not implemented.")
        else:
            expected_output = None

        expected_tool_call = ExpectedToolCall(tool=expected_tool["tool_name"], expected_parameters=expected_parameters if len(expected_parameters)> 0 else None, expected_output=expected_output, turn=expected_tool["turn"] )
        expected_tool_calls.append(expected_tool_call)
    return EvaluationSample(conversation=conversations, expected_tool_calls=expected_tool_calls, sub_goals=sub_goals)

def load_data_sample_dynamic(sample_file_path: str):
    dataset_json_obj = load_json(sample_file_path)

    sub_goals = []

    for sub_goal_json in dataset_json_obj["metadata"]["subgoals"]:
        sub_goals.append(
            SubGoal(
                type=sub_goal_json["type"],
                details=sub_goal_json["details"],
                turn=sub_goal_json["turn"]
            )
        )
    expected_tool_calls = []
    user_instruction = dataset_json_obj["input"][0]["content"]

    for expected_tool in dataset_json_obj["metadata"]["expectedTools"]:
        expected_parameters = [] 
        if "tool_inputs" in expected_tool:
            parameter_ls = expected_tool["tool_inputs"]
            for param in parameter_ls:
                if param["mode"] == "exact":
                    expected_parameter = ToolInputParameter(name=param["name"], value=param["content"])
                elif param["mode"] == "judge": 
                    expected_parameter = ToolInputParameter(name=param["name"], check=param["content"])
                else:
                    raise NotImplementedError("Method not implemented.")
                expected_parameters.append(expected_parameter)
        
        if "tool_output" in expected_tool:
            output_tool = expected_tool["tool_output"]
            if output_tool["mode"] == "judge":
                expected_output = ToolOutput(check=output_tool["content"])
            elif output_tool["mode"] == "exact":
                expected_output = ToolOutput(value=output_tool["content"])
            else:
                raise NotImplementedError("Method not implemented.")
        else:
            expected_output = None

        expected_tool_call = ExpectedToolCall(tool=expected_tool["tool_name"], expected_parameters=expected_parameters if len(expected_parameters)> 0 else None, expected_output=expected_output, turn=expected_tool["turn"] )
        expected_tool_calls.append(expected_tool_call)
    return EvaluationSample(expected_tool_calls=expected_tool_calls, sub_goals=sub_goals, user_instruction=user_instruction)

def remove_previous_turn_tool_calls(turn_traces: List[TurnTrace]):
    new_turn_traces = copy.deepcopy(turn_traces)
    for turn in new_turn_traces[:-1]:
        turn.steps = []
    return new_turn_traces

def remove_current_turn_tool_calls(turn_traces: List[TurnTrace]):
    new_turn_traces = copy.deepcopy(turn_traces)
    new_turn_traces[-1].steps = []
    return new_turn_traces