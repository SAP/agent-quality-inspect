import uuid
import copy
import json
import sys
import threading
import time
from pathlib import Path
from tool_sandbox.common.execution_context import RoleType, set_current_context, get_current_context, DatabaseNamespace
from tool_sandbox.common.message_conversion import Message, serialize_to_conversation
from tool_sandbox.cli.utils import resolve_scenarios, AGENT_TYPE_TO_FACTORY, USER_TYPE_TO_FACTORY
from tool_sandbox.common.tool_discovery import ToolBackend
from tool_sandbox.roles.base_role import BaseRole
from tool_sandbox.roles.execution_environment import ExecutionEnvironment
from tool_sandbox.common.scenario import Scenario, Milestone, Minefield
import polars as pl

class SessionManager:
    def __init__(self):
        self.sessions = {}
    
    def call_with_timeout(self, func, timeout_seconds=180):
        """Call a function with timeout (default 3 minutes)"""
        result = [None]
        exception = [None]
        
        def target():
            try:
                result[0] = func()
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout_seconds)
        
        if thread.is_alive():
            print(f"[ERROR] Agent call timed out after {timeout_seconds} seconds")
            # Note: We can't actually kill the thread, but we can return an error
            raise TimeoutError(f"Agent response timed out after {timeout_seconds} seconds")
        
        if exception[0]:
            raise exception[0]
        
        return result[0]

    def start_session(self, agent_type, scenario_name):
        session_id = str(uuid.uuid4())
        scenarios = resolve_scenarios([scenario_name], ToolBackend.DEFAULT)
        scenario = scenarios[scenario_name]
        # Create agent, environment, user (fix: use RoleType.USER)
        agent = AGENT_TYPE_TO_FACTORY[agent_type]()
        environment = ExecutionEnvironment()
        user = USER_TYPE_TO_FACTORY[RoleType.USER]() if RoleType.USER in USER_TYPE_TO_FACTORY else None
        context = copy.deepcopy(scenario.starting_context)
        set_current_context(context)
        # Handle system messages as in Scenario.play
        sandbox_db = context.get_database(DatabaseNamespace.SANDBOX, drop_sandbox_message_index=False, get_all_history_snapshots=True)
        max_sandbox_message_index = context.max_sandbox_message_index
        for message_index in range(max_sandbox_message_index + 1):
            if (
                sandbox_db["recipient"][message_index] == RoleType.EXECUTION_ENVIRONMENT
                and sandbox_db["sender"][message_index] == RoleType.SYSTEM
            ):
                environment.respond(ending_index=message_index)
        self.sessions[session_id] = {
            "agent_type": agent_type,
            "scenario": scenario,
            "agent": agent,
            "environment": environment,
            "context": context,
        }
        # Do NOT call self.save_trajectory(session_id) here
        return session_id

    def get_session(self, session_id):
        return self.sessions.get(session_id)

    def add_user_message(self, session, message):
        context = session["context"]
        set_current_context(context)
        msg = Message(sender=RoleType.USER, recipient=RoleType.AGENT, content=message)
        BaseRole.add_messages([msg])
        
        # Update session context
        session["context"] = context

    def agent_respond(self, session, session_id=None):
        context = session["context"]
        set_current_context(context)
        agent = session["agent"]
        environment = session["environment"]
        
        # Turn-taking loop as in Scenario.play - exactly match the original implementation
        max_messages = 30  # Default from scenario
        initial_max_index = context.max_sandbox_message_index
        
        while True:
            sandbox_db = context.get_database("SANDBOX", get_all_history_snapshots=True, drop_sandbox_message_index=False)
            
            # Check termination conditions
            conversation_active = sandbox_db["conversation_active"][-1]
            current_message_index = sandbox_db["sandbox_message_index"][-1]
            
            if not conversation_active or current_message_index >= max_messages + initial_max_index:
                break
                
            last_msg_recipient = sandbox_db["recipient"][-1]
            
            if last_msg_recipient == RoleType.AGENT:
                print(f"[DEBUG] Making agent call... (message index: {current_message_index})")
                try:
                    self.call_with_timeout(agent.respond, timeout_seconds=180)  # 3 minutes
                    print(f"[DEBUG] Agent call completed")
                except TimeoutError as e:
                    print(f"[ERROR] {e}")
                    # Break out of the loop on timeout to prevent infinite hanging
                    break
                except Exception as e:
                    print(f"[ERROR] Agent call failed: {e}")
                    # Break out of the loop on other errors too
                    break
            elif last_msg_recipient == RoleType.EXECUTION_ENVIRONMENT:
                print(f"[DEBUG] Making execution environment call...")
                environment.respond()
                print(f"[DEBUG] Execution environment call completed")
            elif last_msg_recipient == RoleType.USER:
                # Update session context before saving trajectory
                session["context"] = get_current_context()
                # Save trajectory after each user turn
                if session_id:
                    self.save_trajectory(session_id)
                # Return the last agent message
                msgs = sandbox_db.to_dicts()
                for msg in reversed(msgs):
                    if msg["sender"] == RoleType.AGENT:
                        return msg["content"]
                return None
            else:
                break
                
            # Refresh the database for the next iteration (exactly like original)
            context = get_current_context()
            set_current_context(context)
        
        # Update session context before returning
        session["context"] = get_current_context()
        
        # If we exit the loop, return the last agent message
        sandbox_db = context.get_database("SANDBOX", get_all_history_snapshots=True, drop_sandbox_message_index=False)
        msgs = sandbox_db.to_dicts()
        for msg in reversed(msgs):
            if msg["sender"] == RoleType.AGENT:
                return msg["content"]
        return None

    def save_trajectory(self, session_id):
        session = self.sessions[session_id]
        context = session["context"]
        scenario = session["scenario"]
        output_dir = Path("data") / session_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure Polars to show full content without truncation
        pl.Config.set_tbl_rows(-1).set_tbl_cols(-1).set_fmt_str_lengths(10000)
        pl.Config.set_tbl_formatting("ASCII_FULL")

        pretty_print_str = (
            "Note that User Simulator few shot messages have been omitted\n"
            + str(
                context.get_database(
                    DatabaseNamespace.SANDBOX,
                    get_all_history_snapshots=True,
                    drop_sandbox_message_index=False,
                )
                .filter(
                    (pl.col("visible_to") != [RoleType.USER])
                    | (pl.col("visible_to").is_null())
                )
                .drop([
                    "openai_tool_call_id",
                    "conversation_active",
                ])
            )
        )
        with open(output_dir / "pretty_print.txt", "w", encoding="utf-8") as f:
            f.write(pretty_print_str)
        # Execution context
        with open(output_dir / "execution_context.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(context.to_dict(serialize_console=False), ensure_ascii=False, indent=4))
        # Conversation
        milestones = getattr(scenario.evaluation, "milestone_matcher", None)
        minefields = getattr(scenario.evaluation, "minefield_matcher", None)
        conversation = serialize_to_conversation(
            context,
            None,
            milestones.milestones if milestones else [],
            minefields.milestones if minefields else [],
        )
        # Group messages into turns: each turn starts with a user message
        turns = []
        current_turn = []
        for msg in conversation:
            if msg.get("role") == "system":
                continue
            if msg.get("role") == "user":
                if current_turn:
                    turns.append(current_turn)
                current_turn = [msg]
            else:
                current_turn.append(msg)
        if current_turn:
            turns.append(current_turn)
        with open(output_dir / "conversation.json", "w", encoding="utf-8") as f:
            json.dump(turns, f, indent=2, ensure_ascii=False)
    
    def get_trajectory(self, session_id):
        """Get the saved trajectory for debugging purposes"""
        output_dir = Path("data") / session_id
        conversation_file = output_dir / "conversation.json"
        if conversation_file.exists():
            with open(conversation_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return []