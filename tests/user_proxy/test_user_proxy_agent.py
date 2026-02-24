
import asyncio
from http import HTTPStatus
from unittest.mock import MagicMock, AsyncMock

import pytest

from agent_inspect.metrics.constants import USE_EXPERT_AGENT
from agent_inspect.exception import UserProxyError
from agent_inspect.models import LLMResponse
from agent_inspect.models.user_proxy import ChatHistory, ConversationTurn, UserProxyMessage, \
    ResponseFromAgent, TerminatingCondition
from agent_inspect.user_proxy import UserProxyAgent


def test_get_system_prompt_expert():
    user_proxy_agent = UserProxyAgent(
        llm_client=MagicMock(),
        task_summary="Test Task Summary",
        terminating_conditions=[TerminatingCondition(
            check="This is just a check."
        )],
        agent_description="Test Agent Description",
        config={ USE_EXPERT_AGENT: True }
    )

    expected_system_prompt = """
You are acting as an expert LLM-simulated user who fully understands the AI assistant system and goal. Always respond naturally in clear, concise language that fits the expert user role and goal. Provide complete and precise information in your responses. Generate one line at a time. Do not give away all the instructions at once. Only provide the information that is necessary for the current step.

You are provided with the following user task summary:
[user_task_summary]
Test Task Summary. This is just a check.

You understand the system well and will provide thorough, accurate responses using only the information provided in the [user_task_summary] section.

If the AI assistant returns output in JSON format, respond only to the content inside the JSON as if the format does not matter.

---
The following provides an overview of the AI assistant if available.
[AI Assistant Description] :
Test Agent Description


---
When you as an expert LLM-simulated user is analysing the real-time chat history, carry out a two-step process as the user: 
first, a Reflection Phase, followed by a Response Generation Phase.
"""

    actual_system_prompt = user_proxy_agent.get_system_prompt()
    assert actual_system_prompt == expected_system_prompt

def test_get_system_prompt_non_expert():
    user_proxy_agent = UserProxyAgent(
        llm_client=MagicMock(),
        task_summary="Test Task Summary",
        terminating_conditions=[TerminatingCondition(
            check="This is just a check."
        )],
        agent_description="Test Agent Description",
        config={ USE_EXPERT_AGENT: False }
    )

    expected_system_prompt = """
You are simulating a clueless, casual NON-expert user who is interacting with an AI assistant. You don’t fully understand how the AI system works, and you tend to give vague or incomplete instructions — often leaving out key steps or context.

When you respond:

Speak naturally, casually, like someone who's unsure how to talk to an AI.

Be brief and only provide part of the needed information.

Do not give a full picture unless the assistant directly asks for it.

Only share details that are directly related to what was just asked or prompted — not more.

Never proactively explain your reasoning or provide background info unless the assistant digs into it.

You are working toward the following general task:
[User Task Summary]
Test Task Summary. This is just a check.

But since you’re not an expert, you’ll just sort of "feel your way through it" and leave lots of gaps in your instructions. NEVER provide COMPLETE instructions. ALWAYS OMIT some variables and missing key context.
If the assistant returns something in structured formats like JSON, you can just react casually to the content. Treat the format like it doesn’t matter.

---
The following provides an overview of the AI assistant if available.
[AI Assistant Description] :
Test Agent Description

---
When you as a clueless, casual NON-expert user is analysing the real-time chat history, carry out a two-step process as the user: 
first, a Reflection Phase, followed by a Response Generation Phase.

When simulating your process during the conversation:
You go through two internal steps each time:

1. Reflection Phase (internal thought):
Take a quick look at the current chat history. Think to yourself:
“Okay, what did the assistant just say or ask? What should I probably say next without overexplaining?”
Remember: you're not confident in how this system works, so don’t try to be precise.

2. Response Generation Phase (your reply):
Now write a short, casual message that gives only partial information based on what the assistant asked. Leave things unclear unless the assistant is persistent.


"""
    actual_system_prompt = user_proxy_agent.get_system_prompt()
    assert actual_system_prompt == expected_system_prompt

def test_get_chat_history_str_from_chat_history_single():
    chat_history = ChatHistory(
        id="chat_history_1",
        conversations=[
            ConversationTurn(
                id="test_id",
                user_message=UserProxyMessage(
                    message_str="Hello, what is the weather today"
                ),
                agent_responses=[
                    ResponseFromAgent(
                        response_str="Hi, How can i help you?"
                    )
                ]
            )
        ]
    )
    expected_chat_history_str = \
"""[LLM-simulated user start]:
Hello, what is the weather today
[LLM-simulated user end]
[AI assistant start]:
Hi, How can i help you?
[AI assistant end]
"""
    actual_chat_history_str = UserProxyAgent.get_chat_history_str_from_chat_history(chat_history)
    assert actual_chat_history_str == expected_chat_history_str

def test_get_chat_history_str_from_chat_history_multiple():
    chat_history = ChatHistory(
        id="chat_history_1",
        conversations=[
            ConversationTurn(
                id="test_id",
                user_message=UserProxyMessage(
                    message_str="Hello, what is the weather today"
                ),
                agent_responses=[
                    ResponseFromAgent(
                        response_str="Hi, How can i help you?"
                    )
                ]
            ),
            ConversationTurn(
                id="test_id_2",
                user_message=UserProxyMessage(
                    message_str="Can you tell me a joke?"
                ),
                agent_responses=[
                    ResponseFromAgent(
                        response_str="Sure! Why did the scarecrow win an award?"
                    ),
                    ResponseFromAgent(
                        response_str="Because he was outstanding in his field!"
                    )
                ]
            )
        ]
    )
    expected_chat_history_str = \
"""[LLM-simulated user start]:
Hello, what is the weather today
[LLM-simulated user end]
[AI assistant start]:
Hi, How can i help you?
[AI assistant end]
[LLM-simulated user start]:
Can you tell me a joke?
[LLM-simulated user end]
[AI assistant start]:
Sure! Why did the scarecrow win an award?
[AI assistant end]
[AI assistant start]:
Because he was outstanding in his field!
[AI assistant end]
"""
    actual_chat_history_str = UserProxyAgent.get_chat_history_str_from_chat_history(chat_history)
    assert actual_chat_history_str == expected_chat_history_str

def test_contains_terminating_message():
    user_proxy_agent = UserProxyAgent(
        llm_client=MagicMock(),
        task_summary="Test Task Summary",
        terminating_conditions=[TerminatingCondition(check="check 1"), TerminatingCondition(check="check 2")],
        agent_description="Test Agent Description",
        config={}
    )

    assert user_proxy_agent._contains_stop_sequence("This is a test message. END_CONVERSATION") is True
    assert user_proxy_agent._contains_stop_sequence("This is a test message. See you later") is False

def test_get_user_message_reflection_200_ok():
    mock_llm_client = MagicMock()
    mock_llm_client.make_request_with_payload = AsyncMock()

    mock_llm_response = LLMResponse(
        status=HTTPStatus.OK,
        completion="This is a just reflection.",
        error_message=None
    )

    mock_llm_client.make_request_with_payload.return_value = mock_llm_response
    terminating_conditions = [TerminatingCondition(
        check="This is a just check."
    )]
    user_proxy_agent = UserProxyAgent(
        llm_client=mock_llm_client,
        task_summary="Test Task Summary",
        terminating_conditions=terminating_conditions,
        agent_description="Test Agent Description",
        config={}
    )
    actual_user_message_reflection = asyncio.run(user_proxy_agent.get_user_message_reflection(chat_history_str="This is chat history string.", stop_sequence="Exit", system_prompt="This is system prompt."))
    expected_user_message_reflection = "This is a just reflection."
    assert actual_user_message_reflection == expected_user_message_reflection
    assert mock_llm_client.make_request_with_payload.call_count == 1
    assert mock_llm_client.make_request_with_payload.call_args.args[0].system_prompt == "This is system prompt."
    assert mock_llm_client.make_request_with_payload.call_args.args[0].user_prompt == """

---
The following [Chat History] (if available) provides context and indicates the CURRENT stage of your conversation as a LLM-simulated user with the AI assistant.
[Chat History]
This is chat history string.
---

Step 1: Reflection Phase

Given the [Chat History] REFLECT carefully on the AI assistant’s last response and what the LLM-simulated user is trying to accomplish based on the [user_task_summary].

Briefly address:
- Your role as the LLM-simulated user.
- The current stage of the conversation. You SHOULD NOT skip any user instructions as mentioned in the [user_task_summary].
- The assistant’s last reply in the [Chat History].

IMPORTANT CLARIFICATION:
- Review the entire [Chat History] and the [user_task_summary] and see what should be your next response as a LLM-simulated user.
- At times, the AI assistant’s last message may overlap with or anticipate a future user turn. In such cases, treat it strictly as the AI assistant response, not a replacement of the user message 

Do NOT generate the LLM-simulated user response yet. RESPOND only with a REFLECTION.
**IMPORTANT** remember your user persona as written in the system prompt (eg: expert user or non-expert) and respond with appropriate reflection.

TERMINATE ONLY IF the conversation is at its FINAL STAGE where the agent has completed all the tasks wanted by the user as shown in the [user_task_summary].
If the conversation has concluded, prepare to respond with Exit in the next response generation phase.
Otherwise, DO NOT consider termination if the current conversation is not at its final stage.
        """

def test_get_user_message_reflection_non_200_ok():
    mock_llm_client = MagicMock()
    mock_llm_client.make_request_with_payload = AsyncMock()

    mock_llm_response = LLMResponse(
        status=HTTPStatus.INTERNAL_SERVER_ERROR,
        completion="",
        error_message="Internal Server Error"
    )

    mock_llm_client.make_request_with_payload.return_value = mock_llm_response
    terminating_conditions = [TerminatingCondition(
        check="This is a just check."
    )]
    user_proxy_agent = UserProxyAgent(
        llm_client=mock_llm_client,
        task_summary="Test Task Summary",
        terminating_conditions=terminating_conditions,
        agent_description="Test Agent Description",
        config={}
    )

    with pytest.raises(UserProxyError, match="Internal Code: 060010, Error Message: Unable to get user message reflection due to status: 500 from LLM client.") as exc_info:
        asyncio.run(user_proxy_agent.get_user_message_reflection(chat_history_str="This is chat history string.", stop_sequence="Goodbye", system_prompt="This is system prompt."))

def test_generate_message_from_chat_history_200_ok_not_terminated():
    mock_llm_client = MagicMock()
    mock_llm_client.make_request_with_payload = AsyncMock()

    mock_llm_response_reflection = LLMResponse(
        status=HTTPStatus.OK,
        completion="This is a just reflection.",
        error_message=None
    )

    mock_llm_response_reply = LLMResponse(
        status=HTTPStatus.OK,
        completion="This is the user proxy message response.",
        error_message=None
    )

    mock_llm_client.make_request_with_payload.side_effect = [
        mock_llm_response_reflection,
        mock_llm_response_reply
    ]

    terminating_conditions = [
        TerminatingCondition(
            check="check 1"
        )
    ]

    user_proxy_agent = UserProxyAgent(
        llm_client=mock_llm_client,
        task_summary="Test Task Summary",
        terminating_conditions=terminating_conditions,
        agent_description="Test Agent Description",
        config={}
    )

    chat_history = ChatHistory(
        id="chat_history_1",
        conversations=[
            ConversationTurn(
                id="test_id",
                user_message=UserProxyMessage(
                    message_str="Hello, what is the weather today",
                ),
                agent_responses=[
                    ResponseFromAgent(
                        response_str="Hi, How can i help you?"
                    )
                ]
            )
        ]
    )

    actual_user_proxy_message = asyncio.run(user_proxy_agent.generate_message_from_chat_history(chat_history))
    expected_user_proxy_message = UserProxyMessage(
        message_str="This is the user proxy message response."
    )
    assert actual_user_proxy_message == expected_user_proxy_message
    assert mock_llm_client.make_request_with_payload.call_count == 2

def test_generate_message_from_chat_history_200_ok_terminated():
    mock_llm_client = MagicMock()
    mock_llm_client.make_request_with_payload = AsyncMock()

    mock_llm_response_reflection = LLMResponse(
        status=HTTPStatus.OK,
        completion="This is a just reflection.",
        error_message=None
    )

    mock_llm_response_reply = LLMResponse(
        status=HTTPStatus.OK,
        completion="It was pleasure serving you, Goodbye",
        error_message=None
    )

    mock_llm_client.make_request_with_payload.side_effect = [
        mock_llm_response_reflection,
        mock_llm_response_reply
    ]

    terminating_conditions = [TerminatingCondition(
        check="This is just a check."
    )]
    user_proxy_agent = UserProxyAgent(
        llm_client=mock_llm_client,
        task_summary="Test Task Summary",
        terminating_conditions=terminating_conditions,
        agent_description="Test Agent Description",
        config={}
    )

    chat_history = ChatHistory(
        id="chat_history_1",
        conversations=[
            ConversationTurn(
                id="test_id",
                user_message=UserProxyMessage(
                    message_str="Hello, what is the weather today"
                ),
                agent_responses=[
                    ResponseFromAgent(
                        response_str="Hi, How can i help you?"
                    )
                ]
            )
        ]
    )

    actual_user_proxy_message = asyncio.run(user_proxy_agent.generate_message_from_chat_history(chat_history))
    expected_user_proxy_message = UserProxyMessage(
        message_str="It was pleasure serving you, Goodbye"
    )
    assert actual_user_proxy_message == expected_user_proxy_message
    assert mock_llm_client.make_request_with_payload.call_count == 2

def test_generate_message_from_empty_chat_history():
    user_proxy_agent = UserProxyAgent(
        llm_client=MagicMock(),
        task_summary="Test Task Summary",
        terminating_conditions=[TerminatingCondition(
            check="This is just a check."
        )],
        agent_description="Test Agent Description",
        config={},
        initial_message="This is the initial message."
    )

    actual_user_proxy_message = asyncio.run(user_proxy_agent.generate_message_from_chat_history(chat_history=None))
    expected_user_proxy_message = UserProxyMessage(
        message_str="This is the initial message.",
    )
    assert actual_user_proxy_message == expected_user_proxy_message

def test_generate_message_from_chat_history_reflection_error():
    mock_llm_client = MagicMock()
    mock_llm_client.make_request_with_payload = AsyncMock()

    mock_llm_response_reflection = LLMResponse(
        status=HTTPStatus.OK,
        completion="This is a just reflection.",
        error_message=None
    )

    mock_llm_response_reply = LLMResponse(
        status=HTTPStatus.INTERNAL_SERVER_ERROR,
        completion="",
        error_message="This is an internal server error."
    )

    mock_llm_client.make_request_with_payload.side_effect = [
        mock_llm_response_reflection,
        mock_llm_response_reply
    ]

    terminating_conditions = [
        TerminatingCondition(
            check="check 1"
        )
    ]

    user_proxy_agent = UserProxyAgent(
        llm_client=mock_llm_client,
        task_summary="Test Task Summary",
        terminating_conditions=terminating_conditions,
        agent_description="Test Agent Description",
        config={}
    )

    chat_history = ChatHistory(
        id="chat_history_1",
        conversations=[
            ConversationTurn(
                id="test_id",
                user_message=UserProxyMessage(
                    message_str="Hello, what is the weather today"
                ),
                agent_responses=[
                    ResponseFromAgent(
                        response_str="Hi, How can i help you?"
                    )
                ]
            )
        ]
    )

    with pytest.raises(UserProxyError, match="Internal Code: 060011, Error Message: Unable to generate user proxy message due to status: 500 from LLM client.") as exc_info:
        asyncio.run(user_proxy_agent.generate_message_from_chat_history(chat_history))
