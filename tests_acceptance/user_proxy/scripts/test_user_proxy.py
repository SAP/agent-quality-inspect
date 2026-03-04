import asyncio

import pytest

from agent_inspect.models.user_proxy import ChatHistory, ConversationTurn, UserProxyMessage, ResponseFromAgent, TerminatingCondition
from agent_inspect.user_proxy import UserProxyAgent
from tests_acceptance.azure_openai_client import AzureOpenAIClient

@pytest.fixture
def opening_chat_history_hello():
    opening_chat_history = ChatHistory(
        id="test_123",
        conversations=[
            ConversationTurn(
                id="1",
                user_message=UserProxyMessage(
                    message_str="Hello, how are you?"
                ),
                agent_responses=[
                    ResponseFromAgent(
                        response_str="I'm doing well, thank you! How can I assist you today?"
                    )
                ]
            )
        ]
    )

    return opening_chat_history

@pytest.fixture
def terminating_chat_history_1_plus_1():
    terminating_chat_history = ChatHistory(
        id="test_123",
        conversations=[
            ConversationTurn(
                id="1",
                user_message=UserProxyMessage(
                    message_str="Hello, how are you?"
                ),
                agent_responses=[
                    ResponseFromAgent(
                        response_str="I'm doing well, thank you! How can I assist you today?"
                    )
                ]
            ),
            ConversationTurn(
                id="2",
                user_message=UserProxyMessage(
                    message_str="What is 1 + 1?"
                ),
                agent_responses=[
                    ResponseFromAgent(
                        response_str="1 + 1 equals 2."
                    )
                ]
            ),
            ConversationTurn(
                id="3",
                user_message=UserProxyMessage(
                    message_str="Thank you for the information."
                ),
                agent_responses=[
                    ResponseFromAgent(
                        response_str="You're welcome!"
                    )
                ]
            )
        ]
    )

    return terminating_chat_history


def test_user_proxy_agent_terminating_condition(terminating_chat_history_1_plus_1):
    azure_openai_client = AzureOpenAIClient(model="gpt-4.1", max_tokens=4096, temperature=0)
    user_proxy_agent = UserProxyAgent(
        llm_client=azure_openai_client,
        task_summary="You task is to ask what is 1 + 1 and get the answer.",
        terminating_conditions=[
            TerminatingCondition(
                check="Ensure that you gets the answer to what is 1 + 1.",
            )
        ],
        agent_description="A helpful AI assistant."
    )

    result_1 = asyncio.run(user_proxy_agent.generate_message_from_chat_history(terminating_chat_history_1_plus_1))
    assert "Ensure that you gets the answer to what is 1 + 1." == result_1.check
    assert "END_CONVERSATION" in result_1.message_str

def test_user_proxy_agent_non_terminating_condition(opening_chat_history_hello):
    azure_openai_client = AzureOpenAIClient(model="gpt-4.1", max_tokens=4096, temperature=0)
    user_proxy_agent = UserProxyAgent(
        llm_client=azure_openai_client,
        task_summary="You task is to ask what is 1 + 1 and get the answer.",
        terminating_conditions=[
            TerminatingCondition(
                check="Ensure that you gets the answer to what is 1 + 1.",
            )
        ],
        agent_description="A helpful AI assistant."
    )

    result_1 = asyncio.run(user_proxy_agent.generate_message_from_chat_history(opening_chat_history_hello))
    assert not result_1.check
    assert not "END_CONVERSATION" in result_1.message_str

def test_user_proxy_agent_no_chat_history_return_initial_message():
    azure_openai_client = AzureOpenAIClient(model="gpt-4.1", max_tokens=4096, temperature=0)
    user_proxy_agent = UserProxyAgent(
        llm_client=azure_openai_client,
        task_summary="You task is to ask what is 1 + 1 and get the answer.",
        terminating_conditions=[
            TerminatingCondition(
                check="Ensure that you gets the answer to what is 1 + 1.",
            )
        ],
        initial_message="This is the initial message.",
        agent_description="A helpful AI assistant."
    )

    result_1 = asyncio.run(user_proxy_agent.generate_message_from_chat_history(None))
    assert not result_1.check
    assert result_1.message_str == "This is the initial message."

def test_user_proxy_agent_no_chat_history_no_initial_message():
    azure_openai_client = AzureOpenAIClient(model="gpt-4.1", max_tokens=4096, temperature=0)
    user_proxy_agent = UserProxyAgent(
        llm_client=azure_openai_client,
        task_summary="You task is to ask what is 1 + 1 and get the answer.",
        terminating_conditions=[
            TerminatingCondition(
                check="Ensure that you gets the answer to what is 1 + 1.",
            )
        ],
        agent_description="A helpful AI assistant."
    )

    result_1 = asyncio.run(user_proxy_agent.generate_message_from_chat_history(None))
    assert not result_1.check
    assert result_1.message_str != ""
