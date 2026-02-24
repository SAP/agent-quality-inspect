# import asyncio
# from http import HTTPStatus
# from unittest.mock import patch, MagicMock
#
# import httpx
# from openai import APIStatusError, RateLimitError
#
# from agent_inspect.constants import MAX_RETRY_ATTEMPTS_EXCEEDED, STATUS_200, STATUS_429
# from tests_acceptance.metrics.scripts.gen_ai_hub_client import GenAiHubClient
#
# @patch("gen_ai_hub.proxy.native.openai.chat.completions.create")
# def test_make_llm_request_successful_response(mock_chat_completion_create):
#     mock_response = MagicMock(choices=[MagicMock(message=MagicMock(content="Generated response"))])
#     mock_chat_completion_create.return_value = mock_response
#
#     client = GenAiHubClient("test-model", 100, 0.1)
#     result = asyncio.run(client.make_llm_request("Test prompt"))
#
#     assert result.status == HTTPStatus.OK
#     assert result.completion == "Generated response"
#
#
# @patch("gen_ai_hub.proxy.native.openai.chat.completions.create")
# def test_make_llm_requests_api_status_error(mock_chat_completion_create):
#     mock_response = httpx.Response(
#         status_code=HTTPStatus.NOT_FOUND,
#         request=httpx.Request("POST", "https://this.is.a.mock.url"),
#         json={"error": {"message": "Invalid input format"}},
#     )
#     mock_chat_completion_create.side_effect = APIStatusError("This is a bad request", response=mock_response, body=None)
#
#     client = GenAiHubClient("test-model", 100, 0)
#     prompt = "Prompt 1"
#     result = asyncio.run(client.make_llm_request(prompt))
#
#     assert result.status == HTTPStatus.NOT_FOUND
#     assert result.completion == ""
#     assert result.error_message == "This is a bad request"
#
# @patch("gen_ai_hub.proxy.native.openai.chat.completions.create")
# def test_make_llm_request_unknown_exception(mock_chat_completion_create):
#     mock_chat_completion_create.side_effect = Exception("Unknown error occurred")
#
#     client = GenAiHubClient("test-model", 100, 0)
#     prompt = "Prompt causing unknown error"
#     result = asyncio.run(client.make_llm_request(prompt))
#
#     assert result.status == HTTPStatus.INTERNAL_SERVER_ERROR
#     assert result.completion == ""
#     assert result.error_message == "Unknown error occurred"
#
# @patch("gen_ai_hub.proxy.native.openai.chat.completions.create")
# def test_make_llm_requests_successful_responses(mock_chat_completion_create):
#     mock_response_1 = MagicMock(choices=[MagicMock(message=MagicMock(content="Response 1"))])
#     mock_response_2 = MagicMock(choices=[MagicMock(message=MagicMock(content="Response 2"))])
#     mock_chat_completion_create.side_effect = [mock_response_1, mock_response_2]
#
#     client = GenAiHubClient("test-model", 100, 0)
#     prompts = ["Prompt 1", "Prompt 2"]
#     results = asyncio.run(client.make_llm_requests(prompts))
#
#     assert len(results) == 2
#     assert results[0].status == STATUS_200
#     assert results[0].completion == "Response 1"
#     assert results[1].status == STATUS_200
#     assert results[1].completion == "Response 2"
#
#
# @patch("gen_ai_hub.proxy.native.openai.chat.completions.create")
# def test_make_llm_requests_max_retry_reached(mock_chat_completion_create):
#     mock_response = httpx.Response(
#         status_code=429,
#         request=httpx.Request("POST", "https://this.is.a.mock.url"),
#         json={"error": {"message": "Rate limited!"}},
#     )
#     mock_chat_completion_create.side_effect = RateLimitError("Rate limit exceeded", response=mock_response, body=None)
#
#     client = GenAiHubClient("test-model", 100, 0)
#     prompts = ["Prompt causing rate limit error"]
#     result = asyncio.run(client.make_llm_request(prompts[0]))
#
#     assert result.status == STATUS_429
#     assert result.completion == ""
#     assert result.error_message == MAX_RETRY_ATTEMPTS_EXCEEDED
