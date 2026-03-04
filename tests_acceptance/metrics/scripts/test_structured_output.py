from tests_acceptance.azure_openai_client import AzureOpenAIClient
from agent_inspect.models import LLMPayload

import asyncio
import json

def test_gen_ai_hub_client_make_request_with_payload():
    client = AzureOpenAIClient(model="gpt-4.1", max_tokens=4096, temperature=0)

    payload = LLMPayload(
        user_prompt="how can I solve 8x + 7 = -23",
        structured_output={
            "type": "json_schema",
            "json_schema": {
                "name": "math_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "explanation": {"type": "string"},
                                    "output": {"type": "string"}
                                },
                                "required": ["explanation", "output"],
                                "additionalProperties": False
                            }
                        },
                        "final_answer": {"type": "string"}
                    },
                    "required": ["steps", "final_answer"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
    )

    response = asyncio.run(client.make_request_with_payload(payload))
    assert response.status == 200

    parsed_response = json.loads(response.completion)
    assert isinstance(parsed_response, dict)
    assert set(parsed_response.keys()) == {"steps", "final_answer"}

    steps = parsed_response["steps"]
    assert isinstance(steps, list)
    assert steps, "Expected at least one reasoning step"
    for step in steps:
        assert isinstance(step, dict)
        assert set(step.keys()) == {"explanation", "output"}
        assert isinstance(step["explanation"], str)
        assert isinstance(step["output"], str)

    assert isinstance(parsed_response["final_answer"], str)
