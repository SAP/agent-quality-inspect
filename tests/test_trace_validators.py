import pytest

from agent_inspect.exception import InvalidInputValueError
from agent_inspect.metrics.utils.trace_validators import TraceValidator
from agent_inspect.models.metrics import TurnTrace, AgentResponse


@pytest.fixture
def mock_turn_trace_missing_input():
    return TurnTrace(
        id="turn_1",
        agent_input="",
        agent_response=AgentResponse(
            response="This is a sample response."
        ),
        steps=[]
    )

@pytest.fixture
def mock_turn_trace_missing_response():
    return TurnTrace(
        id="turn_3",
        agent_input="This is a valid input.",
        agent_response=AgentResponse(
            response=""
        ),
        steps=[]
    )

@pytest.fixture
def mock_turn_trace_valid():
    return TurnTrace(
        id="turn_2",
        agent_input="This is a valid input.",
        agent_response=AgentResponse(
            response="This is a sample response."
        ),
        steps=[]
    )

def test_validate_agent_input_missing_input(mock_turn_trace_missing_input):
    with pytest.raises(InvalidInputValueError, match="Turn :turn_1 is missing agent input."):
        TraceValidator.validate_agent_input(mock_turn_trace_missing_input)

def test_validate_agent_input_valid(mock_turn_trace_valid):
    TraceValidator.validate_agent_input(mock_turn_trace_valid)

def test_validate_agent_response_missing_response(mock_turn_trace_missing_response):
    with pytest.raises(InvalidInputValueError, match="Turn :turn_3 is missing agent response."):
        TraceValidator.validate_agent_response(mock_turn_trace_missing_response)

def test_validate_agent_response_valid(mock_turn_trace_valid):
    TraceValidator.validate_agent_response(mock_turn_trace_valid)

