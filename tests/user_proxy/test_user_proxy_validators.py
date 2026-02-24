import pytest

from agent_inspect.exception import InvalidInputValueError
from agent_inspect.metrics.utils.user_proxy_validators import UserProxyInputValidator
from agent_inspect.models.user_proxy import TerminatingCondition


def test_validate_terminating_condition_validation_fails_empty_string():
    with pytest.raises(InvalidInputValueError, match="Internal Code: 060008, Error Message: Terminating check cannot be an empty string.") as exc_info:

        UserProxyInputValidator.validate_terminating_condition(terminating_conditions=[TerminatingCondition(
            check=""
        )])

def test_validate_terminating_condition_validation_fails_empty_list():
    with pytest.raises(InvalidInputValueError, match="Internal Code: 060008, Error Message: At least one terminating condition must be provided to create User Proxy.") as exc_info:
        UserProxyInputValidator.validate_terminating_condition(terminating_conditions=[])

def test_validate_terminating_condition_validation_passes():
    try:
        UserProxyInputValidator.validate_terminating_condition(terminating_conditions=[TerminatingCondition(
            check="This is a valid terminating condition."
        )])
    except InvalidInputValueError:
        pytest.fail("UserProxyError was raised unexpectedly!")

def test_validate_task_summary_validation_fails():
    with pytest.raises(InvalidInputValueError, match="Internal Code: 060008, Error Message: Task summary cannot be empty to create User Proxy.") as exc_info:
        UserProxyInputValidator.validate_task_summary(task_summary="")

def test_validate_task_summary_validation_passes():
    try:
        UserProxyInputValidator.validate_task_summary(task_summary="This is a valid task summary.")
    except InvalidInputValueError:
        pytest.fail("UserProxyError was raised unexpectedly!")