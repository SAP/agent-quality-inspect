import pytest
from agent_inspect.metrics.constants import K_VALUE, NO_OF_TRIALS
from agent_inspect.models.metrics import NumericalScore
from agent_inspect.metrics.multi_samples import PassAtK, PassHatK
from agent_inspect.exception import EvaluationError

def test_acceptance_pass_at_k_basic():
    metric = PassAtK(config={K_VALUE: 2, NO_OF_TRIALS: 4})
    result = metric.compute([NumericalScore(1), NumericalScore(0), NumericalScore(1), NumericalScore(0)])
    assert abs(result.score - 0.8333333) < 1e-5  # 1 - C(2,2)/C(4,2) = 1 - 1/6 = 5/6

def test_acceptance_pass_at_k_all_success():
    metric = PassAtK(config={K_VALUE: 3, NO_OF_TRIALS: 3})
    result = metric.compute([NumericalScore(1), NumericalScore(1), NumericalScore(1)])
    assert result.score == 1.0

def test_acceptance_pass_at_k_all_fail():
    metric = PassAtK(config={K_VALUE: 2, NO_OF_TRIALS: 3})
    result = metric.compute([NumericalScore(0), NumericalScore(0), NumericalScore(0)])
    assert result.score == 0.0

def test_acceptance_pass_at_k_invalid_k():
    with pytest.raises(EvaluationError, match="k_value .* cannot be greater than num_trials .*"):
        PassAtK(config={K_VALUE: 5, NO_OF_TRIALS: 3})

def test_acceptance_pass_hat_k_basic():
    metric = PassHatK(config={K_VALUE: 2, NO_OF_TRIALS: 4})
    result = metric.compute([NumericalScore(1), NumericalScore(0), NumericalScore(1), NumericalScore(0)])
    assert abs(result.score - (1/6)) < 1e-6  # C(2,2)/C(4,2) = 1/6

def test_acceptance_pass_hat_k_all_success():
    metric = PassHatK(config={K_VALUE: 3, NO_OF_TRIALS: 3})
    result = metric.compute([NumericalScore(1), NumericalScore(1), NumericalScore(1)])
    assert result.score == 1.0

def test_acceptance_pass_hat_k_all_fail():
    metric = PassHatK(config={K_VALUE: 2, NO_OF_TRIALS: 3})
    result = metric.compute([NumericalScore(0), NumericalScore(0), NumericalScore(0)])
    assert result.score == 0.0

def test_acceptance_pass_hat_k_invalid_k():
    with pytest.raises(EvaluationError, match="k_value .* cannot be greater than num_trials .*"):
        PassHatK(config={K_VALUE: 5, NO_OF_TRIALS: 3})

# --- Additional PassAtK Tests ---
def test_acceptance_pass_at_k_no_num_of_trials_given():
    with pytest.raises(EvaluationError, match="num_trials .* must be provided"):
        PassAtK()

def test_acceptance_pass_at_k_no_k_value_given():
    metric = PassAtK(config={NO_OF_TRIALS: 5})
    success_scores = [NumericalScore(1) for _ in range(5)]
    result = metric.compute(success_scores)
    assert result.score == 1.0

def test_acceptance_pass_at_k_empty_success_flags():
    metric = PassAtK(config={K_VALUE: 2, NO_OF_TRIALS: 4})
    success_scores = []
    with pytest.raises(EvaluationError, match="Success scores should have the same length as num_trials .*, but got .*"):
        metric.compute(success_scores)

def test_acceptance_pass_at_k_error_k_zero():
    with pytest.raises(EvaluationError, match="k_value .* must be greater than 0"):
        PassAtK(config={K_VALUE: 0, NO_OF_TRIALS: 4})
    
def test_acceptance_pass_at_k_error_n_trials_zero():
    with pytest.raises(EvaluationError, match="num_trials .* must be provided"):
        PassAtK(config={K_VALUE: 2, NO_OF_TRIALS: 0})
    

# --- Additional PassHatK Tests ---
def test_acceptance_pass_hat_k_no_num_of_trials_given():
    with pytest.raises(EvaluationError, match="num_trials .* must be provided"):
        PassHatK()
    

def test_acceptance_pass_hat_k_no_k_value_given():
    metric = PassHatK(config={NO_OF_TRIALS: 5})
    success_scores = [NumericalScore(1) for _ in range(5)]
    result = metric.compute(success_scores)
    assert result.score == 1.0

def test_acceptance_pass_hat_k_empty_success_flags():
    metric = PassHatK(config={K_VALUE: 2, NO_OF_TRIALS: 4})
    success_scores = []
    with pytest.raises(EvaluationError, match="Success scores should have the same length as num_trials .*, but got .*"):
        metric.compute(success_scores)

def test_acceptance_pass_hat_k_error_k_zero():
    with pytest.raises(EvaluationError, match="k_value .* must be greater than 0"):
        PassHatK(config={K_VALUE: 0, NO_OF_TRIALS: 4})
        
def test_acceptance_pass_hat_k_error_n_trials_zero():
    with pytest.raises(EvaluationError, match="num_trials .* must be provided"):
        metric = PassHatK(config={K_VALUE: 2, NO_OF_TRIALS: 0})