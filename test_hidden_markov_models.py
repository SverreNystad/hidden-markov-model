import pytest
import numpy as np
from hidden_markov_models import HiddenMarkovModel


@pytest.fixture
def umbrella_world_model() -> HiddenMarkovModel:
    transition_matrix = np.array([[0.7, 0.3], [0.3, 0.7]])
    emission_matrix = np.array([[0.9, 0], [0, 0.2]])
    initial_state = [0.5, 0.5]

    return HiddenMarkovModel(
        transition_matrix=transition_matrix,
        emission_matrix=emission_matrix,
        initial_state=initial_state,
    )


def test_forward_algorithm_task_0(umbrella_world_model):
    # Define the HMM model
    observations = [True]

    all_predictions = umbrella_world_model.forward(observations)
    assert isinstance(all_predictions, list)


def test_forward_algorithm_task_1(umbrella_world_model):
    # Define the HMM model
    observations = [True, True]
    expected = 0.883
    margin = 0.001

    all_predictions = umbrella_world_model.forward(observations)
    actual = all_predictions[-1]
    assert actual[0] <= expected + margin
    assert actual[0] >= expected - margin


def test_forward_algorithm_task_2(umbrella_world_model):
    # Define the HMM model
    observations = [True, True, False, True, True]
    expected = 0.883
    margin = 0.001

    all_predictions = umbrella_world_model.forward(observations)
    actual = all_predictions[2]
    assert actual[0] <= expected + margin
    assert actual[0] >= expected - margin
