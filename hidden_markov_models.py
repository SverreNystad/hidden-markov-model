import numpy as np


class HiddenMarkovModel:
    """Hidden Markov Model class"""

    def __init__(
        self,
        transition_matrix: np.ndarray,
        emission_matrix: np.ndarray,
        initial_state: np.ndarray,
    ):
        """
        Constructor for the HiddenMarkovModel class

        Args:
            transition_matrix: the transition matrix for the model
            emission_matrix: the emission matrix for the model
            initial_state: the initial state distribution for the model
        """
        self.transition_matrix = transition_matrix
        self.emission_matrix = emission_matrix
        self.initial_state = initial_state
        self.negated_emission_matrix = 1 - self.emission_matrix

    def forward(self, observations: list[bool]) -> list[np.ndarray[float, float]]:
        """
        Performs the forward algorithm to compute the probabilities of the observation sequence under the model.
        It is a dynamic programming algorithm that computes the probability of being in a specific state at a specific time given the sequence of observations.

        For more information: https://en.wikipedia.org/wiki/Forward_algorithm

        Args:
            observations: list of boolean values representing the observed states
        Returns:
            float: the probability of the observed states given the model
        """
        forward_probabilities = [self.initial_state]
        current_prediction = self.initial_state

        for observation in observations:
            # Predict next state based on current state probabilities
            predicted_probabilities = np.dot(current_prediction, self.transition_matrix)

            # Update predicted state probabilities based on new observation
            if observation:
                observation_probability = np.dot(
                    predicted_probabilities, self.emission_matrix.T
                )
            else:
                observation_probability = np.dot(
                    predicted_probabilities, self.negated_emission_matrix.T
                )

            normalized_probabilities = (
                observation_probability / observation_probability.sum()
            )

            # Prepare for next iteration and collect the probabilities
            current_prediction = normalized_probabilities
            forward_probabilities.append(current_prediction)

        return forward_probabilities

    def format_output(self, forward_probabilities: list[np.ndarray]) -> str:
        """
        Formats the output of the forward probabilities for easy reading.

        Args:
            forward_probabilities (list[np.ndarray]): The list of probability distributions computed by the forward algorithm.

        Returns:
            str: A formatted string representation of the forward probabilities.
        """
        representation = ""
        for time, state_probabilities in enumerate(forward_probabilities):
            representation += f"P(X{time}|e1:{time}) = {state_probabilities}\n"
        return representation.strip()


if __name__ == "__main__":

    hmm = HiddenMarkovModel(
        transition_matrix=np.array([[0.7, 0.3], [0.3, 0.7]]),
        emission_matrix=np.array([[0.9, 0], [0, 0.2]]),
        initial_state=[0.5, 0.5],
    )
    observations = [True, True]
    all_predictions = hmm.forward(observations)
    print("========== Forward Algorithm ==========")
    print("Task 1")
    print(hmm.format_output(all_predictions))
    print("=" * 40)
    print("Task 2")
    observations = [True, True, False, True, True]
    all_predictions = hmm.forward(observations)
    print(hmm.format_output(all_predictions))
