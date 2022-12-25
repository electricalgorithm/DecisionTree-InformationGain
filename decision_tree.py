"""
This module represents the decision tree learner.
"""
from math import log2
import sys
import pandas


class Node:
    """This class represents a node in the decision tree."""

    def __init__(self, attr, vote=None, data=None):
        # Holds which attribute to split.
        self.attribute: str = attr
        # Nodes on the left and right.
        # Always assign the attribute value 1
        # to left child and 0 to right child.
        self.left: Node = None
        self.right: Node = None
        # If leaf, it is the prediction.
        # Else, it is None
        self.vote: str = vote
        # The data in the node.
        self.data: pandas.DataFrame = data


class DecisionTreeLearner:
    """This class represents the decision tree learner."""

    def __init__(self, data: pandas.DataFrame, max_depth: int) -> None:
        self.train_data = data
        self.max_depth: int = max_depth

        # Internal variables.
        self._attribute_count = len(self.train_data.columns) - 1
        self._real_max_depth = min(self._attribute_count, self.max_depth)
        self._root = None
        self._is_trained = False

    def train(self):
        """
        Trains the decision tree.
        """
        if self._is_trained:
            raise Exception("Decision tree is already trained.")

        # Build the tree.
        self._root = self._build_tree(self.train_data, self._real_max_depth)
        self._is_trained = True

    def predict(self, example: dict) -> str:
        """
        It follows the decision tree and returns the predicted value.

        :param example: is a dictionary which holds the attributes and the values of the example.
        :return: The predicted value.
        """
        if not self._is_trained:
            raise Exception("Decision tree is not trained yet.")

        return self._predict(self._root, example)

    def predict_all(self, test_data: pandas.DataFrame) -> list:
        """
        It follows the decision tree and returns the predicted value.

        :param test_data: The test data to test the decision tree.
        :return: The predicted value.
        """
        if not self._is_trained:
            raise Exception("Decision tree is not trained yet.")

        return test_data.apply(self.predict, axis=1)

    def test(self, test_data: pandas.DataFrame) -> float:
        """
        Tests the decision tree on the test data.

        :param test_data: The test data to test the decision tree.
        :return: The accuracy of the decision tree.
        """
        if not self._is_trained:
            raise Exception("Decision tree is not trained yet.")

        # Get the predicted values.
        predicted_values = self.predict_all(test_data)
        # Get the actual values.
        actual_values = test_data.iloc[:, -1]
        # Calculate the error.
        return 1 - (predicted_values == actual_values).sum() / len(test_data)

    def print_tree(self):
        """
        Prints the decision tree.
        """
        if self._is_trained:
            zero_count = len(self._root.data[self._root.data.iloc[:, -1] == 0])
            one_count = len(self._root.data[self._root.data.iloc[:, -1] == 1])

            print(f"[{zero_count} 0/{one_count} 1]")
            self._print_tree(self._root, 0)
            print()

    def get_tree(self):
        """
        Returns the decision tree.
        """
        if self._is_trained:
            return self._root

    def get_max_depth(self):
        """
        Returns the maximum depth of the tree.
        """
        if self._is_trained:
            return self._real_max_depth

    # ################ #
    # INTERNAL METHODS #
    # ################ #
    def _build_tree(self, data: pandas.DataFrame, depth: int) -> Node:
        """
        Builds the decision tree recursively.

        :param data: The data to build the decision tree.
        :param depth: The depth of the tree.
        :return: The root node of the decision tree.
        """
        # If depth is 0 or all the labels are same, return a leaf node.
        # or attribute count is greater then max depth.
        if depth == 0 or len(data.iloc[:, -1].unique()) == 1:
            # Return the most common label.
            most_common_value = data.iloc[:, -1].mode()[0]
            return Node(None, most_common_value, data=data)

        # Find the best attribute to split.
        best_attribute = DecisionTreeLearner.find_best_attribute(data)
        # Create a node with the best attribute.
        node = Node(best_attribute, data=data)
        # Split the data according to the best attribute.
        left_data = data[data[best_attribute] == 1]
        right_data = data[data[best_attribute] == 0]
        # Build the left and right subtrees.
        node.left = self._build_tree(left_data, depth - 1)
        node.right = self._build_tree(right_data, depth - 1)
        return node

    def _predict(self, node: Node, example: dict) -> str:
        """
        It follows the decision tree and returns the predicted value.

        :param node: The node to start the prediction.
        :param example: is a dictionary which holds the attributes and the values of the example.
        :return: The predicted value.
        """
        # If it is a leaf, return the vote.
        if node.vote is not None:
            return node.vote
        # Get the value of the attribute.
        attribute_value = example[node.attribute]
        # If the value is 1, go to the left child.
        if attribute_value == 1:
            return self._predict(node.left, example)
        # Else, go to the right child.
        return self._predict(node.right, example)

    def _print_tree(self, node: Node, depth: int):
        """
        Prints the decision tree recursively in this format:

        :param node: The node to start the print.
        :param depth: The depth of the tree.
        """
        if node.vote is None:
            zero_count_right = len(node.right.data[node.right.data.iloc[:, -1] == 0])
            one_count_right = len(node.right.data[node.right.data.iloc[:, -1] == 1])
            print(
                f"{'| ' * (depth + 1)}{node.attribute} = 0:",
                end=f" [{zero_count_right} 0/{one_count_right} 1]\n",
            )
            self._print_tree(node.right, depth + 1)

            zero_count_left = len(node.left.data[node.left.data.iloc[:, -1] == 0])
            one_count_left = len(node.left.data[node.left.data.iloc[:, -1] == 1])
            print(
                f"{'| ' * (depth + 1)}{node.attribute} = 1:",
                end=f" [{zero_count_left} 0/{one_count_left} 1]\n",
            )
            self._print_tree(node.left, depth + 1)

    # ################ #
    #  STATIC METHODS  #
    # ################ #
    @staticmethod
    def find_best_attribute(data: pandas.DataFrame) -> str:
        """
        Finds the best attribute to split the data.

        :param data: The data to find the best attribute.
        :return: The best attribute to split the data.
        """
        # Calculate the information gain of each attribute.
        information_gains = [
            DecisionTreeLearner.calculate_information_gain(data, attribute_column)
            for attribute_column in range(len(data.columns) - 1)
        ]
        # Find the attribute with the maximum information gain.
        return data.columns[information_gains.index(max(information_gains))]

    @staticmethod
    def calculate_entropy_of_label(data: pandas.DataFrame) -> float:
        """
        Calculates the entropy for given probabilities of each class.
        Note that each element represents the probability of having a class.
        :param probabilities_of_having_class: List of probabilities of each class.
        :return: Entropy of the given probabilities.
        """
        # Get last column of the DataFrame.
        label_column = data.iloc[:, -1]
        # Get the unique values of the last column.
        unique_values = label_column.unique()
        # Calculate the probabilities of each class.
        probabilities_of_having_class = [
            len(label_column[label_column == value]) / len(label_column)
            for value in unique_values
        ]
        # Calculate the entropy.
        entropy = 0
        for probability in probabilities_of_having_class:
            entropy += probability * log2(probability)
        return -entropy

    @staticmethod
    def calculate_conditional_entropy(
        data: pandas.DataFrame, attribute_column: int
    ) -> float:
        """
        Calculates the conditional entropy of the given attribute.
        :param data: The data to calculate the conditional entropy.
        :param attribute: The attribute to calculate the conditional entropy.
        :return: The conditional entropy of the given attribute.
        """
        # Get the attribute column.
        attribute = data.columns[attribute_column]
        # Get the unique values of the attribute.
        unique_values = data[attribute].unique()
        # Calculate the probabilities of each attribute value.
        probabilities_of_attribute = [
            len(data[data[attribute] == value]) / len(data) for value in unique_values
        ]
        # Calculate the conditional entropy.
        return sum(
            [
                probability
                * DecisionTreeLearner.calculate_entropy_of_label(
                    data[data[attribute] == value]
                )
                for probability, value in zip(probabilities_of_attribute, unique_values)
            ]
        )

    @staticmethod
    def calculate_information_gain(
        data: pandas.DataFrame, attribute_column: int
    ) -> float:
        """
        Calculates the information gain of the given attribute.
        :param data: The data to calculate the information gain.
        :param attribute: The attribute to calculate the information gain.
        :return: The information gain of the given attribute.
        """
        return DecisionTreeLearner.calculate_entropy_of_label(
            data
        ) - DecisionTreeLearner.calculate_conditional_entropy(data, attribute_column)


class Application:
    """Application class"""

    @staticmethod
    def argument_parser() -> dict:
        """
        It parses the arguments given.
        :return: The argument dictionary.
        """
        args = sys.argv[1:]
        if len(args) != 6:
            raise SystemExit("[ERROR] Invalid number of arguments.")

        return {
            "max_depth": int(args[2]),
            "train": {
                "input_file": args[0],
                "output_file": args[3],
            },
            "test": {
                "input_file": args[1],
                "output_file": args[4],
            },
            "metrics": {
                "output_file": args[5],
            },
        }

    @staticmethod
    def main():
        """Main function."""
        # Parse the arguments.
        args = Application.argument_parser()
        # Load train and test data.
        train_data = pandas.read_table(args["train"]["input_file"])
        test_data = pandas.read_table(args["test"]["input_file"])

        # Create a decision tree learner and train it.
        decision_tree_learner = DecisionTreeLearner(
            data=train_data, max_depth=args["max_depth"]
        )
        decision_tree_learner.train()
        decision_tree_learner.print_tree()

        # Write the output of train and test data to their output files.
        train_predictions = decision_tree_learner.predict_all(train_data)
        with open(
            args["train"]["output_file"], "w", encoding="UTF-8"
        ) as train_output_file:
            for prediction in train_predictions:
                train_output_file.write(f"{prediction}\n")

        test_predictions = decision_tree_learner.predict_all(test_data)
        with open(
            args["test"]["output_file"], "w", encoding="UTF-8"
        ) as test_output_file:
            for prediction in test_predictions:
                test_output_file.write(f"{prediction}\n")

        # Test the decision tree learner.
        train_error = decision_tree_learner.test(train_data)
        test_error = decision_tree_learner.test(test_data)

        # Write the metrics to the output file.
        with open(args["metrics"]["output_file"], "w", encoding="UTF-8") as output_file:
            output_file.write("error(train): %.6f\n" % train_error)
            output_file.write("error(test): %.6f\n" % test_error)


if __name__ == "__main__":
    Application.main()
