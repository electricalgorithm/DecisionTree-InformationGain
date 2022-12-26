# Decision Tree
This is a Decision Tree implementation with Python which uses information gain to split attributes.

When $Y$ is the target attribute and $X$ is an attribute, the information gain is defined as whera $H$ is the entropy function:
    
```math
\text{InfoGain}(Y, X) = H(Y) - H(Y|X)
```

## Installing
```bash
$ git clone git@github.com:electricalgorithm/DecisionTree-InformationGain.git
$ cd DecisionTree-InformationGain
$ pip install -r requirements.txt
```

## Usage
```
$ python decision_tree.py [args...]
```
Where above [args...] is a placeholder for six command-line arguments: `<train input>`, `<test input>`,
`<max depth>`, `<train out>`, `<test out>`, `<metrics out>`. These arguments are described in detail below:

1. _train input_: Path to the training input .tsv file.
2. _test input_: Path to the test input .tsv file.
3. _max depth_: Maximum depth to which the tree should be built.
4. _train out_` Path of output .txt file to which the predictions on the training data should be written.
5. _test out_: Path of output .txt file to which the predictions on the test data should be written.
6. _metrics out_: Path of the output .txt file to which metrics such as train and test error should be written.

## Output
The output of the program is written to the files specified by the command-line arguments `<train out>`, `<test out>`, and `<metrics out>`. Furthermore,
the tree is also printed to the console with using `DecisionTreeLearner::print_tree()`. 

A minimum tree printing node can be explained as:
```
[X 0/Y 1]
| ATTRIBUTE_A = 0: [X 0/Y 1]
| | ATTRIBUTE_B = 0: [X 0/Y 1]
| | ATTRIBUTE_B = 1: [X 0/Y 1]
| ATTRIBUTE_A = 1: [X 0/Y 1]
| | ATTRIBUTE_C = 0: [X 0/Y 1]

    (X number of samples which equals 0.)
    (Y number of samples which equals 1.)
(...)
```
Note that print functionality only works for binary classification problems for now.

## Further Improvements
- [ ] Add support for printing trees of multi-class classification problems.
- [ ] Add support for continuous attributes.
- [ ] Add support for pruning.
- [ ] Add support for random forests.
- [ ] Add support for cross-validation.

<br>

---

<br>

## Example Usage
```bash
$ python decision_tree.py data/train.tsv data/test.tsv 5 train_out.txt test_out.txt metrics_out.txt

[65 0/135 1]
| F = 0: [42 0/16 1]
| | M2 = 0: [27 0/3 1]
| | | M4 = 0: [22 0/0 1]
| | | M4 = 1: [5 0/3 1]
| | M2 = 1: [15 0/13 1]
| | | M4 = 0: [14 0/7 1]
| | | M4 = 1: [1 0/6 1]
| F = 1: [23 0/119 1]
| | M4 = 0: [21 0/63 1]
| | | M2 = 0: [18 0/26 1]
| | | M2 = 1: [3 0/37 1]
| | M4 = 1: [2 0/56 1]
| | | P1 = 0: [2 0/15 1]
| | | P1 = 1: [0 0/41 1]
```
