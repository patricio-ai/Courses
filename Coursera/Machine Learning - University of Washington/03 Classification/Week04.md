## Overfitting in decision trees
- As you increase the depth you increase the complexity of the decision boundaries and the training error.

- In a perfect decision tree, even though the training error is zero, the true error can shoot up.

**Occam's Razor for decision trees**
*When two trees have similar classification error on the validation set, pick the simpler one.*

How do we pick simpler trees?
- **Early stopping**: stop learning algorithm before tree become too complex
  - **Limit tree depth:** Stop splitting after a certain depth
  - **Classification error:** Do not consider any split that does not cause a sufficient decrease in classification error.
  - **Minimum node size:** Do not split an intermediate node which contains too few data points.

- **Pruning**: simplify tree after learning algorithm terminates
    - The more leaves you have, the more complex the tree is
    $$
    L(T) = \text{# of leaf nodes}
    $$

**Desired total quality format**
Want to balance:
- How well tree fits data
- Complexity of tree
$$
\text{total cost} = \text{measure of fit} + \text{measure of complexity} \\
\text{Total cost } C(T) = \text{Error}(T) + \lambda \ \text{L}(T)
$$

- if $\lambda = 0$
    Standard decision tree learning
- if $\lambda = \infty$
    Root tree, great penalty

$\text{measure of fit}:$ Classification error in the training data. If large, bad fit to training data **underfit**
$\text{measure of complexity}:$ number of leaves. If large, likely to **overfit**

**Handling missing data**
Missing data can impact us both in the training time and at prediction time.

- Purification (training)
    - Throw out missing data
    - Skip features with missing values
- Imputing/filling missing data (prediction)
    - Categorical features use mode: most popular value/mode
    - Numerical features use: average/median
- Modify the learning algorithm to handle missing data (training & prediction)
  - Add missing values to the tree definition with a default branch based on the lowest classification error
