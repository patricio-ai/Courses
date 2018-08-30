## Logistic Regression
### Classification and Representation
#### Classification
The values we now want to predict take on only a small number of discrete values.

Do not use linear regression for a classification problem
* Classification is not actually a linear function
* Outliers shift the fitting, hence a chosen threshold is skewed
* $h_x(\theta)$: can output values smaller than 0, and larger than 1

> Use Logistic/Sigmoid Regression

#### Hypothesis Representation
Hypothesis Output $h_\theta(x) = P(y=1 | x ; \theta)$, probability that $y=1$ given $x$ parameterized by $\theta$
$$
\begin{align*}& h_\theta (x) = g ( \theta^T x ), 0 \leq h_\theta (x) \leq 1 \newline \newline& z = \theta^T x \newline& g(z) = \dfrac{1}{1 + e^{-z}}\end{align*}
$$

The function g(z), maps any real number to the (0, 1) interval

#### Decision Boundary
The shape that separates the area where y = 0 and where y = 1. It is created by our hypothesis and the parameters not the training set. This shape doesn't need to be linear and could be any shape to fit our data

Predict $y=1$ if $z = \theta^T x \geq 0$
$$
\begin{align*}& h_\theta(x) = g(\theta^T x) \geq 0.5 \newline& when \; \theta^T x \geq 0\end{align*}
$$


### Logistic Regression Model
#### Cost Function
How to choose/fit parameters $\theta$?
* Non-convex: multiple local optimums, not guaranteed to converge to the global minimum  
* Gradient descent will converge to the global optimums

> We cannot use the same cost function that we use for linear regression because the Logistic Function will cause the output to be wavy, causing many local optima. In other words, it will not be a convex function.

Cost function for logistic regression:
$$
\begin{align*}& J(\theta) = \dfrac{1}{m} \sum_{i=1}^m \mathrm{Cost}(h_\theta(x^{(i)}),y^{(i)}) \newline & \mathrm{Cost}(h_\theta(x),y) = -\log(h_\theta(x)) \; & \text{if y = 1} \newline & \mathrm{Cost}(h_\theta(x),y) = -\log(1-h_\theta(x)) \; & \text{if y = 0}\end{align*}
$$

Note that writing the cost function in this way guarantees that J(θ) is convex for logistic regression.

if $h_\theta(x) = y$, then $\text{cost}(h_\theta(x),y) = 0$ (for $y=0$ and $y=1$)

if $y=0$ then $\text{cost}(h_\theta(x),y)\rightarrow\infty$ as $h_\theta(x)\rightarrow 1$

Regardless of wheter $y=0$ or $y=1$ if $h_\theta(x)=0.5$, then $\text{cost}(h_\theta(x),y) > 0$

$$
\begin{align*}& \mathrm{Cost}(h_\theta(x),y) = 0 \text{ if } h_\theta(x) = y \newline & \mathrm{Cost}(h_\theta(x),y) \rightarrow \infty \text{ if } y = 0 \; \mathrm{and} \; h_\theta(x) \rightarrow 1 \newline & \mathrm{Cost}(h_\theta(x),y) \rightarrow \infty \text{ if } y = 1 \; \mathrm{and} \; h_\theta(x) \rightarrow 0 \newline \end{align*}
$$

* If our correct answer 'y' is 0, then the cost function will be 0 if our hypothesis function also outputs 0. If our hypothesis approaches 1, then the cost function will approach infinity.

* If our correct answer 'y' is 1, then the cost function will be 0 if our hypothesis function outputs 1. If our hypothesis approaches 0, then the cost function will approach infinity.

#### Simplified Cost Function and Gradient Descent
> Algorithm looks identical to linear regression, but the hypothesis changes

We can compress our cost function's two conditional cases into one case:
$$
\mathrm{Cost}(h_\theta(x),y) = - y \; \log(h_\theta(x)) - (1 - y) \log(1 - h_\theta(x))
$$

We can fully write out our entire cost function as follows:
$$
J(\theta) = - \frac{1}{m} \displaystyle \sum_{i=1}^m [y^{(i)}\log (h_\theta (x^{(i)})) + (1 - y^{(i)})\log (1 - h_\theta(x^{(i)}))]
$$

A vectorized implementation is:
$$
\begin{align*} & h = g(X\theta)\newline & J(\theta) = \frac{1}{m} \cdot \left(-y^{T}\log(h)-(1-y)^{T}\log(1-h)\right) \end{align*}
$$

Gradient Descent:
$$
\theta_j := \theta_j - \alpha \dfrac{\partial}{\partial \theta_j}J(\theta) = \theta_j - \frac{\alpha}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}
$$

A vectorized implementation is:
$$
\theta := \theta - \frac{\alpha}{m} X^{T} (g(X \theta ) - \vec{y})
$$

#### Advanced Optimization
Other algorithms (not Gradient descent)
* Conjugate gradient
* BFGS
* L-BFGS

More complex but no need to manually pitch $\alpha$ and faster

### Multiclass Classification
#### Multiclass Classification: One-vs-all
Now we will approach the classification of data when we have more than two categories

* we divide our problem: TTrain a logistic regression classifier $h_\theta^{(i)}(x)$ for each class $i$ to predict the probability that $y=i$

On a new input $x$ to make a prediction, pick the class $i$ that  $max(h_\theta^{(i)}(x))$


## Regularization
### Solving the Problem of Overfitting
#### The Problem of Overfitting
###### Underfit or High/Strong bias
Structure not captured by the model/hypothesis that is too simple or uses too few features. Bias because it has a strong preconception.

###### Overfit or High variance
Fits the available data but does not generalize well to predict new data. Usually caused by a complicated function with too many features (high polynomial) that creates a lot of unnecessary curves and angles unrelated to the data.

##### Addressing overfitting:
* Reduce the number of features Manually or by a model selection algorithm
* Regularization to keep all but reduce the magnitude of parameters $\theta_j$

#### Cost Function
If we have overfitting from our hypothesis function, we can reduce the weight that some of the terms in our function carry by increasing their cost.

* Add extra regularization parameter ($\lambda$) and regularization term
* $\lambda$: trade-off between, fit the training data and keep parameters $\theta$ small to avoid overfitting
* high $\lambda$ cost: results in underfitting (fails to fit the training set)
$$
min_\theta\ \dfrac{1}{2m}\  \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda\ \sum_{j=1}^n \theta_j^2
$$

#### Regularized Linear Regression
We can apply regularization to both linear regression and logistic regression.

###### Gradient Descent
We will modify our gradient descent function to separate out $\theta_0$ from the rest of the parameters because we do not want to penalize $\theta_0$.

$$
\theta_j := \theta_j(1 - \alpha\frac{\lambda}{m}) - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}
$$
The term $\frac{\lambda}{m}\theta_j$ performs our regularization.

The first term in the above equation will always be less than 1.  Intuitively you can see it as reducing the value of $\theta_j$ by some amount on every update. The second term is the same as before.

###### Normal Equation
Add another term inside the parentheses to the non-iterative normal equation
$$
\begin{align*}& \theta = \left( X^TX + \lambda \cdot L \right)^{-1} X^Ty \newline& \text{where}\ \ L = \begin{bmatrix} 0 & & & & \newline & 1 & & & \newline & & 1 & & \newline & & & \ddots & \newline & & & & 1 \newline\end{bmatrix}\end{align*}
$$
L (dim: (n+1)×(n+1)) is a matrix with 0 at the top left and 1's down the diagonal, with 0's everywhere else.

Recall that if m < n, then $X^TX$ is non-invertible. However, when we add the term λ⋅L, then $X^TX$ + λ⋅L becomes invertible.

#### Regularized Logistic Regression
We can regularize logistic regression in a similar way that we regularize linear regression. As a result, we can avoid overfitting.

Cost Function:
$$
J(\theta) = - \frac{1}{m} \sum_{i=1}^m \large[ y^{(i)}\ \log (h_\theta (x^{(i)})) + (1 - y^{(i)})\ \log (1 - h_\theta(x^{(i)}))\large] + \frac{\lambda}{2m}\sum_{j=1}^n \theta_j^2
$$

The second sum, $\sum_{j=1}^n \theta_j^2$ means to explicitly exclude the bias term, $\theta_0$

$$
\begin{align*}& \text{repeat until convergence:} \; \lbrace \newline \; & \theta_j := \theta_j - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)} \; & \text{for j := 0}\newline \rbrace\end{align*}
$$
$$
\begin{align*}& \text{repeat until convergence:} \; \lbrace \newline \; & \theta_j := \theta_j - \alpha \large[ \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)} + \frac{\lambda}{m} \cdot \theta_j \large]\; & \text{for j := 1...n}\newline \rbrace\end{align*}
$$

###### Conclusion

* One way to fit the data better is to create more features from each data
point
* While the feature mapping allows us to build a more expressive classifier,
it also more susceptible to overfitting.
* Regularization can help combat the overfitting problem
*  Small λ, you should find that the classifier gets almost every training example
correct, but draws a very complicated boundary, thus overfitting the data.
* High λ, not a good fit and the decision boundary will not follow the data so well, thus underfitting the data.
