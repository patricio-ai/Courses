## Linear Regression with Multiple Variables
### Environment Setup Instructions

### Multivariate Linear Regression
#### Multiple Features
$ X^{4}_{1} $: $4^{th}$ row/feature, $1^{st}$ column

For convenience of notation, define  
$ x_0 = 1$ as  $ x^{i}_0 = 1 $

Multivariable form of the hypothesis function with n features:
$
h_{\theta}(x) = \theta_0 x_0 + \theta_1 x_1 + ... + \theta_n x_n
= \theta^T x
$
> $\theta^T$ is a 1 by (n+1) matrix, n+1 dimensional vector

$$
x =
\begin{pmatrix}
  x_{0} \\
  x_{1} \\
  \vdots \\
  x_{n} \\
\end{pmatrix}
\epsilon \enspace \mathbb{R}^{n+1}
\quad
\theta =
\begin{pmatrix}
  \theta_{0} \\
  \theta_{1} \\
  \vdots \\
  \theta_{n} \\
\end{pmatrix}
$$
We now introduce notation for equations where we can have any number of input variables.
$$
\begin{align*}x_j^{(i)} &= \text{value of feature } j \text{ in the }i^{th}\text{ training example} \newline x^{(i)}& = \text{the input (features) of the }i^{th}\text{ training example} \newline m &= \text{the number of training examples} \newline n &= \text{the number of features} \end{align*}
$$

###### Cost function
$$
J(\theta) = \frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})^2
$$
> $J(\theta_0, ..., \theta_n) = J(\theta)$

#### Gradient Descent for Multiple Variables
Repeat until convergence:
$$
\begin{align*}& \text{repeat until convergence:} \; \lbrace \newline \; & \theta_j := \theta_j - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)} \; & \text{for j := 0...n}\newline \rbrace\end{align*}
$$


#### Gradient Descent in Practice - Feature Scaling & Learning Rate
*  We can speed up gradient descent by having each of our input values in roughly the same range.
> This is because θ will descend quickly on small ranges and slowly on large ranges, and so will oscillate inefficiently down to the optimum when the variables are very uneven.

Two techniques to help with this are feature scaling and mean normalization.
* Feature scaling involves dividing the input values by the range (i.e. the maximum value minus the minimum value) of the input variable, resulting in a new range of just 1.
* Mean normalization involves subtracting the average value for an input variable from the values for that input variable resulting in a new average value for the input variable of just zero.

$$
x_i := \dfrac{x_i - \mu_i}{s_i}
$$
Where $μ_i$ is the average of all the values for feature (i) and $s_i$ is the range of values (max - min), or si is the standard deviation.

#### Features and Polynomial Regression
* We can combine multiple features into one. For example, we can combine x1 and x2 into a new feature x3 by taking x1⋅x2.
* We can change the behavior or curve of our hypothesis function by making it a quadratic, cubic or square root function (or any other form). Feature scaling becomes very important.

### Computing Parameters Analytically
#### Normal Equation
Allows us to find the optimum theta without iteration
$$
\theta = (X^T X)^{-1}X^T y
$$

* $m$ examples, $n$ features
* No need to choose $\alpha$
* Don't need to iterate.
* Need to compute $(X^TX)^{-1}_{(n+1) \times (n+1)}$ ~ $O(n^3)$
* Slow if $n$ is very large

> if we have a very large number of features (n > 10.000), go to an iterative process

#### Normal Equation Noninvertibility
What if $X^TX$ is non-invertible? (singular/degenerate)

May be because:
* Redundant features, where two features are very closely related (linearly dependent), delete some features
* Too many features ($m \leq n$), delete some or use regularization

> use psudo inverse to avoid this issue, pinv in octave
