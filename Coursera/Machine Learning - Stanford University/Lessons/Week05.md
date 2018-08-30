## Neural Networks: Learning
### Cost Function and Backpropagation
#### Cost Function
L = total n° of layers in the network
$s_l$ = n° of units (not counting bias) in layer $l$
K = n° of units/classes in the output layer L, $s_L = K$

For classification:
* Binary: $y \in \{0, 1\}$, $h_\theta(x) \in \mathbb{R}^1$, $s_L= K = 1$
* Multi-class: $y \in \mathbb{R}^K$, $h_\theta(x) \in \mathbb{R}^K$, $s_L=K$

In neural networks, we may have many output nodes:
* $h_\theta(x)_k$: hypothesis that results in the $k^{th}$ output

Our cost function for neural networks is going to be a generalization of the one we used for logistic regression:
$$
\begin{gather*} J(\Theta) = - \frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K \left[y^{(i)}_k \log ((h_\Theta (x^{(i)}))_k) + (1 - y^{(i)}_k)\log (1 - (h_\Theta(x^{(i)}))_k)\right] + \frac{\lambda}{2m}\sum_{l=1}^{L-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_{l+1}} ( \Theta_{j,i}^{(l)})^2\end{gather*}
$$
* The double sum simply adds up the logistic regression costs calculated for each cell in the output layer. Loops through the number of output nodes.
* The triple sum simply adds up the squares of all the individual Θs in the entire network.
* The i in the triple sum does not refer to training example i

For multiple theta matrices:
* Number of columns: number of nodes in our current layer (including the bias unit)
* Number of rows: number of nodes in the next layer (excluding the bias unit)

To minimize we need:
$J(\Theta)$ and the (partial) derivate terms $\frac{\partial}{\partial\Theta_{ij}^{(l)}}$ for every $i,j,l$

#### Backpropagation Algorithm & Intuition
*Backpropagation* is neural-network terminology for minimizing our cost function: $\min_\Theta J(\Theta)$

We start by computing $\delta$ for the output layer and we go back.

$\delta_j^{(l)}$, captures the error of node $j$ in layer $l$

e.g:
$\delta_j^{(l)} = a_j^{(4)} - y_j$, with $a_j^{(4)} = (h_\theta(x))_j$ as the hypothesis output
Or:
$\delta^{(4)} = a^{(4)} - y$
As vectors with dimensions equal to the output number.

Compute:
$$
\delta_j^{(4)} = a_j^{(4)}-y_j \\
\delta_j^{(3)} = (\theta^{(3)})^T  \delta_{(4)} .* g'(z^{(3)}) \\
\delta_j^{(2)} = (\theta^{(2)})^T \delta_{(3)} .* g'(z^{(2)})
$$

> There no is $\delta^{(1)}$, the first layer correspond to the input, the features we observed.

Where derivate of the activation function $g$ evaluated at the input values, equal to:
$$
g'(z^{(3)}) = a^{(3)} .* (1 - a^{(3)})
$$

Note if ignoring $\lambda$ or $\lambda = 0$, the partial derivatives correspond to:
$$
\frac{\partial}{\partial\Theta_{ij}^{(l)}} J(\theta) =  a_j^{(l)} \delta_i^{(l+1)}
$$

The partial derivatives correspond to $\triangle$:
$$
\triangle_{ij}^{(l)} := \triangle_{ij}^{(l)} + a_j^{(l)} \delta_i^{(l+1)} \\
\triangle^{(l)} := \triangle^{(l)} + \delta^{(l+1)} (a^{(l)})^T
$$

Algorithm:
* Set: $\triangle_{ij}^{(l)} = 0$, for all $l, i, j$

For training example $t = 1$ to $m$:
1. Set: $a^{(1)} := x^{(t)}$
1. Perform forward propagation to compute $a^{(l)}$ for $l = 2, 3, ..., L$
1. Using $y^{(t)}$ compute: $\delta^{(L)} = a^{(L)}-y^{(t)}$
1. Steps back from right to left, compute $\delta^{(L-1)}, \delta^{(L-2)},\dots,\delta^{(2)}$ using:
$$
\delta^{(l)} = ((\Theta^{(l)})^T \delta^{(l+1)})\ .*\ a^{(l)}\ .*\ (1 - a^{(l)})
$$
The delta values of layer $l$ are calculated by multiplying the delta values in the next layer with the theta matrix of layer $l$ multiplied element-wise with g-prime, derivative of the activation function g evaluated with the input values given by $z^{(l)}$.
$$
g'(z^{(l)}) = a^{(l)}\ .*\ (1 - a^{(l)})
$$
1. Update our new $\Delta$ matrix, $\Delta^{(l)}_{i,j} := \Delta^{(l)}_{i,j} + a_j^{(l)} \delta_i^{(l+1)}$, as a vector:
$$
\Delta^{(l)} := \Delta^{(l)} + \delta^{(l+1)}(a^{(l)})^T
$$
The capital-delta matrix D is used as an "accumulator" to add up our values as we go along and eventually compute our partial derivative. Thus we get $\frac \partial {\partial \Theta_{ij}^{(l)}} J(\Theta)$:
$$
D^{(l)}_{i,j} := \dfrac{1}{m}\left(\Delta^{(l)}_{i,j} + \lambda\Theta^{(l)}_{i,j}\right)\text{, if j≠0}
$$
$$
D^{(l)}_{i,j} := \dfrac{1}{m}\Delta^{(l)}_{i,j} \text{, if j=0}
$$
$j = 0$, correspond to the bias term

> Our "error values" for the last layer are simply the differences of our actual results in the last layer and the correct outputs in y.

e.g:
* FP using $x^{(1)}$ followed by BP using $y^{(1)}$
* Then, FP using $x^{(2)}$ followed by BP using $y^{(2)}$

To calculate the "error" $\delta_j^{(l)}$ for $a^{(l)}_j$ (unit $j$ in layer $l$), we start from right to left, e.g:
$$
\delta_2^{(3)} = \Theta_{12}^{(3)} \delta_1^{(4)} \\
\delta_2^{(2)} = \Theta_{12}^{(2)} \delta_1^{(3)} + \Theta_{22}^{(2)} \delta_2^{(3)}
$$


### Backpropagation in Practice
#### Implementation Note: Unrolling Parameters
* In Neural Networks, our parameters $\theta^{(i)}$ and $\Delta^{(i)}$ are matrices so we must *unroll* them into vectors to use optimizing functions such as "fminunc()"
```matlab
thetaVector = [ Theta1(:); Theta2(:); Theta3(:); ]
deltaVector = [ D1(:); D2(:); D3(:) ]
```
If the dimensions of Theta1 is 10x11, Theta2 is 10x11 and Theta3 is 1x11, then we can get back our original matrices from the "unrolled" versions as follows:
```matlab
Theta1 = reshape(thetaVector(1:110),10,11)
Theta2 = reshape(thetaVector(111:220),10,11)
Theta3 = reshape(thetaVector(221:231),1,11)
```

#### Gradient Checking
Calculate the gradient approximation:
$$m = \frac{y_2-y_1}{x_2-x_1}$$

We can approximate the derivative of our cost function with:
$$
\dfrac{\partial}{\partial\Theta}J(\Theta) \approx \dfrac{J(\Theta + \epsilon) - J(\Theta - \epsilon)}{2\epsilon}
$$

With multiple theta matrices, we can approximate the derivative with respect to $Θ_j$:
$$
\dfrac{\partial}{\partial\Theta_j}J(\Theta) \approx \dfrac{J(\Theta_1, \dots, \Theta_j + \epsilon, \dots, \Theta_n) - J(\Theta_1, \dots, \Theta_j - \epsilon, \dots, \Theta_n)}{2\epsilon}
$$

Hence, we are only adding or subtracting epsilon to the $\Theta_j$ matrix. In octave
```matlab
epsilon = 1e-4;
for i = 1:n,
  thetaPlus = theta;
  thetaPlus(i) += epsilon;
  thetaMinus = theta;
  thetaMinus(i) -= epsilon;
  gradApprox(i) = (J(thetaPlus) - J(thetaMinus))/(2*epsilon)
end;
```

Implementation:
1. Implement backprop to compute DVec (unrolled $\Delta$), deltaVector
1. Implement numerical *gradient check*, gradApprox.
1. Make sure they give similar values, gradApprox ≈ deltaVector
1. **Turn off** gradient checking and use backprop for learning. To compute gradApprox can be very slow.

#### Random Initialization
We need an initial value for $\theta$, and we use Symmetry breaking. Initialize each $\theta_{ij}^{(l)}$ to a random value in $[-\epsilon, \epsilon]$:

```matlab
If the dimensions of Theta1 is 10x11, Theta2 is 10x11 and Theta3 is 1x11.

Theta1 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta2 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta3 = rand(1,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
```

> The $\epsilon$ used above is unrelated to the one in *Gradient Checking*

Problem of simetric weights:  
$\theta_{ij}^{(l)} = r$ for all $i,j,l$. After each update, all nodes will update to the same value repeatedly.

#### Putting It Together
Pick a network architecture; choose the layout of your neural network, including how many hidden units in each layer and how many layers in total you want to have:
* n° of input units: Dimension of features $x^{(i)}$
* n° of output units: Number of classes
* n° of hidden layers: n° of units the same in every hidden layer. More is better but more complex. Bigger than the number of features is ok.

Training a Neural Network:
1. Randomly initialize weights
1. Implement forward propagation to get $h_\theta(x^{(i)})$ for any $x^{(i)}$
1. Implement the cost function $J(\theta)$
1. Implement backpropagation to compute partial derivatives of  $J(\theta)$ with respect to the parameters $\theta_{jk}^{(l)}$.
**Inside a for loop (for 1:m):**  
We perform forward propagation and backpropagation using $(x^{(i)}, y^{(i)})$ to get activations $a^{(l)}$, $\delta^{(l)}$ for $l = 2, ..., L$ and the delta acumulation terms $\Delta^{(l)}:= \Delta^{(l)} + \delta^{(l+1)}(a^{(l)})^T$
**Outside the foor loop:**  
Compute the partial derivate terms.
1. Use gradient checking to compare the numerical estimate of gradient of $J(\theta)$ vs the partial derivatives calculated in backpropagation. Disable gradient checking.
1. Use gradient descent or advanced optimization with backpropagation to minimize $J(\theta)$ as a function of parameters $\theta$.

Backpropagation computes the direction of the gradient and gradient descent takes the steps. Together try to minimize $J(\theta)$ as a function of $\theta$ to get $h_\Theta(x^{(i)}) \approx y^{(i)}$.

> In Neural Networks $J(\Theta)$ is not convex and thus we can end up in a local minimum instead.

###### Matlab/Octave
Recall that whereas the original labels (in the variable y) were 1, 2, ..., 10, for the purpose of training a neural network, we need to recode the labels as vectors containing only values 0 or 1.

For example, if $x^(i)$ is an image of the digit 5, then the corresponding $y^(i)$ (that you should use with the cost function) should be a 10-dimensional vector with $y_5 = 1$, and the other elements equal to 0.
