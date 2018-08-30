## Neural Networks: Representation
### Motivations
#### Non-linear Hypotheses
50x50 = 2500 px
1px-> 0-255
Number of combinations of k elements chosen from n, repetition allowed:
$$
{{n+k-1}\choose{k}} = \frac{(n+k-1)!}{k!(n-1)!}
$$
Answer: C(2500, 2) = 2500*2499/2 = 3,123,750.

Intuition: We want a 2 terms. There are 2500 ways to pick the first term, leaving 2499 possible second terms. However, for each pair we have a repeat. So we divide 2500*2499 by 2.

### Neural Networks
#### Neural Networks
The neural network will be able to represent complex models that form non-linear hypotheses

#### Model Representation
Hypothesis representation using neural networks.
> At a very simple level, neurons are basically computational units that take inputs (*dendrites*) as electrical inputs (*spikes*) that are channeled to outputs (*axons*).

* In this model our $x_0$ input node is sometimes called the "bias unit" and it is always equal to 1
* We use the same logistic function sometimes call it a sigmoid (logistic) activation function
* $\theta$ parameters are called *weights*

Layers:
* Input layer: input nodes
* Intermediate/Hidden layer: between the input and output layers.  
  Hidden layer nodes (*activation units*): $a^2_0 \cdots a^2_n$
* Output layer: outputs the hypothesis function

$$
\begin{align*}& a_i^{(j)} = \text{activation of unit $i$ in layer $j$} \newline& \Theta^{(j)} = \text{matrix of weights controlling function mapping from layer $j$ to layer $j+1$}\end{align*}
$$
If we had one hidden layer, it would look like:
$$
\begin{bmatrix}x_0 \newline x_1 \newline x_2 \newline x_3\end{bmatrix}\rightarrow\begin{bmatrix}a_1^{(2)} \newline a_2^{(2)} \newline a_3^{(2)} \newline \end{bmatrix}\rightarrow h_\theta(x)
$$

The values for each of the *activation* nodes is obtained as follows:
$$
\begin{align*} a_1^{(2)} = g(\Theta_{10}^{(1)}x_0 + \Theta_{11}^{(1)}x_1 + \Theta_{12}^{(1)}x_2 + \Theta_{13}^{(1)}x_3) \newline a_2^{(2)} = g(\Theta_{20}^{(1)}x_0 + \Theta_{21}^{(1)}x_1 + \Theta_{22}^{(1)}x_2 + \Theta_{23}^{(1)}x_3) \newline a_3^{(2)} = g(\Theta_{30}^{(1)}x_0 + \Theta_{31}^{(1)}x_1 + \Theta_{32}^{(1)}x_2 + \Theta_{33}^{(1)}x_3) \newline h_\Theta(x) = a_1^{(3)} = g(\Theta_{10}^{(2)}a_0^{(2)} + \Theta_{11}^{(2)}a_1^{(2)} + \Theta_{12}^{(2)}a_2^{(2)} + \Theta_{13}^{(2)}a_3^{(2)}) \newline \end{align*}
$$


Our hypothesis output is the logistic function applied to the sum of the values of our activation nodes, which have been multiplied by yet another parameter matrix $\Theta^{(2)}$ containing the weights for our second layer of nodes.

Each layer gets its own matrix of weights $\Theta^{(j)}$:
$$
\text{If network has $s_j$ units in layer $j$ and $s_{j+1}$ units in layer $j+1$, then $\Theta^{(j)}$ will be of dimension $s_{j+1} \times (s_j + 1)$.}
$$

> The +1 comes from the addition in $\Theta^{(j)}$ of the "bias nodes. In other words the output nodes will not include the bias nodes while the inputs will.

Example: If layer 1 has 2 input nodes and layer 2 has 4 activation nodes. Dimension of $\Theta^{(1)}$ is going to be 4×3 where $s_j = 2$ and $s_{j+1} = 4$ , so $s_{j+1} \times (s_j + 1) = 4 \times 3$

$z_k^{(j)}$ encompasses the parameters inside our g function and our activation nodes for layer $j$ are:  
$$
a^{(j)} = g(z^{(j)})
$$

We can then add a bias unit equal to 1. To compute our final hypothesis:
$$
z^{(j+1)} = \Theta^{(j)}a^{(j)}
$$

$$
h_\Theta(x) = a^{(j+1)} = g(z^{(j+1)})
$$


### Applications
#### Examples and Intuitions
Our functions will look like:
$$
\begin{align*}\begin{bmatrix}x_0 \newline x_1 \newline x_2\end{bmatrix} \rightarrow\begin{bmatrix}g(z^{(2)})\end{bmatrix} \rightarrow h_\Theta(x)\end{align*}
$$
> Where $x0$ is our bias variable and is always 1.

Theta matrix as:
$$
\Theta^{(1)} =\begin{bmatrix}-30 & 20 & 20\end{bmatrix}
$$

Hypothesis output:
$$
\begin{align*}& h_\Theta(x) = g(-30 + 20x_1 + 20x_2) \newline \newline & x_1 = 0 \ \ and \ \ x_2 = 0 \ \ then \ \ g(-30) \approx 0 \newline & x_1 = 0 \ \ and \ \ x_2 = 1 \ \ then \ \ g(-10) \approx 0 \newline & x_1 = 1 \ \ and \ \ x_2 = 0 \ \ then \ \ g(-10) \approx 0 \newline & x_1 = 1 \ \ and \ \ x_2 = 1 \ \ then \ \ g(10) \approx 1\end{align*}
$$
So we have constructed one of the fundamental operations in computers by using a small neural network rather than using an actual AND gate.

Matrices for AND, NOR, and OR are:
$$
\begin{align*}AND:\Theta^{(1)} &=\begin{bmatrix}-30 & 20 & 20\end{bmatrix} \newline NOR:\Theta^{(1)} &= \begin{bmatrix}10 & -20 & -20\end{bmatrix} \newline OR:\Theta^{(1)} &= \begin{bmatrix}-10 & 20 & 20\end{bmatrix} \newline\end{align*}
$$
We can combine these to get the XNOR logical operator:
$$
\begin{align*}\begin{bmatrix}x_0 \newline x_1 \newline x_2\end{bmatrix} \rightarrow\begin{bmatrix}a_1^{(2)} \newline a_2^{(2)} \end{bmatrix} \rightarrow\begin{bmatrix}a^{(3)}\end{bmatrix} \rightarrow h_\Theta(x)\end{align*}
$$
For the transition between the first and second layer, that combines the values for AND and NOR:
$$
\Theta^{(1)} =\begin{bmatrix}-30 & 20 & 20 \newline 10 & -20 & -20\end{bmatrix}
$$
For the transition between the second and third layer, that uses the value for OR:
$$
\Theta^{(2)} =\begin{bmatrix}-10 & 20 & 20\end{bmatrix}
$$
Write out the values for all our nodes:
$$
\begin{align*}& a^{(2)} = g(\Theta^{(1)} \cdot x) \newline& a^{(3)} = g(\Theta^{(2)} \cdot a^{(2)}) \newline& h_\Theta(x) = a^{(3)}\end{align*}
$$

#### Multiclass Classification
Our hypothesis function return a vector of values. Each $y(i)$ represents a different image corresponding to either a car, pedestrian, truck, or motorcycle.

Our resulting hypothesis for one set of inputs may look like:
$$
h_\Theta(x) =\begin{bmatrix}0 \newline 0 \newline 1 \newline 0 \newline\end{bmatrix}
$$
In which case our resulting class is the third one down, or $h_\Theta(x)_3$, which represents the motorcycle.
