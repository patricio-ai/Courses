## Introduction
### Welcome
#### Welcome to Machine Learning!

### Introduction
> Machine learning is the science of getting computers to learn, without being explicitly programmed.

Examples:
* Database mining: Large datasets from growth of automation
* Applications can't program by hand: Computer Vision, NLP, recognition
* Self-customizing programs: Product recommendations

#### Welcome
#### What is Machine Learning?
> Tom Mitchell: A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E.
* Example: playing checkers:  
  E = the experience of playing many games of checkers  
  T = the task of playing checkers.  
  P = the probability that the program will win the next game.

###### Algorithms
* Supervised learning: Teach the computer with the right answers
* Unsupervised learning: Learn by itself
* Others: Reinforcement learning and recommender systems

#### Supervised Learning
We are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output.

Supervised learning problems are categorized into:
* Regression: predict results within a continuous output (quantity)
* Classification: predict results in a discrete value output (categories)

#### Unsupervised Learning
Problems where we are not told what the desired output is. We can derive structure from data where we don't necessarily know the effect of the variables, there is no feedback based on the prediction results.

Unsupervised learning problems are categorized into:
* Clustering Algorithms: Group and organize data that are somehow similar or related by different variables
* Non-clustering: The "Cocktail Party Algorithm", find structure in a chaotic environment


## Linear Regression with One Variable
### Model and Cost Function
#### Model Representation
* Notation:  
  m = Number of training examples
  x = input variable, features
  y = output variable, target
  (x(i), y(i)); i=1,...,m list of training examples

* Workflow  
  Training set: Data to learn from  
  -> Feed to Learning Algorithm, output Hypothesis (h) function  
  -> h, maps from x's to y's (estimated)  

* Linear regression with one variable (Univariate)  
  ho(x) = o1*x + o0
  Choose o0, 01, so ho(x) is close to y for our training samples (x, y)

#### Cost Function
We can measure the accuracy of our hypothesis function by using a cost function  
For linear regression problems: J(o0, 01) = Squared error function
The mean is halved as a convenience for the computation of the gradient descent, as the derivative term of the square function will cancel out the 1/2 term

* The best possible line will be such so that the average squared vertical distances of the scattered points from the line will be the least

* Ideally, the line should pass through all the points of our training data set. In such a case, the value of J(θ0,θ1) will be 0


### Parameter Learning
#### Gradient Descent
So we have our hypothesis function and we have a way of measuring how well it fits into the data. Now we need to estimate the parameters in the hypothesis function. That's where gradient descent comes in.  
The slope of the tangent is the derivative, tangential line to a function of our cost function, at that point and it will give us a direction to move towards. We make steps down the cost function in the direction with the steepest descent. The size of each step is determined by the parameter $\alpha$, which is called the learning rate.  
Depending on where one starts on the graph, one could end up at different points.

* Start with some o0 = 0, 01 = 0, for the cost function as a function of the parameter estimates, J(o0, o1)
* Repeat changing o0, o1 until convergence to minimize J(o0, o1) (local optimum)
* The direction in which the step is taken is determined by the partial derivative of J(o0, o1)
* As we approach a local minimum, gradient descent will automatically take smaller steps, so we can work with Alpha fixed, the gradient descent can converge to a local minimum
* Alpha: learning rate (step size), if too small, gradient can be slow, if too large it may fail to converge
* Implement simultaneous update

> Also called, Batch Gradient Descent.  
  Batch: Each step of gradient descent uses all the training examples

> Recall our definition of the cost function was J(θ0,θ1)=12m∑mi=1(hθ(x(i))−y(i))2

#### Gradient Descent For Linear Regression
Recall that in linear regression, our hypothesis is hθ(x)=θ0+θ1x, and we use m to denote the number of training examples.

* Always a bowl-shaped, Convex function only has global optimum


## Linear Algebra Review
### Linear Algebra Review
#### Matrices and Vectors
* Dimension of matrix: n° Rows x n° Columns
* Aij refers to the element in the ith row and jth column of matrix A
* A vector with 'n' rows is referred to as an 'n'-dimensional vector.
* vi refers to the element in the ith row of the vector.
* In general, all our vectors and matrices will be 1-indexed. Note that for some programming languages, the arrays are 0-indexed.
* Rn refers to the set of n-dimensional vectors of real numbers.

#### Addition and Scalar Multiplication
* Addition and subtraction are element-wise, so you simply add or subtract each corresponding element
* To add or subtract two matrices, their dimensions must be the same
* Result in same dimension

#### Matrix Scalar Addition/Multiplication
* add/multiply every element by the scalar value
* Result in same dimension

#### Matrix Multiplication Properties
* To multiply two matrices, the number of columns of the first matrix must equal the number of rows of the second matrix
* Matrices are not commutative: A∗B≠B∗A
* Matrices are associative: (A∗B)∗C=A∗(B∗C)

#### Inverse and Transpose
* The inverse of a matrix A is denoted A−1. Multiplying by the inverse results in the identity matrix
* A non square matrix does not have an inverse matrix
* Matrices that don't have an inverse are singular or degenerate
* Only if A is an m x m (squared matrix) could have an inverse
* A * (A^-1) = A^-1 * A = I
* Tshe rows of a matrix become the columns

> in octave with the pinv(A) function and in Matlab with the inv(A) function
> in matlab with the transpose(A) function or A'

###### Identity Matrix
* The identity matrix, when multiplied by any matrix of the same dimensions, results in the original matrix
* Denoted Inxn, simply has 1's on the diagonal (upper left to lower right diagonal) and 0's elsewhere
* A * I = I * A = A
