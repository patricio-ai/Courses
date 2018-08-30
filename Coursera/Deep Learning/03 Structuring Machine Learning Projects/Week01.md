## ML Strategy
### Introduction to ML Strategy
I hope that these guidelines help you set up your whole team to have a well-defined target that you can iterate efficiently towards improving performance.

#### Why ML Strategy
Too much things to try:
- Collect more data
- Collect more diverse training set
- Train algorithm longer with gradient descent
- Try Adam
- Try bigger/smaller network
- Try dropout
- Add $L_2$ regularization
- Network architecture
  - Activation functions
  - \# hidden units...
- etc..

#### Orthogonalization
- Fit training/dev/test set well on cost function
- Performs well in real cost

Diagnose what exactly is the bottleneck to your system's performance. As well as identify the specific set of knobs you could use to tune your system to improve that aspect of its performance.

### Setting up your goal
#### Single number evaluation metric
*Evaluation metric.*

- Precision  
  Of the examples that your classifier recognizes as *cats*, What percentage actually are *cats*?
- Recall
  what percentage of actual *cats*, Are correctly recognized?

> Combine precision and recall *F1 score*, to get a single number.

#### Satisficing and Optimizing metric
- **Optimizing** metric:  
  Want to do as well as possible

- **Satisficing** metric:
    After they reach some threshold, you don't care how much better it is

#### Train/dev/test distributions
*dev set*: is sometimes called the hold out cross validation set.

In machine learning you try a lot of ideas, train up different models on the *training set*, and then use the *dev set* to evaluate the different ideas and pick one. And, keep innovating to improve dev set performance until, finally, you have one that you're happy with that you then evaluate on your *test set*.

> Make your dev and test sets come from the same distribution. Shuffle the data from the dev and test set.

#### Size of the dev and test sets
Set your test set to be big enough to give high confidence in the overall performance of your system.

I think the old rule of thumb of a 70/30 is that, that no longer applies. And the trend has been to use more data for training and less for dev and test, especially when you have a very large data sets.

> If you have a very large dev set so that you think you won't overfit. Maybe it's not totally unreasonable to just have a train dev set.

#### When to change dev/test sets and metrics
The goal of the evaluation metric is accurately tell you, given two classifiers, which one is better for your application.

So don't keep coasting with an error metric you're unsatisfied with, instead try to define a new one that you think better captures your preferences in terms of what's actually a better algorithm.

**Orthogonalization**
1. **Define one metric** to evaluate classifiers, (placing the target).
1. Worry separately about **how to do well** on this metric (aiming and shooting).

If doing well on your metric and your current dev sets, if that does not correspond to doing well on the application you actually care about, then change your metric and your dev test set, so that your data better reflects the type of data you actually need to do well on.

### Comparing to human-level performance
#### Why human-level performance?
*Bayes optimal error:* The very best theoretical function for mapping from $x$ to $y$ that can never be surpassed.

As long as ML is worse than humans, you can:
- Get labeled data from humans
- Gain insight from manual error analysis
- Better analysis of bias/variance

#### Avoidable bias
|   Human error  |       1%      |        7.5%       |
|:--------------:|:-------------:|:-----------------:|
| Training error |       8%      |         8%        |
|    Dev error   |      10%      |        10%        |
|                | Focus on bias | Focus on variance |

**Avoidable bias**
- Level of error that you just cannot get below.
- Difference between Human and Training error

**Variance**
- Difference between Training and Dev error.

> Human level error as an estimate for Bayes error.

#### Understanding human-level performance
- You can use human-level error as a proxy or as a approximation for Bayes error.
- The difference between your estimate of Bayes error tells you how much avoidable bias is a problem, how much avoidable bias there is.
- The difference between training error and dev error, that tells you how much variance is a problem, whether your algorithm's able to generalize from the training set to the dev set.

For problems where the data is noisy, having a better estimate for Bayes error can help you better estimate avoidable bias and variance. And therefore make better decisions on whether to focus on bias reduction tactics, or on variance reduction tactics.

These techniques will tend to work well until you surpass human-level performance, whereupon you might no longer have a good estimate of Bayes error that still helps you make this decision really clearly.

#### Surpassing human-level performance
**Examples:**
- Online advertising
- Product recommendations
- Logistics (predicting transit time)
- Loan approvals

**Based on:**
- Structured data
- Not natural perception
- Huge amount of data

> I hope that maybe someday you manage to get your deep learning system to also surpass human-level performance.

#### Improving your model performance
The two fundamental assumptions fo supervised learning
1. You can fit the training set pretty well, *low avoidable bias*.
1. The training set performance generalizes pretty well to the dev/test set, *low variance*.

**Human-level vs Training error (Avoidable Bias):**
- Train bigger model
- Train longer/better optimization algorithms
  - Momemtum
  - RMSprop
  - Adam
- NN architecture/hyperparameters search

**Training error vs Dev error (Variance):**
- More data
  - Data augmentation
- Regularization
  - $L_2$
  - Dropout
- NN architecture/hyperparameters search

### Machine Learning flight simulator
There are a lot of decisions to make:
- What is the evaluation metric?
- How do you structure your data into train/dev/test sets?

### Heroes of Deep Learning - Andrej Karpathy
> I think I felt very strongly that basically, this technology was transformative in that a lot of people want to use it. It's almost like a hammer. And what I wanted to do, I was in a position to randomly hand out this hammer to a lot of people. And I just found that very compelling.

> And so that's something that I keep advising people is that you not work with flow or something else. You can work with it once you have written at something yourself on the lowest detail, you understand everything under you, and now you are comfortable to.
