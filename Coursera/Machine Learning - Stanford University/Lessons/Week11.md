## Application Example: Photo OCR
#### Problem Description and Pipeline
*Photo OCR:* Photo Optical Character Recognition, read text from images.
*Pipeline:* A system with many stages / components, several of which may use machine learning.

In many complex machine learning systems, these sorts of pipelines are common, where you can have multiple modules--in this example, the text detection, character segmentation, character recognition modules--each of which may be machine learning component, or sometimes it may not be a machine learning component but to have a set of modules that act one after another on some piece of data in order to produce the output you want.

> Break the problem down into a sequence of different modules.

#### Sliding Windows
Define a step-size/stride and slide this window over the different locations in the image (scanning) and run these patches through the classifier and identify positive/negative examples.

Example:
The classifier on a 200x200 image using 20x20 patches and step of 4 pixels each time, it will take around 2500 steps on a single image.

#### Getting Lots of Data and Artificial Data
Artificial data synthesis:
- Creating new data from scratch
- Use a small training set to turn that into a larger training set (amplify).

> Remember to have a low bias classifier (plot the learning curves)

#### Ceiling Analysis: What Part of the Pipeline to Work on Next
How much could you possibly gain if one of the modules became absolutely perfect?. The upper bound on the performance of that system.

When we plug in the ground-truth labels for one of the components, the performance of the overall system improves very little:
- We should not dedicate significant effort to collecting more data for that component.

> Improving which component will make a big difference?.

### Conclusion
#### Summary
**Supervised Learning:**
- Linear regression
- Logistic regression
- Neural networks
- SVMs

**Unsupervised Learning:**
- K-means
- PCA
- Anomaly detection

**Special Applications/Topics:**
- Recommender systems
- Large scale machine learning

**Advice on building a ML system:**
- Bias/Variance
- Regularization
- Debugging
    - Evaluation of learning algorithms
    - Learning curves
    - Error analysis
    - Ceiling analysis

Use machine learning to build cool systems and cool applications and cool products. And I hope that you find ways to use machine learning not only to make your life better but maybe someday to use it to make many other people's life better as well.
