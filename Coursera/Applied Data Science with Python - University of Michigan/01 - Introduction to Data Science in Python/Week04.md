## Statistical Analysis in Python and Project
**Distribution**
Set of all possible random variables

```Python
x = np.random.binomial(20, .5, 10000)
print((x>=15).mean())
```

- The mean value depends upon the samples that we've taken, and converges to the expected value given a sufficiently large sample set.

- Variance is a measure of how badly values of samples are spread out from the mean.

- The standard deviation is a measure of how different each item, in our sample, is from the mean.

- *Kurtosis:* shape of the tale of the distribution
  - A negative value means the curve is slightly more flat than a normal distribution,
  - A positive value means the curve is slightly more peaky than a normal distribution.
- *Skew:* push the peak of the curve one way or the other.

**Hypothesis Testing**
*Statement to test*
- Alternative Hypothesis
  - Our idea
  - e.g. there is a difference between groups
- Null Hypothesis
  - The alternative of our idea
  - e.g. there is no difference between groups

If we find that there is a difference between groups, then we can reject the null hypothesis and we accept our alternative.

We aren't saying that our hypothesis is true per se, but we're saying that there's evidence against the null hypothesis.

*Critical value $\alpha$*:
- The threshold as to how much chance your are willing to accept
- typical values in social sciences are 0.1, 0.05, 0.01

> A T test is one way to compare the means of two different population.

**P-hacking or Dredging**
- Doing many tests until you find one which is of statistical significance
- At a confidence level of 0.05, we expect to find one positive result 1 over 20 tests
- Remedies
  - Bonferroni correction
  - Hold-out sets
  - Investigation pre-registration
  
