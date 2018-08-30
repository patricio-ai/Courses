## Applied Visualizations
### Pandas Visualizations
```Python
df.plot()

# returns a matplotlib.axes._subplot
ax = df.plot.scatter('A', 'C', c='B', s=df['B'], colormap='viridis')
ax.set_aspect('equal')

# compare between columns
pd.tools.plotting.scatter_matrix(...);

# see any patterns or clustering
pd.tools.plotting.parallel_coordinates(..., 'Name');
```

### Seaborn
*Seaborn is really just a wraparound matplotlib. It adds styles to make default data visualizations much more visually appealing and makes creation of specific types of complicated plots much simpler.*

Some of the plotting functions in Seaborn return a matplotlib axis object.

Others operate on an entire figure and produce plots with several panels, returning a Seaborn grid object.

```Python
import seaborn as sns

# plot a kernel density estimation over a stacked barchart
plt.figure()
plt.hist([v1, v2], histtype='barstacked', normed=True);
v3 = np.concatenate((v1,v2))
sns.kdeplot(v3);

# in seaborn
# we can pass keyword arguments for each individual component of the plot
sns.distplot(v3, hist_kws={'color': 'Teal'}, kde_kws={'color': 'Navy'});

# visualize the distribution of the two variables individually. As well as relationships between the variables.
sns.jointplot(v1, v2, alpha=0.4);

# Hexbin plots are the bivariate counterpart to histograms. Hexbin plots show the number of observations that fall within hexagonal bins.
sns.jointplot(v1, v2, kind='hex');

# You can think of two dimensional KDE plots as the continuous version of the hexbin jointplot.
sns.set_style('white')
sns.jointplot(v1, v2, kind='kde', space=0);

# scatter_matrix
sns.pairplot(iris, hue='Name', diag_kind='kde', size=2);

# box-plot, The violinplot is like box plot with a rotated kernel density estimation on each side.
plt.figure(figsize=(8,6))
plt.subplot(121)
sns.swarmplot('Name', 'PetalLength', data=iris);
plt.subplot(122)
sns.violinplot('Name', 'PetalLength', data=iris);
```
