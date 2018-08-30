## Charting Fundamentals
**Subplots**
Allows you to create axis to different portions of this grid.

Arguments of `ax1 = plt.subplot(1, 2, 2)`:
1. number of rows
1. number of columns
1. plot number
    - number from left to right and top to bottom

```Python
# pass sharey=ax1 to ensure the two subplots share the same y axis
ax2 = plt.subplot(1, 2, 2, sharey=ax1)
# create a 3x3 grid of subplots with shared axis
fig, ((ax1,ax2,ax3), (ax4,ax5,ax6), (ax7,ax8,ax9)) = plt.subplots(3, 3, sharex=True, sharey=True)
# plot the linear_data on the 5th subplot axes
ax5.plot(linear_data, '-')
```

**Histograms**
*Probability density function*

```Python
# 10 bins by default
ax.hist(sample)
```

**GridSpec**
Allows to map axes over multiple cells in a grid

```Python
# use gridspec to partition the figure into subplots
import matplotlib.gridspec as gridspec
# over all shape, indexed as rows and columns
gspec = gridspec.GridSpec(3, 3)

# pass the elements of the GridSpec object which we wish to cover, it's a list so index starts at zero, get axis
top_histogram = plt.subplot(gspec[0, 1:])
side_histogram = plt.subplot(gspec[1:, 0])
lower_right = plt.subplot(gspec[1:, 1:])
```

> Matplotlib requires that we share axes when creating plots, there is no post hoc sharing

**Box/Whisker Plots**
Shows, for each sample, the median of each value, the minimum and maximum of the samples, and the interquartile range (variation).

```Python
# assign the output to a variable to supress output. Default 'whis': 1.5*interquartile (IQR) whiskers with outliers
_ = plt.boxplot([ df['normal'], df['random'], df['gamma'] ], whis='range')
```

**Inset**
Overlay an axes on top of another within a figure.
```Python
import mpl_toolkits.axes_grid1.inset_locator as mpl_il
# overlay axis on top of another
ax2 = mpl_il.inset_axes(plt.gca(), width='60%', height='40%', loc=2)
ax2.hist(df['gamma'], bins=100)
ax2.margins(x=0.5)
```

**Heatmaps**
A two-dimensional histogram where the x and the y values indicate potential points and the color plotted is the frequency of the observation.

Heatmaps break down is when there's no continuous relationship between dimensions, e.g. categorical data; misleads the viewer into looking for patterns and ordering through spatial proximity and any such patterns would be purely spurious.

```Python
_ = plt.hist2d(X, Y, bins=25)
```

**Animation**
*Re-drawn*

```Python
# create the function that will do the plotting, where curr is the current frame
def update(curr):
    # if last frame,stop the animation 'a'
    if curr == n:
        a.event_source.stop()
    plt.cla()
    ...
    # annotate at a certain position in the chart
    plt.annotate('n = {}'.format(curr), [3,27])

# interval in msec
fig = plt.figure()
a = animation.FuncAnimation(fig, update, interval=100)
```

**Interactivity**
Events
```Python
#
def onclick(event):
    plt.cla()
    plt.plot(data)
    plt.gca().set_title('Event at pixels {},{} \nand data {},{}'.format(event.x, event.y, event.xdata, event.ydata))

# Event listener (wiring it up), pass a 'button_press_event' into onclick when the event is detected
plt.gcf().canvas.mpl_connect('button_press_event', onclick)
```
Pick
Respond when the user is actually clicked on a visual element in the figure.
```Python
plt.figure()
# picker=5 means the mouse doesn't have to click directly on an event, but can be up to 5 pixels away
plt.scatter(df['height'], df['weight'], picker=5)

def onpick(event):
    origin = df.iloc[event.ind[0]]['origin']
    plt.gca().set_title('Selected item came from {}'.format(origin))

# tell mpl_connect we want to pass a 'pick_event' into onpick when the event is detected
plt.gcf().canvas.mpl_connect('pick_event', onpick)
```
