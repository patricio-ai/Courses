## Basic Charting
**Matplotlib Architecture**
- Backend Layer
  - Deals with the rendering of plots to screen or files
  - In Jupyter notebooks we use the inline backend
- Artist Layer
  - Contains container such as Figure, Subplot and Axes
  - Contains primitives, such as a Line2D Rectangle, collections (PathCollection)
- Scripting Layer (pyplot)
  Simplifies access to the Artist and Backend layers

> Matplotlib's pyplot is an example of a procedural (command after command) method for building visualizations while SVG, HTML, are declarative methods of creating visualizations

**Basic Plotting with Matplotlib**
```Python
import matplotlib.pyplot as plt

# interactive
%matplotlib notebook
```

**Plots**
Matplotlib calls to the scripting interface to create figures, subplots, and axis.

Then load those axis up with various artists, which the back-end renders to the screen or some other medium like a file.

The scripting layer is really a set of convenience functions on top of the object layer.

```Python
import matplotlib.pyplot as plt
import numpy as np

plt.figure()

languages =['Python', 'SQL', 'Java', 'C++', 'JavaScript']
pos = np.arange(len(languages))
popularity = [56, 39, 34, 34, 29]

# change the bar colors to be less bright blue
bars = plt.bar(pos, popularity, align='center', linewidth=0, color='lightslategrey')
# make one bar, the python bar, a contrasting color
bars[0].set_color('#1F77B4')

# soften all labels by turning grey
plt.xticks(pos, languages alpha=0.8)
plt.ylabel('% Popularity', alpha=0.8)
plt.title('Top 5 Languages for Math & Data \nby % popularity on Stack Overflow', alpha=0.8)

# remove all the ticks (both axes), and tick labels on the Y axis
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

# remove the frame of the chart
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.show()
```
