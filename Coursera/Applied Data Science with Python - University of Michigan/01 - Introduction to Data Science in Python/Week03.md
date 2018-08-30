## Advanced Python Pandas
### Merging Dataframes
How:
- outer: union
- inner: intersection
- left/right: keep

### Pandas Idioms
An idiomatic solution is often one which has both high performance and high readability.

- Chain indexing
  - ```df.loc['a']['b']```
  > Generally bad, pandas could return a copy of a view depending upon numpy

- Method chaining
  - Every method on an object returns a reference to that object

In applymap, you provide some function which should operate on each cell of a DataFrame, and the return set is itself a DataFrame.

*Apply* maps across all of the rows in a DataFrame.

### Group by
```Python
df = df.set_index('STNAME')

def fun(item):
    if item[0]<'M':
        return 0
    if item[0]<'Q':
        return 1
    return 2

for group, frame in df.groupby(fun):
    print('There are ' + str(len(frame)) + ' records in group ' + str(group) + ' for processing.')
```

**Split apply combine**
Split your data, apply some function, then you combine the results.

```Python
print(df.groupby('Category').apply(lambda df,a,b: sum(df[a] * df[b]), 'Weight (oz.)', 'Quantity'))
```

> A potential issue using the aggregate method of group by objects. You see, when you pass in a dictionary it can be used to either to identify the columns to apply a function on or to name an output column if there's multiple functions to be run. The difference depends on the keys that you pass in from the dictionary and how they're named.

### Scales
- Ratio
  - units are equally spaced
  - mathematical operations of +-\*/ are all valid
  - e.g. height vs weight
- Interval scale
  - units are equally spaced, but there is no true zero or absence of value
  - mathematical operations of +-\*/ are NOT valid
  - e.g. temperature
- Ordinal
  - The order of the units is important, buy not evenly spaced
  - e.g. grades such as: A+, A
- Nominal
  - categories of data with no order with respect to another
  - e.g teams of a sport


```Python
grades = df['Grades'].astype('category',
                         categories=['D', 'D+', 'C-', 'C', 'C+', 'B-', 'B', 'B+', 'A-', 'A', 'A+'],
                           ordered=True)

# Bin and add labels for the sizes [Small < Medium < Large].
s = pd.Series([168, 180, 174, 190, 170, 185, 179, 181, 175, 169, 182, 177, 180, 171])
pd.cut(s, 3, labels=['Small', 'Medium', 'Large'])
```

### Pivot Tables & Date Functionality
A pivot table is itself a data frame, where the rows represent one variable that you're interested in, the columns another, and the cells some aggregate value.

Dates as index, with Timestamp, Periods, DatetimeIndex, PeriodIndex, Timedeltas, etc.

### Goodhart's Law
When a measure becomes a target, it ceases to be a good measure. Behavior changes, manipulating a metric.
