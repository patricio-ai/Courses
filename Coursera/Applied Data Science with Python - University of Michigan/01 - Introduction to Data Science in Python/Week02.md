## Basic Data Processing with Pandas
### The Series Data Structure
Two columns of data
- The first is the special index
- The second is your actual data

You need to use special functions to test for the presence of not a number, such as the Numpy library is NAN.

### Querying a Series
- To query by numeric location, starting at zero, use the iloc attribute.

- To query by the index label, you can use the loc attribute

*Broadcasting:* You can apply an operation to every value in the series, changing the series.

> The append method doesn't actually change the underlying series. It instead returns a new series which is made up of the two appended together

### The DataFrame Data Structure
*You can think of the DataFrame itself as simply a two-axes labeled array.*

```Python
df.loc['Store 1']['Cost']
```
Chaining tends to cause Pandas to return a copy of the DataFrame instead of a view on the DataFrame.

use: ```df.loc[:,['Name', 'Cost']]```

### DataFrame Indexing and Loading
```Python
df = pd.read_csv('olympics.csv', index_col = 0, skiprows=1)
```
### Querying a DataFrame
A Boolean mask is an array which can be of one dimension like a series, or two dimensions like a data frame, where each of the values in the array are either true or false.

The where function takes a Boolean mask as a condition, applies it to the data frame or series, and returns a new data frame or series of the same shape.

> The output of two Boolean masks being compared with logical operators is another Boolean mask.

### Indexing Dataframes
We can get rid of the index completely by calling the function reset_index. This promotes the index into a column and creates a default numbered index.

When you use a MultiIndex, you must provide the arguments in order by the level you wish to query.

Inside of the index, each column is called a level and the outermost column is level zero. 

```Python
df = df.set_index([df.index, 'Name'])
df.index.names = ['Location', 'Name']
df = df.append(pd.Series(data={'Cost': 3.00, 'Item Purchased': 'Kitty Food'}, name=('Store 2', 'Kevyn')))
```

### Missing Values
When you use statistical functions on DataFrames, these functions typically ignore missing values
