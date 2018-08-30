## Working with text in python
- Data continues to grow exponentially
- Approximately 80% of all data is estimated to be unstructured, text-rich data

What can be done with text:
- Find/Identify/Extract relevant information from text
- Classify text documents
- Search for relevant text documents
- Sentiment analysis (positive/negative)
- Topic modeling

### Handling text in python
Primitive constructs in text:
- Sentences/Input strings
- Words (tokens)
- Characters
- Document, larger files

Concepts:
- Handling text sentences
- Splitting sentences into words and words into characters
- Finding unique words
- Handling text from documents

> When you split on a particular character, that character is not included in the result.

### Regular Expressions
```python
[w for w in text if w.startswith('#')]

import re
# one or more times that's between square brackets
[w for w in text if re.search('@\w+', w)]
```

**Meta-character**
\. wildcard, matches a single character
\^ start of a string
\$ end of a string
\[\] matches one of the set of characters within it
[a-z] matches one of the range of characters
[^abc] matches a character that is not a, b or c
a | b matches either a or b
() scoping for operators
\\ Escape for special characters
\b Matches word boundary
\d Any digit, equivalent to [0-9]
\D Any non-digit, equivalent to [^0-9]
\s Any whitespace, equivalent to [ \t\n\r\f\v]
\S Any non-whitespace, equivalent to [^ \t\n\r\f\v]
\w Alphanumeric character, equivalent to [a-zA-Z0-9_]
\W Non-alphanumeric character, equivalent to [a-zA-Z0-9_]
\* matches zero or more occurrences
\+ matches one or more occurrences
\? matches zero or one occurrences
{n} exactly n repetitions, $n \geq 0$
{n, } at Least n repetitions
{, n} at most n repetitions
{m, n} at least m and at most n repetitions

### Internationalization and Issues with Non-ASCII Characters
- Diacritics
- International languages

**Unicode**
- Industry stander for encoding and representing text
- Can be implemented by different character encodings (i.e. UTF-8)

**UTF-8**
- Unicode Transformational Format 8-bits
- Variable length encoding, one to four bytes
- Backward compatible with ASCII
- Dominant character encoding for the web
