## Topic Modeling
Finding similarity between words and text is non-trivial

**Semantic similarity**
- Grouping similar words into semantinc concepts
- As a building block in natural language understading task
  - Textual entailment
  - Paraphrasing

**WordNet**
- Semantic dictionary of words interlinked by semantic relations
- Includes rich linguistic information
  - part of speech, word senses, synonyms, ...
- Organized information in a hierarchy

**Path Similarity**
- Find the shortest path between the two concepts
- Similarity measure inversely related to path distance

**Lowest common subsumer (LCS)**
- Find the closest ancestor to both concepts

**Lin Similarity**
- Similarity measure based on the information contained in the LCS of the two concepts
$$
\text{LinSim}(u, v) = 2 \frac{\log P(\text{LCS}(u,v))}{\log{P(u)} + \log P(v)}
$$
$P(u)$: is given by the information content learnt over a large corpus

```Python
import nltk
from nltk.corpus import wordnet as wn

# find appropriate sense of the words
deer = wn.synset('deer.n.01')
elk = wn.synset('elk.n.01')

# find path similarity
deer.path_similarity(elk)
deer.path_similarity(horse)

# Lin similarity
from nltk.corpus import wordnet_ic
brown_ic = wordnet_ic.ic('ic-brown.dat')

deer.lin_similarity(elk, brown_ic)
deer.lin_similarity(hose, brown_ic)
```

**Collocations and Distributional similarity**
- Two words that frequently appears in similar contexts are more likely to be semantically related
- Words before, after, within a small window
- Parts of speech of words before, after, in a small window
- Specific syntactic relation to the target word
- Words in the same sentence, same document
- How frequent are these, similar if occur together often
- Also important to see how frequent are individual words
  - Pointwise Mutual Information (PMI)
  $$
  \text{PMI}(w, c) = \log\bigg(\frac{P(w, c)}{P(w)P(c)}\bigg)
  $$

```Python
import nltk
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(text)
# get top 10
finder.nbest(bigram_measures.pmi, 10)
```

### Topic Modeling
*Documents as a mixture of topics*
- A coarse-level analysis of what's in a text collection
- Topic: the subject/theme of a discourse
- Topics are represented as a word distribution

What's known:
  - The text collection or corpus
  - Number of topics

What's not known:
  - The actual topics
  - Topic distribution for each document

Approaches
- Probabilistic Latent Semantic Analysis (PLSA)
- Latent Dirichlet Allocation (LDA)

> Essentially, this is a text clustering problem with documents and words clustered simultaneously

### Generative models and LDA
In the generation process, you have a model that gives out words, and then you use those words to generate the document.

When you are using four documents to infer your models, you need to infer four models. And you need to somehow infer what was the combination of words coming from these four topics.

**Latent Dirichlet Allocation**
- Generative model for a document d
  - Choose length of document d
  - Choose a mixture of topics for document

In practice:
- How many topics?
- Interpreting topics
  - Topics are just word distributions
- Pre-processing text
  - Tokenize, normalize (lowercase)
  - Stop word removal (depends on context)
  - Stemming (same root)
  - Convert tokenized documents to a document - term matrix
  - Build LDA models on the doc-term matrix

```Python
import gensim
form gensim import corpora, models

dictionary = corpora.Dictionary(doc_set)
corpus = [dictionary.doc2bow(doc) for doc in doc_set]
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=4, id2word=dictionary, passes=50)
print(ldamodel.prin_topics(num_topics=4, num_words=5))
```

### Information Extraction
*Identify and extract fields of interest from free text, unstructured to structured form?*

**Field of interest**
- Named entities
  - News: People, Places, Dates
  - Finance: Money, Companies
  - Medicine, Diseases, Drugs, Procedures
- Relations
  - What happened to how, when, where

**Named Entity Recognition**
- Named entities:
  Noun phrases that are of specific type and refer to specific individuals, places, organizations
- Named Entity Recognition:
  Techniques to identify all mentions of pre-defined named entities in text
  - Identify the mention/phrase: Boundary detection
  - Identify the type: Tagging / classification

**Relation extraction**
- Identify relationships between named entities

**Co-reference resolution**
- Disambiguate mentions and group mentions together

**Question Answering**
- Given a question, find the most appropriate answer from the text
- Builds on named entity recognition, relation extraction, and co-reference resolution
