# Language Vectorization



## Pre-requisites

This work assumes you are using 

1. Python3, 
2. A unix based system and 
3. Are familiar with pip package management.

## How to use

Below we include a simple example for how to use the software:

```python
from language_vectorizer import langvec

# Import some text data
with open ('Data/bible.txt', "r") as myfile:
    text = myfile.read()

# Initialize the mode with the data. 
lv = langvec(text = text)

# You can also initilize the model with clean sentences
# raw_text_sentences = text.split('.')
# s = [gensim.utils.to_unicode(word).split() for word in raw_text_sentences]
#lv = langvec(sentences = s)

# Pre-processes the raw text data to create clean sentences
lv.preprocess_text()

# Update any word2vec parameters you might care about.
# See __init__ for all the feature defaults.
lv.word2vec_params['features_size'] = 10

# Trains word to vec using internal sentences
lv.trian_word2vec_model(save_model_as = 'bible', save_tensors_as = 'bible' )
```

## Pretrained doc2vec model at the sentence level available here

Based on the work of Jey Han Lau and Timothy Baldwin (2016). An Empirical Evaluation of doc2vec with Practical Insights into Document Embedding Generation. In Proceedings of the 1st Workshop on Representation Learning for NLP, 2016.
https://github.com/jhlau/doc2vec

```
wget https://ibm.box.com/s/3f160t4xpuya9an935k84ig465gvymm2
```


### OUTPUTS: 

The software outputs a model file that ends in `.w2vmodel`, and (optionally) two tensor files ending with `_tensors.tsv` and `_metadata.tsv`. The tensor files can be imported into http://projector.tensorflow.org/ for visualization purposes.


