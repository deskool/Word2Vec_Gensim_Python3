# WORD2VEC USING PYTHON 3
# Author: Mohammad M. Ghassemi and Tuka Alhanai
# Massachusetts Institute of Technology

###############################################################
# IMPORT EXTERNAL LIBRARIES
###############################################################
import gensim
from gensim.models import word2vec
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

###############################################################
# CHOOSE EXTERNAL .TXT FILES
###############################################################
with open ("Data/allnotes.txt", "r") as myfile:
    raw_text=myfile.read()

#additional stopwords
with open ("Other/my_stopwords.txt", "r") as myfile:
    more_stopwords=myfile.read()
my_stopwords = stopwords.words('english')
my_stopwords.extend(more_stopwords.split(','))

###############################################################
# SET Gensim Word2Vec Parameters
###############################################################

# min_count = ignore all words with total frequency lower than this.
words_min_count = 10

# number of features
features_size = 100

# window size
# window is the maximum distance between the current and predicted word within a sentence.
words_distance_window = 4

#  if 1 (default), sort the vocabulary by descending frequency before assigning word indexes.
sorted_vocab = 1

# sg defines the training algorithm. By default (sg=0), CBOW is used. Otherwise (sg=1), skip-gram is employed.
train_algorithm = 1

#negative = if > 0, negative sampling will be used, the int for negative specifies how many “noise words” should be drawn (usually between 5-20). 
# Default is 5. If set to 0, no negative samping is used.
negative_sampling = 10

#  limit RAM during vocabulary building; if there are more unique words than this, then prune the infrequent ones. 
# Every 10 million word types need about 1GB of RAM. Set to None for no limit (default).
max_vocab_memory_size = None

# for other parameters
# @see https://radimrehurek.com/gensim/models/word2vec.html


###############################################################
# PRE-PROCESS THE DATA
###############################################################
#REMOVE PUNCTUATION
raw_text = raw_text.replace('\n', ' ')
raw_text = raw_text.replace('  ',' ')
raw_text = raw_text.replace('"','')
raw_text = raw_text.replace('*','')
raw_text = raw_text.replace(',','')
raw_text = raw_text.replace(':','')
raw_text = raw_text.replace(';','')
raw_text = raw_text.replace('0','')
raw_text = raw_text.replace('1','')
raw_text = raw_text.replace('2','')
raw_text = raw_text.replace('3','')
raw_text = raw_text.replace('4','')
raw_text = raw_text.replace('5','')
raw_text = raw_text.replace('6','')
raw_text = raw_text.replace('7','')
raw_text = raw_text.replace('8','')
raw_text = raw_text.replace('9','')
raw_text = raw_text.replace('?','.')
raw_text = raw_text.replace('!','.')
raw_text = raw_text.replace('(','')
raw_text = raw_text.replace(')','')
raw_text = raw_text.lower()

#REMOVE STOPWORDS
pattern = re.compile(r'\b(' + r'|'.join(my_stopwords) + r')\b\s*')
raw_text = pattern.sub('', raw_text)

#SPLIT THE DATA INTO SENTENCES
raw_text_sentences = raw_text.split('.')
sentences = [gensim.utils.to_unicode(word).split() for word in raw_text_sentences]

###############################################################
# TRAIN THE MODEL
###############################################################
model = word2vec.Word2Vec(sentences, 
    sg = train_algorithm, 
    min_count=words_min_count, 
    size=features_size, 
    window=words_distance_window, 
    sorted_vocab=sorted_vocab,
    negative=negative_sampling,
    max_vocab_size=max_vocab_memory_size)

#SAVE THE MDODEL
model.save('Model/word2vec_gensim_model')

###############################################################
# SAVE THE WORD VECTORS
###############################################################
with open( 'Outputs/allnotes_tensors.tsv', 'w+') as tensors:
    with open( 'Outputs/allnotes_metadata.tsv', 'w+') as metadata:
         for word in model.wv.index2word:
           encoded=word
           metadata.write(encoded + '\n')
           vector_row = '\t'.join(map(str, model[word]))
           tensors.write(vector_row + '\n')


