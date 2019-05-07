####################################################################
# IMPORT EXTERNAL LIBRARIES
####################################################################
import gensim
from gensim.models import word2vec, doc2vec
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
import string
import regex as re
from nltk.tokenize import word_tokenize
import numpy as np
from gensim.test.utils import common_texts

#--------------------------------------------------------------------
# MULTIPROCESSING TO SPEED UP TRAINING.
#--------------------------------------------------------------------
import multiprocessing
CORES = multiprocessing.cpu_count()

#--------------------------------------------------------------------
# PARAMETERS TO CLEAN UP THE TEXT
#--------------------------------------------------------------------
STOPLIST    = set(stopwords.words('english') + ["n't", "'s", "'m", "ca"] + list(ENGLISH_STOP_WORDS))
re.DEFAULT_VERSION = re.VERSION1
split_chars = lambda char: list(char.strip().split(' '))
WHITESPACE  = ["", " ", "  ", "   ","    ", "\t", "\n", "\r\n", "\n\n"]
_quotes     = r'\' \'\' " ” “ `` ` ‘ ´ ‘‘ ’’ ‚ , „ » « 「 」 『 』 （ ） 〔 〕 【 】 《 》 〈 〉'
QUOTES      = split_chars(_quotes)
_hyphens    = '- – — -- --- —— ~'
HYPHENS     = split_chars(_hyphens)

#---------------------------------------------------------------
# A TOOL THAT VECTORIZES TEXT DOCUMENTS AND CLAUSES
#---------------------------------------------------------------
class langvec:
    def __init__(self, name, text = None, sentences = None):
        # BASIC TEXT INFORMATION
        self.name      = name
        self.text      = text                                   # 
        self.sentences = sentences                              # cleaned sentences in list of list formats
        
        # WORD2VEC RELATED
        # for other parameters see: https://radimrehurek.com/gensim/models/word2vec.html
        self.word2vec_model  = None;
        self.word2vec_params = { 'words_min_count'      : 10,   # min_count = ignore all words with total frequency lower than this.
                                 'features_size'        : 100,  # number of features
                                 'words_distance_window': 4,    # window is the maximum distance between the current and predicted word within a sentence.
                                 'sorted_vocab'         : 1,    # if 1 (default), sort the vocabulary by descending frequency before assigning word indexes.
                                 'train_algorithm'      : 1,    # defines the training algorithm. By default (sg=0), CBOW is used. Otherwise (sg=1), skip-gram is employed.
                                 'negative_sampling'    : 10,   # if > 0, negative sampling will be used, the int for negative specifies how many “noise words” should be drawn (usually between 5-20). # Default is 5. If set to 0, no negative samping is used.
                                 'max_vocab_size'       : None  # limit RAM during vocabulary building; if there are more unique words than this, then prune the infrequent ones. # Every 10 million word types need about 1GB of RAM. Set to None for no limit (default).
                               }

        self.doc2vec_model  = None;
        self.doc2vec_params = { 'dm'          : 1,    # ({1,0}, optional) – Defines the training algorithm. If dm=1, ‘distributed memory’ (PV-DM) is used. Otherwise, distributed bag of words (PV-DBOW) is employed.
                                'dm_mean'     : 1,    # ({1,0}, optional) – If 0 , use the sum of the context word vectors. If 1, use the mean. Only applies when dm is used in non-concatenative mode.
                                'vector_size' : 200,  # (int, optional) – Dimensionality of the feature vectors
                                'window'      : 8,    # (int, optional) – The maximum distance between the current and predicted word within a sentence.
                                'min_count'   : 19,   # (int, optional) – Ignores all words with total frequency lower than this.
                                'epochs'      : 10,   # (int, optional) – Number of iterations (epochs) over the corpus.
                                'workers'     : CORES # (int, optional) – Use these many worker threads to train the model (=faster training with multicore machines).
                              }
        # INITIALIZE THE WORD2VEC MODEL

    #---------------------------------------------------------------
    # PRE-PROCESS THE DATA
    #---------------------------------------------------------------
    def preprocess_text(self):

        # CLEAN THE TEXT
        clean_text = self.text
        clean_text = text.lower()


        #REPLACE THINGS
        clean_text = clean_text.replace('?','.')
        clean_text = clean_text.replace('!','.')
        clean_text = clean_text.replace(';','.')
        for i in range(10):
            clean_text = clean_text.replace('..','.')

        clean_text = clean_text.replace('. .','.')


        #REMOVE STOPWORDS AND OTHER PARTS OF TEXT
        clean_text = clean_text.replace(',','')

        pattern = re.compile(r'\b(' + r'|'.join(STOPLIST) + r')\b\s*')
        clean_text = pattern.sub('', clean_text)

        pattern = re.compile(r'\b(' + r'|'.join(QUOTES) + r')\b\s*')
        clean_text = pattern.sub('', clean_text)

        pattern = re.compile(r'\b(' + r'|'.join(HYPHENS) + r')\b\s*')
        clean_text = pattern.sub('', clean_text)

        self.text = clean_text


        #SPLIT THE DATA INTO SENTENCES
        raw_text_sentences = clean_text.split('.')
        self.sentences     = [gensim.utils.to_unicode(word).split() for word in raw_text_sentences]


	#---------------------------------------------------------------
	# TRAIN A WORD2VEC MODEL
	#---------------------------------------------------------------
    def train_word2vec(self, epochs = 10, save_tensors = True ):
        


        def recursive_len(item):
            if type(item) == list:
                return sum(recursive_len(subitem) for subitem in item)
            else:
                return 1

        # INITIALIZE THE MODEL
        model = word2vec.Word2Vec(self.sentences, 
                                  sg             = self.word2vec_params['train_algorithm'], 
                                  min_count      = self.word2vec_params['words_min_count'], 
                                  size           = self.word2vec_params['features_size'], 
                                  window         = self.word2vec_params['words_distance_window'], 
                                  sorted_vocab   = self.word2vec_params['sorted_vocab'],
                                  negative       = self.word2vec_params['negative_sampling'],
                                  max_vocab_size = self.word2vec_params['max_vocab_size'])

        # SAVE THE MDODEL
        self.word2vec_mdoel = model
        model.save(self.name + '.w2vmodel')


        # OPTIONALLY, SAVE TENSORS FOR VISUALIZATION
        if save_tensors is True:
            with open( self.name + '_tensors.tsv', 'w+') as tensors:
                with open( self.name + '_metadata.tsv', 'w+') as metadata:
                    for word in model.wv.index2word:
                        encoded = word
                        metadata.write(encoded + '\n')
                        vector_row = '\t'.join(map(str, model[word]))
                        tensors.write(vector_row + '\n')

    #---------------------------------------------------------------
    # LOAD A WORD2VEC MODEL
    #---------------------------------------------------------------
    def load_word2vec_model(self, model_name):
        print('Loading model', model_name, 'into .work2vec_model')
        self.word2vec_model = word2vec.Word2Vec.load(model_name)



    #---------------------------------------------------------------
    # TRAIN A DOC2VEC MODEL
    #---------------------------------------------------------------
    def train_doc2vec(self):
        documents = [doc2vec.TaggedDocument(doc, [i]) for i, doc in enumerate(self.sentences)]
        self.doc2vec_model = doc2vec.Doc2Vec(documents   = documents,
                                             dm          = self.doc2vec_params['dm'], 
                                             dbow_words  = self.doc2vec_params['dm_mean'], 
                                             vector_size = self.doc2vec_params['vector_size'],
                                             window      = self.doc2vec_params['window'],
                                             min_count   = self.doc2vec_params['min_count'],
                                             epochs      = self.doc2vec_params['epochs'],
                                             workers     = self.doc2vec_params['workers']
                                            )
        #self.doc2vec_model.train(self.sentences)
        self.doc2vec_model.save(self.name + '.d2vmodel')

    #---------------------------------------------------------------
    # LOAD A DOC2VEC MODEL
    #---------------------------------------------------------------
    def load_doc2vec_model(self, model_name):
        print('Loading model', model_name, 'into .doc2vec_model')
        self.doc2vec_model = doc2vec.Doc2Vec.load(model_name)


#---------------------------------------------------------------
# EXAMPLE USEAGE
#---------------------------------------------------------------
if __name__ == "__main__":

    #-----------------------------------------------------------
    # INTIAILIZAING
    #-----------------------------------------------------------
    # IMPORT THE DATA
    with open ('Data/bible.txt', "r") as myfile:
        text = myfile.read()

    # INITIALIZE THE LANGVEC CLASS
    lv = langvec(name = 'bible', 
                 text = text)
    
    # lv = langvec(sentences = sentences)

    # PRE-PROCESS THE TEXT DATA
    lv.preprocess_text()

    #-----------------------------------------------------------
    # TRAIN VECTORIZERS
    #-----------------------------------------------------------
    lv.train_word2vec(save_tensors = True)
    lv.train_doc2vec()


    #-----------------------------------------------------------
    # LOADING DOC2VEC MODEL TRAINED USING WIKIPEDIA
    #------------------------------------------------------------
    lv.load_doc2vec_model("Data/pretrained_model/enwiki_dbow/doc2vec.bin")
    
    # SANITY CHECK TO MAKE SURE THAT SIMMILAR SENTENCES ARE 'CLOSER'
    water   = lv.doc2vec_model.infer_vector("fish swim in lakes".split(' '))
    water2  = lv.doc2vec_model.infer_vector("dolphins swim in the ocean".split(' '))
    land    = lv.doc2vec_model.infer_vector("snakes slither in forests".split(' '))
    land2   = lv.doc2vec_model.infer_vector("serpents slide through water".split(' '))

    # CHECK THAT THE EUCLIDIAN DISTANCE PLACES PHRASES THAT ARE SIMILAR CLOSER TOGETHER
    print(np.linalg.norm(water-water2))
    print(np.linalg.norm(water-land))
    print(np.linalg.norm(land-land2))
    

    



