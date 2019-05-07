###############################################################
# IMPORT EXTERNAL LIBRARIES
###############################################################
import gensim
from gensim.models import word2vec
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
import string
import regex as re
from nltk.tokenize import word_tokenize

#--------------------------------------------------------------------
# PARAMETERS TO CLEAN UP THE TEXT
#--------------------------------------------------------------------
STOPLIST = set(stopwords.words('english') + ["n't", "'s", "'m", "ca"] + list(ENGLISH_STOP_WORDS))
re.DEFAULT_VERSION = re.VERSION1
split_chars = lambda char: list(char.strip().split(' '))
WHITESPACE = ["", " ", "  ", "   ","    ", "\t", "\n", "\r\n", "\n\n"]
_quotes = r'\' \'\' " ” “ `` ` ‘ ´ ‘‘ ’’ ‚ , „ » « 「 」 『 』 （ ） 〔 〕 【 】 《 》 〈 〉'
QUOTES  = split_chars(_quotes)
_hyphens = '- – — -- --- —— ~'
HYPHENS  = split_chars(_hyphens)

#---------------------------------------------------------------
# A TOOL THAT VECTORIZES TEXT DOCUMENTS AND CLAUSES
#---------------------------------------------------------------
class langvec:
    def __init__(self, text = None, sentences = None):
        # BASIC TEXT INFORMATION
        self.text      = text
        self.sentences = sentences;                            # generates cleaned sentences from the text 
        
        # WORD2VEC RELATED
        self.word2vec_model  = None;
        self.word2vec_params = {'words_min_count'      : 10,   # min_count = ignore all words with total frequency lower than this.
                                'features_size'        : 100,  # number of features
                                'words_distance_window': 4,    # window is the maximum distance between the current and predicted word within a sentence.
                                'sorted_vocab'         : 1,    # if 1 (default), sort the vocabulary by descending frequency before assigning word indexes.
                                'train_algorithm'      : 1,    # defines the training algorithm. By default (sg=0), CBOW is used. Otherwise (sg=1), skip-gram is employed.
                                'negative_sampling'    : 10,   # if > 0, negative sampling will be used, the int for negative specifies how many “noise words” should be drawn (usually between 5-20). # Default is 5. If set to 0, no negative samping is used.
                                'max_vocab_size'       : None  # limit RAM during vocabulary building; if there are more unique words than this, then prune the infrequent ones. # Every 10 million word types need about 1GB of RAM. Set to None for no limit (default).
                               }

         
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
	# TRAIN THE MODEL
	#---------------------------------------------------------------
    def trian_word2vec_model(self, save_model_as, save_tensors_as = None ):
        model = word2vec.Word2Vec(self.sentences, 
                                  sg             = self.word2vec_params['train_algorithm'], 
                                  min_count      = self.word2vec_params['words_min_count'], 
                                  size           = self.word2vec_params['features_size'], 
                                  window         = self.word2vec_params['words_distance_window'], 
                                  sorted_vocab   = self.word2vec_params['sorted_vocab'],
                                  negative       = self.word2vec_params['negative_sampling'],
                                  max_vocab_size = self.word2vec_params['max_vocab_size']
                                  )

        self.word2vec_mdoel = model
        
        # SAVE THE MDODEL
        model.save(save_model_as + '.w2vmodel')

        # OPTIONALLY, SAVE TENSORS FOR VISUALIZATION
        if save_tensors_as is not None:
            with open( save_tensors_as + '_tensors.tsv', 'w+') as tensors:
                    with open( save_tensors_as + 'metadata.tsv', 'w+') as metadata:
                             for word in model.wv.index2word:
                                 encoded=word
                                 metadata.write(encoded + '\n')
                                 vector_row = '\t'.join(map(str, model[word]))
                                 tensors.write(vector_row + '\n')


if __name__ == "__main__":

    # IMPORT THE DATA
    with open ('Data/bible.txt', "r") as myfile:
        text = myfile.read()

    # INITIALIZE THE LANGVEC CLASS
    lv = langvec(text = text)
    # lv = langvec(sentences = sentences)

    # PRE-PROCESS THE TEXT DATA
    lv.preprocess_text()

    # TRAIN WORD TO VEC
    lv.trian_word2vec_model(save_model_as = 'bible', save_tensors_as = 'bible' )



