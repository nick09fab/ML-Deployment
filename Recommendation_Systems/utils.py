# A collection of helper functions
# nltk packages
import nltk
import ssl
import re
import numpy as np
import pandas as pd

""" Download all nltk packages through external ssl"""
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

# nltk libraries
from nltk.corpus import stopwords 
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn


#Function to calculate null_count percentage for each column
def calc_missing_rowcount(df):
    columns = df.columns
    null_count = [df[col].isnull().sum() for col in columns]
    null_perc = [round((val/df.shape[0]) * 100,2) for val in null_count]
    _df = pd.DataFrame({'Columns':np.array(columns),
                        'Count': np.array(null_count),
                        'Percent':np.array(null_perc)})
    # If round is not an option, formatting options in pandas and numpy will 
    # pd.options.display.float_format = '{:.2f}'.format
    # np.set_printoptions(suppress=True)
                            
    return _df.sort_values("Percent",ascending=False)


# Function to remove the stopwords
def clean_stopwords(sent):
    stopwords_set = set(stopwords.words("english"))
    sent = sent.lower() # Text to lowercase
    words = word_tokenize(sent) # Split sentences into words
    text_nostopwords = " ".join( [each_word for each_word in words if each_word not in stopwords_set] )
    return text_nostopwords

# Function to clean the text and remove all the unnecessary elements.
def clean_punctuation(sent):
    sent = sent.lower() # Text to lowercase
    pattern = '[^\w\s]' # Removing punctuation
    sent = re.sub(pattern, '', sent)
    return sent

# This is a helper function to map NTLK position tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('V'):
        return wn.VERB
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    else:
        return wn.NOUN

def remove_stopword(text):
    stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if word.isalpha() and word not in stop_words]
    return " ".join(words)

# Lemmatize the sentence
def clean_lemma(text):
    lemmatizer = WordNetLemmatizer()
    word_pos_tags = nltk.pos_tag(word_tokenize(remove_stopword(text))) # Get position tags
    # Map the position tag and lemmatize the word/token
    words =[lemmatizer.lemmatize(tag[0], get_wordnet_pos(tag[1])) 
            for _, tag in enumerate(word_pos_tags)] 
    return " ".join(words)

