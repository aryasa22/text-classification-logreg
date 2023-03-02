import numpy as np 
import pandas as pd 
import re
import nltk
import itertools
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# Functions
## Lowercasing
def lowercase(text):
  lower = str.lower(text)
  return lower

## Replace New Line
def rep_newline(text):
  newline = text.replace("\\n", " ")
  newline = newline.replace("\n", " ")
  return newline

## Remove Kaskus Formatting
def rmv_kaskus(text):
  clean = re.sub('\[', ' [', text)
  clean = re.sub('\]', '] ', clean)
  clean = re.sub('\[quote[^ ]*\].*?\[\/quote\]', ' ', clean)
  clean = re.sub('\[[^ ]*\]', ' ', clean)
  clean = re.sub('&quot;', ' ', clean)
  return clean

## Remove Twitter Formatting
def rmv_twitter(text):
  clean = re.sub(r'@[A-Za-z0-9]+', '', text)
  clean = re.sub(r'\brt\b', '', clean)
  return clean

## Remove URL
def rmv_url(text):
  clean = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', text)
  return clean

## Remove Additional Whitespaces
def rmv_whitespace(text):
  clean = re.sub('  +', ' ', text)
  return clean

## Remove Punctuation
def rmv_punct(text):
  clean = re.sub(r'[^\w\s]', ' ', text)
  return clean

## Replace Slang Words
slangs = pd.read_csv('../data/slangword.csv')
slangs_dict = slangs.set_index('original').to_dict()
slangs_dict = slangs_dict['translated']

def replace_slangs(text):
    word_list = text.split()
    word_list_len = len(word_list)
    transformed_word_list = []
    i = 0
    while i < word_list_len:
        if (i + 1) < word_list_len:
            two_words = ' '.join(word_list[i:i+2])
            if two_words in slangs_dict:
                transformed_word_list.append(slangs_dict[two_words])
                i += 2
                continue
        transformed_word_list.append(slangs_dict.get(word_list[i], word_list[i]))
        i += 1
    return ' '.join(transformed_word_list)

## Remove Non Alphabeth
def rmv_nonalphabeth(text):
  clean = re.sub('[^a-zA-Z ]+', '', text)
  return clean

## Remove Repeated Letters
def rmv_repeat(text):
  clean = ''.join(''.join(s)[:1] for _, s in itertools.groupby(text))
  return clean

## Tokenizing
from nltk import word_tokenize
nltk.download('punkt')


## Remove Stopwords
from nltk.corpus import stopwords
nltk.download('stopwords')

def rmv_stopwords(tokens):
  nostop = []
  for token in tokens:
    if not token in set(stopwords.words('indonesian')):
      nostop.append(token)
  return nostop

## Join Tokens
def join_tokens(tokens):
  joined = ' '.join(tokens)
  return joined

## Stemming
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def stemming(text):
  factory = StemmerFactory()
  stemmer = factory.create_stemmer()
  return stemmer.stem(text)

## Vectorize
with open('vectorizer.pickle', 'rb') as f:
    vect = pickle.load(f)

# Preprocessing
def preprocessing(inputs):
  clean_inputs = lowercase(inputs)
  clean_inputs = rep_newline(clean_inputs)
  clean_inputs = rmv_kaskus(clean_inputs)
  clean_inputs = rmv_twitter(clean_inputs)
  clean_inputs = rmv_url(clean_inputs)
  clean_inputs = rmv_whitespace(clean_inputs)
  clean_inputs = rmv_punct(clean_inputs)
  clean_inputs = replace_slangs(clean_inputs)
  clean_inputs = rmv_nonalphabeth(clean_inputs)
  clean_inputs = rmv_repeat(clean_inputs)
  clean_inputs = word_tokenize(clean_inputs)
  clean_inputs = rmv_stopwords(clean_inputs)
  clean_inputs = join_tokens(clean_inputs)
  clean_inputs = stemming(clean_inputs)
  clean_inputs = vect.transform([clean_inputs])
  clean_inputs = clean_inputs.toarray()

  return clean_inputs