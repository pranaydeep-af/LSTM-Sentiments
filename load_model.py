import multiprocessing
import numpy as np
import os
import re
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout


regex = re.compile('[^a-zA-Z]')
np.random.seed(1337)
# set parameters:
vocab_dim = 300
maxlen = 100
n_iterations = 10
n_exposures = 30
window_size = 7
batch_size = 32
n_epoch = 2
input_length = 100
cpu_count = multiprocessing.cpu_count()
data_locations = {'./Data/test_neg.txt': 'TEST_NEG',
                  './Data/test_pos.txt': 'TEST_POS',
                  './Data/train_neg.txt': 'TRAIN_NEG',
                  './Data/train_pos.txt': 'TRAIN_POS'}





def import_tag(datasets = None):
  ''' Imports the datasets into one of two dictionaries

      Dicts:
          train & test

      Keys:
          values > 12500 are "Negative" in both Dictionaries

      '''
  if datasets is not None:
      train = {}
      test = {}
      for k, v in datasets.items():
          with open(k) as fpath:
              data = fpath.readlines()
          for val, each_line in enumerate(data):
              if v.endswith("NEG") and v.startswith("TRAIN"):
                  train[val] = each_line
              elif v.endswith("POS") and v.startswith("TRAIN"):
                  train[val + 12500] = each_line
              elif v.endswith("NEG") and v.startswith("TEST"):
                  test[val] = each_line
              else:
                  test[val + 12500] = each_line
      return train, test
  else:
      print('Data not found...')


def tokenizer(text):
  ''' Simple Parser converting each document to lower-case, then
      removing the breaks for new lines and finally splitting on the
      whitespace
  '''
  text = [document.lower().replace('\n', '').split() for document in text]
  for i,l1 in enumerate(text):
      text[i]=[regex.sub('',x) for x in l1]
  return text


def create_dictionaries(train = None,
                      test = None,
                      model = None):
  ''' Function does are number of Jobs:
      1- Creates a word to index mapping
      2- Creates a word to vector mapping
      3- Transforms the Training and Testing Dictionaries

  '''
  if (train is not None) and (model is not None) and (test is not None):
      gensim_dict = Dictionary()
      gensim_dict.doc2bow(model.vocab.keys(),
                          allow_update=True)
      w2indx = {v: k+1 for k, v in gensim_dict.items()}
      w2vec = {word: model[word] for word in w2indx.keys()}

      def parse_dataset(data ):
          ''' Words become integers
          '''
          for key in data.keys():
              txt = data[key].lower().replace('\n', '').split()
              new_txt = []
              for word in txt:
                  try:
                      new_txt.append(w2indx[word])
                  except:
                      new_txt.append(0)
              data[key] = new_txt
          return data
      train = parse_dataset(train )
      test = parse_dataset(test )
      return w2indx, w2vec, train, test
  else:
      print('No data provided...')





from keras.models import model_from_json
print('Loading Keras Model...')
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.summary()
loaded_model.load_weights('model.h5')
print("Loaded Weights from Disk")
print('Compiling the Model...')
loaded_model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model = Word2Vec.load('vectorizer.w2v')
print 'Vectorizer Imported'


new_stuff = {1: 'bad bad good'}
index_dict, word_vectors, train, test = create_dictionaries(train = new_stuff, test = {}, model=model)
train = sequence.pad_sequences(train.values(), maxlen = maxlen)
print loaded_model.predict(train)
