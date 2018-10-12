import re
import random
import time
#import config
print('Library versions:')

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import casual_tokenize
from sklearn.externals import joblib

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Input, LSTM, Dropout, Embedding, RepeatVector, concatenate, \
    TimeDistributed
from keras.utils import np_utils
from keras.models import model_from_json


import keras
print('keras:{}'.format(keras.__version__))
import pandas as pd
print('pandas:{}'.format(pd.__version__))
import sklearn
print('sklearn:{}'.format(sklearn.__version__))
import nltk
print('nltk:{}'.format(nltk.__version__))
import numpy as np
print('numpy:{}'.format(np.__version__))



# 8192 - large enough for demonstration, larger values make network training slower
MAX_VOCAB_SIZE = 2**13
# seq2seq generally relies on fixed length message vectors - longer messages provide more info
# but result in slower training and larger networks
MAX_MESSAGE_LEN = 30  
# Embedding size for words - gives a trade off between expressivity of words and network size
EMBEDDING_SIZE = 100
# Embedding size for whole messages, same trade off as word embeddings
CONTEXT_SIZE = 100
# Larger batch sizes generally reach the average response faster, but small batch sizes are
# required for the model to learn nuanced responses.  Also, GPU memory limits max batch size.
BATCH_SIZE = 4
# Helps regularize network and prevent overfitting.
DROPOUT = 0.2
# High learning rate helps model reach average response faster, but can make it hard to 
# converge on nuanced responses
LEARNING_RATE=0.005

TRAINING_TIME=60

# Tokens needed for seq2seq
UNK = 0  # words that aren't found in the vocab
PAD = 1  # after message has finished, this fills all remaining vector positions
START = 2  # provided to the model at position 0 for every response predicted

# Implementaiton detail for allowing this to be run in Kaggle's notebook hardware
SUB_BATCH_SIZE = 1000

class twitterbot:
    def __init__(self):
        self.analyzer = {}
        self.vocab = {}
        self.reverse_vocab = {}


    

    def to_word_idx(self, sentence):
        full_length = [self.vocab.get(tok, UNK) for tok in self.analyzer(sentence)] + [PAD] * MAX_MESSAGE_LEN
        return full_length[:MAX_MESSAGE_LEN]

    def from_word_idx(self, word_idxs):
        return ' '.join(self.reverse_vocab[idx] for idx in word_idxs if idx != PAD).strip()

    #REPLACE WITH SQLITE CODE
    #------------------------------------------------------------------------------------------------------------

    def read_csv_data(self, filename):
        tweets = pd.read_csv(filename)
        #tweets = pd.read_csv('/floyd/input/data/twcs.csv')

        first_inbound = tweets[pd.isnull(tweets.in_response_to_tweet_id) & tweets.inbound]

        inbounds_and_outbounds = pd.merge(first_inbound, tweets, left_on='tweet_id', 
                                        right_on='in_response_to_tweet_id').sample(frac=1)

        # Filter to only outbound replies (from companies)
        inbounds_and_outbounds = inbounds_and_outbounds[inbounds_and_outbounds.inbound_y ^ True]

        self.texts = inbounds_and_outbounds.text_x
        self.responses = inbounds_and_outbounds.text_y

        print('Data shape: {}'.format(inbounds_and_outbounds.shape))
        print('t shape: {}'.format(self.texts.shape))
        print('r shape: {}'.format(self.responses.shape))

    def build_vocab(self):
        count_vec = CountVectorizer(tokenizer=casual_tokenize, max_features=MAX_VOCAB_SIZE - 3)
        print("Fitting CountVectorizer on X and Y text data...")
        count_vec.fit(self.texts, self.responses)
        joblib.dump(count_vec, 'count_vec.joblib') 
        
        self.analyzer = count_vec.build_analyzer()
        #joblib.dump(self.analyzer, 'analyzer.joblib')
        self.vocab = {k: v + 3 for k, v in count_vec.vocabulary_.items()}
        self.vocab['__unk__'] = UNK
        self.vocab['__pad__'] = PAD
        self.vocab['__start__'] = START
        # Used to turn seq2seq predictions into human readable strings
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        print("Learned vocab of {} items.".format(len(self.vocab)))
        joblib.dump(self.vocab, 'vocab.joblib')

    def build_vectors(self):

        print("Calculating word indexes for X...")
        x = pd.np.vstack(self.texts.apply(self.to_word_idx).values)
        print("Calculating word indexes for Y...")
        y = pd.np.vstack(self.responses.apply(self.to_word_idx).values)



        all_idx = list(range(len(x)))
        train_idx = set(random.sample(all_idx, int(0.8 * len(all_idx))))
        test_idx = {idx for idx in all_idx if idx not in train_idx}

        self.train_x = x[list(train_idx)]
        self.test_x = x[list(test_idx)]
        self.train_y = y[list(train_idx)]
        self.test_y = y[list(test_idx)]

        
        print('Training data of shape {} and test data of shape {}.'.format(self.train_x.shape,self.test_x.shape))
        



    
    def build_model(self):
        shared_embedding = Embedding(
            output_dim=EMBEDDING_SIZE,
            input_dim=MAX_VOCAB_SIZE,
            input_length=MAX_MESSAGE_LEN,
            name='embedding',
        )
        
        # ENCODER
        
        encoder_input = Input(
            shape=(MAX_MESSAGE_LEN,),
            dtype='int32',
            name='encoder_input',
        )
        
        embedded_input = shared_embedding(encoder_input)
        
        # No return_sequences - since the encoder here only produces a single value for the
        # input sequence provided.
        encoder_rnn = LSTM(
            CONTEXT_SIZE,
            name='encoder',
            dropout=DROPOUT
        )
        
        context = RepeatVector(MAX_MESSAGE_LEN)(encoder_rnn(embedded_input))
        
        # DECODER
        
        last_word_input = Input(
            shape=(MAX_MESSAGE_LEN, ),
            dtype='int32',
            name='last_word_input',
        )
        
        embedded_last_word = shared_embedding(last_word_input)
        # Combines the context produced by the encoder and the last word uttered as inputs
        # to the decoder.
        decoder_input = concatenate([embedded_last_word, context], axis=2)
        
        # return_sequences causes LSTM to produce one output per timestep instead of one at the
        # end of the intput, which is important for sequence producing models.
        decoder_rnn = LSTM(
            CONTEXT_SIZE,
            name='decoder',
            return_sequences=True,
            dropout=DROPOUT
        )
        
        decoder_output = decoder_rnn(decoder_input)
        
        # TimeDistributed allows the dense layer to be applied to each decoder output per timestep
        next_word_dense = TimeDistributed(
            Dense(int(MAX_VOCAB_SIZE / 2), activation='relu'),
            name='next_word_dense',
        )(decoder_output)
        
        next_word = TimeDistributed(
            Dense(MAX_VOCAB_SIZE, activation='softmax'),
            name='next_word_softmax'
        )(next_word_dense)
        
        self.model = Model(inputs=[encoder_input, last_word_input], outputs=[next_word])
        optimizer = Adam(lr=LEARNING_RATE, clipvalue=5.0)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')



    def add_start_token(self, y_array):
        #Adds the start token to vectors.  Used for training data. 
        return np.hstack([
            START * np.ones((len(y_array), 1)),
            y_array[:, :-1],
        ])

    def binarize_labels(self, labels):
        # Helper function that turns integer word indexes into sparse binary matrices for 
        #    the expected model output.
        
        return np.array([np_utils.to_categorical(row, num_classes=MAX_VOCAB_SIZE)
                        for row in labels])


    def respond_to(self, text):
        # Helper function that takes a text input and provides a text output. 
        input_y = self.add_start_token(PAD * np.ones((1, MAX_MESSAGE_LEN)))
        idxs = np.array(self.to_word_idx(text)).reshape((1, MAX_MESSAGE_LEN))
        for position in range(MAX_MESSAGE_LEN - 1):
            prediction = self.model.predict([idxs, input_y]).argmax(axis=2)[0]
            input_y[:,position + 1] = prediction[position]
        return self.from_word_idx(self.model.predict([idxs, input_y]).argmax(axis=2)[0])


    def train_mini_epoch(self, start_idx, end_idx):
        # Batching seems necessary in Kaggle Jupyter Notebook environments, since
        #    `model.fit` seems to freeze on larger batches (somewhere 1k-10k).
        
        b_train_y = self.binarize_labels(self.train_y[start_idx:end_idx])
        input_train_y = self.add_start_token(self.train_y[start_idx:end_idx])
        
        self.model.fit(
            [self.train_x[start_idx:end_idx], input_train_y], 
            b_train_y,
            epochs=1,
            batch_size=BATCH_SIZE,verbose=2
        )
        
        rand_idx = random.sample(list(range(len(self.test_x))), SUB_BATCH_SIZE)
        print('Test results:', self.model.evaluate(
            [self.test_x[rand_idx], self.add_start_token(self.test_y[rand_idx])],
            self.binarize_labels(self.test_y[rand_idx])
        ))
        
    def train(self,filename):
            self.read_csv_data(filename)
            self.build_vocab()
            self.build_vectors()
            self.build_model()
            
            self.train_mini_epoch(0, SUB_BATCH_SIZE)  

    def load_model_from_disk(self,model_file, weights_file):
        json_file = open(model_file, 'r')
        lm_json = json_file.read()
        json_file.close()
        lm = model_from_json(lm_json)   
        # load weights into new model
        lm.load_weights(weights_file)
        print("Loaded model from disk")
        return lm



    def create_responder(self):
        count_vec = joblib.load('count_vec.joblib')
        self.analyzer = count_vec.build_analyzer()
        self.vocab = joblib.load('vocab.joblib')
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.model = self.load_model_from_disk('s2s_model.json', 's2s_model.h5')



"""
tb = twitterbot()
tb.train('../data/twcs.csv')
"""



tb = twitterbot()
tb.create_responder()



input_strings = [
            "@AppleSupport I fix I this I stupid I problem I",
            "@AmazonHelp I hadnt expected that such a big brand like amazon would have such a poor customer service.",
        ]
        
for input_string in input_strings:
    output_string = tb.respond_to(input_string)
    print('> "{}"\n< "{}"'.format(input_string, output_string))




