import re
import random
import time


from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

from keras.models import model_from_json
import secrets
import json

print('Library versions:')
import numpy as np
print('numpy:{}'.format(np.__version__))
MAX_MESSAGE_LEN = 30  

# Tokens needed for seq2seq
UNK = 0  # words that aren't found in the vocab
PAD = 1  # after message has finished, this fills all remaining vector positions
START = 2  # provided to the model at position 0 for every response predicted

from sklearn.externals import joblib
count_vec = joblib.load('count_vec.joblib')
analyzer = count_vec.build_analyzer()
vocab = joblib.load('vocab.joblib')

# Used to turn seq2seq predictions into human readable strings
reverse_vocab = {v: k for k, v in vocab.items()}
print("Learned vocab of {} items.".format(len(vocab)))

def to_word_idx(sentence):
    full_length = [vocab.get(tok, UNK) for tok in analyzer(sentence)] + [PAD] * MAX_MESSAGE_LEN
    return full_length[:MAX_MESSAGE_LEN]

def from_word_idx(word_idxs):
    return ' '.join(reverse_vocab[idx] for idx in word_idxs if idx != PAD).strip()


def add_start_token(y_array):
    """ Adds the start token to vectors.  Used for training data. """
    return np.hstack([
        START * np.ones((len(y_array), 1)),
        y_array[:, :-1],
    ])

def respond_to(model, text):
    """ Helper function that takes a text input and provides a text output. """
    input_y = add_start_token(PAD * np.ones((1, MAX_MESSAGE_LEN)))
    idxs = np.array(to_word_idx(text)).reshape((1, MAX_MESSAGE_LEN))
    for position in range(MAX_MESSAGE_LEN - 1):
        prediction = model.predict([idxs, input_y]).argmax(axis=2)[0]
        input_y[:,position + 1] = prediction[position]
    return from_word_idx(model.predict([idxs, input_y]).argmax(axis=2)[0])


def load_model_from_disk(model_file, weights_file):
    json_file = open(model_file, 'r')
    lm_json = json_file.read()
    json_file.close()
    lm = model_from_json(lm_json)
    # load weights into new model
    lm.load_weights(weights_file)
    print("Loaded model from disk")
    return lm


class StdOutListener(StreamListener):
    def on_data(self, data):
#        print(data)
        obj = json.loads(data)
        print('> {}'.format(obj["text"]))
        print('< {}'.format(respond_to(loaded_model, obj["text"])))
#        blob = TextBlob(obj["text"])
#        for sentence in blob.sentences:
#            print(sentence.sentiment.polarity)
        return True
    def on_error(self, status):
        print(status)



loaded_model = load_model_from_disk('s2s_model.json', 's2s_model.h5')
listener = StdOutListener()
auth = OAuthHandler(secrets.consumer_key, secrets.consumer_secret)
auth.set_access_token(secrets.access_token_key, secrets.access_token_secret)
stream = Stream(auth, listener)
stream.filter(track=['@AmazonHelp'])