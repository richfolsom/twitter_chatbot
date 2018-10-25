
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import tweepy
import sqlite3
import secrets
import json


def create_connection(db_file):
    try:
        conn = sqlite3.connect(db_file)
        conn.execute('create table if not exists tweets(tweet text, response text)')
        return conn
    except Error as e:
        print(e)
 
    return None


def insert_tweet(conn, tweet):
    sql = ''' INSERT INTO tweets(tweet, response)
              VALUES(?,?) '''
    cur = conn.cursor()
    cur.execute(sql, tweet)
    return cur.lastrowid



class AmazonListener(StreamListener):
    def on_data(self, data):
        try:
            j = json.loads(data)
            if str(j['user']['id']) == '85741735':
                reply_to = api.get_status(str(j['in_reply_to_status_id']))
                if reply_to:
                    print('{}\n\t{}'.format(reply_to.text,j['text']))
                    with conn:
                        record = (reply_to.text, j['text'])
                        insert_tweet(conn, record)
            return True
        except Exception as E:
            print(E)
            pass    

    def on_error(self, status):
        print(status)
        with open('results.json', 'w') as outfile:
            json.dump(self.r, outfile)

auth = OAuthHandler(secrets.consumer_key, secrets.consumer_secret)
auth.set_access_token(secrets.access_token_key, secrets.access_token_secret)
api = tweepy.API(auth)

conn = create_connection('test.db')
listener = AmazonListener()
stream = Stream(auth, listener)
stream.filter(follow=['85741735'])