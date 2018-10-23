
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import tweepy
import traceback

import secrets
import json

import sqlite3
import pandas as pd

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    try:
        conn = sqlite3.connect(db_file)
        conn.execute('create table if not exists tweets(tweet text, response text)')
        return conn
    except Error as e:
        print(e)
 
    return None


def insert_tweet(conn, tweet):
    """
    Create a new project into the projects table
    :param conn:
    :param project:
    :return: project id
    """
    sql = ''' INSERT INTO tweets(tweet, response)
              VALUES(?,?) '''
    cur = conn.cursor()
    cur.execute(sql, tweet)
    return cur.lastrowid




auth = OAuthHandler(secrets.consumer_key, secrets.consumer_secret)
auth.set_access_token(secrets.access_token_key, secrets.access_token_secret)
api = tweepy.API(auth)



class StdOutListener(StreamListener):

    def __init__(self):
        self.max=100
        
        self.i = 0

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
            #if tweet:
            #    if self.i % 10 == 0:
            #        print(self.i)
            #    with conn:
            #        t = json.loads(tweet)
            #        record = (j['text'], t['text'])
            #        insert_tweet(conn, record)
            #    self.i=self.i+1
                
            #    if self.i==self.max:
            #        exit()
            return True
        except Exception as E:
            traceback.print_stack()
            print(E)
            pass    

    def on_error(self, status):
        print(status)
        with open('results.json', 'w') as outfile:
            json.dump(self.r, outfile)

'po8765'
conn = create_connection('test.db')
listener = StdOutListener()
stream = Stream(auth, listener)
stream.filter(follow=['85741735'])

#df = pd.read_sql_query('select * from tweets limit 5', conn)
#print(df.head())
