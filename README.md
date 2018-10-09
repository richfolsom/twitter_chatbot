# twitter_chatbot


## This is my first attempt at a chatbot for twitter

I borrowed from this notebook:  https://www.kaggle.com/soaxelbrooke/twitter-basic-seq2seq to get started then modified it to run on Floydhub for training from the command line.  I also added a Twitter listener.  I plan on writing up the whole process so stay tuned.

High level:
1.  Check out the code to a local directory.
2.  Sign up for Twitter and get an API Key.
3.  Fill in the blanks from the secrets.py file with your API info.
4.  Install all the dependencies (I need to make a requirements.txt file)
5.  From the command line, run 'python3 train.py' (For the first run, I'd update line 19 to be really short to make sure everything works).
6.  Step 5 will create 4 files (s2s_model.json, s2s_model.h5, vocab.joblib, count_vec.joblib)  These files are the configuration and weights for the trained model we created on step 5.
7.  Now you can run 'python3 client.py', which will actually run the chatbot but just print the responses to stdout.  If you want to filter on anything else besides AmazonHelp, you can edit the last line of client.py.
8.  At this point, the responses will be gibberish because the model isn't fully trained.
9.  If you have a GPU local, you can just run 'python3 train.py' for much longer (the original notebook suggested 6 hours) by updating line 19.
10. I didn't have a local GPU, so I signed up for Floydhub http://floydhub.com and ran it there.  I then downloaded all the generated files.
11. Now, you should get more meaningful responses to the tweets, and if you're ready, you can actually tweet the responses vs. write them to the screen.