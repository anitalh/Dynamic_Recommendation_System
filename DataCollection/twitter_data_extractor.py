import tweepy
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import requests


class TweetsListner(StreamListener):

	def __init__(self,consumer_key,consumer_secret,access_token,access_token_secret):
		self.auth = OAuthHandler(consumer_key, consumer_secret)
		self.auth.set_access_token(access_token, access_token_secret)
 
	def on_data(self, data):
		try:
			tweet_text = json.loads(data)
			print (tweet_text)
			tweet_text = tweet_text["text"]
			tweet_id = tweet_text["id_str"]
			return True
		except BaseException as e:
		    print("Error on_data: %s" % str(e))
		return True
 
	def on_error(self, status):
		print(status)
		return True

@classmethod
def parse(cls, api, raw):
    status = cls.first_parse(api, raw)
    setattr(status, 'json', json.dumps(raw))
    return status

def extract_tweet():
	consumer_key ="pUVDi94pxaNgc214PQrhpPUwa"
	consumer_secret ="U8juTSRI2RazqQRIqlqp1qoWcaYVxH7bi2Ka3BKhdk5rPYcIVD"
	access_token ="994457489709125632-7jsHa4jzK2lN7ZoEi6Muv7aeNIWPDmp"
	access_token_secret ="FiJ0Budbxd3bvXi596vSK8Gc2ShmwoCeVTvtPDxLZTBo1"
	auth = OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)
	api = tweepy.API(auth)
	twitter_stream = Stream(auth, TweetsListner())
	twitter_stream.filter(follow=["994457489709125632","1048372701788868608","1048374435370164230"])
	
extract_tweet()