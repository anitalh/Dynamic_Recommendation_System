from textblob import TextBlob
import json
from nltk import sent_tokenize
from textblob.taggers import NLTKTagger
import requests
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer
import spacy
import urllib3
from pprint import pprint
import time
import sys
sys.path.append("..")
from kafkaProducer import Producer

class sentimentsAnalyser:
  '''
  ""TODO"" : 1. Tokanize the sentence based on context (like but,and) or based on punctuation (",","." etc)
             2.  Extracting a noun  from the sentence
             3. Associate a sentmentints to known 
             4. Generate a Dynamic profile

  '''
  def __init__(self):
        self.genre_list = ['action', 'adventure','comedy','crime','animation','anime','biopic','childrens','detective','spy','documentary','drama','horror','family','fantasy','historical','medical','musical','paranormal','romance','sport','fiction','sci-fi','mystery','thriller','suspense','war','western']
        with open("../MovieData/actor1_handle.json",'r') as a1f:
          #data = a1f.read()
          self.actor1_dict = json.load(a1f)
        with open("../MovieData/actor2_handle.json",'r') as a2f:
          #data = a2f.read()
          self.actor2_dict = json.load(a2f)
        with open("../MovieData/director_handle.json",'r') as df:
          #data = df.read()
          self.director_dict = json.load(df)
        with open("../MovieData/movie_handle.json",'r') as mf:
          #data = mf.read()
          self.movie_dict = json.load(mf)

  def punCtuateSentence(self,msg):
    http = urllib3.PoolManager()
    payload = "text="+ msg
    data = {'text': msg}
    print (payload)
    resp = http.request(
    "POST", "http://bark.phon.ioc.ee/punctuator", 
    body=payload)
    pprint (vars(resp))
    print (resp.content)
    print (resp.text)
    print (resp.headers)
    #print (json.loads(resp.data.decode('utf-8')))
    '''
    payload = "text="+ msg
    headers = {'Content-Type': 'application/json', 'Accept-Encoding': None}
    print (payload)
    r = requests.post("http://bark.phon.ioc.ee/punctuator",  headers=headers, data=json.dumps(payload), verify =False)
    print (type(r))
    print (r.headers)
    print ("resp from punct API",str(r.text),"len",len(r.text))
    print ("json",r.text)
    resp= json.loads(r.text)
    print (resp)
    return resp
    '''
  '''
  add all conjunction to the sentence boundries 
  '''
  def set_custom_boundaries(self,doc):
    for token in doc[:-1]:
      if token.text in ('but' , 'yet' , 'and','however','or','though','although','eventhough','while' ):
        doc[token.i+1].is_sent_start = True
    return doc


  def spacySentenceSeggrigator(self,msg):
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe(self.set_custom_boundaries, before='parser')
    doc = nlp(msg)
    print('seggrgated sentences:', [sent.text for sent in doc.sents])
    return  [sent.text for sent in doc.sents]

  '''
  returns all the nouns in the original text and screen_name corresponding 
  to the original tweeter so that sentiments can be assocated to them

  if retweet us empty or postive (single sentece) without noun
    assign polarity of original tweet

  elif negtive tweet without tags
    assign opposit polarity of original tweet
  elseif :
    take polarity from retweet and those tags not in rtweet take tags original polarity.
  else:
    matching tags compare polariry and take poliryt of users if tags not in 
  '''

  def computeNounTagsFromrtweetRelation(self,retweet_text,msg,noun_sentiment_dict):
      original_tweet = msg['tweet_data']['quoted_status']['text']
      print(original_tweet)
      print("original")
      #correctedMsg = TextBlob(original_tweet).correct()
      #print (correctedMsg)
      # correct or add punctuation marks to the sentence so that splitting is correct.
      #punctuated_msg = self.punCtuateSentence(str(correctedMsg))
      #punctuated_msg = str(correctedMsg)
      original_items =[]
      retweet_items =[]
      original_items = self.extractAndAssignPolarity(original_tweet.lower())
      if len(retweet_text) == 0:
        return original_items
      else:
        retweet_items = self.extractAndAssignPolarity(retweet_text.lower())
        if len(retweet_items) == 0:
          analysed_sent = TextBlob(retweet_text.lower())
          sentiment = analysed_sent.sentiment.polarity
          if(sentiment<0):
            for orig in original_items:
              orig['sentiment'] = orig['sentiment']*(-1)
          
        else:
          for orig in original_items:
            for retweet in retweet_items:
              if orig['value'] == retweet['value']:
                if (orig['sentiment'] > 0 and retweet['sentiment'] < 0) or (orig['sentiment'] < 0 and retweet['sentiment'] > 0 ):
                  orig['sentiment'] = retweet['sentiment']

        print(original_items)
        return original_items


      if len(original_tweet) > 1:
        split_words = original_tweet.split()
        for word in split_words:
          if word.startswith("@") or (word in self.genre_list):
            nouns_list.append(msg['quoted_status']['user']['screen_name'])
        if msg['quoted_status']['user']['screen_name']:
           print ("getNounsFromRetweet appending",msg['quoted_status']['user']['screen_name'])
           nouns_list.append(msg['quoted_status']['user']['screen_name'])
        print ("nouns_list",nouns_list)
        return nouns_list


  def getNounsFromRetweet(self,msg):
    nouns_list =[]
    if 'quoted_status' in msg:
      original_tweet = msg['quoted_status']['text']
      split_words = original_tweet.split()
      for word in split_words:
        if word.startswith("@") or (word in self.genre_list):
          nouns_list.append(msg['quoted_status']['user']['screen_name'])
      if msg['quoted_status']['user']['screen_name']:
         print ("getNounsFromRetweet appending",msg['quoted_status']['user']['screen_name'])
         nouns_list.append(msg['quoted_status']['user']['screen_name'])
    print ("nouns_list",nouns_list)
    return nouns_list

  '''
  {
  "u1":{
        "a":{"a1":[],"a2":[]},
        "c":{"c1":[],"c2":[]}
  },
  "u2":{
        "a":{"a1":{sp:1.0,dp:1,time =56655},"a2":[]},
        "c":{"c1":[],"c2":[]}
  }
    
  }
  if timediff < 24 hrs
    replace DP 
    set new current time

  else:
    if DP > SP:
      set Sp = sp + |sp*dp|
    else:
      set Sp = sp + |sp-dp|
  
  set dep = dp new
  '''


  '''
  def searchAndUpdateDynamicProfile(self,userId,criteria_list):
    with open("../ProfileData/"+str(userId)+".json") as f:
      user_data = json.load(f)
    computation_duration = (1*60)
    print(userId,criteria_list)
    #user_data = profile_data['userId']
    for criteria in criteria_list:
      print("Inside0")
      if criteria['role'] in user_data:
        foundcriteria = False
        print("Inside1")
        for key,val in user_data.items():
          if key == criteria['role']:
            print ("macthed criteria['value']",criteria['value'])
            foundcriteria = True
            key = criteria['value']
            print (time.time() - user_data[criteria['role']][key]['time'])
            print(computation_duration)
            if (time.time() - user_data[criteria['role']][key]['time']) >= computation_duration:
              if user_data[criteria['role']][key]['DP'] >= user_data[criteria['role']][key]['SP']:
                print ("user_data[criteria['role']][key]['DP'] is greater than SP")
                user_data[criteria['role']][key]['SP'] += abs((user_data[criteria['role']][key]['DP'])*(user_data[criteria['role']][key]['SP'])*(0.1))
                if user_data[criteria['role']][key]['SP'] > 1:
                  user_data[criteria['role']][key]['SP'] = 1
              else:
                user_data[criteria['role']][key]['SP'] -= abs((user_data[criteria['role']][key]['DP'])*(user_data[criteria['role']][key]['SP'])*(0.1))
                if user_data[criteria['role']][key]['SP'] < -1:
                  user_data[criteria['role']][key]['SP'] = -1

            print(user_data[criteria['role']][key])
            user_data[criteria['role']][key]['DP'] = criteria['sentiment'] 
            user_data[criteria['role']][key]['time'] = time.time()
            print(user_data[criteria['role']][key])

            break
        if foundcriteria == False:
           user_data[criteria['role']][criteria['value']] = {}
           user_data[criteria['role']][criteria['value']]['SP'] = criteria['sentiment']
           user_data[criteria['role']][criteria['value']]['DP'] = criteria['sentiment']
           user_data[criteria['role']][criteria['value']]['time'] = time.time()
        
      else:
         #add criteria
         print ("aading criteria")
         user_data[criteria['role']] = {}
         user_data[criteria['role']][criteria['value']] = {}
         user_data[criteria['role']][criteria['value']]['SP'] = criteria['sentiment']
         user_data[criteria['role']][criteria['value']]['DP'] = criteria['sentiment']
         user_data[criteria['role']][criteria['value']]['time'] = time.time()
    with open("../ProfileData/"+str(userId)+".json","w") as f:
      user_data = json.dumps(user_data)
      f.write(user_data)
  '''

  def extractAndAssignPolarity(self,text):
      seggrgated_sentence_list = self.spacySentenceSeggrigator(text)

      #extracting a nouns 
      nltk_tagger = NLTKTagger()

      #use below code which will not analyse everytime and hence makes it faster.
      
      '''
      from textblob import Blobber
      from textblob.sentiments import NaiveBayesAnalyzer
      tb = Blobber(analyzer=NaiveBayesAnalyzer())

      print tb("sentence you want to test")
      '''
     
      criteria_list = []
      for sentence in seggrgated_sentence_list:
        #invoke sentiment analyser
        analysed_sent = TextBlob(sentence)
        sentiment = analysed_sent.sentiment
        print (sentiment)
        #blob = TextBlob(sentence, pos_tagger=nltk_tagger)
        #tokens_list = blob.pos_tags
          
        # logic to idenityfy character starting with @ and considering them as noun
        split_words = sentence.split()
        


        for word in split_words:
          found = False
          noun_sentiment_dict ={}
          if word.startswith("@"):
            noun= word[1:]
            try:
              if self.actor1_dict[noun]:
                noun_sentiment_dict['role'] = 'actor'
                found = True
            except:
              pass
            if not(found):
              try:
                if self.actor2_dict[noun]:
                  noun_sentiment_dict['role'] = 'actor'
                  found = True
              except:
                pass
            if not(found):
              try:
                if self.director_dict[noun]:
                  noun_sentiment_dict['role'] = 'director'
                  found = True
              except:
                pass
            if not(found):
              try:
                if self.movie_dict[noun]:
                  noun_sentiment_dict['role'] = 'movie'
                  found = True
              except:
                pass
            if found:
              noun_sentiment_dict['sentiment'] = sentiment.polarity
              noun_sentiment_dict['value'] = noun
              criteria_list.append(noun_sentiment_dict)
          elif word in self.genre_list:
              noun_sentiment_dict['role'] = 'genre'
              found = True
              noun_sentiment_dict['value'] = word
              noun_sentiment_dict['sentiment'] = sentiment.polarity
              criteria_list.append(noun_sentiment_dict)
      return criteria_list



  def analyseSentiments(self,msg):
    '''
    we need to do pre processing before doing sentimentalAnalhysis like
    1. Spelling correction
    2. grammer correction
    3. Translate to english if in other language
    4. addiung punctuation if required! ( if we add this splitting might be easier)???

    '''

    #sentence correction
    if 'extended_tweet' in msg['tweet_data'] :
        print( "extended_tweet")
        tweet_text = msg['tweet_data']['extended_tweet']['full_text']
        print ("extended_tweet",tweet_text)
    else:
        try:
          tweet_text = msg['tweet_data']['text']
        except:
          return
    userId = msg['tweet_data']["user"]["id_str"]
    #print (type(tweet_text))
    #correctedMsg = TextBlob(tweet_text).correct()
    #print (correctedMsg)

    # correct or add punctuation marks to the sentence so that splitting is correct.
    #punctuated_msg = self.punCtuateSentence(str(correctedMsg))
    #print ("punctuated_msg",punctuated_msg)
    punctuated_msg = str(tweet_text)

    #print (type(punctuated_msg))
    
    #handle retweet case
    noun_sentiment_dict = {}
    criteria_list =[]
    output_data ={}
    if 'quoted_status' in msg['tweet_data']:
      criteria_list = self.computeNounTagsFromrtweetRelation(punctuated_msg,msg,noun_sentiment_dict)
    else:
      criteria_list = self.extractAndAssignPolarity(punctuated_msg.lower())

    print (criteria_list)
    output_data['type'] = 'sentimentsAnalyser'
    output_data['msg'] = criteria_list
    output_data['userid'] = userId


    Producer().produceMessage(json.loads(json.dumps(output_data)),"sentiments")
    #self.searchAndUpdateDynamicProfile(userId,criteria_list)
    '''
      split senetence
      punkt_param.abbrev_types = set(['dr', 'vs', 'mr', 'mrs', 'prof', 'inc'])


      # Textblob sentence tokanizer 
      
      punkt_param = PunktParameters()
      punkt_param.abbrev_types = set(['dr', 'vs', 'mr', 'mrs', 'prof', 'inc'])
      sentence_splitter = PunktSentenceTokenizer(punkt_param)
      sentences = sentence_splitter.tokenize(punctuated_msg)


      #sentence_list = sent_tokenize(correctedMsg)



      #Spacy sentence tokanizer

      seggrgated_sentence_list = self.spacySentenceSeggrigator(punctuated_msg.lower())

      #extracting a nouns 
      nltk_tagger = NLTKTagger()

      #use below code which will not analyse everytime and hence makes it faster.
      
      from textblob import Blobber
      from textblob.sentiments import NaiveBayesAnalyzer
      tb = Blobber(analyzer=NaiveBayesAnalyzer())

      print tb("sentence you want to test")
    
      for sentence in seggrgated_sentence_list:
        #invoke sentiment analyser
        analysed_sent = TextBlob(sentence, analyzer=NaiveBayesAnalyzer())
        sentiment = analysed_sent.sentiment
        #blob = TextBlob(sentence, pos_tagger=nltk_tagger)
        #tokens_list = blob.pos_tags

        
        with open("actor1_handle.json",'r') as a1f:
          data = a1f.read()
          actor1_dict = json.load(data)
        with open("actor2_handle.json",'r') as a2f:
          data = a2f.read()
          actor2_dict = json.load(data)
        with open("director_handle.json",'r') as df:
          data = df.read()
          director_dict = json.load(data)
        with open("movie_handle.json",'r') as mf:
          data = mf.read()
          movie_dict = json.load(data)
          
        # logic to idenityfy character starting with @ and considering them as noun
        split_words = sentence.split()
        criteria_list = []

        print (sentence,tokens_list)
        for word in split_words:
          if word.startswith("@"):
            found = False
            noun= word[1:]
            try:
              if actor1_dict[noun]:
                noun_sentiment_dict['role'] = 'actor'
                found = True
            except:
              pass
            if not(found):
              try:
                if actor2_dict[noun]:
                  noun_sentiment_dict['role'] = 'actor'
                  found = True
              except:
                pass
            if not(found):
              try:
                if director_dict[noun]:
                  noun_sentiment_dict['role'] = 'director'
                  found = True
              except:
                pass
            if not(found):
              try:
                if movie_dict[noun]:
                  noun_sentiment_dict['role'] = 'movie'
                  found = True
              except:
                pass
            noun_sentiment_dict['value'] = noun
          elif word in self.genre_list:
              noun_sentiment_dict['role'] = 'genre'
              found = True
              noun_sentiment_dict['value'] = noun
          if sentiment.classification == 'pos':
              noun_sentiment_dict['sentiment'] = sentiment. p_pos
          else:
               noun_sentiment_dict['sentiment'] = (sentiment. p_neg*(-1))
          noun_sentiment_dict['value'] = word

        for items in tokens_list:
          print (items[0],":",items[1])

          if items[1] in ("NNP" , "NN", "NNS","NNP" , "NNPS"):
            if sentiment.classification == 'pos':
              noun_sentiment_dict[items[0]] = sentiment. p_pos
            else:
               noun_sentiment_dict[items[0]] = (sentiment. p_neg*(-1))
        # iterate over nouns in retweet and assign sentiments to it
        if len(nouns_in_retweet) > 0:
          for nouns in nouns_in_retweet:
            if sentiment.classification == 'pos':
              noun_sentiment_dict[nouns] = sentiment. p_pos
            else:
               noun_sentiment_dict[nouns] = sentiment. p_neg* (-1)

      #self.searchAndUpdateDynamicProfile(userId,noun_sentiment_dict)
    '''
      
