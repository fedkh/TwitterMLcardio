from TwitterAPI import TwitterAPI
import csv
import json
from textblob import TextBlob
import re
import twitter_credentials

#Cleaning function: enleve les  caractere speciaux , les hashtags'#' , enleve les @username , enleve les liens
def clean_tweet(tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

#L'authentification
api = TwitterAPI(twitter_credentials.CONSUMER_KEY,twitter_credentials.CONSUMER_SECRET,
                 twitter_credentials.ACCESS_TOKEN,twitter_credentials.ACCESS_TOKEN_SECRET)

#********Full-archive endpoint********#
QUERY = "(heart disease OR cardiovascular OR heart failure OR coronary OR ischemic) place_country:US lang:en"
PRODUCT = 'fullarchive'
#L'environement de developement de l'application 
LABEL = 'PFE'
#La requete
r = api.request('tweets/search/%s/:%s' % (PRODUCT, LABEL), 
            {'query':QUERY , 
             'fromDate':'201802060000',
             'toDate':'201802070000'
            }
            )
#Ouverture du fichier Json
jsonFile= open('tweets.json','a') 
#Ouverture du fichier CSV
tweets = open('tweets.csv','a',newline='')
csvWriter = csv.writer(tweets)

#Parcours du resultat obtenu par la requete           
for item in r:

#Teste si le tweet contient 140 caracteres ou plus
  if('extended_tweet' in item):
         text=item['extended_tweet']['full_text']
  else: 
        text=item['text'] 
#Le cleaning des tweets (le fichier CSV n'accepte pas les caracteres speciaux)  
  cleanedTweets=TextBlob(clean_tweet(text))
#Ajout des cleaned tweets dans le fichier csv
  csvWriter.writerow([cleanedTweets])

#Ajout des tweets dans le fichier Json (tous les champs)
  json.dump(item,jsonFile)
  
#Fermeture des fichier Json et csv 
jsonFile.close() 
tweets.close()



