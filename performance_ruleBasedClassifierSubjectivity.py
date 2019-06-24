
from pyspark import SparkConf,SparkContext
import sklearn
from sklearn.metrics import accuracy_score

conf=SparkConf().setMaster("local").setAppName("PFE")
sc=SparkContext(conf=conf)

# generate labeled dataset
lines=sc.textFile("file:////workingdatasets/subjectivity.tff")

champs=lines.map(lambda x:x.split(' '))
results=champs.collect()
labels=[]
words=[]
for result in results:
    champ0=result[0]
    len1=len(champ0)
    label=champ0[5:len1]
    labels.append(label)

    champ2=result[2]
    len2=len(champ2)
    word=champ2[6:len2]
    words.append(word)

keys =words
values =labels
dictionary = dict(zip(keys, values))

def check_label(word):
    label=''
    for key,value in dictionary.items():
    
        if(word==key):
            label=value
    return label

tweetsFile=sc.textFile("file:////workingdatasets/manualLabeling_preprocessedTweets.csv")
tweets=tweetsFile.map(lambda x:x.split())
resultats=tweets.collect()


classifier_input=[]
manual_labels=[]

for tweet in resultats :
    
    #words=tweet.split(':')
    w=tweet[0].split(':')
    if(w[0]=='s'):
        manual_labels.append('subjective')  
    else:
        manual_labels.append('objective')
      
    tweet[0]=w[1]
    input_tweet=" ".join(tweet)
    classifier_input.append(input_tweet)



classifier_labels=[]
for tweet in classifier_input :
    nbrSubj=0
    nbrObj=0
    words=tweet.split(' ') 
    for word in words:
           
           for key,value in dictionary.items():
            if(word==key):
                label=check_label(word)
                if(label=='strongsubj'):
                    nbrSubj=nbrSubj+1
                if(label=='weaksubj'):
                    nbrObj=nbrObj+1

    if(nbrObj<=1):
        classifier_labels.append('subjective')
        
        
    else:
        classifier_labels.append('objective')
 

accuracy=sklearn.metrics.accuracy_score(manual_labels, classifier_labels) 
        

print(accuracy)

