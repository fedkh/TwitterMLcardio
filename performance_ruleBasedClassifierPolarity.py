
from pyspark import SparkConf,SparkContext
import sklearn
from sklearn.metrics import accuracy_score

conf=SparkConf().setMaster("local").setAppName("PFE")
sc=SparkContext(conf=conf)

#Creation of polarity corpus using senticnet5
lines=sc.textFile("file:////workingdatasets/senticnet5.txt")
champs=lines.map(lambda x:x.split())
results=champs.collect()

words=[]
labels=[]
for result in results[1:len(results)-1]:
    words.append(result[0])
    labels.append(result[1])
    
keys =words
values =labels
dictionary = dict(zip(keys, values))

def check_label(word):
    label=''
    for key,value in dictionary.items():
    
        if(word==key):
            label=value
    return label

    
tweetsFile=sc.textFile("file:////workingdatasets/label_personalTweets.csv")
tweets=tweetsFile.map(lambda x:x.split())
resultats=tweets.collect()


classifier_input=[]
manual_labels=[]

for tweet in resultats :
    
    #words=tweet.split(':')
    w=tweet[0].split(':')
    if(w[0]=='n'):
        manual_labels.append('negative')  
    else:
        manual_labels.append('positive')
      
    tweet[0]=w[1]
    input_tweet=" ".join(tweet)
    classifier_input.append(input_tweet)


labels_classifier=[]        
for tweet in classifier_input :
    
    nbrNeg=0
    nbrPos=0
    words=tweet.split(' ')
     
    for word in words:
           
          for key,value in dictionary.items():
            if(word==key):
                label=check_label(word)
                if(label=='negative'):
                    nbrNeg=nbrNeg+1
                if(label=='positive'):
                    nbrPos=nbrPos+1

    if(nbrNeg>3):
        labels_classifier.append('negative')
        
    else:
        labels_classifier.append('positive')

accuracy=sklearn.metrics.accuracy_score(manual_labels, labels_classifier) 
        

print(accuracy)
