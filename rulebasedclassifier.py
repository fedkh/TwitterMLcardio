from pyspark import SparkConf,SparkContext
import csv
conf=SparkConf().setMaster("local").setAppName("PFE")
sc=SparkContext(conf=conf)

# generate labeled dataset
lines=sc.textFile("subjectivity.tff")

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

tweetsFile=sc.textFile("preprocessedTweets.csv")
tweets=tweetsFile.map(lambda x:x.split())
resultats=tweets.collect()

subjectiveTweets = open('personalTweets.csv','a',newline='')
objectiveTweets = open('nonPersonalTweets.csv','a',newline='')
csvWriter1 = csv.writer(subjectiveTweets,delimiter=' ')
csvWriter2 = csv.writer(objectiveTweets,delimiter=' ')


for tweet in resultats :
    nbrSubj=0
    nbrObj=0
    tweets=" ".join(tweet)
    
    for word in tweet:
           for key,value in dictionary.items():
            if(word==key):
                label=check_label(word)
                if(label=='strongsubj'):
                    nbrSubj=nbrSubj+1
                if(label=='weaksubj'):
                    nbrObj=nbrObj+1

    if(nbrSubj>=2):
        sentence=tweets.split()
        csvWriter1.writerow(sentence)
        
    else:
        sentence=tweets.split()
        csvWriter2.writerow(sentence)
 
subjectiveTweets.close()
objectiveTweets.close()