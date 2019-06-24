from pyspark import SparkConf,SparkContext
import csv
conf=SparkConf().setMaster("local").setAppName("PFE")
sc=SparkContext(conf=conf)

lines=sc.textFile("file:////workingdatasets/tweets2016-2019.csv")
champs=lines.map(lambda x:x.split())
results=champs.collect()

tweets = open('preprocessedTweets.csv','a',newline='')
csvWriter = csv.writer(tweets,delimiter=' ')

for result in results:
    if(result[0]!='RT'):
       tweet=" ".join(result)
       sentence=tweet.lower()
       sentence=sentence.split()
       csvWriter.writerow(sentence)
    
tweets.close()   
               
