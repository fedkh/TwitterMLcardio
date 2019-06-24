# -*- coding: utf-8 -*-

from pyspark import SparkConf,SparkContext
import pandas 
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm,ensemble
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

conf=SparkConf().setMaster("local").setAppName("PFE")
sc=SparkContext(conf=conf)


lines=sc.textFile("personalTweets.csv")
subjLines=lines.map(lambda x:x.split())
personalTweets=subjLines.collect()

labels=[]
tweets=[]
for tweet in personalTweets:
    tweet_str=" ".join(tweet)
    tweets.append(tweet_str)
    label='personal'
    labels.append(label)

lines=sc.textFile("nonPersonalTweets.csv")
objLines=lines.map(lambda x:x.split())
nonpersonalTweets=objLines.collect()

for tweet in nonpersonalTweets:
    tweet_str=" ".join(tweet)
    tweets.append(tweet_str)
    label='nonpersonal'
    labels.append(label)
    

# create a dataframe using tweets and lables
trainDF = pandas.DataFrame()
trainDF['tweets'] = tweets
trainDF['labels'] = labels

# split the dataset into training and test datasets 
train_x, test_x, train_y, test_y = model_selection.train_test_split(trainDF['tweets'], trainDF['labels'],train_size=0.8, test_size=0.2 ,stratify=trainDF['labels'])

#encode test_y and train_y
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
test_y = encoder.fit_transform(test_y)

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=280)
tfidf_vect.fit(trainDF['tweets'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xtest_tfidf =  tfidf_vect.transform(test_x)

# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(trainDF['tweets'])
# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xtest_count =  count_vect.transform(test_x)


#recuperer le code de la classe personal
codeSubj=0
for i,item in enumerate(encoder.classes_):
    if(item=='personal'):
        codeSubj=i

def train_model(classifier, feature_vector_train, label, feature_vector_test):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    # predict the labels on test dataset
    predictions = classifier.predict(feature_vector_test)
    
    # associate each tweet in the test dataset to its predicted class       
    keys = test_x.values
    values =predictions
    dictionary = dict(zip(keys, values))
    
    predictedPersonaltweets=[]
    for cle, valeur in dictionary.items():
        if(valeur==codeSubj):
            predictedPersonaltweets.append(cle)
   
    return metrics.accuracy_score(predictions, test_y),predictedPersonaltweets

# Naive Bayes on Word Level TF IDF Vectors
accuracy,mnbPersonalTweets = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xtest_tfidf)
print ("NB, WordLevel TF-IDF: ", accuracy)

accuracy,mnbPersonalTweets = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xtest_count)
print ("NB, countVectorizer : ", accuracy)

# RF on Word Level TF IDF Vectors
accuracy,rfPersonalTweets = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf, train_y, xtest_tfidf)
print ("RF, WordLevel TF-IDF: ", accuracy)
#print(rfPersonalTweets)
accuracy,rfPersonalTweets = train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y, xtest_count)
print ("RF, countVectorizer : ", accuracy)

# SVM on Word Level TF IDF Vectors
accuracy,svmPersonalTweets = train_model(svm.SVC(C=1.0, kernel='linear'), xtrain_tfidf, train_y, xtest_tfidf)
print ("SVM, WordLevel TF-IDF: ", accuracy)

accuracy,svmPersonalTweets = train_model(svm.SVC(C=1.0, kernel='linear'), xtrain_count, train_y, xtest_count)
print ("SVM, countVectorizer : ", accuracy)

# Linear Classifier on Word Level TF IDF Vectors (logistic regression)
accuracy,lrPersonalTweets = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xtest_tfidf)
print ("LR, WordLevel TF-IDF: ", accuracy)
#print(lrPersonalTweets)

accuracy,lrPersonalTweets = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xtest_count)
print ("LR, countVectorizer : ", accuracy)
###################Polarity##############################


#Enregistrer les tweets de training et de test avec leur classe encodé
train_tweets=train_x.values
train_labels=train_y

trainingPersonalTweets=[]
keys =train_tweets
values =train_labels
dictionary = dict(zip(keys, values))

#Garder les tweets personel du training dataset 
for cle, valeur in dictionary.items():
    if(valeur==codeSubj):
        trainingPersonalTweets.append(cle)

#Creation of polarity corpus using senticnet5
lines=sc.textFile("senticnet5.txt")
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
    

#polarity labeling of training dataset
labelTraining=[]        
      
for tweet in trainingPersonalTweets :
    
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
        labelTraining.append('negative')
        
    else:
        labelTraining.append('positive')

#polarity labeling of test dataset (svm)
labelsvmPersonalTweets=[]        
for tweet in svmPersonalTweets :
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
        labelsvmPersonalTweets.append('negative')
        
    else:
        labelsvmPersonalTweets.append('positive')

#polarity labeling of test dataset (multinomial naive bayes)
labelmnbPersonalTweets=[]        
for tweet in mnbPersonalTweets :
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
        labelmnbPersonalTweets.append('negative')
        
    else:
        labelmnbPersonalTweets.append('positive')


    
#polarity labeling of test dataset (Random Forest)
labelrfPersonalTweets=[]        
for tweet in rfPersonalTweets :
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
        labelrfPersonalTweets.append('negative')
        
    else:
        labelrfPersonalTweets.append('positive')




#polarity labeling of test dataset (Logistic Regression)
labellrPersonalTweets=[]        
for tweet in lrPersonalTweets :
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
        labellrPersonalTweets.append('negative')
        
    else:
        labellrPersonalTweets.append('positive')


############ML classifier(polarity)###########
#Pour SVM
DF=pandas.DataFrame()
DF['tweets']=trainingPersonalTweets+svmPersonalTweets
DF['labels']=labelTraining+labelsvmPersonalTweets

#encode test_y and train_y
encoder = preprocessing.LabelEncoder()
train_yP = encoder.fit_transform(labelTraining)
test_yP = encoder.fit_transform(labelsvmPersonalTweets)


# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word',token_pattern=r'\w{1,}',max_features=280)
tfidf_vect.fit(DF['tweets'])
xtrain_tfidf =  tfidf_vect.transform(trainingPersonalTweets)
xtest_tfidf =  tfidf_vect.transform(svmPersonalTweets)

# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(DF['tweets'])
# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(trainingPersonalTweets)
xtest_count =  count_vect.transform(svmPersonalTweets)

#recuperer le code de la classe Negative
codeNeg=0
for i,item in enumerate(encoder.classes_):
    if(item=='negative'):
        codeNeg=i
    
def train_model_polarity(classifier, feature_vector_train, label, feature_vector_test):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    # predict the labels on test dataset
    predictions = classifier.predict(feature_vector_test)
    
    # associate each tweet in the test dataset to its predicted class       
    keys = test_x.values
    values =predictions
    dictionary = dict(zip(keys, values))
    
    predictedPersonalNegativetweets=[]
    for cle, valeur in dictionary.items():
        if(valeur==codeNeg):
            predictedPersonalNegativetweets.append(cle)
   
    return metrics.accuracy_score(predictions, test_yP),predictedPersonalNegativetweets


#SVM on Word Level TF IDF Vectors
accuracy,svmNegativeTweets = train_model_polarity(svm.SVC(C=1.0, kernel='linear'), xtrain_tfidf, train_yP, xtest_tfidf)
print ("SVM, TF idf : ", accuracy)
#print(svmNegativeTweets)
accuracy,svmNegativeTweets = train_model_polarity(svm.SVC(C=1.0, kernel='linear'), xtrain_count, train_yP, xtest_count)
print ("svm, countVectorizer : ", accuracy)


########Pour Multinomial Naive Bayes
DF=pandas.DataFrame()
DF['tweets']=trainingPersonalTweets+mnbPersonalTweets
DF['labels']=labelTraining+labelmnbPersonalTweets

#encode test_y and train_y
encoder = preprocessing.LabelEncoder()
train_yP = encoder.fit_transform(labelTraining)
test_yP = encoder.fit_transform(labelmnbPersonalTweets)

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', max_features=280)
tfidf_vect.fit(DF['tweets'])
xtrain_tfidf =  tfidf_vect.transform(trainingPersonalTweets)
xtest_tfidf =  tfidf_vect.transform(mnbPersonalTweets)


# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(DF['tweets'])
# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(trainingPersonalTweets)
xtest_count =  count_vect.transform(mnbPersonalTweets)

#recuperer le code de la classe personal
codeNegative=0
n=0
for i,item in enumerate(encoder.classes_):
    if(item=='negative'):
        codeNegative=i
def train_model_polarity(classifier, feature_vector_train, label, feature_vector_test):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    # predict the labels on test dataset
    predictions = classifier.predict(feature_vector_test)
    
    # associate each tweet in the test dataset to its predicted class       
    keys = test_x.values
    values =predictions
    dictionary = dict(zip(keys, values))
    
    predictedPersonalNegativetweets=[]
    for cle, valeur in dictionary.items():
        if(valeur==codeNeg):
            predictedPersonalNegativetweets.append(cle)
   
    return metrics.accuracy_score(predictions, test_yP),predictedPersonalNegativetweets
# Naive Bayes on Word Level TF IDF Vectors
accuracy,mnbNegativeTweets = train_model_polarity(naive_bayes.MultinomialNB(), xtrain_tfidf, train_yP, xtest_tfidf)
print ("NB, WordLevel TF-IDF: ", accuracy)

accuracy,mnbNegativeTweets = train_model_polarity(naive_bayes.MultinomialNB(), xtrain_count, train_yP, xtest_count)
print ("NB, countVectorizer : ", accuracy)

#Pour Random Forest

DF=pandas.DataFrame()
DF['tweets']=trainingPersonalTweets+rfPersonalTweets
DF['labels']=labelTraining+labelrfPersonalTweets

#encode test_y and train_y
encoder = preprocessing.LabelEncoder()
train_yP = encoder.fit_transform(labelTraining)
test_yP = encoder.fit_transform(labelrfPersonalTweets)

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', max_features=280)
tfidf_vect.fit(DF['tweets'])
xtrain_tfidf =  tfidf_vect.transform(trainingPersonalTweets)
xtest_tfidf =  tfidf_vect.transform(rfPersonalTweets)

# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(DF['tweets'])
# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(trainingPersonalTweets)
xtest_count =  count_vect.transform(rfPersonalTweets)

codeNeg=0
for i,item in enumerate(encoder.classes_):
    if(item=='negative'):
        codeNeg=i
    
def train_model_polarity(classifier, feature_vector_train, label, feature_vector_test):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    # predict the labels on test dataset
    predictions = classifier.predict(feature_vector_test)
    
    # associate each tweet in the test dataset to its predicted class       
    keys = test_x.values
    values =predictions
    dictionary = dict(zip(keys, values))
    
    predictedPersonalNegativetweets=[]
    for cle, valeur in dictionary.items():
        if(valeur==codeNeg):
            predictedPersonalNegativetweets.append(cle)
   
    return metrics.accuracy_score(predictions, test_yP),predictedPersonalNegativetweets   
# RF on Word Level TF IDF Vectors
accuracy,rfNegativeTweets = train_model_polarity(ensemble.RandomForestClassifier(), xtrain_tfidf, train_yP, xtest_tfidf)
print ("RF, WordLevel TF-IDF: ", accuracy)
#print(rfNegativeTweets)

accuracy,rfNegativeTweets = train_model_polarity(ensemble.RandomForestClassifier(), xtrain_count, train_yP, xtest_count)
print ("RF, countVectorizer : ", accuracy)


##########"Pour logistic regression
DF=pandas.DataFrame()
DF['tweets']=trainingPersonalTweets+lrPersonalTweets
DF['labels']=labelTraining+labellrPersonalTweets

#encode test_y and train_y
encoder = preprocessing.LabelEncoder()
train_yP = encoder.fit_transform(labelTraining)
test_yP = encoder.fit_transform(labellrPersonalTweets)

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', max_features=280)
tfidf_vect.fit(DF['tweets'])
xtrain_tfidf =  tfidf_vect.transform(trainingPersonalTweets)
xtest_tfidf =  tfidf_vect.transform(lrPersonalTweets)

# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(DF['tweets'])
# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(trainingPersonalTweets)
xtest_count =  count_vect.transform(lrPersonalTweets)

codeNeg=0
for i,item in enumerate(encoder.classes_):
    if(item=='negative'):
        codeNeg=i
def train_model_polarity(classifier, feature_vector_train, label, feature_vector_test):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    # predict the labels on test dataset
    predictions = classifier.predict(feature_vector_test)
    
    # associate each tweet in the test dataset to its predicted class       
    keys = test_x.values
    values =predictions
    dictionary = dict(zip(keys, values))
    
    predictedPersonalNegativetweets=[]
    for cle, valeur in dictionary.items():
        if(valeur==codeNeg):
            predictedPersonalNegativetweets.append(cle)
   
    return metrics.accuracy_score(predictions, test_yP),predictedPersonalNegativetweets
# Linear Classifier on Word Level TF IDF Vectors (logistic regression)
accuracy,lrNegativeTweets = train_model_polarity(linear_model.LogisticRegression(), xtrain_tfidf, train_yP, xtest_tfidf)
print ("LR, WordLevel TF-IDF: ", accuracy)
#print(lrNegativeTweets)

accuracy,lrNegativeTweets = train_model_polarity(linear_model.LogisticRegression(), xtrain_count, train_yP, xtest_count)
print ("LR, countVectorizer : ", accuracy)


########Le word count

fichier = open("wordCountAllTweets.txt", "a")
rdd=sc.textFile("preprocessedTweets.csv")
words = rdd.flatMap(lambda x:x.split())
wordCounts=words.countByValue()
for word,count in wordCounts.items():
	#mot=word+" "+str(count)
    mot=word
    mot=mot+" "
    mot=mot+str(count)
    mot=mot+'\n'
    fichier.write(mot)
fichier.close()



fichier = open("WordCountsvmNegativeTweets.txt", "a")
rdd=sc.parallelize(svmNegativeTweets)
words = rdd.flatMap(lambda x:x.split())
wordCounts=words.countByValue()
for word,count in wordCounts.items():
	#mot=word+" "+str(count)
    mot=word
    mot=mot+" "
    mot=mot+str(count)
    mot=mot+'\n'
    fichier.write(mot)
fichier.close()

