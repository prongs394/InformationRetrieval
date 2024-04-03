import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.svm import SVC

#first SandersPosNeg
def load_dataset(filename, cols):
    dataset = pd.read_csv(filename, encoding='latin-1')
    dataset.columns = cols
    return dataset
dataset = load_dataset("SandersPosNeg.csv", ['line'])
arraydata = np.array(dataset['line'])

#make two arrays on for classes of tweets = classoftweet, one for the tweet itsefl = tweet
classoftweet=[]
tweet=[]
for i in range(len(arraydata)):
    x = arraydata[i][0]   # 0 or 4
    classoftweet.append(x)

    #now add the tweet:
    y = arraydata[i]
    z = y[1:len(y)]
    tweet.append(z)

dataa = []   #data is like this: each array has two values in it: class of tweet and the tweet itself
for i in range(len(tweet)):
    x = classoftweet[i]
    y = tweet[i]
    a=[]
    a.append(x)
    a.append(y)
    dataa.append(a)

#dataa is list. we want to convert it to pandas dataframe
df = pd.DataFrame(dataa, columns=['target', 'text'])   #target is 0 or 4 and text is the tweet

#now we can work with this dataset


#preprocessing:
stop_words = set(stopwords.words('english'))   #some words like in,and,or,... are not needed

def preprocess(tweet):

    # delete links:
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)

    # lorwercasing letters:
    tweet.lower()

    # delete nametags:
    tweet = re.sub(r'\@\w+', '', tweet)

    # delete stopwords:
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [w for w in tweet_tokens if not w in stop_words]

    result = " ".join(filtered_words)
    return result

# get feature vector for tweets with tf-idf:
def get_feature_vector(train_fit):
    vector = TfidfVectorizer(sublinear_tf=True)
    vector.fit(train_fit)
    return vector

df.text = df['text'].apply(preprocess)  #text means tweet. so here we'll do preorocessing on the tweets
tf_vector = get_feature_vector(np.array(df.iloc[:, 1]).ravel())  #getting features from tweets

# second column: tweet
X = tf_vector.transform(np.array(df.iloc[:, 1]).ravel())

# first column: 0 and 4
y = np.array(df.iloc[:, 0]).ravel()

#cross validation into 10 groups:
# shuffle data (because in our data set first half of data is in class 0 and the second half in class 4)
#devide data to 10 parts and classify it 10 times and everytime taking a different part(one of those 10) as test and rest(9) as train
cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
SVC_model = SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(SVC_model, X, y, cv=cv)

print("SandersPosNeg:", scores, sep='\t')
x=0
for i in range (len(scores)):
    x = x+ scores[i]
x = x/10
print(x)



#____________________________________________________________________
#now for OMD.csv:
import csv
file = open("OMD.csv")
csvreader = csv.reader(file)
header = next(csvreader)
rows = []
for row in csvreader:
    rows.append(row)
#print(rows)
file.close()
#now in rows arrays we have some arrays which each arrays contains one line

dat=[]
for i in range(len(rows)):   #want to make dat an array like: array[i][0] = class of tweet i and array[i][1] = the i'th tweet(the text).
    x=[]
    x.append(rows[i][0])
    y = ''
    for j in range(1, len(rows[i])):
        y = y + rows[i][j]
    x.append(y)
    dat.append(x)
#now dat is a 2d array: it has 1905 smaller arrays and each small array has 2 elements: first one is the class of the tweet and second is the tweet itself.

#now we will make it into pandas data frame:
df = pd.DataFrame(dat, columns=['target', 'text'])

#and now we can do the exact same thing we did on SandersPosNeg.csv
df.text = df['text'].apply(preprocess)
tf_vector = get_feature_vector(np.array(df.iloc[:, 1]).ravel())

# second column : tweet
X = tf_vector.transform(np.array(df.iloc[:, 1]).ravel())
# first column : 0 and 4
y = np.array(df.iloc[:, 0]).ravel()
# shuffle data (because in our data set first half of data is in class 0 and the second half in class 4)
#devide data to 10 parts and classify it 10 times and everytime taking a different part(one of those 10) as test and rest(9) as train
cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
SVC_model = SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(SVC_model, X, y, cv=cv)

#SVM for OMD:
print("OMD:", scores, sep='\t')
x=0
for i in range (len(scores)):
    x = x+ scores[i]
x = x/10
print(x)

