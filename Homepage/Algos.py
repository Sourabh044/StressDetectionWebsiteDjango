import pandas as pd
import numpy as np
url = 'https://raw.githubusercontent.com/Sourabh044/StressDetectonUsingText/master/Processed%20Data/preprocessedNP.csv'
data = pd.read_csv(url)
data.columns = ['text', 'label']

#data.drop('useless',axis=1,inplace=True)
# Cleaning
import nltk
import re
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword=set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
data["text"] = data["text"].apply(clean)

data["label"] = data["label"].map({0: "No Stress", 1: "Stress"})
data = data[["text", "label"]]

# train test split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

x = np.array(data["text"])
y = np.array(data["label"])

cv = CountVectorizer()
X = cv.fit_transform(x)
xtrain, xtest, ytrain, ytest = train_test_split(X, y,test_size=0.33,random_state=42)

# naive bayes
def naive(inputext):
    from sklearn.naive_bayes import BernoulliNB
    model = BernoulliNB()
    model.fit(xtrain, ytrain)
    data = cv.transform([inputext]).toarray()
    output = model.predict(data)
    return output[0]
#  Logistic 
def logistict(inputext):
    from sklearn.linear_model import LogisticRegression
    model2 = LogisticRegression()
    model2.fit(xtrain,ytrain)
    data = cv.transform([inputext]).toarray()
    output = model2.predict(data)
    return output[0]
# DecisionTree
def decisionTree(inputext):
    from sklearn.tree import DecisionTreeClassifier
    model3 = DecisionTreeClassifier()
    model3.fit(xtrain,ytrain)
    data = cv.transform([inputext]).toarray()
    output = model3.predict(data)
    return output[0]
# KNN
def knn(inputext):
    from sklearn.neighbors import KNeighborsClassifier
    model4= KNeighborsClassifier()
    model4.fit(xtrain,ytrain)
    data = cv.transform([inputext]).toarray()
    output = model4.predict(data)
    return output[0]
# SVM
def svm(inputext):
    from sklearn.svm import SVC
    model5 = SVC(C=2)
    model5.fit(xtrain,ytrain)
    data = cv.transform([inputext]).toarray()
    output = model5.predict(data)
    return output[0]
# Randomforest
def rf(inputext):
    from sklearn.ensemble import RandomForestClassifier
    model6 = RandomForestClassifier()
    model6.fit(xtrain,ytrain)
    data = cv.transform([inputext]).toarray()
    output = model6.predict(data)
    return output[0]
