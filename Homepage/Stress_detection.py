#!/usr/bin/env python
# coding: utf-8

# ***Hello World***

# In[249]:


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
url = 'https://raw.githubusercontent.com/Sourabh044/StressDetectonUsingText/master/Processed%20Data/preprocessedNP.csv'
data = pd.read_csv(url)
data.columns = ['text', 'label']


# In[256]:


data.head(5)


# In[251]:


print(data.isnull().sum())


# In[252]:


import nltk
import re
# nltk.download('stopwords')
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


# ***Now, Labeling the the Text, [Stress,No Stress]***

# In[253]:


data["label"] = data["label"].map({0: "No Stress", 1: "Stress"})
data = data[["text", "label"]]
print(data.head())


# In[254]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

x = np.array(data["text"])
y = np.array(data["label"])

cv = CountVectorizer()
X = cv.fit_transform(x)
xtrain, xtest, ytrain, ytest = train_test_split(X, y,test_size=0.33,random_state=42)


# In[142]:


from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
model = BernoulliNB()
model.fit(xtrain, ytrain)
# print("Accuracy of Logistic Regression model is:",
# metrics.accuracy_score(ytest, ytrain)*100)


# In[143]:


user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = model.predict(data)
print(output)


# In[175]:


scores = []
scores.append(model.score(xtest,ytest))
# dict = {'Algorithm':'Naive Bayes','Score':model.score(xtest,ytest)}
# scoresdf  = scoresdf.append(dict, ignore_index = True)
model.score(xtest,ytest)*100


# In[145]:


from sklearn.metrics import classification_report
pred_bern = model.predict(xtest)
print(classification_report(ytest, pred_bern))


# In[146]:


from sklearn.metrics import confusion_matrix
confusion_matrix(ytest,pred_bern)


# In[147]:


from sklearn.linear_model import LogisticRegression
model2 = LogisticRegression()
model2.fit(xtrain,ytrain)


# In[148]:


scores.append(model2.score(xtest,ytest))
model2.score(xtest,ytest)*100
dict = {'Algorithm':'LogisticRegression','Score':model2.score(xtest,ytest)}
scoresdf  = scoresdf.append(dict, ignore_index = True)


# In[149]:


user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = model2.predict(data)
print(output)


# In[150]:


from sklearn.metrics import confusion_matrix
pred_logis = model2.predict(xtest)
confusion_matrix(ytest,pred_logis)


# In[151]:


from sklearn.tree import DecisionTreeClassifier
model3 = DecisionTreeClassifier()
model3.fit(xtrain,ytrain)


# In[152]:


scores.append(model3.score(xtest,ytest))
model3.score(xtest,ytest)*100


# In[153]:


user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = model3.predict(data)
print(output)


# In[154]:


pred_decs = model3.predict(xtest)
from sklearn.metrics import confusion_matrix
confusion_matrix(ytest,pred_decs)


# In[155]:


plt.figure(figsize=(10,10))
sn.heatmap(confusion_matrix(ytest,pred_decs),annot=True)
plt.xlabel('prediction')
plt.ylabel('truth')


# In[156]:


from sklearn.neighbors import KNeighborsClassifier
model4= KNeighborsClassifier()
model4.fit(xtrain,ytrain)


# In[157]:


scores.append(model4.score(xtest,ytest))
model4.score(xtest,ytest)*100


# In[158]:


user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = model4.predict(data)
print(output)


# In[159]:


pred_knn = model4.predict(xtest)


# In[160]:


from sklearn.metrics import confusion_matrix
confusion_matrix(ytest,pred_knn)


# In[161]:


from sklearn.svm import SVC
model5 = SVC(C=2)
model5.fit(xtrain,ytrain)


# In[162]:


model5.score(xtest,ytest)*100


# In[163]:


user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = model5.predict(data)
print(output)


# In[164]:


pred_svm=model5.predict(xtest)


# In[165]:


from sklearn.metrics import confusion_matrix
confusion_matrix(ytest,pred_svm)


# In[166]:


from sklearn.ensemble import RandomForestClassifier
model6 = RandomForestClassifier()
model6.fit(xtrain,ytrain)


# In[167]:


model6.score(xtest,ytest)


# In[168]:


pred_rand = model6.predict(xtest)


# In[169]:


user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = model6.predict(data)
print(output)


# In[170]:


from sklearn.metrics import confusion_matrix
confusion_matrix(ytest,pred_rand)


# In[196]:


dict = {'Algo':['NaiveBayes','LogisticRegression','DecisionTree','KNN','SVC','RandomForest'],
        'Score':[model.score(xtest,ytest),model2.score(xtest,ytest),model3.score(xtest,ytest),model4.score(xtest,ytest),
                 model5.score(xtest,ytest),model6.score(xtest,ytest)]}
scoresdf = pd.DataFrame.from_dict(dict)


# In[200]:


scoresdf


# In[248]:


plt.figure(figsize=(8, 8), dpi=80)
plt.scatter(scoresdf['Algo'],scoresdf['Score'],s=500)
plt.title("Scores 0f Different Algos")
plt.show()


# In[246]:


x,y = scoresdf['Algo'],scoresdf['Score']
f, ax = plt.subplots(figsize=(18,5))
# y = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.1])
plt.barh(x, y)
plt.ylabel("Algo")
plt.xlabel('Scores')
plt.title("Scores 0f Different Algos")
plt.show()


# In[ ]:





# In[ ]:




