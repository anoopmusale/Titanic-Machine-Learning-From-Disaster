# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 20:39:20 2018

@author: Anoop
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#Data Preprocessing
def get_title(string):
    import re
    regex = re.compile(r'Mr|Don|Major|Capt|Jonkheer|Rev|Col|Dr|Mrs|Countess|Dona|Mme|Ms|Miss|Mlle|Master', re.IGNORECASE)
    results = regex.search(string)
    if results != None:
        return(results.group().lower())
    else:
        return(str(np.nan))
        
title_dictionary = {
    "capt":"Officer", 
    "col":"Officer", 
    "major":"Officer", 
    "dr":"Officer",
    "jonkheer":"Royalty",
    "rev":"Officer",
    "countess":"Royalty",
    "dona":"Royalty",
    "lady":"Royalty",
    "don":"Royalty",
    "mr":"Mr",
    "mme":"Mrs",
    "ms":"Mrs",
    "mrs":"Mrs",
    "miss":"Miss",
    "mlle":"Miss",
    "master":"Master",
    "nan":"Mr"
}


train['Title'] = train['Name'].apply(get_title)
test['Title'] = test['Name'].apply(get_title)
train['Title'] = train['Title'].map(title_dictionary)
test['Title'] = test['Title'].map(title_dictionary)

mean_title = train.groupby('Title')['Age'].mean()
mean_test = test.groupby('Title')['Age'].mean()
title_list = ['Mr','Miss','Mrs','Master', 'Royalty', 'Officer']


def age_replace(means, data, title_list):
    for title in title_list:  
        temp = data['Title'] == title 
        data.loc[temp, 'Age'] = data.loc[temp, 'Age'].fillna(means[title])
        

age_replace(mean_title, train, title_list)
age_replace(mean_test, test, title_list)

train.drop('Ticket', 1, inplace=True)
test.drop('Ticket', 1, inplace=True)
train.drop('Name', 1, inplace=True)
test.drop('Name', 1, inplace=True)
train.drop('Cabin', 1, inplace=True)
test.drop('Cabin', 1, inplace=True)
train.drop('Title', 1, inplace=True)
test.drop('Title', 1, inplace=True)

train['Embarked'].fillna('S', inplace=True)
test['Embarked'].fillna('S', inplace=True)
test['Fare'].fillna(test['Fare'].mean(), inplace=True)

from sklearn.preprocessing import MinMaxScaler
numericals_list = ['Age','Fare']
for column in numericals_list:
    sc = MinMaxScaler(feature_range=(0, 1))
    sc.fit(train[column].values.reshape(-1,1))
    train[column] = sc.transform(train[column].values.reshape(-1,1))
    test[column] = sc.transform(test[column].values.reshape(-1,1))
    
from sklearn.preprocessing import LabelEncoder
categorical_classes_list = ['Sex','Embarked']

encoding_list = []
for column in categorical_classes_list:
    le = LabelEncoder()
    le.fit(train[column])
    encoding_list.append(train[column].unique())
    encoding_list.append(list(le.transform(train[column].unique())))
    train[column] = le.transform(train[column])
    test[column] = le.transform(test[column])
    
train = pd.get_dummies(train, columns=['Embarked'])
test = pd.get_dummies(test, columns=['Embarked'])

finalTest = pd.read_csv('gender_submission.csv')

X_train = train.iloc[:, [2,3,4,5,6,7,8,9,10]].values
y_train = train.iloc[:, 1].values
X_test = test.iloc[:, [1,2,3,4,5,6,7,8,9]].values
y_FinalTest = finalTest.iloc[:, 1].values

'''
*******************************************************************************
                                DECISION TREE
*******************************************************************************
'''
# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifierDecisionTree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifierDecisionTree.fit(X_train, y_train)

# Predicting the Test set results
y_predDecisionTree = classifierDecisionTree.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_DecisionTree = confusion_matrix(y_FinalTest, y_predDecisionTree)

from sklearn.metrics import accuracy_score
ac_DecisionTree = accuracy_score(y_FinalTest, y_predDecisionTree)

'''
*******************************************************************************
                            LOGISTIC REGRESSION
*******************************************************************************
'''
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifierLogisticRegression = LogisticRegression(random_state = 0, solver = 'lbfgs')
classifierLogisticRegression.fit(X_train, y_train)

# Predicting the Test set results
y_predLogisticRegression = classifierLogisticRegression.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_LogisticRegression = confusion_matrix(y_FinalTest, y_predLogisticRegression)

from sklearn.metrics import accuracy_score
ac_LogisticRegression = accuracy_score(y_FinalTest, y_predLogisticRegression)

'''
*******************************************************************************
                            RANDOM FOREST
*******************************************************************************
'''
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifierRandomForest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifierRandomForest.fit(X_train, y_train)

# Predicting the Test set results
y_predRandomForest = classifierRandomForest.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_RandomForest = confusion_matrix(y_FinalTest, y_predRandomForest)


from sklearn.metrics import accuracy_score
ac_RandomForest = accuracy_score(y_FinalTest, y_predRandomForest)
'''
*******************************************************************************
                            Naive Bayes
*******************************************************************************
'''
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifierNaiveBayes = GaussianNB()
classifierNaiveBayes.fit(X_train, y_train)

# Predicting the Test set results
y_predNaiveBayes = classifierNaiveBayes.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_NaiveBayes = confusion_matrix(y_FinalTest, y_predNaiveBayes)

from sklearn.metrics import accuracy_score
ac_NaiveBayes = accuracy_score(y_FinalTest, y_predNaiveBayes)

'''
*******************************************************************************
                        SUPPORT VECTOR MACHINE
*******************************************************************************
'''

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifierSVM = SVC(kernel = 'poly', random_state = 0)
classifierSVM.fit(X_train, y_train)


# Predicting the Test set results
y_predSVM = classifierSVM.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_SVM = confusion_matrix(y_FinalTest, y_predSVM)

from sklearn.metrics import accuracy_score
ac_SVM = accuracy_score(y_FinalTest, y_predSVM)

'''
*******************************************************************************
                                KNN
*******************************************************************************
'''

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifierKNN = KNeighborsClassifier(n_neighbors = 10, metric = 'minkowski', p = 2)
classifierKNN.fit(X_train, y_train)

# Predicting the Test set results
y_predKNN = classifierKNN.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_KNN = confusion_matrix(y_FinalTest, y_predKNN)


from sklearn.metrics import accuracy_score
ac_KNN = accuracy_score(y_FinalTest, y_predKNN)

'''
*******************************************************************************
                            PLOTTING
*******************************************************************************
'''

Accuracy = [(ac_DecisionTree*100),(ac_KNN*100),(ac_LogisticRegression*100),(ac_NaiveBayes*100),(ac_RandomForest*100),(ac_SVM*100)]
Algorithms = ['Decision Tree', 'KNN', 'Logistic Regression','Naive Bayes', 'Random Forest', 'SVM']
plt.figure(figsize=(16,8))
plt.bar(Algorithms, Accuracy, label='Accuracy Chart', color = 'blue')
plt.xlabel('Algorithms')
plt.ylabel('Accuracy %')
#plt.savefig('AccuracyGraph.png')
plt.show()

RecallRate = [((cm_DecisionTree[0][0]*100)/(cm_DecisionTree[0][0]+cm_DecisionTree[1][0])),((cm_KNN[0][0]*100)/(cm_KNN[0][0]+cm_KNN[1][0])),((cm_LogisticRegression[0][0]*100)/(cm_LogisticRegression[0][0]+cm_LogisticRegression[1][0])),((cm_NaiveBayes[0][0]*100)/(cm_NaiveBayes[0][0]+cm_NaiveBayes[1][0])),((cm_RandomForest[0][0]*100)/(cm_RandomForest[0][0]+cm_RandomForest[1][0])),((cm_SVM[0][0]*100)/(cm_SVM[0][0]+cm_SVM[1][0]))]
plt.figure(figsize=(16,8))
plt.bar(Algorithms, RecallRate, label='Recall Chart', color = 'green')
plt.xlabel('Algorithms')
plt.ylabel('Recall %')
plt.savefig('RecallGraph.png')
#plt.show()
