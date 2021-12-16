import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics 
#from imblearn.over_sampling import SMOTE

data=pd.read_csv(r"Dataset_For_Covid19.csv")
data.head()

data=data.iloc[:,[1,2,3,4,5,6,7,8,9]] 
print(data)

data.dtypes

data.isnull().sum()

sns.heatmap(data.corr(),annot= True)

le=LabelEncoder()
data['Gender']=le.fit_transform(data.Gender)
data['Ventilated']=le.fit_transform(data.Ventilated)
data['Outcome']=le.fit_transform(data.Outcome)
data

data.corr()
sns.heatmap(data.corr(),annot=True)

data['Outcome'].value_counts()

print("Gender distribution:\n",data['Gender'].value_counts(),"\n")
print("Ventilated  distribution:\n",data['Ventilated'].value_counts(),"\n")
print("Outcome distribution:\n",data['Outcome'].value_counts(),"\n")




X,Y=data.iloc[:,0:7],data.iloc[:,8]
train_x,test_x,train_y,test_y= train_test_split(X,Y,test_size=0.2,random_state=55)
print("shape of train_x=",train_x.shape)
print("shape of test_x=",test_x.shape)
print("shape of train_y=",train_y.shape)
print("shape of test_y=",test_y.shape)


#LOGISTIC REGRESSION
LR = LogisticRegression(max_iter=1000)
LR.fit(train_x,train_y)
pred= LR.predict(test_x)
print("Confusion matrix of Logisitc Regression Model\n",metrics.confusion_matrix(test_y,pred))
print("Accuracy of Logisitc Regression Model         \t",metrics.accuracy_score(test_y,pred))
print("Recall score of Logisitc Regression Model     \t",metrics.recall_score(test_y,pred))
print("Precision Score of Logisitc Regression Model  \t",metrics.precision_score(test_y,pred))
print("f1 score of Logisitc Regression Model         \t",metrics.f1_score(test_y,pred))

#RANDOM FOREST
rf=RandomForestClassifier()
rf.fit(train_x,train_y)
pred=rf.predict(test_x)
print("Confusion matrix of Random Forest Classifier Model\n",metrics.confusion_matrix(test_y,pred))
print("Accuracy of Random Forest Classifier Model        \t",metrics.accuracy_score(test_y,pred))
print("Recall score of Random Forest Classifier Model    \t",metrics.recall_score(test_y,pred))
print("Precision Score of Random Forest Classifier Model \t",metrics.precision_score(test_y,pred))
print("f1 score of Random Forest Classifier Model        \t",metrics.f1_score(test_y,pred))
