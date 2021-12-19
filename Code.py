#IMPORT LIBRARIES
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics 


#IMPORT DATASET
data=pd.read_csv(r"dataset_for_covid.csv")
data.head()


#DISPLAY SELECTED ATTRIBUTES
data=data.iloc[:,[0,1,2,3,4,5,6,7,8]] 
print(data)

#LOOKING FOR NULL VALUES
data.isnull().sum()

#HEAT MAP VISUALIZATION
sns.heatmap(data.corr(),annot= True)


#LABEL ENCODING THE ATTRIBUTES TO NUMERIC VALUES
le=LabelEncoder()
data['Sex']=le.fit_transform(data.Sex)
data['Outcome']=le.fit_transform(data.Outcome)
data


#SPLIT DATASET INTO TEST-TRAIN SET
X,Y=data.iloc[:,0:7],data.iloc[:,8]
train_x,test_x,train_y,test_y= train_test_split(X,Y,test_size=0.2,random_state=50)
print("shape of train_x=",train_x.shape)
print("shape of test_x=",test_x.shape)
print("shape of train_y=",train_y.shape)
print("shape of test_y=",test_y.shape)


#LOGISTIC REGRESSION MODEL
LR = LogisticRegression(max_iter=1000)
LR.fit(train_x,train_y)
pred= LR.predict(test_x)
print("Confusion matrix of Logisitc Regression Model\n",metrics.confusion_matrix(test_y,pred))
print("Accuracy of Logisitc Regression Model         \t",metrics.accuracy_score(test_y,pred))
print("Recall score of Logisitc Regression Model     \t",metrics.recall_score(test_y,pred))
print("Precision Score of Logisitc Regression Model  \t",metrics.precision_score(test_y,pred))
print("f1 score of Logisitc Regression Model         \t",metrics.f1_score(test_y,pred))


#RANDOM FOREST MODEL
rf=RandomForestClassifier()
rf.fit(train_x,train_y)
pred=rf.predict(test_x)
print("Confusion matrix of Random Forest Classifier Model\n",metrics.confusion_matrix(test_y,pred))
print("Accuracy of Random Forest Classifier Model        \t",metrics.accuracy_score(test_y,pred))
print("Recall score of Random Forest Classifier Model    \t",metrics.recall_score(test_y,pred))
print("Precision Score of Random Forest Classifier Model \t",metrics.precision_score(test_y,pred))
print("f1 score of Random Forest Classifier Model        \t",metrics.f1_score(test_y,pred))
