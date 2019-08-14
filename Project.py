import numpy as np
import  pandas as pd
import sklearn as sk
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

xlfile=pd.ExcelFile('TakenMind-Python-Analytics-Problem-case-study-1-1.xlsx')

dframe1=xlfile.parse('Existing employees')
dframe1['left']=np.zeros((len(dframe1.index),), dtype=np.int)
#print(dframe1)

dframe2=xlfile.parse('Employees who have left')
dframe2['left']=np.ones(len(dframe2.index),dtype=np.int)
#print(dframe2)

employee=pd.concat([dframe1,dframe2],ignore_index=True)
#print(employee)
employee.to_csv('dataset.csv',sep=',')

#creating labelEncoder
le = preprocessing.LabelEncoder()

employee['salary']=le.fit_transform(employee['salary'])
employee['dept']=le.fit_transform(employee['dept'])

X=employee[['Emp ID','satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','dept','salary']]  # Features
y=employee['left']  # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)


#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision
print("Precision:",metrics.precision_score(y_test, y_pred))
# Model Recall
print("Recall:",metrics.recall_score(y_test, y_pred))

#array of feature names for employee dataset
feature_names=['Emp ID','satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','dept','salary']

#fnding important features from employee dataset
feature_imp = pd.Series(clf.feature_importances_,index=feature_names).sort_values(ascending=False)
print(feature_imp)

# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
#plt.legend()
plt.show()


#removing features having less importance (dept,Work_accident,salary,promotion_last_5years)

print('Removing less important features from dataset')

# Split dataset into features and labels
X=employee[['Emp ID','satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company']]  # Features
y=employee['left']  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70, random_state=5) # 70% training and 30% test

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)


#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision
print("Precision:",metrics.precision_score(y_test, y_pred))
# Model Recall
print("Recall:",metrics.recall_score(y_test, y_pred))


forest_cm = metrics.confusion_matrix(y_pred, y_test, [1,0])
sns.heatmap(forest_cm, annot=True, fmt='.2f',xticklabels = ["Left", "Stayed"] , yticklabels = ["Left", "Stayed"] )
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Random Forest Algorithm')
plt.savefig('random_forest')

#check employees prone to leave next
test=dframe1[['Emp ID','satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company']]
pred=clf.predict(test)
print(pred)

