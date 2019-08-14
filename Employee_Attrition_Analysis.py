import numpy as np
import  pandas as pd
import seaborn as sns
import sklearn as sk
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.cluster import KMeans

xlfile=pd.ExcelFile('TakenMind-Python-Analytics-Problem-case-study-1-1.xlsx')

dframe1=xlfile.parse('Existing employees')
dframe1['left']=np.zeros((len(dframe1.index),), dtype=np.int)
#print(dframe1)

dframe2=xlfile.parse('Employees who have left')
dframe2['left']=np.ones(len(dframe2.index),dtype=np.int)
#print(dframe2)

#creating dataset having both existing and employees who have left
employee=pd.concat([dframe1,dframe2],ignore_index=True)

#find out the number of employees who left the company and those who didn’t:
print(employee['left'].value_counts())

mean1=employee.groupby('left').mean()
mean1.to_csv('mean1.csv',sep=',')

pd.crosstab(employee.dept,employee.left).plot(kind='bar')
plt.title('Turnover Frequency for Department')
plt.xlabel('Department')
plt.ylabel('Frequency of Turnover')
plt.savefig('department_bar_chart')

table=pd.crosstab(employee.salary, employee.left)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Salary Level vs Turnover')
plt.xlabel('Salary Level')
plt.ylabel('Proportion of Employees')
plt.savefig('salary_bar_chart')

#Histograms for numeric variables during the exploratory phrase.
num_bins = 10
employee.hist(bins=num_bins, figsize=(20,15))
plt.savefig("histogram_plots")
plt.show()


# Filter data
left_emp = employee[['satisfaction_level', 'last_evaluation']][employee.left == 1]
# Create groups using K-means clustering.
kmeans = KMeans(n_clusters = 3, random_state = 0).fit(left_emp)
# Add new column "label" annd assign cluster labels.
left_emp['label'] = kmeans.labels_
# Draw scatter plot
plt.scatter(left_emp['satisfaction_level'], left_emp['last_evaluation'], c=left_emp['label'],cmap='Accent')
plt.xlabel('Satisfaction Level')
plt.ylabel('Last Evaluation')
plt.title('Clusters of employees who left')
plt.show()

