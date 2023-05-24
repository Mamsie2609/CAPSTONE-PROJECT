#!/usr/bin/env python
# coding: utf-8

# # To analyse customer behavior and identify patterns and trends that can help understand the influence of customer demographics, such as age and educational level, on their attitude toward defaulting. By leveraging customer data, the objective is to develop a model that can predict the likelihood of deposits from customers.

# In[1]:


# IMPORT REQUIRED LIBRARIES

# FOR DATA ANALYSIS

import pandas as pd
import numpy as np

# FOR DATA VISUALISATION

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'notebook'


# MACHINE LEARNING AND EVALUATION

from sklearn.cluster import KMeans 
from scipy.cluster.hierarchy import dendrogram, linkage 
from sklearn.metrics import silhouette_score, homogeneity_score
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# CLASSIFIER LIBRARIES

from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier 
from sklearn.svm import LinearSVC, SVC 
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier

# EVALUATION METRICS

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score 
from sklearn.metrics import confusion_matrix

import warnings 
warnings.filterwarnings ("ignore")


# In[2]:


# LOAD DATASET

data = pd.read_csv(r"/Users/mamsie/Desktop/Data Science/CAPSTONE PROJECT/bank.csv")
data.head()


# In[3]:


# DIMENSIONALITY OF THE DATA - THE NUMBER OF ROWS AND COLUMNS

data.shape


# In[4]:


# NUMERICAL STATISTICAL ANALYSIS

data.describe()


# # DATA CLEANING AND PRE - PROCESSING

# In[5]:


# CHECK FOR MISSING VALUES

print(data.isnull().sum())

# VISUALISING THE MISSING DATA

plt.figure(figsize = (10,5))
sns.heatmap(data.isnull(), cbar=True, cmap="Blues_r");


# In[6]:


# CHECK FOR DUPLICATES

print(data.duplicated().sum())


# # EXPLORATORY DATA ANALYSIS

# In[7]:


# AGE BRACKET

def Age_bracket(age): 
    if age <= 35:
        return "Young adults"
    elif age <= 55:
        return "Middle-aged adults" 
    elif age <= 65:
        return "Senior citizens" 
    else:
        return "Elderly"
data['Age_bracket'] = data['age'].apply(Age_bracket)

# INVESTIGATING THE AGE GROUP OF CUSTOMERS

# Sets the size of the plot to be 10 inches in width and 5 inches in height.
plt.figure(figsize = (10, 5))

# Creates a countplot using the Seaborn library, with 'age_bracket' on the x-axis 
sns.countplot (x='Age_bracket', data=data) 

# Sets the label for the x-axis as 'Age Group'.
plt.xlabel('Age Group')

# Sets the label for the y-axis as 'Count of Age Group'.
plt.ylabel('Count of Age Group') 

# Sets the title of the plot as 'Total Number of Patients'.
plt.title('Total Number of Customers');


# * The customer base is primarily composed of middle-aged adults, with the highest number of customers falling into this age group. Following middle-aged customers, young adults constitute the second-largest group in terms of customer count. Senior citizens represent another segment of customers, although their numbers are comparatively lower. The elderly population comprises the smallest segment of customers.

# In[8]:


# INVESTIGATING THE JOB OF CUSTOMERS

plt.figure(figsize = (15, 15))
sns.countplot (x='job', data=data)
plt.xlabel('Profession')
plt.ylabel('Count of Customers')
plt.title('Total Number of Customers');


# * The chart indicates that the highest number of customers are employed in blue-collar and management professions. These two job categories have the largest representation among the customers. On the other hand, the dataset contains a smaller number of customers whose profession is listed as unknown.

# In[9]:


# INVESTIGATING THE DISTRIBUTION OF MARITAL STATUS OF CUSTOMERS

plt.figure(figsize = (10, 10))
sns.countplot (x='marital', data=data)
plt.xlabel('Marital Status')
plt.ylabel('Count of Customers')
plt.title('Total Number of Customers');


# * The dataset comprises predominantly married individuals, indicating a substantial presence of married customers. Single individuals also constitute a considerable portion of the dataset. Conversely, the dataset includes the fewest number of customers who are divorced.

# In[10]:


# INVESTIGATING THE EDUCATION DISTRIBUTION OF CUSTOMERS

default_counts = data['education'].value_counts()
plt.figure(figsize=(10, 5))
plt.pie(default_counts, labels=default_counts.index, autopct='%1.1f%%', startangle=30)
plt.title('Distribution of Customers by Education')
plt.axis('equal');


# * The distribution of customer education indicates that the highest number of customers have a secondary educational background, followed by tertiary and primary education. The dataset contains the least number of customers with unknown educational status.

# In[11]:


# INVESTIGATING THE DISTRIBUTION OF CUSTOMERS WHO DEFAULTED ON LOAN REPAYMENT

default_counts = data['default'].value_counts()
plt.figure(figsize=(10, 5))
plt.pie(default_counts, labels=default_counts.index, autopct='%1.1f%%', startangle=30)
plt.title('Customer Distribution by Loan Default')
plt.axis('equal');


# * The majority of customers in the dataset have not defaulted on loans.

# In[12]:


# INVESTIGATING THE ACCOUNT BALANCE OF CUSTOMERS

data['balance'].plot(kind='hist', bins=20)
plt.xlabel('Account balance')
plt.ylabel('Frequency')
plt.title('Distribution of Balance')
plt.show();


# * The majority of customers have their balance ranging from 0 to 20,000.

# In[13]:


# INVESTIGATING THE DISTRIBUTION OF HOUSING LOANS FROM CUSTOMERS

# Count the number of customers for each housing loan category
housing_counts = data['housing'].value_counts()

# Create a doughnut chart
plt.figure(figsize=(5, 5))
plt.pie(housing_counts, labels=housing_counts.index, autopct='%1.1f%%', startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})

# Draw a white circle at the center to create the doughnut effect
center_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(center_circle)

# Set aspect ratio to be equal to ensure a circular shape
plt.axis('equal')
plt.title('Distribution of Housing Loans from Customers')

plt.show();


# * A significant proportion of customers have taken housing loans.

# In[14]:


# INVESTIGATING THE LOAN DISTRIBUTION OF CUSTOMERS

plt.figure(figsize = (10, 10))
sns.countplot (x='loan', data=data)
plt.xlabel('Loan')
plt.ylabel('Count of Customers')
plt.title('Total Number of Customers');


# * A higher number of customers have not taken out personal loans.

# In[15]:


# INVESTIGATING THE DISTRIBUTION OF CUSTOMER FOR EACH CONTACT TYPE.

plt.figure(figsize = (10, 10))
sns.countplot (x='contact', data=data)
plt.xlabel('Contact')
plt.ylabel('Count of Customers')
plt.title('Total Number of Customers');


# * The majority of customers are contacted using the cellular method of communication.

# In[16]:


# INVESTIGATING THE OUTCOME OF THE PREVIOUS MARKETING CAMPAIGN.

plt.figure(figsize = (10, 10))
sns.countplot(x='poutcome', data=data)
plt.xlabel('Outcome')
plt.title('Outcome of the previous marketing campaign,');


# * The previous marketing campaign's outcome indicates that the highest number of customers had an unknown outcome, followed by failure, while success had the least representation.

# In[17]:


# INVESTIGATING THE DISTRIBUTION OF DEPOSITS FROM CUSTOMERS

plt.figure(figsize = (10, 10))
sns.countplot (x='deposit', data=data)
plt.xlabel('Deposit')
plt.ylabel('Count of Customers')
plt.title('Total Number of Customers');


# * It was observed that a small proportion of the customers had made deposits.

# In[18]:


# INVESTIGATING THE DISTRIBUTION OF CUSTOMER BALANCE AND DEPOSIT

sns.barplot(x ='deposit',y ='balance',data=data, ci=None);


# * Customers who have made deposits have higher account balance.

# In[19]:


# INVESTIGATING THE DISTRIBUTION OF BALANCE BY AGE

sns.barplot('Age_bracket', 'balance', data = data, ci=None)
plt.xlabel('Age Group')
plt.ylabel('Balance')
plt.title('Relationship between age and balance')
plt.show()


# * The elderly customers have the highest account balance, followed by senior citizens, while young adults have the lowest balance among the customers.

# In[20]:


# INVESTIGATING THE DISTRIBUTION OF DEPOSIT BY AGE

cross_tab = pd.crosstab(data['Age_bracket'], data['deposit'])
cross_tab.plot(kind='bar', stacked=True, figsize=(10, 10))
plt.xlabel('Age Bracket')
plt.ylabel('Count')
plt.title('Relationship between Age Bracket and Deposit')
plt.legend(title='Deposit', loc='upper right')
plt.show()


# * The highest number of customers who have made no deposits are found within the middle-aged adults and young adults age groups.

# In[21]:


# INVESTIGATING THE DISTRIBUTION OF DEPOSIT BASED ON EDUCATIONAL BACKGROUND

stacked_bar = pd.crosstab(data['education'], data['deposit'])
stacked_bar.plot(kind='bar', stacked=True, figsize=(10, 10))
plt.xlabel('Education')
plt.ylabel('Count of Customers')
plt.title('Count of Customers by Education and Deposit')
plt.legend(title='Deposit')
plt.show()


# * Customers with a secondary educational background have the highest number of individuals who have made no deposits, while customers with an unknown educational background have the least number of individuals who have made no deposits.

# In[22]:


# INVESTIGATING THE DISTRIBUTION OF BALANCE BY JOB

grouped_data = data.groupby('job')['balance'].mean().reset_index()
grouped_data['balance_rounded'] = grouped_data['balance'].round(1)
fig = go.Figure(data=go.Bar(
    x=grouped_data['job'],
    y=grouped_data['balance'],
    text=grouped_data['balance_rounded'],  
    textposition='outside'  
))

fig.update_layout(
    xaxis_title='Profession',
    yaxis_title='Balance',
    title='Distribution of Balance by Job'
)

pio.show(fig)


# * The chart reveals that retired individuals have the highest balance among the customers, indicating a significant representation of this job category. Additionally, the dataset shows a smaller number of customers working in the services industry.

# In[23]:


# INVESTIGATING THE DISTRIBUTION OF DEPOSIT BASED ON MARITAL STATUS

grouped_bar = pd.crosstab(data['marital'], data['deposit'])
grouped_bar.plot(kind='bar', figsize=(10, 10))
plt.xlabel('Marital Status')
plt.ylabel('Count of Customers')
plt.title('Count of Customers by Deposit and Marital Status')
plt.legend(title='Deposit')
plt.show()


# * The chart illustrates that married individuals have the highest number of customers with no deposits, while divorced individuals have the least number of customers who made deposits. 

# In[24]:


# VISUALISE THE RELATIONSHIP BETWEEN AGE GROUPS, BALANCE AND DEPOSIT

sns.barplot('Age_bracket', 'balance',hue='deposit', data = data, ci=None)
plt.xlabel('Age_bracket')
plt.ylabel('Balance')
plt.title('Relationship between Age Group, Balance and Deposit')
plt.show()


# * Among the customers, the elderly who have made deposits have the highest balance, while young adults who have made no deposits have the least balance.

# In[25]:


# INVESTIGATING THE DISTRIBUTION OF BALANCE BY AGE AND PERSONAL LOAN

sns.barplot('Age_bracket', 'balance', data = data, hue='loan', ci=None)
plt.xlabel('Age_bracket')
plt.ylabel('Balance')
plt.title('Relationship between Age Group, Balance and Personal Loan')
plt.show()


# * Among the customers, the elderly with the highest balance do not have personal loans, whereas young adults with the least balance have personal loans.

# In[26]:


# INVESTIGATING THE DISTRIBUTION OF BALANCE BY AGE AND HOUSING LOAN

sns.barplot('Age_bracket', 'balance', data = data, hue='housing', ci=None)
plt.xlabel('Age group')
plt.ylabel('Balance')
plt.title('Relationship between Age Group, Balance and Housing Loan')
plt.show()


# * Among the customers, the elderly with the highest balance have housing loans, while young adults with the least balance also have housing loans.

# In[27]:


# INVESTIGATING THE DISTRIBUTION OF BALANCE BY MARITAL STATUS AND EDUCATION

sns.barplot('marital', 'balance', data = data, hue='education', ci = None)
plt.xlabel('marital')
plt.ylabel('Balance')
plt.title('Relationship between marital status, education and balance')
plt.show()


# * Customers with the highest balance are married individuals with a tertiary educational background, while those with the least balance are divorced individuals with a secondary educational background.

# In[28]:


# INVESTIGATING THE CORRELATION BETWEEN FEATURES

plt.figure(figsize=(10, 10))
sns.heatmap(data.corr(), annot=True, cmap='cool')
plt.title('Correlation Matrix')
plt.show()


# * The features "previous" and "pdays" show a moderate positive correlation of 0.45, indicating a relationship between the number of contacts made to the customer before the current campaign and the number of days since the customer was last contacted in a previous campaign. However, this correlation is not considered strong.

# In[29]:


# VISUALISE THE RELATIONSHIP BETWEEN JOB, BALANCE AND MARITAL STATUS

plt.figure(figsize=(20, 20))
sns.boxplot(x='job', y='balance', hue='marital', data=data)
plt.xlabel('Profession')
plt.ylabel('Balance')
plt.title('Distribution of Balance by Job and Marital Status')
plt.legend(title='Marital Status')
plt.show()


# * The boxplot reveals numerous outliers in terms of balance across all job categories.

# In[30]:


# INVESTIGATING THE DISTRIBUTION OF DEFAULT BY AGE

cross_tab = pd.crosstab(data['Age_bracket'], data['default'])
cross_tab.plot(kind='bar', stacked=True, figsize=(10, 10))
plt.xlabel('Age Bracket')
plt.ylabel('Count')
plt.title('Relationship between Age Bracket and Default')
plt.legend(title='Default', loc='upper right')
plt.show()


# * In comparison to other age groups, middle-aged adults tend to default on loans more frequently.

# In[31]:


# INVESTIGATING THE DISTRIBUTION OF DEFAULT BY EDUCATIONAL BACKGROUND

cross_tab = pd.crosstab(data['education'], data['default'])
cross_tab.plot(kind='bar', stacked=True, figsize=(10, 10))
plt.xlabel('Education')
plt.ylabel('Count')
plt.title('Relationship between Education and Default')
plt.legend(title='Default', loc='upper right')
plt.show()


# * Compared to customers with other educational backgrounds, those with secondary education tend to have a higher default rate on loans.

# In[32]:


# INVESTIGATING THE DISTRIBUTION OF DEFAULT BY MARITAL STATUS

cross_tab = pd.crosstab(data['marital'], data['default'])
cross_tab.plot(kind='bar', stacked=True, figsize=(10, 10))
plt.xlabel('Marital status')
plt.ylabel('Count')
plt.title('Relationship between Marital status and Default')
plt.legend(title='Default', loc='upper left')
plt.show()


# * Compared to customers with other marital statuses, married individuals exhibit a relatively higher default rate on loans.

# In[33]:


# INVESTIGATING DEFAULT BASED ON PROFESSION

grouped_bar = pd.crosstab(data['job'], data['deposit'])
grouped_bar.plot(kind='bar', figsize=(20, 20))
plt.xlabel('Profession')
plt.ylabel('Count of Customers')
plt.title('Count of Customers by Default and Profession')
plt.legend(title='Default')
plt.show()


# * Compared to customers in other occupations, individuals in management roles have a relatively higher default rate on loans.

# # MACHINE LEARNING

# In[34]:


# CONVERTING CATEGORICAL VARIABLES TO NUMERICAL

encoder = LabelEncoder () 

for col in data. columns [1:]:
    if (data[col].dtype=='object'):
        data[col] = encoder.fit_transform(data[col])
    else:
        data[col] = data[col]
data.head()


# In[35]:


# DETERMINE OPTIMAL NUMBER OF CLUSTERS 

wcss = []
for i in range(1, 11): 
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state = 42)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss, color='red', marker='o') 
plt.title('Elbow Method') 
plt.xlabel('Number of clusters') 
plt.ylabel ('WCSS') 
plt. show()


# In[36]:


# FIT K-means CLUSTERING MODEL 

kmeans = KMeans (n_clusters=5, init= 'k-means++', random_state=42)
kmeans.fit(data)

data ['cluster'] = kmeans.labels_


# In[37]:


# VISUALISE CLUSTER RESULTS 

plt.scatter (data['balance'], data['age'], c=data['cluster'], label='Centroids') 
plt.xlabel('Balance')
plt.ylabel ('Age') 
plt.title('Cluster Visualisation')
plt. show ()

# Calculate silhouette score
from sklearn.metrics import silhouette_score
silhouette_score (data, kmeans.labels_)


# * This suggests that the clusters have a reasonable degree of separation 

# In[38]:


# EXCLUDE THE 'DEPOSIT' COLUMN

data1 = data.drop(columns=['deposit'])  
label = data[['deposit']]


# In[39]:


# SPLIT THE DATASET INTO TRAINING AND TESTING SETS 

X_train, X_test, y_train, y_test = train_test_split(data1, label, test_size=0.1, random_state=42)


# In[40]:


# NORMALISE THE DATA 

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[41]:


# TRANSFORM THE SCALED DATA INTO DATAFRAME 

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)


# In[42]:


# CLASSIFIER MODELS

classifiers = [[XGBClassifier(), 'XGB Classifier'],
              [RandomForestClassifier(), 'Random Forest'],
              [KNeighborsClassifier(),'K-Nearest Neighbors'],
              [SGDClassifier(), 'SGD Classifier'],
              [SVC(), 'SVC'],
              [GaussianNB(),'Naive Bayes'],
              [DecisionTreeClassifier(random_state = 42), "Decision tree"],
              [LogisticRegression(),'Logistic Regression']
              ]


# In[43]:


acc_list = {}
precision_list = {}
recall_list = {}
roc_list = {}
cm_dict = {}
f1_list = {}

for classifier in classifiers:
    model = classifier[0]
    model.fit(X_train_scaled, y_train) 
    model_name = classifier[1]
    
    pred = model.predict(X_test_scaled)
    
    a_score = accuracy_score (y_test, pred)
    p_score = precision_score(y_test, pred)
    r_score = recall_score(y_test, pred)
    roc_score = roc_auc_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    
    acc_list[model_name] = ([str(round (a_score*100, 2)) + '%'])
    precision_list[model_name] = ([str(round(p_score*100, 2)) + '%'])
    recall_list[model_name] = ([str(round(r_score*100, 2)) + '%'])
    roc_list[model_name] = ([str(round(roc_score*100, 2)) + '%'])
    f1_list[model_name] = [str(round(f1*100, 2)) + '%']  
    
    cm = confusion_matrix(y_test, pred)
    cm_dict[model_name] = cm

    if model_name != classifiers[-1][1]:
       print('')        


# In[44]:


print("Accuracy Score")
s1 = pd.DataFrame (acc_list)
s1. head()


# * The XGB Classifier has the highest accuracy score of 90.14%, followed closely by the Random Forest with 90.11%.

# In[45]:


print("Precision Score")
s2 = pd.DataFrame (precision_list)
s2. head()


# * The SGD Classifier has the highest precision score of 70.0%, indicating a higher proportion of correctly predicted positive instances (customers who make deposits) out of all instances predicted as positive.
# 

# In[46]:


print("Recall")
s3 = pd.DataFrame (recall_list)
s3. head()


# * XGB Classifier has the highest recall score of 48.01%, indicating a higher proportion of correctly predicted positive instances out of all actual positive instances.

# In[47]:


print("ROC Score")
s4 = pd.DataFrame (roc_list)
s4. head()


# * The XGB Classifier has the highest ROC score of 72.02%, followed by the Random Forest with 68.74%.

# In[48]:


f1_list


# * The XGB Classifier has the highest F1 score of 54.4%, indicating a balanced performance between precision and recall.
# * The Random Forest and Decision Tree models also show relatively good F1 scores of 48.38% and 45.83% respectively.
# * The K-Nearest Neighbors, Logistic Regression, and Naive Bayes models have F1 scores ranging from 28.3% to 38.39%.
# * The SGD Classifier and SVC models have the lowest F1 scores of 3.18% and 24.71% respectively.

# In[49]:


cm_dict


# ### The bank should be more concerned with minimising the resources spent on targeting customers who are unlikely to make deposits (false positives),hence the most important metric to consider would be precision. Considering this objective, the SGD Classifier would be the most suitable model as it achieves the highest precision score. Minimising false positives (incorrectly targeting customers) is crucial to optimise resource allocation.
