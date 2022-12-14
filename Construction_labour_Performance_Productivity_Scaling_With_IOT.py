### Loading The Necessary
### For data manipulations
import pandas as pd
import numpy as np

### For data Visualizations
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')



# Let's Read the Dataset for Analysis The Productivity
labour=pd.read_excel(r"C:\Users\sony\FlaskDeployment-20221214T182518Z-001\FlaskDeployment\labour.xlsx")


# To Know the Information of the Dataset
labour.info()



# let's check the shape of the Dataset
print("Shape of the Dataset:", labour.shape)



# Let's check the Head of the Dataset
labour.head()



# To know the Names of Each Column in Dataset
labour.columns


# Let's Check the Age present in this Dataset

labour['Age'].value_counts()
### With the Help of Value-counts we can see that at the age of 21 there are 19 Workers are working ,at the age of 26 there are 17 Workers are Working, and  
### at the age of 25 there are 16 workers are working......... 



# Let's Check the Gender present in this Dataset
labour['Gender'].value_counts()
### With the Help of Value-counts we can see that the Male Workers are 175 working at Site,
#### and The Female Workers are 101 Working at Site.............



# Let's Check the Nationality present in this Dataset
labour['Nationality'].value_counts()
### With the Help of Value-counts we can see that the Maximum Workers are from China =188 and working at Site,
#### and The Second Highest Workers are 52 from Philippines and Working at Site.............




# Let's Check the Designation present in this Dataset
labour['Designation'].value_counts()
### With the Help of Value-counts we can see that the Semi-Skilled Workers are 100 working at Site,
#### and The Un-Skilled Workers are 96 Working at Site
#### and Skilled Workers are 80 Working at Site  .............



# Lets Check The Heart Beat Present in the Dataset
labour['HeartRateSensor'].value_counts()
### With the Help of Value-counts we can see that the Workers Heart Rate is 120 beats per minute for 
#### 31 Workers working at Site,
#### and other hand side The Workers Heart Rate is 123 beats per minute
##### for 24 Workers Working at Site........



# To know the stats about the dataset
labour.describe()



# Checking the Null Values in the Dataset
labour.isna().sum()



labour.columns



# Checking the Duplicate Values in the Dataset
duplicate = labour.duplicated()
duplicate
sum(duplicate)



# EXPLORATORY DATA ANALYSIS
# The Histogram of all The Numerical columns in combined
plt.figure()
labour.hist(figsize=(20,16),xrot=40)
plt.show()



# The For Loop to get the countplot of each column
for column in labour.select_dtypes(include='object'):
    if labour[column].nunique() < 10:
        sns.countplot(y=column,data=labour)
        plt.show()


        
# For loop to plot the boxplot of all numerical columns in one loop
columns = ['Age','Experience' ,'Body Temperature','GalvanicSkinResponseSensor' ,'SkinTemperatureSensor' ,'BloodVolumePulse','RespirationRateSensor','HeartRateSensor']
for col in columns:
    plt.figure(figsize=(8,8))
    sns.boxplot(labour[col])
    plt.show()
    


# Univarite Analysis
plt.figure()
sns.catplot(data=labour, x="Gender", y="Age", kind="box")
plt.show()
### With the Help of Value-counts we can see that the Male Workers are between 
#### 31 Workers working at Site,
#### and other hand side The Workers Heart Rate is 123 beats per minute 
##### for 24 Workers Working at Site........



sns.catplot(data=labour, x="Gender", y="Experience", kind="box")


sns.catplot(data=labour, x="Gender", y="HeartRateSensor", kind="box")


plt.figure()
sns.catplot(data=labour, x="Gender", y="RespirationRateSensor", kind="box")
plt.show()


sns.catplot(data=labour, x="Gender", y="Body Temperature", kind="box")


labour.columns


plt.figure(figsize=(10,7))
labour['Gender'].value_counts().plot(kind="pie", autopct="%.2f")
plt.show()
#### From the below Piechart We can see that the 63.43% of male Workers are working at site...
#### and 36.59% Female Workers are working at site ..
##### As compare to male their are less female Workers Working at Site.....


plt.figure(figsize=(29,23))
labour['Age'].value_counts().plot(kind="pie", autopct="%.2f")
plt.show()
### From the Below Pie Chart we can see That the workers with 21 years old are 6.88% and 
### 26 Years old are 6.16 % and 25 years old are 5.80 % which is highest as compare to all age groups.



plt.figure(figsize=(18,12))
labour['Nationality'].value_counts().plot(kind="pie", autopct="%.2f")
plt.show()
##### From the below PieChart we can see that the maximum numbers of workers are from China = 68.12%
#### and Second from Philipiness =18.84% which is very less compare to China............




plt.figure(figsize=(15,13))
labour['Designation'].value_counts().plot(kind="pie", autopct="%.2f")
plt.show()
####### From the below PieChart we can see that the 36.23% of Workers are Semi-Skilled
#### and 34.78% of Workers are Un-Skilled , 28.99% of Workers are Skilled ............




plt.figure(figsize=(16,12))
labour['Experience'].value_counts().plot(kind="pie", autopct="%.2f")
plt.show()
### From the below PieChart we can see that the Construction Workers have 5,18 and 4 years of experience ..
##### In percentage it 7.25% for 4years and 7.25% for 18years and 5.88% for 4Years of experience which is the Highest Experience
##### As compare to other Workers ...................




plt.figure(figsize=(15,11))
labour['Performance'].value_counts().plot(kind="pie", autopct="%.2f")
plt.show()
####### From the below PieChart we can see that the 85.14% of Work are Doing Productive and 14.86% of Work are Doing 
#### Non-Productivity Work and Those Labels are Classified with the help of Motion Sensor Labels




plt.figure(figsize=(15,11))
labour['Site'].value_counts().plot(kind="pie", autopct="%.2f")
plt.show()
### From the below PieChart We can see that the 36.23% Workers are Working at Site-1
### And 35.14% Workers are Working at Site-2, 28.62% Workers are Working at Site-3 which is very less compare to site-1 & Site-2.




plt.figure(figsize=(21,13))
labour['MotionSensor'].value_counts().plot(kind="pie", autopct="%.2f")
plt.show()
### From the below PieChart we can see that the Workers Works are classified into various labels with the help of motion sensor
### and we can say that 11.23% of workers are welding , 3.62% of workers are wiring, 4.71%  of workers are Placing Rebar,
#### 5.43% of Workers are Lashing Rope, 5.80% of Workers are Curing, 5.80% of Workers are Fetching Mortar,
#### 5.80% of Workers are Moving Stones , 6.16% of Workers are Chatting Only but related to Work only ,
### 6.88% of Workers are Excavation , 7.25% of Workers are Adjusting Bricks, 8.33% of Workers are Cutting Rebar,
### 9.06% of Workers are Cutting Bricks , 9.06% of Workers are Moving Bricks , 10.87% of Workers are Drinking Water ..........





plt.figure(figsize=(12,9))
labour['Gas Sensor'].value_counts().plot(kind="pie", autopct="%.2f")
plt.show()
### from the below PieChart we can say that the Gas Sensor dedected in 46.01% and Not Dedected in 53.99%  





plt.figure(figsize=(12,10))
labour['Noise Detection'].value_counts().plot(kind="pie", autopct="%.2f")
plt.show()
### From the below Piechart we can say that the Noise detection Sensor is detected in 49.28% and Not Dedected in 50.72%.....




plt.figure(figsize=(30,22))
labour['HeartRateSensor'].value_counts().plot(kind="pie", autopct="%.2f")
plt.show()
### From the below PieChart We can say that the 11.23% of Workers having 120 Beats Per Minute and 8.70% of Workers having 123 Beats Per Minute...




plt.figure(figsize=(12,9))
labour['Martial Status'].value_counts().plot(kind="pie", autopct="%.2f")
plt.show()
#### From the Below PieChart We can say that the 39.13% of Workers are Married and 34.78% of Workers are Un-Married 
#### and 26.09% of Workers are Divorced ..............



# Bivariate Analysis
plt.figure(figsize=(9,8))
sns.barplot(labour['Gender'], labour['Age'])
plt.show()




plt.figure(figsize=(12,9))
sns.barplot(labour['Age'], labour['Nationality'])
plt.show()



plt.figure(figsize=(8,7))
sns.barplot(labour['Gender'], labour['HeartRateSensor'])
plt.show()
### From the below Barplot we can say that the Male Workers are Working at site having 128 Beats Per minute 
##### and Female Workers are working at site having 120 Beats Per minute ..............




plt.figure(figsize=(12,8))
sns.barplot(labour['Gender'], labour['RespirationRateSensor'])
plt.show()




plt.figure(figsize=(14,9))
sns.barplot(labour['Age'], labour['GalvanicSkinResponseSensor'])
plt.show()
##### From the below Countplot we can say that the Galvanic Skin Response Sensor Tells Each Workers Stress and more the reading
### Less the Stress and Less the Reading of galvanic Skin Response sensor it indicates of more Stress of any individual worker.




plt.figure(figsize=(14,9))
sns.barplot(labour['Nationality'], labour['HeartRateSensor'])
plt.xticks(rotation=40)
plt.tight_layout()
plt.show()
### The Workers's or Labours around the World of all age groups and male n female are exceeding 
### 60 to 100 heart beats per minute(Normal Person) that we can see and understand from below barplot
##### The Labour's from bangladesh having more than 150 Beats Per minute 
######## And the Labour's from Malaysia having more than 140 Beats Per minute which is less than bangladesh Workers..........





plt.figure(figsize=(14,9))
sns.barplot(labour['Nationality'], labour['Body Temperature'])
plt.xticks(rotation=40)
plt.tight_layout()
plt.show()
##### From the below Barplot we can say that the Workers from around the World having normal body temperature 
### Which is less than 100.00 Fahrenheit it indicates good health of each workers becoz less than 100.00 fahrenheit 
#### and If for example the people having more than 100.00 Fahrenheit degree it tells us the particular worker is having Fever,Diarrhea, Severe Headache ........



plt.figure(figsize=(14,9))
sns.barplot(labour['Age'], labour['HeartRateSensor'])
plt.xticks(rotation=40)
plt.tight_layout()
plt.show()
### The People's or Labours around the World of all age groups and male n female are exceeding 
### the 60 to 100 heart beats per minute(normal person) that we can see and understand from below barplot we can say ....
#### The 36 Years old Workers are having more than 140 beats per minute it is very dangerous for workers........



plt.figure(figsize=(14,9))
sns.barplot(labour['Age'], labour['Experience'])
plt.xticks(rotation=40)
plt.tight_layout()
plt.show()
### The People's or Labours around the World of all age groups and male n female are having 
### 46 and 53 years old workers are having 18 years of work experience related to Construction Field .....



plt.figure(figsize=(8,7))
sns.barplot(labour['Gender'], labour['Body Temperature'])
plt.show()



plt.figure(figsize=(15,9))
sns.barplot(labour['Age'], labour['MotionSensor'])
plt.yticks(rotation=24)
plt.tight_layout()
plt.show()



plt.figure(figsize=(8,8))
sns.barplot(labour['Experience'], labour['Gender'], hue = labour["Gender"])
plt.show()
### From the Above Barplot we can see that . The Female Labours Having the more Work Experience in Construction Field 
### As Compare to the Male Labours in the COnstruction Field...............



plt.figure(figsize=(9,6))
sns.boxplot(labour['Age'], labour["Nationality"], labour["Gender"])
plt.yticks(rotation=26)
plt.tight_layout()
plt.show()
#### We Can See from the Above Boxplot that . The Female Workers are more from Japan and Male Workers are Less
#####  and Female age group is  Between 26 to 37 age. 
##### from China also the same as Compare to the Japan but the age Group is different...............




plt.figure(figsize=(16,10))
sns.boxplot(labour['Age'], labour["Nationality"], labour["Designation"])
plt.yticks(rotation=26)
plt.tight_layout()
plt.show()
### From The Above Barplot the Japan have more Un-Skilled Labour and China Also Have Little-More Un-Skilled Labours 


# Mutlivariate Analysis
labour.columns

sns.set_theme()
f, ax = plt.subplots(figsize=(12,10))
corr = labour.corr()
sns.heatmap(corr,
           xticklabels=corr.columns.values,
           yticklabels=corr.columns.values,cmap='spring_r')



sns.pairplot(labour)



# Auto-Exploratory Data Analysis
import sweetviz as sv
report=sv.analyze(labour)
report.show_html('sweet_report.html')



from pandas_profiling import ProfileReport
profile=ProfileReport(labour,explorative=True)
profile.to_file('output.html')



plt.figure(figsize=(8,7))
sns.countplot(x=labour['Performance'],data=labour)
plt.xticks(rotation=22)
plt.tight_layout()
plt.show()



plt.figure(figsize=(9, 6))
sns.countplot(x=labour['Nationality'],data=labour)
plt.xticks(rotation=22)
plt.tight_layout()
plt.show()



plt.figure(figsize=(16,10))
sns.countplot(x=labour['Age'],data=labour)
plt.show()




plt.figure(figsize=(15,9))
sns.countplot(x=labour['Gender'],data=labour)
plt.show()




plt.figure(figsize=(8,7))
sns.countplot(x=labour['Site'],data=labour)
plt.xticks(rotation=24)
plt.tight_layout()
plt.show()




plt.figure()
fig, ax = plt.subplots(figsize=(9,8))
sns.stripplot(x = "Nationality",
              y = "Age",
              data = labour,
              jitter = True,
              ax = ax,
              s = 8)
sns.despine(right = True)
plt.xticks(rotation=22)
plt.tight_layout()
plt.show()



labour.columns

#### Checking the Duplicate Values Present in the Labour Dataset
duplicate = labour.duplicated()
duplicate
sum(duplicate)


##### Importing Neccessaary Libraries which are required for columntransformer, PipeLine, SimpleImputer, MinMaxScaling, OneHotEncoding
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransforme
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder


### To Know the Unique features Present in the Target Variable
labour['Performance'].unique()

### To Know the Values Present in the Target Variable
labour['Performance'].value_counts()

labour.info()

# FEATURE ENGINEERING
# Dropping the Columns Which are Un-necessary
labour_1 = labour.drop(['Emp ID','Name','Nationality','Martial Status','Designation','Experience','Work Date','Site','Rfid','Latitude','Longitude','Work Started Time','Noise Detection','Gas Sensor'], axis=1)
labour_1.columns

### To know the data-types of Each variable Present in DataSet
labour_1.dtypes

### To know the Information of the labour Dataset
labour_1.info()

## Checking the null Values or Missing Values Present in the labour Dataset
labour_1.isnull().sum()


####### Partitioning the Labour_1 into X=Predictors and Y=Target
X = labour_1.iloc[:,:10]

X

### To know the X Variables 
X.columns 

X.head


Y = labour_1['Performance']

Y

## Taking Only The Numeric Features for Simple Imputation, Winsorization, MinMaxScaling 
numeric_features = X.select_dtypes(exclude=['object']).columns

numeric_features

## Taking Only The Categorical Features for OneHotEncoding 
categorical_features = X.select_dtypes(include=['object']).columns

categorical_features

# creating instance of Pipeline for SimpleImputer = Mean
num_pipeline = Pipeline(steps = [('impute', SimpleImputer(strategy= "mean"))])

### Transforming the Imputed numeric Features into Preprocessor
preprocessor = ColumnTransformer(transformers = [('num', num_pipeline, numeric_features)])

#### After Columnstransformation and mean imputation fitting the X Values
imputation = preprocessor.fit(X)

### Dumping the Mean Imputation File For Backend Purpose For Deployment Use. 
import joblib

joblib.dump(imputation, 'meanimpute')

### Creating the dataframe of mean imputation of Numeric X Values and putting Those values into imputation1
imputation1 = pd.DataFrame(imputation.transform(X), columns = numeric_features)

imputation1


### Indentifying the Outliers of Numeric features with the help of Boxplot 
import matplotlib.pyplot as plt

# Multiple boxplots in a single visualization.
# Columns with larger scales affect other columns. 
# Below code ensures each column gets its own y-axis.

# pandas plot() function with parameters kind = 'box' and subplots = True

import matplotlib.pyplot as plt

imputation1.plot(kind = 'box', subplots = True, sharey = False, figsize = (12,10))

'''sharey True or 'all': x- or y-axis will be shared among all subplots.
False or 'none': each subplot x- or y-axis will be independent.'''

# increase spacing between subplots
plt.subplots_adjust(wspace = 0.50) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()


X.columns

#### Winsorizing Outliers with capping method 
from feature_engine.outliers import Winsorizer

winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                        tail='both', # cap left, right or both tails 
                        fold=1.5,
                        variables=['Age','GalvanicSkinResponseSensor', 'SkinTemperatureSensor','BloodVolumePulse', 'RespirationRateSensor', 'HeartRateSensor'])


#### Replacing the Outlier Values Present in Below Columns 
clean = winsor.fit(imputation1[['Age','GalvanicSkinResponseSensor', 'SkinTemperatureSensor','BloodVolumePulse', 'RespirationRateSensor', 'HeartRateSensor']])


### Dumping the Winsorization File For Backend Purpose For Deployment
import joblib

joblib.dump(clean, 'winsor')


imputation1[['Age','GalvanicSkinResponseSensor', 'SkinTemperatureSensor','BloodVolumePulse', 'RespirationRateSensor', 'HeartRateSensor']] = clean.transform(imputation1[['Age','GalvanicSkinResponseSensor', 'SkinTemperatureSensor','BloodVolumePulse', 'RespirationRateSensor', 'HeartRateSensor']])


imputation1


### Identifying the Outliers After Winsorization with the Help of Boxplot
imputation1.plot(kind = 'box', subplots = True, sharey = False, figsize = (12,10)) 

# increase spacing between subplots
plt.subplots_adjust(wspace = 0.50)# ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()

# creating instance of Pipeline for MinMaxScaling
scale_pipeline = Pipeline([['scale', MinMaxScaler()]])


### Transforming the MinMaxScaling numeric Features into Scale_columntransformer
scale_columntransformer = ColumnTransformer([('scale', scale_pipeline, numeric_features)])

#### After Columnstransformation for MinMaxScaling fitting the imputation1 numeric values into scale
scale = scale_columntransformer.fit(imputation1)

### Dumping the MinMaxScaling File For Backend Purpose For Deployment Use.
import joblib

joblib.dump(scale, 'minmax')

### Creating the Dataframe of minmaxscale values of imputation1 into scaled_data
scaled_data = pd.DataFrame(scale.transform(imputation1))


scaled_data

# creating instance of Pipeline for OneHotEncoding
encoding_pipeline = Pipeline([('onehot', OneHotEncoder())])

### Transforming the OneHotEncoding categorical Features into Preprocess_pipeline
preprocess_pipeline = ColumnTransformer([('categorical', encoding_pipeline, categorical_features)])

#### After Columnstransformation for OneHotEncoding fitting the preprocess_pipeline categorical values into clean
clean = preprocess_pipeline.fit(X)

### Dumping the onehotencoding File For Backend Purpose For Deployment Use.
import joblib

joblib.dump(clean, 'encoding')

### Creating the Dataframe of OneHotEncoding values of clean into encoded_data
encoded_data = pd.DataFrame(clean.transform(X).todense())


encoded_data

### Concating the Scaled_data, Encoded_data into Clean_data
clean_data = pd.concat([scaled_data, encoded_data], axis=1 , ignore_index = True)


##### Splitting the labour Dataset into Train_test For ModelBuilding Purpose
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(clean_data, Y, test_size = 0.2, random_state = 21, stratify = Y)

### Checking the Shape of the Xtrain, and Xtest
X_train.shape, X_test.shape

### Checking the Shape of the Ytrain, and Ytest
Y_train.shape, Y_test.shape


print(Y_train.value_counts()/220)

print("\n")

print(Y_test.value_counts()/56)

# Model Building 
## 1. Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier as DT

###### Creating the Instance of Decision Tree Classifier
model = DT(criterion = 'entropy')

# Fit the Decision Tree CLassifier model to the Input Training instances
model.fit(X_train, Y_train)

# Predicting the Test Results
preds = model.predict(X_test)


preds


from sklearn.metrics import accuracy_score

# Evaluating the Decision Tree CLassifier-Model for model's Performance
print(accuracy_score(Y_test, preds))
pd.crosstab(Y_test, preds, rownames = ['Actual'], colnames = ['Predictions'])



# error on train data
pred_train = model.predict(X_train)
print(accuracy_score(Y_train, pred_train))
pd.crosstab(Y_train, pred_train, rownames=['Actual'], colnames = ['Predictions'])


### Importing the GridSearchCV to know the Best Estimator in Decision Tree CLassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

### The Hyperparameter Tunning Details are given below

#######criterion : string, optional (default=”gini”)
####Possible options are “gini” and “entropy”. Both “gini-index” and “cross-entropy” are values to show the node purity.
### When the node is purer, value of gini-index or cross-entropy is smaller and close to zero. 
#### Decision tree algorithm splits nodes as long as this value decreases till it reaches zero or 
#### there is no other parameter to stop it. For now, lets continue with gini.

##### max_depth : int or None, optional (default=None)
### The maximum depth of the tree. If None, then nodes are expanded until all nodes are pure or until all nodes 
#####  contain less than min_samples_split samples.
param_grid = { 'criterion':['gini','entropy'],'max_depth': np.arange(3,15)}


###### Creating the Instance of Decision Tree Classifier for GridSearchCV to know the Best Estimator
dtree_model = DecisionTreeClassifier()

##### taking the parameter_grid , dtree_model, cross validation to fit the clean_data and Y
dtree_gscv = GridSearchCV(dtree_model, param_grid, cv=5, scoring = 'accuracy', return_train_score = False, verbose = 1)

##### Fit the clean_data , Y with the help of dtree_gscv
dtree_gscv.fit(clean_data, Y)

##### Checking the Best_estimator
DT_best = dtree_gscv.best_estimator_


### Dumping the Decision Tree CLassifier pickle File For Backend Purpose For Deployment Use.
DT_best

#### 
import pandas as pd
import pickle
import joblib

pickle.dump(DT_best, open('DT.pkl','wb'))



### 2.K-Nearest Neighbors-Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# Initialising the KNN
knn_cl = KNeighborsClassifier(n_neighbors = 21)


# Fit the model to the Input Training instances 
knn_cl.fit(X_train, Y_train)


# Predicting the Test Results
pred = knn_cl.predict(X_test)
pred


# Evaluating the KNN-Model for model's Performance
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, pred))
pd.crosstab(Y_test, pred, rownames = ['Actual'], colnames= ['Predictions']) 


# error on train data
pred_train = knn_cl.predict(X_train)
print(accuracy_score(Y_train, pred_train))
pd.crosstab(Y_train, pred_train, rownames=['Actual'], colnames = ['Predictions'])




# 3.logistic  regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



# Initialising the Logistic Regression
lr = LogisticRegression()


# Fit the model to the Input Training instances 
lr.fit(X_train, Y_train)


# Predicting the Test Results
Y_pred = lr.predict(X_test)


# Evaluating the Logistic Regression-Model for model's Performance
lr_train_acc = accuracy_score(Y_train, lr.predict(X_train))
lr_test_acc = accuracy_score(Y_test, Y_pred)



# Checking the Accuracy of train and test To know the Difference
print(f"Training Accuracy of Logistic Regression Model is {lr_train_acc}")
print(f"Test Accuracy of Logistic Regression Model is {lr_test_acc}")




# 4.Support Vector Classification
# Training the model
from sklearn.svm import SVC


# Initialising the Support Vector Classification Model
svm_cl = SVC(kernel='rbf', C=100, random_state=10).fit(X_train,Y_train)


# Fit the model to the Input Training instances 
svm_cl.fit(X_train, Y_train)


# Predicting The Test Results
Y_pred = svm_cl.predict(X_test)


# Evaluating the Support Vector Classification-model for model's Performance
svm_train_acc = accuracy_score(Y_train, svm_cl.predict(X_train))
svm_test_acc = accuracy_score(Y_test, Y_pred)


# Checking the Accuracy of Train and Test To know the Difference
print(f"Training Accuracy of Support Vector Classification Model is {svm_train_acc}")
print(f"Test Accuracy of Support Vector Classification Model is {svm_test_acc}")


# 5.Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier


# Initializing the Random Forest Classifier Model
rand_clf = RandomForestClassifier(criterion = 'gini', max_depth = 8, max_features = 'sqrt', min_samples_leaf = 4, min_samples_split = 5, n_estimators = 150)


# Fit the model to the training Instances
rand_clf.fit(X_train, Y_train)


# Predicting on The Test Results
Y_pred = rand_clf.predict(X_test)


# Evaluating the Random Forest Classifier model for performance
rand_clf_train_acc = accuracy_score(Y_train, rand_clf.predict(X_train))
rand_clf_test_acc = accuracy_score(Y_test, Y_pred)


# Checking the Random Forest Classifier Accuracy of train and test to know the difference
print(f"Training Accuracy of Random Forest Classifier Model is {rand_clf_train_acc}")
print(f"Test Accuracy of Random Forest Classifier Model is {rand_clf_test_acc}")



import pandas as pd
import pickle
import joblib

### Reading the Test Data 
data_new = pd.read_excel(r"C:\Users\sony\my first project\labour _test.xlsx")

### Deleting the columns which are present in the test data
labour_new1 = data_new.drop(['Emp ID','Name','Nationality','Martial Status','Designation','Experience','Work Date','Site','Rfid','Latitude','Longitude','Work Started Time','Noise Detection','Gas Sensor'], axis=1)
labour_new1.columns

### Loading all the pickle and Joblib files 
model = pickle.load(open('DT.pkl','rb'))
impute = joblib.load('meanimpute')
winsor = joblib.load('winsor')
encoding = joblib.load('encoding')
scale = joblib.load('minmax')

## Taking Only The Numeric Features for Simple Imputation, Winsorization, MinMaxScaling 
numeric_features = labour_new1.select_dtypes(exclude=['object']).columns

numeric_features

## Taking Only The Categorical Features for OneHotEncoding 
categorical_features = labour_new1.select_dtypes(include=['object']).columns

categorical_features

impute = pd.DataFrame(impute.transform(labour_new1),columns=numeric_features)

impute[['Age','GalvanicSkinResponseSensor', 'SkinTemperatureSensor','BloodVolumePulse', 'RespirationRateSensor', 'HeartRateSensor']] = winsor.transform(impute[['Age','GalvanicSkinResponseSensor', 'SkinTemperatureSensor','BloodVolumePulse', 'RespirationRateSensor', 'HeartRateSensor']])

clean2=pd.DataFrame(scale.transform(impute))
clean3=pd.DataFrame(encoding.transform(labour_new1).todense())
clean_data=pd.concat([clean2,clean3],axis=1,ignore_index=True)
prediction=pd.DataFrame(model.predict(clean_data),columns=['Performance'])
final_data=pd.concat([prediction,data_new],axis=1)
