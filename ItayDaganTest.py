#!/usr/bin/env python
# coding: utf-8

# # DS Home Test - Itay Dagan
# 
# 
# ### initialization

# In[ ]:


#Importing relevant libraries for DS regression problem

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge


# In[ ]:


#Import the files (press, items, and demand)

press=pd.read_csv("C:/Users/dagani/OneDrive - HP Inc/Itay General/DS_Test/press.csv")
item=pd.read_csv("C:/Users/dagani/OneDrive - HP Inc/Itay General/DS_Test/Item attribute.csv")
demand=pd.read_csv("C:/Users/dagani/OneDrive - HP Inc/Itay General/DS_Test/demand_test.csv")


# ### Dealing with missing values

# In[ ]:


# checking % missing data in item file

item.isnull().sum()/len(item)


# In[ ]:


item.Criticality.value_counts()


# In[ ]:


#less then 2% missing data for "Criticality", i'll fill them with "A" based on other data in the set

item['Criticality']=item['Criticality'].fillna('A')


# In[ ]:


#making sure not other missing values in dataset

item.isnull().sum()/len(item)


# In[ ]:


# checking missing data in item file

demand.isnull().sum()/len(demand)


# In[ ]:


# deleting the few rows where country+region is missing (les then 0.5%)
# also deleting empty last unneseccesery column, and rows with 0 orders
demand=demand.drop('Unnamed: 5',axis=1)
demand=demand.dropna(subset=['Country Name', 'SPO Region'], how='all')
demand=demand[demand['Quantity Ordered']>0]
len(demand)


# ### Create one unite table with all relevenat fields from the other tables

# In[ ]:


# almost 10% missing data in importand field, i'll fill it based on the most frequent region for the country
# in each row (there are some countries that were belong to diffrent region in the database) 
df=pd.merge(left=demand,right=item,how='left',on='Item Number')
coun_Reg=demand.drop(['Item Number','Order Date','Quantity Ordered'],axis=1)
coun_Reg=coun_Reg.groupby(['Country Name']).agg(lambda x:x.value_counts().index[0])
coun_Reg


# In[ ]:


#add new region column to the table, in order to fix wrong region in the db and in order to fill the missing values
df=pd.merge(left=df,right=coun_Reg,how='left',on='Country Name')
def calc(y,x):
    if pd.isna(y):
        return x
    else:
        return y
    
df['SPO_Region']=df[['SPO Region_y','SPO Region_x']].apply(lambda df: calc(df['SPO Region_y'],df['SPO Region_x']),axis=1)


# In[ ]:


# update column names and drop unnecessary columns
df.rename(columns = {'Item Number':'Item_Number', 'Order Date':'Order_Date','Quantity Ordered':'Quantity_Ordered','purchase or manufacture':'purchase_or_manufacture'}, inplace = True)
df=df.drop(['SPO Region_x','SPO Region_y','sub assembly in the press','Country Name'],axis=1)


# In[ ]:


# update date type for order_date, create fields for year+quarter, 
df['Order_Date']=pd.to_datetime(df.Order_Date)
df['Year']=pd.DatetimeIndex(df['Order_Date']).year
df['Quarter']=pd.DatetimeIndex(df['Order_Date']).quarter


# In[ ]:


df.info()


# In[ ]:


# want to add press information (install base per quarter per press) to the main table, this field important for the model
press_unpivot=press.melt(id_vars='Press',var_name='Quarter', value_name='amount')
press_unpivot.head(10)


# In[ ]:


#create key in the press table (quarter+year+press)
press_unpivot['key']=press_unpivot['Quarter']+'_'+press_unpivot['Press']


# In[ ]:


# create same key in the df (quarter+year+press)
def fun_key(y,q,p):   
    return 'q'+q+'-'+y[-2:]+'_'+p   

df['key']=df[['Year','Quarter','Press']].apply(lambda df: fun_key(str(df['Year']),str(df['Quarter']),df['Press']),axis=1)


# In[ ]:


#merge df/press in order to add instal base
df=pd.merge(left=df,right=press_unpivot,how='left',on='key')


# In[ ]:


#Organize final fields in the df
df=df.drop(['Press_y','key','Quarter_y',],axis=1)
df.rename(columns = {'amount':'Install_Base','Quarter_x':'Quarter','Press_x':'Press'}, inplace = True)
df


# In[ ]:


#give up on filed "purchase or manu..", don'n think it relevant, i also group by Q+Y
df=df.groupby(['Item_Number','SPO_Region','Year','Quarter','Press','Criticality','Install_Base'])['Quantity_Ordered'].sum().reset_index()


# In[ ]:


#Demand by time graph
demand.plot(x='Order Date',y='Quantity Ordered',figsize=(20,10))


# In[ ]:


#final df
df.info()


# In[ ]:


#demand.plot.scatter(x='Order Date',y='Quantity Ordered',figsize=(20,10))


# In[ ]:


#before making other manipulation on df (coverting string to numbers), load the file with data to forecast
toForcast=pd.read_csv("C:/Users/dagani/OneDrive - HP Inc/Itay General/DS_Test/toForecast.csv")
dfFull=pd.concat([df,toForcast],axis=0)


# In[ ]:


dfFull.info()


# In[ ]:


#checking the labels (string) that need to convert
for label, content in dfFull.items():
    if pd.api.types.is_string_dtype(content):
       print (label)


# In[ ]:


#first i convert string to categort
for label, content in dfFull.items():
    if pd.api.types.is_string_dtype(content):
       dfFull[label]=content.astype("category").cat.as_ordered()

dfFull.info()


# In[ ]:


## turn categorial variable into numbers
for label, content in dfFull.items():
   if not pd.api.types.is_numeric_dtype(content):
        dfFull[label]=pd.Categorical(content).codes
dfFull.info()


# In[ ]:


#seperate the 2 db
df=dfFull[dfFull['Quantity_Ordered']>0]

mf_1 = dfFull['Quarter'] > 2
mf_2 = dfFull['Year'] > 2020

dfForecast=dfFull.loc[mf_1 & mf_2]

dfForecast.info()
df.info()


# In[ ]:


sns.pairplot(df)


# # Creating some machine learning algorithms

# ## Building an evaluation function

# In[ ]:


#create evaluation func
#create root mean squered log error between pred and true
def rmsle(y_test,test_prediction):
    return np.sqrt(mean_absolute_error(y_test,test_prediction))

#create func to evaluate model on a few different levels
def show_scored(model):
    train_preds=model.predict(x_train)
    test_prediction=model.predict(x_test)
    scores= {"Training MAE": mean_absolute_error(y_train,train_preds),
             "Valid MAE": mean_absolute_error(y_test,test_prediction),
             "Training RMSLE": rmsle(y_train,train_preds),
             "Valid RMSLE": rmsle(y_test,test_prediction),
             "Training R^2": r2_score(y_train,train_preds),
             "Valid R^2": r2_score(y_test,test_prediction)
             }
    return scores
    


# ## Machine learning - RandomForest regressor

# In[ ]:


get_ipython().run_cell_magic('time', '', "# first algo is randomForest, and i sepert the df by periods (old data is for training, new for testing)\n\n\nmodel=RandomForestRegressor(n_jobs=-1,\n                           n_estimators=100)\n\ndf_train=df[df['Year']<2020]\ndf_test=df[df['Year']>2019]\n\nx_train,y_train=df_train.drop('Quantity_Ordered',axis=1), df_train.Quantity_Ordered\nx_test,y_test=df_test.drop('Quantity_Ordered',axis=1), df_test.Quantity_Ordered\n\nmodel.fit(x_train,y_train)\ny_pred=model.predict(x_test)\n\nmean_absolute_error(y_pred,y_test)\nr2_score(y_test,y_pred)\n\nshow_scored(model)")


# In[ ]:


## very bad results when seperate train/test data by periods


# In[ ]:


#now i'll create the test sample randomize

x=df.drop('Quantity_Ordered',axis=1)
y=df['Quantity_Ordered']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=27)

model=RandomForestRegressor(n_jobs=-1,
                           n_estimators=100,
                           random_state=28)
model.fit(x_train,y_train)

test_prediction=model.predict(x_test)
r2_score(y_test,test_prediction)

show_scored(model)


# ### Much better score when the test samples are randomize! R^2 of 82%

# In[ ]:


df.Quantity_Ordered.mean()


# In[ ]:


#trying some other machine learning model - linear regression 

x=df.drop('Quantity_Ordered',axis=1)
y=df['Quantity_Ordered']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=42)

model=LinearRegression()
model.fit(x_train,y_train)

test_prediction=model.predict(x_test)
mean_absolute_error(y_test,test_prediction)

r2_score(y_test,test_prediction)


# In[ ]:


#trying some other machine learning model - Ridge model 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=42)

ridge_model=Ridge(alpha=100)
ridge_model.fit(x_train,y_train)
test_prediction=ridge_model.predict(x_test)
MAE=mean_absolute_error(y_test,test_prediction)

r2_score(y_test,test_prediction)


# ## Hyperparameter tuning with RandomizedSearchCV
# 
# #### i'll try to imporve the score with randomForest

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n#Different RandomForestRegressor hyperparameters\nrf_grid={"n_estimators": np.arange(10,100,10),\n        "max_depth": [None,3,5,10],\n         "min_samples_split": np.arange(2,20,2),\n         "min_samples_leaf": np.arange(1,20,2),\n         "max_features": [0.5,1,"sqrt","auto"]     \n        }\n\n# Instantiate RandomizedSearchCV\nrs_model=RandomizedSearchCV(RandomForestRegressor(n_jobs=--1,\n                                                 random_state=27),\n                                                 param_distributions=rf_grid,\n                                                 n_iter=2,\n                                                 cv=7,\n                                                 verbose=True)\nrs_model.fit(x_train,y_train)')


# In[ ]:


#find the best model hyperparameterss
rs_model.best_params_


# x=df.drop('Quantity_Ordered',axis=1)
# y=df['Quantity_Ordered']
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=27)
# 
# model=RandomForestRegressor(n_jobs=-1,
#                             n_estimators=5,
#                             min_samples_split= 12,
#                             min_samples_leaf= 13,
#                             max_features= 'auto',
#                             max_depth= 10,
#                             random_state=28)
# model.fit(x_train,y_train)
# 
# test_prediction=model.predict(x_test)
# r2_score(y_test,test_prediction)
# 
# show_scored(model)

# In[ ]:


##testing the model with the new hyperparameterss 

x=df.drop('Quantity_Ordered',axis=1)
y=df['Quantity_Ordered']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=27)

model=RandomForestRegressor(n_jobs=-1,
                            random_state=27,
                            n_estimators=10,
                            min_samples_split=16,
                            min_samples_leaf=9,
                            max_features='auto',
                            max_depth= None)

model.fit(x_train,y_train)

test_prediction=model.predict(x_test)
r2_score(y_test,test_prediction)

show_scored(model)


# 
# ## Finaly - make prediction for q3-21 & q4-21 data!
# ### I'll train the data on all of the demand set
# 

# In[ ]:


#no train/test split now
x=df.drop('Quantity_Ordered',axis=1)
y=df['Quantity_Ordered']
x_test=dfForecast.drop('Quantity_Ordered',axis=1)

model=RandomForestRegressor(n_jobs=-1,
                           n_estimators=100,
                           random_state=28)
model.fit(x,y)
test_prediction=model.predict(x_test)


# In[ ]:


#Create the forecast file
toForcast2=pd.read_csv("C:/Users/dagani/OneDrive - HP Inc/Itay General/DS_Test/toForecast.csv")
toForcast2['Quantity_Ordered']=test_prediction
toForcast2.to_csv("ForecastDemandPerReg_Q.csv")


# In[ ]:




