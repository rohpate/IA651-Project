### Project Done By Rohit patel - rohpate@clarkson.edu and Tarun sharma - tasharm@clarkson.edu


# PREDICTION OF TEMPERATURE AROUND THE GLOBE
#### INTRODUCTION
Every continent has the different average temperature based on region and thier location  our main aim for this project is to predict the temperature of that region,city and year based on different models

# DATA COLLECTION
Every continent has the different average temperature based on region and thier location our main aim for this project is to predict the temperature of that region
The Dataset is extracted from the data world bank and referred from University of Dayton daily average temperatures for 157 U.S. and 167 international cities. The Data updated on a regular basis and contain data from January 1, 1995 to present.
Source data for this site are from the National Climatic Data Center. The data is available for research and non-commercial purposes only.
Reference : https://academic.udayton.edu/kissock/http/Weather/default.htm

# DATA DESCRIPTION AND DIRECTORY
-> The overall data contains 2.9 millons rows and 8 columns 
1) REGION : Continent name such as Africa,Asia
2) COUNTRY : Country Name across the globe 
3) STATE : Territory of that Country
4) CITY : City of that state in the country
5) MONTH : Month of that year ranges from 1 to 12
6) DAY : Day of that Month present ranges 
7) YEAR : Year of which the details were recorded
8) AVERAGE TEMPERATURE : Avgerage temperature recorded on that date by city and region

# IMPORTING ALL THE NECESSARY PYTHON LIBRARIES 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# READING CSV FILE AND GOING THROUGH THE DATA
df=pd.read_csv("city_temperature.csv")
df

OUTPUT :

![image](https://user-images.githubusercontent.com/93997961/234970304-776fd02f-b452-4228-a8f8-27b67ea92306.png)

-> Dropping the duplicate rows

df=df.drop_duplicates()

-> see how null value exist in each column
df.isna().sum()

OUTPUT:  

![image](https://user-images.githubusercontent.com/93997961/234971850-db3d9b20-5c3b-46d9-a2fa-24e02264baff.png)


df.describe()

OUTPUT : 

![image](https://user-images.githubusercontent.com/93997961/234972279-5cc2d420-931b-42f5-b8a0-33e9be2c1554.png)

#### Removing the rows that contains 200 and 201 in the year and contains 0 in the day
df =df[ (df['Year'] != 200) & (df['Year'] != 201) & (df['Day'] != 0) ]

#### Transform the Average Temperature from Fahrenheit to Celsius
df["AvgTemperature"]=(df["AvgTemperature"]-32)*(5/9)

# Add a datetime column to use it in plotting
df['Date'] = pd.to_datetime(df[['Year','Month','Day']])

####  Remove values that are less than -50 and year equal or greater than 2020 since there exists some random drops in it
df =df[(df['AvgTemperature'] >= -50) & (df['Year'] < 2020)]

# PLOTTING THE DATASET

### PLOTTING BY REGION


![image](https://user-images.githubusercontent.com/93997961/234974857-2328df5d-904b-40e4-93be-698a5c8f659b.png)

## The hottest Average Temperature in the Dataset
df.sort_values(by = ['AvgTemperature'], ascending  = False).head(1)


![image](https://user-images.githubusercontent.com/93997961/234975164-e814f627-501f-4a1b-a36e-0d1602bed2a2.png)

## The coldest Average Temperature in the Dataset
df.sort_values(by = ['AvgTemperature'], ascending  = True).head(1)

![image](https://user-images.githubusercontent.com/93997961/234975481-43ceaa66-d431-468d-8f3f-91a2ef7823ff.png)


## PLOTTING SOME CITIES

![image](https://user-images.githubusercontent.com/93997961/234975668-fcd8951f-6c21-4906-a1ec-cfa80d51ec64.png)

### plotting the temperature of every country in a region


![image](https://user-images.githubusercontent.com/93997961/234976740-7c61e3dd-4284-4d0b-b363-d90becd4a046.png)


# plotting the temperature of every city in a country
So we have taken United Kingdom as an example

![image](https://user-images.githubusercontent.com/93997961/234977986-eb01c985-18d5-4517-b8f2-7163b04afa86.png)

## PLOTTING BASIC VISUALIZATION BY USING TABLEAU 
#### AVG Temperature of each by region

![image](https://user-images.githubusercontent.com/93997961/235270015-fb7e13e1-7d14-4e18-aa67-efa468e60d50.png)

#### AVG Temperature of some major cities from each region

![image](https://user-images.githubusercontent.com/93997961/235270500-b13a700c-2117-422c-b847-88b0dcfa8240.png)

#### AVG Temperature of top 3 region from 2018 to 2020

![image](https://user-images.githubusercontent.com/93997961/235270771-49775369-348e-4b98-8afd-a010c82a907e.png)


### Using Label Encoder (Transform non-numerical Data to numerical)
le=LabelEncoder()

df["Region"]=le.fit_transform(df["Region"])

region = dict(zip(le.classes_, range(len(le.classes_))))

df["Country"]=le.fit_transform(df["Country"])

country = dict(zip(le.classes_, range(len(le.classes_))))

df["State"]=le.fit_transform(df["State"])

state = dict(zip(le.classes_, range(len(le.classes_))))

df["City"]=le.fit_transform(df["City"])

city = dict(zip(le.classes_, range(len(le.classes_))))

![image](https://user-images.githubusercontent.com/93997961/234978562-d2884f74-0e5c-4891-81c9-1d59f83f4827.png)

# SPLTTING THE DATA INTO TRAIN AND TEST 
80% train and 20% test

split=2491106

train=df[:split]

test=df[split:]

![image](https://user-images.githubusercontent.com/93997961/234979401-48a3699f-f52f-4f77-bda8-c7adf054b6df.png)

## USING XGBOOST MODEL
xgbr=xgb.XGBRegressor(booster="dart",objective="reg:squarederror",n_estimators=151)

xgbr.fit(X_train,y_train)

xgbrPredic=xgbr.predict(X_test)

## EVALUATING MODEL RESULTS AND ACCURACY  

RMSE
print("Root Mean Squared Error (RMSE) score XGBoost:"+str(np.sqrt(mean_squared_error(y_test,xgbrPredic))))

Root Mean Squared Error (RMSE) score XGBoost:3.8601992844359025

### FROM THE BELOW RESULTS WE GOT NEARLY 86% ACCURACY FOR THIS MODEL
R^2: 
r2 = r2_score(y_test, xgbrPredic)

print("R-squared score: {:.2f}".format(r2))

R-squared score: 0.86

# PLOTTING THE RESULTS

test["prediction"]=xgbrPredic

df=df.set_index("Date")

ax=df['AvgTemperature'][(df['City'] ==city["London"])].plot(figsize=(15,5))

df["prediction"][(df['City'] ==city["London"])].plot(ax=ax,style=".")

plt.legend(["Real Data","Predictions"])

ax.set_title("Daily Average Temperature in London")

plt.xlabel("Date")

plt.ylabel("Average Temperature")

plt.show()

![image](https://user-images.githubusercontent.com/93997961/234982504-9065b73e-4459-49e5-9c2f-7fd3a67cddd9.png)

## USING GRADIENT BOOSTING MODEL 

gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

gbm.fit(X_train, y_train)

y_pred = gbm.predict(X_test)

## EVALUATING MODEL RESULTS AND ACCURACY 

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("RMSE:", rmse)

RMSE: 5.841273980934234

### FROM THE BELOW RESULTS WE GOT NEARLY 70% ACCURACY FOR THIS MODEL

r2 = r2_score(y_test,y_pred)

print("R-squared score: {:.2f}".format(r2))

R-squared score: 0.69

## LIMITATIONS
Before Trying the XGBOOST and Gradient Boosting algorithm I got only 3% accuracy for Linear Regression and after understanding the reason behind it I got to know that our model and data set is non-linear model to overcome this I used XGBoost which is a powerful algorithm that can handle complex nonlinear relationships between the features and the target variable.

### CONCLUSION
After Doing different model evaluation and finding out the accuracy by using different algorithm .XGBoost is a robust and flexible algorithm that is well-suited for temperature prediction tasks, making it a popular choice for machine learning practitioners  



















