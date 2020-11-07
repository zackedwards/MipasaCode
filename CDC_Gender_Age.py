import mipasa
import dateutil
from datetime import datetime as dt
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.dates as mdates
import time
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import statistics
import statsmodels.api as sm

client = mipasa.Client()
feed = client.get_feed_by_name('CDC Provisional COVID-19 Death Counts')
data = feed.get_file('Death Counts_Source.csv').get_as_dataframe()

data = data.rename(columns={'COVID-19 Deaths': 'deaths'})
data = data.rename(columns={'Age group': 'age'})
data = data.dropna(axis=0,how='any')

data = data.astype({'deaths': int})

data = data.drop(['Data as of','Start week','End Week','Total Deaths','Pneumonia Deaths','Pneumonia and COVID-19 Deaths','Influenza Deaths','Pneumonia, Influenza, or COVID-19 Deaths'],axis=1)

states = data.State.unique()
states2 = []
for i in states:
  if 'Total' in i:
    continue
  else:
    states2.append(i)
    
us_data = data[data.values == 'United States']
all_sex = us_data[us_data.values == 'All Sexes']
female = us_data[us_data.values == 'Female']
male = us_data[us_data.values == 'Male']
x = np.arange(len(male.age))
width = 0.4

fig1, ax1 = plt.subplots(figsize=(15,20))
plt.bar(x + width/2, height = male.deaths, width = width, align = 'center', color = 'darkturquoise',tick_label = male.age, label = 'Men')
plt.bar(x - width/2, height = female.deaths, width = width, align = 'center', color = 'pink',tick_label = female.age, label = 'Women')
plt.title('US Male VS Female COVID19 Deaths By Age',fontsize=20)
plt.ylabel("Deaths",fontsize=20)
plt.xlabel("Age Group",fontsize=20)
plt.xticks(rotation = 45,fontsize = 15)
plt.yticks(rotation = 0, fontsize = 20)
plt.legend(loc = 2, fontsize = 'xx-large')
plt.show()

male.reset_index()
female.reset_index()
temp= []
for i in male.deaths:
  temp.append(i)
total_list = []
for i in all_sex.deaths:
  total_list.append(i)
female_temp = []
for i in female.deaths:
  female_temp.append(i)
male_height = [float(i)/float(j) for i,j in zip(temp, total_list)]
female_height = [float(i)/float(j) for i,j in zip(female_temp, total_list)]

fig1, ax1 = plt.subplots(figsize=(15,20))
plt.bar(x + width/2, height = male_height, width = width, align = 'center', color = 'darkturquoise',tick_label = male.age, label = 'Men')
plt.bar(x - width/2, height = female_height, width = width, align = 'center', color = 'pink',tick_label = female.age, label = 'Women')
plt.title('US Male VS Female COVID19 Deaths as Percentage By Age',fontsize=20)
plt.ylabel("Deaths",fontsize=20)
plt.xlabel("Age Group",fontsize=20)
plt.xticks(rotation = 45,fontsize = 15)
plt.yticks(rotation = 0, fontsize = 20)
plt.legend(loc = 2, fontsize = 'xx-large')
plt.show()

fig, ax = plt.subplots(figsize=(15,20))
plt.bar(x, height = all_sex.deaths, color = 'slategray',tick_label = all_sex.age)
plt.title('US All Gender COVID19 Deaths By Age',fontsize=20)
plt.ylabel("Deaths",fontsize=20)
plt.xlabel("Age Group",fontsize=20)
plt.xticks(rotation = 45,fontsize = 15)
plt.yticks(rotation = 0, fontsize = 20)
plt.legend()
plt.show()