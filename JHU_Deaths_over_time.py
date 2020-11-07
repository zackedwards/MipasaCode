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

#getting the data
client = mipasa.Client()
feed = client.get_feed_by_name('JHU CSSE')
confirmed = feed.get_file('Output_JHU_Confirmed.csv').get_as_csv()
deaths = feed.get_file('Output_JHU_Deaths.csv').get_as_csv()
recov = feed.get_file('Output_JHU_Recovered.csv').get_as_csv()

#organizing the data
confirm = pd.DataFrame(confirmed[1:], columns = confirmed[0])
deaths = pd.DataFrame(deaths[1:], columns = deaths[0])
recov = pd.DataFrame(recov[1:], columns = recov[0])
temp = []
for i in confirm.cases:
  x = int(i)
  temp.append(x)
confirm.cases = temp
temp = []
for i in deaths.cases:
  x = int(i)
  temp.append(x)
deaths.cases = temp
temp = []
for i in recov.cases:
  x = int(i)
  temp.append(x)
recov.cases = temp

worldwide_cases = confirm.groupby('date')['cases'].sum().reset_index()
worldwide_deaths = deaths.groupby('date')['cases'].sum().reset_index()
worldwide_recov = recov.groupby('date')['cases'].sum().reset_index()

#creating a worldwide sum
worldwide_sum = pd.DataFrame()
worldwide_sum['active'] = worldwide_deaths['cases'] / worldwide_cases['cases']
worldwide_sum['date'] = pd.to_datetime(worldwide_cases['date'])
worldwide_sum['date']=worldwide_sum['date'].map(dt.toordinal)

#setting up the X,Y and the linear regression under Y_pred
X = worldwide_sum.date.to_numpy().reshape(-1,1)
Y = worldwide_sum.active.to_numpy().reshape(-1,1)
#print("x ",X)
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions

#setting up the plot
fig1, ax1 = plt.subplots(figsize=(10,5))
sns.set_style("whitegrid")
plt.scatter(X,Y)
plt.plot(X, Y_pred, color='red')

sns.set_style("whitegrid")
plt.title('Worldwide Ratio of Deaths Per Cases Over Time',fontsize=20)
plt.ylabel("Ratio (Deaths / Cases)",fontsize=15)
plt.xlabel("Date: 1/22/20-current",fontsize=15)
plt.tick_params(labelbottom=False, bottom=False)
plt.show()