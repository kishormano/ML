import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

df = pd.read_csv("housing.csv")
df

df.info()

df.dropna(inplace = True)

df.info()

from sklearn.model_selection import train_test_split

x = df.drop(['median_house_value'], axis=1)
y = df['median_house_value']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

train_df = x_train.join(y_train)
train_df

train_df.hist(figsize=(15,8))

plt.figure(figsize=(15,8))
sb.heatmap(train_df.corr(), annot=True, cmap="YlGnBu")

train_df['total_rooms'] = np.log(train_df['total_rooms'] + 1)
train_df['total_bedrooms'] = np.log(train_df['total_bedrooms'] + 1)
train_df['population'] = np.log(train_df['population'] + 1)
train_df['households'] = np.log(train_df['households'] + 1)

train_df.hist(figsize=(15, 8))

train_df = train_df.join(pd.get_dummies(train_df.ocean_proximity)).drop(['ocean_proximity'], axis=1)

plt.figure(figsize=(15,8))
sb.heatmap(train_df.corr(), annot=True, cmap="YlGnBu")

plt.figure(figsize=(15, 8))
sb.scatterplot(x="latitude", y="longitude", data=train_df, hue="median_house_value", palette="coolwarm")

train_df['bedroom_ratio'] = train_df['total_bedrooms'] / train_df['total_rooms']
train_df['household_ratio'] = train_df['total_bedrooms'] / train_df['households']

plt.figure(figsize=(15,8))
sb.heatmap(train_df.corr(), annot=True, cmap="YlGnBu")

from sklearn.linear_model import LinearRegression
x_train, y_train = train_df.drop(['median_house_value'], axis=1), train_df['median_house_value']
reg = LinearRegression()
reg.fit(x_train, y_train)

test_df = x_test.join(y_test)

test_df['total_rooms'] = np.log(test_df['total_rooms'] + 1)
test_df['total_bedrooms'] = np.log(test_df['total_bedrooms'] + 1)
test_df['population'] = np.log(test_df['population'] + 1)
test_df['households'] = np.log(test_df['households'] + 1)

test_df = test_df.join(pd.get_dummies(test_df.ocean_proximity)).drop(['ocean_proximity'], axis=1)

test_df['bedroom_ratio'] = test_df['total_bedrooms'] / test_df['total_rooms']
test_df['household_ratio'] = test_df['total_bedrooms'] / test_df['households']

x_test, y_test = test_df.drop(['median_house_value'], axis=1), test_df['median_house_value']

reg.score(x_test, y_test)

