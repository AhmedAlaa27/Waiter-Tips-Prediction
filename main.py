import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
data = pd.read_csv('Requirements/tips.csv')

# Group the data by day and calculate the total tips for each day
tips_by_day = data.groupby('day')['tip'].sum()

# Create a pie chart
plt.figure(figsize=(8, 6))  # Set the size of the figure
plt.pie(tips_by_day, labels=tips_by_day.index, autopct='%1.1f%%', startangle=140)

# Add title
plt.title('Distribution of Tips by Day')

# Show the plot
plt.show()

data['day'] = data['day'].map({'Thur': 0, 'Fri': 1, 'Sat': 2, 'Sun': 3})
data['sex'] = data['sex'].map({'Male': 1, 'Female': 0})
data['smoker'] = data['smoker'].map({'Yes': 1, 'No': 0})
data['time'] = data['time'].map({'Dinner': 1, 'Lunch': 0})

x = np.array(data.drop(['tip'], axis=1))
y = np.array(data['tip'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.50)

scalar = StandardScaler()
x_norm = scalar.fit_transform(x_train)

model = LinearRegression()
model.fit(x_norm, y_train)

y_predict = model.predict(x_norm)

# Features = data['total_bill', 'sex', 'smoker', 'day', 'time', 'size']
features = np.array([[250, 1, 0, 0, 1, 4]])
y_hat = model.predict(features)
print(f'The amount of tips predicted ${y_hat[0]: .2f}')

score = model.score(x_norm, y_train)
print(f'Score = {score: .2f}')
