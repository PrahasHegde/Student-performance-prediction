import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression



df = pd.read_csv("Student_Performance.csv")

print(df.head())
print(df.shape)
print(df.info())

#split the dataset into features and labels.

y = df['Performance Index'].copy()
X = df.drop(columns='Performance Index').copy()

#One-Hot encoding to handle categorical values in extracurricular activities
X = pd.get_dummies(X, columns=['Extracurricular Activities'])

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)


#model
#Support Vector Regressor(SVR)
svr = SVR()
svr.fit(X_train, y_train)
svr_prediction = svr.predict(X_test)

#metrics
svr_mae = mean_absolute_error(y_test, svr_prediction)
print(svr_mae) # 1.8384445448641034



#scatter plot for actual vs predicted
actual = y_test
predicted = svr_prediction

# Plot the actual values as a scatter plot
plt.scatter(range(len(actual)), actual, color='blue', label='Actual')

# Plot the predicted values as a line
plt.scatter(range(len(actual)), predicted, color='red', label='Predicted')

# A line between the actual point and predicted point
for i in range(len(actual)):
    plt.plot([i, i], [actual.iloc[i], predicted[i]], color='green', linestyle='--')

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Actual vs Predicted Values (Stock price prediction)')
plt.legend()
plt.show()


# #Linear Regression
# lr = LinearRegression()
# lr.fit(X_train, y_train)
# lr_prediction = lr.predict(X_test)
# lr_mae = mean_absolute_error(y_test, lr_prediction)
# lr_mae # 1.6313121603176193