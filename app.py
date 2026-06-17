import pandas as pd
import matplotlib.pyplot as plt
from utils.regression import LinearRegression


df = pd.read_csv("data/salary_data.csv")
X = df[['YearsExperience']].values
y = df['Salary'].values


split = int(0.67 * len(df))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


model = LinearRegression(learning_rate=0.01, n_iters=5000)
model.fit(X_train, y_train)


preds = model.predict(X_test)


from sklearn.metrics import mean_squared_error
print("MSE:", mean_squared_error(y_test, preds))


plt.scatter(X_test, y_test, color="blue")
plt.plot(X_test, preds, color="red")
plt.show()
