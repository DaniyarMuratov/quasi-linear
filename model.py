import pandas as pd
import xgboost as xgb
import cowsay
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

data = pd.read_csv('wfp_food_prices_kaz.csv')

data['year'] = pd.to_datetime(data['date']).dt.year
data['month'] = pd.to_datetime(data['date']).dt.month
data['day'] = pd.to_datetime(data['date']).dt.day

data = pd.get_dummies(data, columns=['market', 'commodity', 'date'])
X = data.drop('price', axis=1)
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=68)

model_xg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5,
                         learning_rate=0.2, random_state=84)
model_xg.fit(X_train._get_numeric_data(), y_train)
y_pred_xg = model_xg.predict(X_test._get_numeric_data())
mse_xg = mean_squared_error(y_test, y_pred_xg)
accuracy_xg = model_xg.score(X_test, y_test)

cowsay.cow(f"Линейная регрессия\nСреднеквадратичная ошибка: {round(mse_xg, 2)}")

model_tree = DecisionTreeRegressor()
model_tree.fit(X_train, y_train)
y_pred_tree = model_tree.predict(X_test)
mse_tree = mean_squared_error(y_test, y_pred_tree)

cowsay.tux(f"Решающее дерево\nСреднеквадратичная ошибка: {round(mse_tree, 2)}")

model_rf = RandomForestRegressor()
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)


cowsay.stegosaurus(f"Random Forest\nСреднеквадратичная ошибка: {round(mse_rf, 2)}")


y_pred = (y_pred_xg + y_pred_tree + y_pred_rf) / 3
mse = mean_squared_error(y_test, y_pred)
cowsay.dragon(f"Среднеквадратичная ошибка для комбинированных\nпредсказаний: {round(mse, 2)}")