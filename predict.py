# predict.py
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

model = joblib.load("model.joblib")
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

predictions = model.predict(X_test)
print("R2 Score from predict.py:", r2_score(y_test, predictions))
