from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pickle


X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=500)
model.fit(X, y)

with open("app/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as app/model.pkl")
