from flask import Flask, request, jsonify
import pickle
import numpy as np


app = Flask(__name__)

with open("model.pkl","rb") as f :
    model = pickle.load(f)

@app.route('/')
def index():
    return "ML model is ready"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1,-1)
    prediction = int(model.predict(features)[0])
    return jsonify({"Predictions" : prediction})

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0")
