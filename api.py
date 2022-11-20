from flask import Flask
from flask import request
from joblib import load
import numpy as np
import os

class Model:
    def __init__(self):
        self.model_path = ""
        self.model = None
        
    def load(self, model_path):
        if model_path == self.model_path:
            print("Using existing model")
            return
        else:
            print("Loading new model")
            self.model = load(model_path)
            self.model_path = model_path
            return

app = Flask(__name__)
app.config['MODEL'] = Model()

def find_best_model_implementation():
    print("Reading all the files")
    best_model_path = ""
    best_model_metric = -1.0
    for file in os.listdir("./results"):
        if file.endswith(".txt"):
            result_file = os.path.join("./results", file)
            with open(result_file) as f:
                lines = f.readlines()
            macro_f1_score = float(lines[1].split(" ")[-1])
            if macro_f1_score > best_model_metric:
                best_model_metric = macro_f1_score
                best_model_path = lines[2].split(" ")[-1]
    return best_model_path

@app.route("/predict", methods=['POST'])
def predict_digit():
    image = request.json['image']
    model_name = request.json['model_name']
    if model_name == "":
        app.config['MODEL'].load(find_best_model_implementation())
    else:
        app.config['MODEL'].load(model_name)
    predicted = app.config['MODEL'].model.predict([
        np.array(image)
    ])
    return {"y_predicted":int(predicted[0])}

@app.route("/find_best_model", methods=['GET'])
def find_best_model():
    best_model = find_best_model_implementation()
    return f"<h1>{str(best_model)}</h1>"
