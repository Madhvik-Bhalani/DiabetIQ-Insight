from flask import Flask, request, render_template
from Models.predict_new_data import predict_new_data
from Models.age_group_data_analysis.age_group_data import calculate_avg_data
import numpy as np
import pandas as pd

app = Flask(__name__, static_folder="static", static_url_path="/static")


@app.route("/")
def Home():
    return render_template("index.html", title="Home | DiabetIQ Insight")


@app.route("/predict_diabetes", methods=["GET", "POST"])
def predict_diabetes():
    if request.method == "GET":
        return render_template(
            "predict_diabetes.html", title="Assess Your Diabetes | DiabetIQ Insight"
        )
    if request.method == "POST":
        # convert form value into array
        features = [float(x) for x in request.form.values()]

        # make 2D array for Standard Scaller
        f_features = np.array(features).reshape(1, -1)

        # make predication with multiple model
        predicted_data = predict_new_data(f_features)

        # calculate age group wise avg data
        age = int(request.form.get("age"))
        age_group_avg_data = calculate_avg_data(age)

        return render_template(
            "predict_diabetes.html",
            title="Assess Your Diabetes | DiabetIQ Insight",
            datas=[predicted_data, features, age_group_avg_data],
        )


@app.route("/explore_dataset", methods=["GET"])
def explore_dataset():
    data = pd.read_csv("Dataset/diabetes.csv")
    return render_template(
        "dataset.html", title="Explore Dataset | DiabetIQ Insight", datas=data
    )


@app.route("/trained_models", methods=["GET"])
def trained_models():
    return render_template(
        "trained_models.html", title="Trained Models | DiabetIQ Insight"
    )


if __name__ == "__main__":
    app.run()
