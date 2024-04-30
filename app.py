from flask import Flask,request,render_template
import pickle

app = Flask(__name__, static_folder='static', static_url_path='/static')

@app.route("/")
def Home():
    return render_template('index.html',title="Home | DiabetIQ Insight");

@app.route("/predict_diabetes")
def predict_diabetes():
    return render_template('predict_diabetes.html',title="Assess Your Diabetes | DiabetIQ Insight");

if __name__ == '__main__':
    app.run()
