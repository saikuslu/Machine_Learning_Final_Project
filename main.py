import datetime
import smtplib
from SVM_diabetes_predicition import diabetesPrediction
from flask import Flask, request, render_template, session, redirect


app = Flask(__name__)
app.secret_key = "Diabetes"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/Prediction", methods=['post'])
def diabetesDetails():
    # Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
    Pregnancies = int(request.form.get('Pregnancies'))
    Glucose = int(request.form.get('Glucose'))
    BloodPressure = int(request.form.get('BloodPressure'))
    SkinThickness = int(request.form.get('SkinThickness'))
    Insulin = int(request.form.get('Insulin'))
    BMI = float(request.form.get('BMI'))
    DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
    Age = int(request.form.get('Age'))
    input_data=(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)
    print(input_data)
    if diabetesPrediction(input_data):
        # submission_successful = True #or False. you can determine this.
        # return render_template("index.html", submission_successful=submission_successful,message="Diabetes",form="form")
        return render_template("index.html", message="Diabetes", color="red")
    else:
        # submission_successful = True #or False. you can determine this.
        # return render_template("index.html", submission_successful=submission_successful,message="No Diabetes",form="form")
        return render_template("index.html", message="No Diabetes", color="green")       

if __name__=="__main__":
    app.run(debug=True,port=5004)