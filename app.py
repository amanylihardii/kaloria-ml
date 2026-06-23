from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("rf_calories.pkl")
encoder = joblib.load("gender_encoder.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    level = None
    apple = None
    rice = None
    steps = None

    if request.method == "POST":

        gender = request.form["gender"]
        age = float(request.form["age"])
        height = float(request.form["height"])
        weight = float(request.form["weight"])
        duration = float(request.form["duration"])
        heart_rate = float(request.form["heart_rate"])
        body_temp = float(request.form["body_temp"])

        gender_encoded = encoder.transform([gender])[0]

        data = pd.DataFrame([{
            "Gender": gender_encoded,
            "Age": age,
            "Height": height,
            "Weight": weight,
            "Duration": duration,
            "Heart_Rate": heart_rate,
            "Body_Temp": body_temp
        }])

        prediction = round(float(model.predict(data)[0]), 2)

        if prediction < 150:
            level = "Low"
        elif prediction < 300:
            level = "Medium"
        else:
            level = "High"

        apple = round(prediction / 80, 1)
        rice = round(prediction / 25, 1)
        steps = round(prediction * 25)

    return render_template(
        "index.html",
        prediction=prediction,
        level=level,
        apple=apple,
        rice=rice,
        steps=steps
    )


if __name__ == "__main__":
    app.run(debug=True)