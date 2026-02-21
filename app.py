from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained pipeline
with open("loan_model.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/")
def home():
    return render_template("frontend.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get form data
    data = request.form.to_dict()

    # DataFrame for model input
    df = pd.DataFrame([data])

    # Prediction
    proba = model.predict_proba(df)[0][1]
    prediction = "Loan Approved" if proba > 0.6 else "Loan Not Approved"

    return render_template(
        "result.html",
        prediction=prediction,
        probability=round(float(proba), 3),
        previous_values=data       # send all previous values
    )


if __name__ == "__main__":
    app.run(debug=True)
