from flask import Flask, render_template, request
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
print("🧪 Python version:", sys.version)


# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('offensive_stats_unscl_model.pkl')

# Route to handle the form and prediction
@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # Get the input values from the form
        try:
            inputs = [float(request.form['input1']),
                      float(request.form['input2']),
                      float(request.form['input3']),
                      float(request.form['input4']),
                      float(request.form['input5']),
                      float(request.form['input6'])]

            # Make the prediction
            prediction = model.predict_proba(np.array(inputs).reshape(-1, 6))

            # Get the lose/win probabilities
            prob_lose = np.round(prediction[0][0] * 100)
            prob_win = np.round(prediction[0][1] * 100)

            # Return prediction result to the template
            return render_template("index.html", prob_lose=prob_lose, prob_win=prob_win,
                                   input1=request.form['input1'], input2=request.form['input2'],
                                   input3=request.form['input3'], input4=request.form['input4'],
                                   input5=request.form['input5'], input6=request.form['input6'])

        except ValueError:
            return render_template("index.html", error="Please enter valid float numbers.",
                                   input1=request.form['input1'], input2=request.form['input2'],
                                   input3=request.form['input3'], input4=request.form['input4'],
                                   input5=request.form['input5'], input6=request.form['input6'])

    # For GET requests, just render the form without any values
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
