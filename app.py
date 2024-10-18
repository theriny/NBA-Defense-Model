from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('offensive_stats_model.pkl')

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

            # Convert input to a numpy array and reshape for the model
            input_data = np.array(inputs).reshape(1, -1)

            # Make the prediction (will return a vector with 2 values (prob of losing, prob of winning))
            prediction = model.predict_proba(input_data)

            #get the lose/win probabilities
            prob_lose = np.round(prediction[0][0]*100)
            prob_win = np.round(prediction[0][1]*100)

            result = f"Given the offensive stats you have provided, your team has a {prob_win}% chance of winning and a {prob_lose}% chance of losing."

            # Return prediction result to the template
            return render_template("index.html", result=result)

        except ValueError:
            return render_template("index.html", error="Please enter valid float numbers.")
    
    # For GET requests, just render the form
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
