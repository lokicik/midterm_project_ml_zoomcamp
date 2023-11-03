# docker build -t midterm_project .
# docker run -it --rm -p 9696:9696 midterm_project


import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pickle
from flask import Flask
from flask import request
from flask import jsonify
import numpy as np

input_drinking = 'drinking.bin'
with open(input_drinking, 'rb') as f_in:
    drinking_model = pickle.load(f_in)

input_smoking = 'smoking.bin'
with open(input_smoking, 'rb') as f_in:
    smoking_model = pickle.load(f_in)


app = Flask('pred')

@app.route('/predict', methods=['POST'])
def predict():
    patient = request.get_json()

    pretty_json = json.dumps(patient, indent=4)
    print(f"For input:\n{pretty_json}\n")
    input_array = np.array([list(patient.values())])

    # Reshape the array to have 2 dimensions
    X = input_array.reshape(1, -1)

    drinking_prediction = drinking_model.predict(X)
    predicted_class_drinking = drinking_prediction[0]
    y_pred_drinking = drinking_model.predict_proba(X)[0, 1]

    smoking_prediction = smoking_model.predict(X)
    predicted_class_smoking = smoking_model.classes_[smoking_prediction[0]]
    y_pred_smoking = smoking_model.predict_proba(X)[0]

    result = {
        'DRK_YN': {
            'Description' : '1 for Drinker / 0 for Not Drinker',
            'Prediction': int(predicted_class_drinking),
            'Predicted_Probability': float(y_pred_drinking)
        },
        'SMK_stat_type_cd': {
            'Description': '0 for Never Smoked / 1 for Used to Smoke / 2 for Still Smoking',
            'Prediction': int(predicted_class_smoking),
            'Predicted_Probability': y_pred_smoking[int(predicted_class_smoking)]
        }
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=7860)



