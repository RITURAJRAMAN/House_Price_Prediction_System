import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route("/")
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    input_data = [float(x) for x in request.form.values()]
    variables_data = [np.array(input_data)]

    features_name = ['YearBuilt', 'Area', 'NumBedRooms', 'AreaBedroom', 'BedroomCond', 'NumKitch', 'AreaKitch',
                     'KitchCond', 'Garage', 'GarageArea', 'Electricity', 'AirConditioning', 'NumHearth',
                     'HouseCondition', 'Pool', 'Garden']

    df = pd.DataFrame(variables_data, columns=features_name)
    house_price = int(model.predict(df))
    return render_template('index.html', prediction_text='The Estimated Price for the House is ₹{}'.format(house_price))


if __name__ == "__main__":
    app.run(debug=True)
