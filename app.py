from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

"""creating instance for the flask app"""
app = Flask(__name__)

encoder = pickle.load(open('airbnb_en.pkl', 'rb'))

model = pickle.load(open('airbnb_model.pkl', 'rb'))


@app.route('/', methods=['GET', 'POST'])
def root():
    """Root returns base.html
    """
    message = 'Predicting Airbnb Optimal Price'
    return render_template('base.html', message=message)


@app.route('/request')
def request_data():
    """Receives data and returns a json file
    """
    
    accomodates = [int(request.args.get('accomodates'))]
    bathrooms = [float(request.args.get('bathrooms'))]
    bedrooms = [float(request.args.get('bedrooms'))]
    beds = [float(request.args.get('beds'))]
    bed_type = [int(request.args.get('bed_type'))]
    instant_bookable = [int(request.args.get('instant_bookable'))]
    minimum_nights = [int(request.args.get('minimum_nights'))]
    neighborhood = [int(request.args.get('neighborhood'))]
    room_type = [int(request.args.get('room_type'))]
    wifi = [int(request.args.get('wifi'))]

    # Defaulted values
    security_deposit = 0
    cleaning_fee = 10
# 
                #
                #'wifi': wifi, 'cleaning_fee': cleaning_fee
    features = {'accomodates': accomodates, 'bathrooms': bathrooms, 'bedrooms': bedrooms,'beds': beds, 'bed_type': bed_type, 'instant_bookable': instant_bookable,
                'minimum_nights': minimum_nights, 'neighborhood': neighborhood,
                'room_type': room_type, 'wifi': wifi, 'cleaning_fee': cleaning_fee, 'security_deposit': security_deposit,
                }

    # Creating a Dataframe.
    predict_data = pd.DataFrame(features, index=[1])
    features_encoder = encoder.transform(predict_data)

    # fitting the df to the model.
    prediction = model.predict(features_encoder)

    # Prediction.
    res =  [format(prediction[0], '.2f'),features]
    return (render_template('predictor.html', res=res))


if __name__ == '__main__':
    app.run(debug=True)
