from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

app = Flask(__name__)

# Load Models & Historical Data
model_bin = joblib.load('flight_delay_binary_model.pkl')
model_multi = joblib.load('flight_delay_severity_model.pkl')
airline_avg = joblib.load('airline_avg.pkl')
origin_avg = joblib.load('origin_avg.pkl')
route_avg = joblib.load('route_avg.pkl')
route_dist = joblib.load('route_distance.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Grab data from the HTML form 
    airline = request.form['airline'].upper()
    origin = request.form['origin'].upper()
    dest = request.form['dest'].upper()
    flight_date_str = request.form['flight_date']
    dep_hour = int(request.form['dep_hour'])
    weather = request.form['weather'] 

    # Process Dates, Times, and Route
    flight_date = datetime.strptime(flight_date_str, '%Y-%m-%d')
    month = flight_date.month
    day = flight_date.day
    day_of_week = flight_date.weekday() + 1
    
    route = f"{origin}_{dest}"
    peak_hour = 1 if dep_hour in [6,7,8,9,17,18,19,20] else 0
    is_weekend = 1 if day_of_week in [6, 7] else 0
    
    if 5 <= dep_hour < 12: time_block = 'Morning'
    elif 12 <= dep_hour < 17: time_block = 'Afternoon'
    elif 17 <= dep_hour < 21: time_block = 'Evening'
    else: time_block = 'Night'

    # AUTOMATIC LOOKUPS
    distance = route_dist.get(route, 1000.0) 
    avg_a = airline_avg.get(airline, 10.0)
    avg_o = origin_avg.get(origin, 10.0)
    avg_r = route_avg.get(route, 10.0)

    # Build the DataFrame for the Model (No Departure Delay cheat code!)
    input_df = pd.DataFrame([{
        'MONTH': month, 'DAY': day, 'DAY_OF_WEEK': day_of_week,
        'AIRLINE': airline, 'AIRLINE_NAME': airline, 
        'ORIGIN_AIRPORT': origin, 'DESTINATION_AIRPORT': dest,
        'ORIGIN_CITY': 'Unknown', 'ORIGIN_STATE': 'Unknown',
        'DESTINATION_CITY': 'Unknown', 'DESTINATION_STATE': 'Unknown',
        'ROUTE': route, 'SCHEDULED_DEPARTURE': dep_hour * 100,
        'DISTANCE': distance, 'DEP_HOUR': dep_hour,
        'PEAK_HOUR': peak_hour, 'IS_WEEKEND': is_weekend,
        'TIME_BLOCK': time_block, 'AIRLINE_AVG_DELAY': avg_a,
        'ORIGIN_AVG_DELAY': avg_o, 'ROUTE_AVG_DELAY': avg_r,
        'WEATHER_CONDITION': weather, 
        'AIR_SYSTEM_DELAY': 0, 'LATE_AIRCRAFT_DELAY': 0, 
        'WEATHER_DELAY': 0, 'SECURITY_DELAY': 0
    }])

    # Run Prediction
    pred_binary = model_bin.predict(input_df)[0]
    pred_multi = model_multi.predict(input_df)[0]

    # Format the Results
    if pred_binary == 0:
        result_text = "✅ Flight is expected to be ON TIME."
        severity_text = ""
    else:
        final_range = '0-30 min' if pred_multi == 'On Time' else pred_multi
        result_text = "⚠️ Flight is likely to be DELAYED."
        severity_text = f"Expected Delay Range: {final_range}"

    distance_text = f"Auto-calculated route distance: {int(distance)} miles."
    
    return render_template('index.html', prediction=result_text, severity=severity_text, calc_dist=distance_text)

if __name__ == '__main__':
    app.run(debug=True)