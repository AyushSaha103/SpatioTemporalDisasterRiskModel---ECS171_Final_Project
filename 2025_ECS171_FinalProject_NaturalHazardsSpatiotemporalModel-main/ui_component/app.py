import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# model structure taken from Gavin's model.ipynb
class DisasterLSTM(nn.Module):
    def __init__(self, input_size, output_size):
        super(DisasterLSTM, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, batch_first=True, num_layers=3)
        self.fc2 = nn.Linear(64, output_size)  
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = x.unsqueeze(1)
        _, (h_n, _) = self.lstm(x)
        x = self.fc2(h_n[-1]) 
        x = self.sigmoid(x)
        return x
    
# using forams function from disasterpred_random_forest.py
def RF_predict_disasters_by_region(models, feature_columns):
    regions = ['North East', 'North West', 'Central East', 'Central West', 'South East', 'South West']
    months = range(1, 13)
    
    all_predictions = []
    
    for region in regions:
        if 'North' in region:
            lat = 42
        elif 'Central' in region:
            lat = 37
        else:
            lat = 32
            
        if 'East' in region:
            lon = -85
        else:
            lon = -110
        
        for month in months:
            # Create base features
            feature_dict = {
                'latitude': lat,
                'longitude': lon,
                'month': month,
                'month_sin': np.sin(2 * np.pi * month/12),
                'month_cos': np.cos(2 * np.pi * month/12)
            }
            
            # Add one-hot encoded region columns
            for col in feature_columns:
                    if col.startswith('region_'):
                        feature_dict[col] = 1 if col == f'region_{region.replace(" ", "_")}' else 0
            
            features_df = pd.DataFrame([feature_dict])
            
            prediction_row = {
                'region': region,
                'month': month
            }
            
            # Predict probabilities for each disaster type
            for disaster, model in models.items():
                prob = model.predict_proba(features_df)[0][1]
                prediction_row[disaster] = prob
            
            all_predictions.append(prediction_row)
    
    return pd.DataFrame(all_predictions)


st.set_page_config(page_title="Natural Hazards Prediction", layout="wide")

st.title("Natural Hazards Prediction Model")
st.write("Predict the probability of natural hazards based on location and time.")

@st.cache_resource
def load_pytorch_model():
	model = DisasterLSTM(input_size=3,output_size=1)
	model.load_state_dict(torch.load("ui_component/saved_prediction_models/disaster_lstm.pth"))
	model.eval()
	scaler = joblib.load("ui_component/saved_prediction_models/scaler.pkl")
	return model, scaler

def load_disaster_type_sklearn_model():
    models = {}
    disasters = ['disasterType_storm', 'disasterType_flood', 'disasterType_rain',
                     'disasterType_ice', 'disasterType_snow', 'disasterType_blizzard',
                     'disasterType_hurricane']
    for disaster in disasters:
        models[disaster] = joblib.load(f"ui_component/saved_prediction_models/rfc_disaster_models/random_forest_{disaster}.joblib")
    return models

def load_funding_type_sklearn_model():
    model = joblib.load("ui_component/saved_prediction_models/rf_funding_model.joblib")
    return model

def predicting_funding_type(model, year, region_index, disaster_encoding, state_code, county_code, time_days, month):
    fips_to_state_alphabetical = {
        1: (1, "AL"),  # Alabama
        2: (35, "AK"),  # Alaska
        4: (4, "AZ"),  # Arizona
        5: (2, "AR"),  # Arkansas
        6: (5, "CA"),  # California
        15: (5, "HI"),  # Hawaii (same as California)
        8: (6, "CO"),  # Colorado
        9: (7, "CT"),  # Connecticut
        10: (8, "DE"),  # Delaware
        11: (8, "DC"),  # District of Columbia (same code as Maryland)
        12: (9, "FL"),  # Florida
        13: (10, "GA"), # Georgia
        16: (12, "ID"), # Idaho
        17: (13, "IL"), # Illinois
        18: (14, "IN"), # Indiana
        19: (11, "IA"), # Iowa
        20: (15, "KS"), # Kansas
        21: (16, "KY"), # Kentucky
        22: (17, "LA"), # Louisiana
        23: (20, "ME"), # Maine
        24: (19, "MD"), # Maryland
        25: (18, "MA"), # Massachusetts
        26: (21, "MI"), # Michigan
        27: (31, "MN"), # Minnesota
        28: (24, "MS"), # Mississippi
        29: (23, "MO"), # Missouri
        30: (25, "MT"), # Montana
        31: (28, "NE"), # Nebraska
        32: (32, "NV"), # Nevada
        33: (29, "NH"), # New Hampshire
        34: (30, "NJ"), # New Jersey
        35: (30, "NM"), # New Mexico
        36: (33, "NY"), # New York
        37: (26, "NC"), # North Carolina
        38: (27, "ND"), # North Dakota
        39: (34, "OH"), # Ohio
        40: (35, "OK"), # Oklahoma
        41: (36, "OR"), # Oregon
        42: (37, "PA"), # Pennsylvania
        72: (38, "PR"), # Puerto Rico
        44: (39, "RI"), # Rhode Island
        45: (40, "SC"), # South Carolina
        46: (41, "SD"), # South Dakota
        47: (42, "TN"), # Tennessee
        48: (43, "TX"), # Texas
        49: (44, "UT"), # Utah
        50: (48, "VT"), # Vermont
        51: (45, "VA"), # Virginia
        53: (47, "WA"), # Washington
        54: (49, "WV"), # West Virginia
        55: (50, "WI"), # Wisconsin
        56: (51, "WY")  # Wyoming
    }
    new_data = {
        'fyDeclared': [year],
        'region': [region_index],
        'fipsStateCode': [state_code],
        'fipsCountyCode': [county_code],
        'disasterType_categoricalEncoding': [disaster_encoding],
        'incidentBeginDate' : [time_days],
        'month' : [month],
    }

    data = [0] * (12 + 51 + 6)

    data[0] = new_data['fyDeclared'][0]
    data[1] = new_data['region'][0]
    data[2] = new_data['fipsStateCode'][0]
    data[3] = new_data['fipsCountyCode'][0]
    data[4] = new_data['disasterType_categoricalEncoding'][0]
    data[5] = new_data['incidentBeginDate'][0]
    data[fips_to_state_alphabetical[new_data['fipsStateCode'][0]][0] + 5] = 1
    data[56 + new_data['month'][0]] = 1

    df = pd.DataFrame(data)

    single_example = df.T.values
    single_prediction = model.predict(single_example)
    return single_prediction
    

try:
    model, scaler = load_pytorch_model()
    st.success("LSTM Model loaded successfully!")
    rfc_disaster_models = load_disaster_type_sklearn_model()
    st.success("Random Forest Model loaded successfully!")
    funding_model = load_funding_type_sklearn_model()
    st.success("Funding Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Location Input")
    
    latitude = st.number_input(
        "Latitude",
        min_value=24.5,
        max_value=49.5,
        value=37.0,
        help="Enter latitude (24.5°N to 49.5°N for continental US)"
    )
    
    longitude = st.number_input(
        "Longitude",
        min_value=-125.0,
        max_value=-66.5,
        value=-95.0,
        help="Enter longitude (-125°W to -66.5°W for continental US)"
    )

    state_code = st.number_input(
        "State Code",
        min_value=1.0,
        max_value=78.0,
        value=30.0,
        help="Enter valid state code"
    )

    county_code = st.number_input(
        "County Code",
        min_value=1.0,
        max_value=999.0,
        value=100.0,
        help="Enter valid county code"
    )

with col2:
    st.subheader("Time Input")
    
    time_days = st.number_input(
        "Time (days from reference)",
        min_value=0,
        max_value=30000,
        value=20000,
        help="Enter the time in days from the reference date"
    )

    year = st.number_input(
        "Year",
        min_value=1969,
        max_value=3000,
        value=2025,
        help="Enter the year of the disaster"
    )

region_options = ['North East', 'North West', 'Central East', 'Central West', 'South East', 'South West']

with col3:
    st.subheader("Month Input")
    
    month_input = st.number_input(
        "Month",
        min_value=1,
        max_value=12,
        value=1,
        help="Enter the month of year as a number between 1 and 12"
    )

    st.subheader("Region Selection")
    selected_region = st.selectbox(
        "Select Region",
        options=region_options,
        index=0,
        help="Select a region of the United States"
    )

    region_index = region_options.index(selected_region)

if st.button("Predict"):
    input_data = np.array([[latitude, longitude, time_days]])
    
    scaled_input = scaler.transform(input_data)
    
    input_tensor = torch.tensor(scaled_input, dtype=torch.float32)
    
    with torch.no_grad():
        prediction = model(input_tensor)
        probability = prediction.item()
    
    rf_predictions = RF_predict_disasters_by_region(rfc_disaster_models, ['latitude', 'longitude', 'month', 'month_sin', 'month_cos',
       'region_Central East', 'region_Central West', 'region_North East',
       'region_North West', 'region_South East', 'region_South West'])
    
    predicted_funding = predicting_funding_type(funding_model, year, region_index, 1.0, state_code, county_code, time_days, month_input)

    st.header("Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    
    funding_types = ['Individual Housing Program', 'Individual Assistance Program', 'Public Assistance Program', 'Hazard Mitigation Program']

    with col1:
        st.metric(
            label="Disaster Probability",
            value=f"{probability:.2%}"
        )
        st.subheader("Funding Type")
        for prediction in predicted_funding:
            for i in range(len(prediction)):
                if prediction[i] == 1:
                    st.text(f"{funding_types[i]}")
    
    with col2:
        st.subheader("Local Area Heatmap")
        
        lat_range = np.linspace(latitude - 2, latitude + 2, 20)
        lon_range = np.linspace(longitude - 2, longitude + 2, 20)
        grid_lat, grid_lon = np.meshgrid(lat_range, lon_range)
        
        grid_points = np.column_stack((
            grid_lat.flatten(),
            grid_lon.flatten(),
            np.full(grid_lat.size, time_days)
        ))
        
        scaled_points = scaler.transform(grid_points)
        input_tensor = torch.tensor(scaled_points, dtype=torch.float32)
        
        with torch.no_grad():
            predictions = model(input_tensor)
        
        prob_grid = predictions.numpy().reshape(20, 20)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(prob_grid, cmap='YlOrRd')
        plt.title(f'Local Area Disaster Probability')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        st.pyplot(fig)
    
    with col3:
        st.subheader("Disaster Probabilities")
        filtered_predictions = rf_predictions[rf_predictions['region'] == selected_region] 
        filtered_predictions = filtered_predictions[filtered_predictions['month'] == month_input]
        for disaster in rfc_disaster_models.keys():
            prob = filtered_predictions[disaster].iloc[0]
            disaster_name = disaster.replace('disasterType_', '').capitalize()
            st.metric(
                label=f"{disaster_name}",
                value=f"{prob:.2%}"
            )
