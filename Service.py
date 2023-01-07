import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor

# Load the data that we want to make predictions on from an online source
data_url = "https://univcadiayyad-my.sharepoint.com/:x:/g/personal/essowedeofrederic_faya_edu_uca_ma/EVPEGZFrStlMs_bIJ8tp33kBrczbMRzt2YH5pCqpEIeoxA?rtime=wX16wCfw2kg"
data = pd.read_csv(requests.get(data_url).text)

# Split the data into features and target
X = data.drop("target", axis=1)
y = data["target"]




def add_predictions(data_url):
    # Load the data that we want to make predictions on from the online source
    data = pd.read_csv(requests.get(data_url).text)

    # Make predictions on the input data
    predictions = model.predict(data)

    # Add the predictions to the input data as a new column
    data["prediction"] = predictions

    # Save the modified data back to the online source
    requests.post(data_url, data=data.to_csv())

    return data
