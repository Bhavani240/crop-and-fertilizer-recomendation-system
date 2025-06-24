import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Sample Data (Replace with your actual dataset)
data = pd.read_csv("crop_data.csv")  # Ensure you have a dataset
X = data[["N","P","K","temperature","humidity","ph","rainfall"
]]
y = data["label"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the Model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully as model.pkl")
