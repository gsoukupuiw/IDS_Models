import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.keras.models import load_model
import numpy as np

def print_feature_info(df, feature_names):
    print("Number of selected features:", len(feature_names))
    print("Names of the features:", feature_names)
    print("DataFrame preview:\n", df.head())


# Function to calculate and print F1 score
def calculate_f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f'Precision: {precision}, Recall: {recall}, F1 Score: {f1}')

# Define selected features
selected_features = [
    'frame.time_epoch', 'frame.len', 'radiotap.channel.freq', 'radiotap.datarate',
     'wlan_radio.data_rate','wlan_radio.signal_dbm', 'wlan.sa', 'wlan.da', 'Label'
]

# Initialize an empty DataFrame to store all data
all_data = pd.DataFrame()

# Loop through all CSV files and append them to the DataFrame
for i in range(33):
    file_path = f'/home/cell/Downloads/1.Deauth/Deauth_{i}.csv'
    temp_df = pd.read_csv(file_path, usecols=selected_features, low_memory=False)
    temp_df.replace({"?": None}, inplace=True)  # Replace '?' with None
    all_data = pd.concat([all_data, temp_df], ignore_index=True)

# Drop columns with more than 50% null data
null_columns = all_data.columns[all_data.isnull().mean() >= 0.5]
all_data.drop(null_columns, axis=1, inplace=True)

# Drop rows with any null values
all_data.dropna(inplace=True)

# Convert all columns to numeric where possible
for col in all_data.columns:
    all_data[col] = pd.to_numeric(all_data[col], errors='ignore')

# Convert all categorical columns to string type
categorical_cols = all_data.select_dtypes(include=['object']).columns
all_data[categorical_cols] = all_data[categorical_cols].astype(str)

# Label encode categorical features
label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    all_data[col] = label_encoders[col].fit_transform(all_data[col])

# Print feature information
print_feature_info(all_data, selected_features)


# Features and Labels
X = all_data.drop(['Label'], axis=1)
y = all_data['Label']

# Split data into training (70%), testing (20%), and final evaluation set (10%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_test, X_final_eval, y_test, y_final_eval = train_test_split(X_temp, y_temp, test_size=(2/3), random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_final_eval_scaled = scaler.transform(X_final_eval)  # Scale the final evaluation set

# One-hot encode labels
y_train_onehot = to_categorical(y_train)
y_test_onehot = to_categorical(y_test)
y_final_eval_onehot = to_categorical(y_final_eval)  

# DNN Architecture for Increased Depth
model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25)) # Added an additional layer with 128 neurons
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train_onehot, epochs=8, batch_size=512, validation_split=0.5)

# Predict and calculate F1 score for the test set
y_pred_test = model.predict(X_test_scaled)
y_pred_test_labels = np.argmax(y_pred_test, axis=1)
calculate_f1_score(y_test, y_pred_test_labels)

# Predict and calculate F1 score for the final evaluation set
y_pred_final_eval = model.predict(X_final_eval_scaled)
y_pred_final_eval_labels = np.argmax(y_pred_final_eval, axis=1)
calculate_f1_score(y_final_eval, y_pred_final_eval_labels)

# Save the model
model.save("DNNIncreasedDepth.h5")