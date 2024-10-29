import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.inspection import permutation_importance
from pgmpy.models import BayesianNetwork
import io

# Original dataset
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Encoding the dataset for Naive Bayes
le_outlook = LabelEncoder()
le_temperature = LabelEncoder()
le_humidity = LabelEncoder()
le_wind = LabelEncoder()
le_play = LabelEncoder()

df['Outlook'] = le_outlook.fit_transform(df['Outlook'])
df['Temperature'] = le_temperature.fit_transform(df['Temperature'])
df['Humidity'] = le_humidity.fit_transform(df['Humidity'])
df['Wind'] = le_wind.fit_transform(df['Wind'])
df['PlayTennis'] = le_play.fit_transform(df['PlayTennis'])

# Features and target
X = df[['Outlook', 'Temperature', 'Humidity', 'Wind']]
y = df['PlayTennis']

# Sidebar options for train-test split
st.sidebar.header("Model Training Options")
test_size = st.sidebar.slider("Test Size", min_value=0.1, max_value=0.5, value=0.2)

# Train-test split and model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
model = CategoricalNB()
model.fit(X_train, y_train)

# Streamlit app layout
st.title("Play Tennis Prediction")

# Show the dataset
st.subheader("Dataset")
st.write(pd.DataFrame(data))

# Input fields for prediction
st.subheader("Make a Prediction")
new_outlook = st.selectbox("Outlook", ['Sunny', 'Overcast', 'Rain'])
new_temperature = st.selectbox("Temperature", ['Hot', 'Mild', 'Cool'])
new_humidity = st.selectbox("Humidity", ['High', 'Normal'])
new_wind = st.selectbox("Wind", ['Weak', 'Strong'])

if st.button("Predict PlayTennis"):
    try:
        # Transform input values for prediction
        encoded_instance = [
            le_outlook.transform([new_outlook])[0],
            le_temperature.transform([new_temperature])[0],
            le_humidity.transform([new_humidity])[0],
            le_wind.transform([new_wind])[0]
        ]
        prediction = model.predict([encoded_instance])[0]
        play_prediction = le_play.inverse_transform([prediction])[0]
        st.write(f"Prediction: PlayTennis = {play_prediction}")
    except ValueError as e:
        st.write(f"Error: {str(e)}. Please check your inputs.")

# Display model performance
st.subheader("Model Performance")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

if st.checkbox("Show Confusion Matrix"):
    cm = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:", cm)

# Plot feature distributions
st.subheader("Feature Distributions")
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

axs[0, 0].hist(data['Outlook'], bins=3, color='skyblue', edgecolor='black')
axs[0, 0].set_title('Outlook Distribution')

axs[0, 1].hist(data['Temperature'], bins=3, color='lightgreen', edgecolor='black')
axs[0, 1].set_title('Temperature Distribution')

axs[1, 0].hist(data['Humidity'], bins=2, color='salmon', edgecolor='black')
axs[1, 0].set_title('Humidity Distribution')

axs[1, 1].hist(data['Wind'], bins=2, color='orange', edgecolor='black')
axs[1, 1].set_title('Wind Distribution')

plt.tight_layout()
st.pyplot(fig)

# Customizable Bayesian Belief Network
st.subheader("Customize Bayesian Network")
show_nodes = st.multiselect("Select nodes to include:", ['Outlook', 'Temperature', 'Humidity', 'Wind', 'PlayTennis'], default=['Outlook', 'Temperature', 'Humidity', 'Wind', 'PlayTennis'])

# Define Bayesian Network structure
bayesian_model = BayesianNetwork([('Outlook', 'PlayTennis'), 
                                  ('Temperature', 'PlayTennis'), 
                                  ('Humidity', 'PlayTennis'), 
                                  ('Wind', 'PlayTennis')])

# Filter edges based on selected nodes
filtered_edges = [(u, v) for u, v in bayesian_model.edges() if u in show_nodes and v in show_nodes]
G_filtered = nx.DiGraph(filtered_edges)

plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G_filtered)
nx.draw(G_filtered, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=12, font_weight='bold', arrows=True)
plt.title("Bayesian Belief Network")
st.pyplot(plt)

# Feature Importances (Permutation Importance)
st.subheader("Feature Importances using Permutation Importance")
perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

fig, ax = plt.subplots()
ax.barh(X.columns, perm_importance.importances_mean, color='skyblue')
ax.set_xlabel("Mean Importance")
ax.set_title("Permutation Importance of Features")
st.pyplot(fig)

# Batch Prediction Export
st.subheader("Batch Prediction Export")
if st.button("Export Predictions"):
    predictions = model.predict(X_test)
    prediction_df = X_test.copy()
    prediction_df['Predicted_PlayTennis'] = le_play.inverse_transform(predictions)
    
    # Convert to CSV
    csv = prediction_df.to_csv(index=False)
    st.download_button("Download Predictions as CSV", data=csv, file_name="predictions.csv", mime="text/csv")

# Upload New Data for Real-Time Analysis
st.subheader("Upload New Data")
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    new_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", new_data.head())
    
    # Encode new data for prediction
    for col in new_data.columns:
        if col == 'Outlook':
            new_data[col] = le_outlook.transform(new_data[col])
        elif col == 'Temperature':
            new_data[col] = le_temperature.transform(new_data[col])
        elif col == 'Humidity':
            new_data[col] = le_humidity.transform(new_data[col])
        elif col == 'Wind':
            new_data[col] = le_wind.transform(new_data[col])
    
    predictions = model.predict(new_data)
    new_data['Predicted_PlayTennis'] = le_play.inverse_transform(predictions)
    st.write("Predictions for Uploaded Data:", new_data)
