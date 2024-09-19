import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load both train.csv and test.csv datasets
@st.cache_data
def load_data():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    return train, test

train_data, test_data = load_data()

# Display the dataset
st.title("Titanic Survival Prediction")
st.write("Train Dataset Overview:")
st.write(train_data.head())

# Selecting features for the model
st.sidebar.header("Features Selection")
selected_features = st.sidebar.multiselect(
    "Select the features for training the model", 
    ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
    default=['Pclass', 'Sex', 'Age', 'Fare']
)

# Preprocessing function
def preprocess_data(data, selected_features, is_train=True):
    if is_train:
        data = data[selected_features + ['Survived']].copy()
    else:
        data = data[selected_features].copy()

    # Only apply transformations if the feature is selected
    if 'Sex' in selected_features:
        data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    
    if 'Embarked' in selected_features:
        data['Embarked'] = data['Embarked'].fillna('S').map({'S': 0, 'C': 1, 'Q': 2})
    
    if 'Age' in selected_features:
        data['Age'] = data['Age'].fillna(data['Age'].median())
    
    if 'Fare' in selected_features:
        data['Fare'] = data['Fare'].fillna(data['Fare'].median())
    
    return data

# Check if at least one feature is selected
if len(selected_features) > 0:
    processed_train_data = preprocess_data(train_data, selected_features)

    # Split the data
    X = processed_train_data[selected_features]
    y = processed_train_data['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Predictions on train data
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.write(f"Model Accuracy on Train Data: {accuracy:.2f}")

    # Preprocess the test data
    processed_test_data = preprocess_data(test_data, selected_features, is_train=False)

    # Make predictions on test.csv
    test_predictions = model.predict(processed_test_data)

    # Prepare the submission file
    submission = test_data.copy()
    submission['Survived'] = test_predictions

    # Show submission data
    st.write("Predictions on Test Data:")
    st.write(submission.head())

    # Download the predictions as a CSV file
    csv = submission.to_csv(index=False)
    st.download_button(
        label="Download Predictions",
        data=csv,
        file_name='titanic_predictions.csv',
        mime='text/csv',
    )

    # Predict survival for custom input
    st.sidebar.header("Custom Prediction")
    input_data = {}
    for feature in selected_features:
        if feature == 'Sex':
            input_data['Sex'] = st.sidebar.selectbox("Sex", ('male', 'female'))
        elif feature == 'Embarked':
            input_data['Embarked'] = st.sidebar.selectbox("Embarked", ('S', 'C', 'Q'))
        else:
            input_data[feature] = st.sidebar.number_input(feature, min_value=0.0)

    # Converting input to DataFrame
    input_df = pd.DataFrame([input_data])

    if 'Sex' in selected_features:
        input_df['Sex'] = input_df['Sex'].map({'male': 0, 'female': 1})
    if 'Embarked' in selected_features:
        input_df['Embarked'] = input_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # Make prediction on custom input
    if st.sidebar.button("Predict"):
        prediction = model.predict(input_df[selected_features])
        st.write(f"Prediction (1 = Survived, 0 = Not Survived): {prediction[0]}")
else:
    st.write("Please select at least one feature for training the model.")
