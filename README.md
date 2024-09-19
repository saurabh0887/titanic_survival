# Titanic Survival Prediction with Streamlit

This project uses a **Random Forest Classifier** to predict the survival of passengers from the Titanic disaster, based on their features such as age, sex, class, etc. It also allows users to experiment with different features and make predictions on a custom input via a **Streamlit** web app.

## Features

- **Model Training**: Select different features to train the Random Forest model.
- **Custom Prediction**: Input custom passenger details and predict survival using the trained model.
- **Kaggle Submission**: Download predictions for the test dataset in the format required for Kaggle's Titanic competition.

## Live App

You can interact with the deployed app directly on Streamlit:

DEMO: [Deployed Streamlit App](https://titanicsurvivalprediction.streamlit.app/)

## Dataset

This project uses the classic Titanic dataset:

- **train.csv**: Training dataset that includes survival information for passengers.
- **test.csv**: Test dataset for making predictions.

You can find the Titanic dataset on Kaggle: [Titanic Dataset](https://www.kaggle.com/c/titanic/data)

## Project Structure

```bash
titanic-streamlit/
│
├── data/
│   ├── train.csv           # Titanic training dataset
│   ├── test.csv            # Titanic test dataset
│
├── app.py                  # Main Streamlit app
├── requirements.txt        # Required Python libraries
└── README.md               # Project documentation
