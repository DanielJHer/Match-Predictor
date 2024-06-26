import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
dataset = pd.read_csv('Online_Dating_Behavior_Dataset.csv')

# Define features and target variable
X = dataset.drop(columns=['Matches'])
y = dataset['Matches']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)

# Save the trained model using pickle
with open('linear_regression_model.pkl', 'wb') as model_file:
    pickle.dump(linear_regressor, model_file)

print("Model trained and saved as 'linear_regression_model.pkl'")