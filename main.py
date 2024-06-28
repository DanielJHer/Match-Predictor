import tkinter as tk
from tkinter import messagebox
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the trained model
with open('linear_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the dataset for visualization
dataset = pd.read_csv('Online_Dating_Behavior_Dataset.csv')

# Define features and target variable
X = dataset.drop(columns=['Matches'])
y = dataset['Matches']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Make predictions on the test set using the loaded model
y_pred = model.predict(X_test)

def predict_matches():
    try:
        gender = int(gender_entry.get())
        purchased_vip = int(purchased_vip_entry.get())
        income = float(income_entry.get())
        children = int(children_entry.get())
        age = int(age_entry.get())
        attractiveness = int(attractiveness_entry.get())

        # Prepare the feature vector for prediction
        features = np.array([[gender, purchased_vip, income, children, age, attractiveness]])
        prediction = model.predict(features)[0]

        result_label.config(text=f'Predicted Matches: {prediction:.2f}')
    except ValueError:
        messagebox.showerror("Input error", "Please enter valid input values.")

def show_histogram():
    fig, ax = plt.subplots()
    dataset['Matches'].hist(bins=20, ax=ax)
    ax.set_title('Histogram of Matches')
    ax.set_xlabel('Number of Matches')
    ax.set_ylabel('Frequency')

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=10, columnspan=2, pady=10)

def show_scatter_plot():
    fig, ax = plt.subplots()
    ax.scatter(dataset['Attractiveness'], dataset['Matches'], alpha=0.5)
    ax.set_title('Scatter Plot of Attractiveness vs. Matches')
    ax.set_xlabel('Attractiveness')
    ax.set_ylabel('Number of Matches')

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=11, columnspan=2, pady=10)

def show_box_plot():
    fig, ax = plt.subplots()
    dataset.boxplot(column='Income', by='Age', ax=ax)
    ax.set_title('Box Plot of Income by Age')
    ax.set_xlabel('Age')
    ax.set_ylabel('Income')
    fig.suptitle('')  # Suppress the automatic title to make it cleaner

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=12, columnspan=2, pady=10)

# Create the GUI window
root = tk.Tk()
root.title("Match Prediction")

# Create and place widgets
tk.Label(root, text="Gender (0 or 1):").grid(row=0, column=0, padx=10, pady=5)
gender_entry = tk.Entry(root)
gender_entry.grid(row=0, column=1, padx=10, pady=5)

tk.Label(root, text="Purchased VIP (0 or 1):").grid(row=1, column=0, padx=10, pady=5)
purchased_vip_entry = tk.Entry(root)
purchased_vip_entry.grid(row=1, column=1, padx=10, pady=5)

tk.Label(root, text="Income:").grid(row=2, column=0, padx=10, pady=5)
income_entry = tk.Entry(root)
income_entry.grid(row=2, column=1, padx=10, pady=5)

tk.Label(root, text="Children:").grid(row=3, column=0, padx=10, pady=5)
children_entry = tk.Entry(root)
children_entry.grid(row=3, column=1, padx=10, pady=5)

tk.Label(root, text="Age:").grid(row=4, column=0, padx=10, pady=5)
age_entry = tk.Entry(root)
age_entry.grid(row=4, column=1, padx=10, pady=5)

tk.Label(root, text="Attractiveness (1-10):").grid(row=5, column=0, padx=10, pady=5)
attractiveness_entry = tk.Entry(root)
attractiveness_entry.grid(row=5, column=1, padx=10, pady=5)

predict_button = tk.Button(root, text="Predict", command=predict_matches)
predict_button.grid(row=6, columnspan=2, pady=10)

result_label = tk.Label(root, text="", font=("Helvetica", 14))
result_label.grid(row=7, columnspan=2, pady=10)

hist_button = tk.Button(root, text="Show Histogram of Matches", command=show_histogram)
hist_button.grid(row=8, columnspan=2, pady=5)

scatter_button = tk.Button(root, text="Show Scatter Plot", command=show_scatter_plot)
scatter_button.grid(row=9, columnspan=2, pady=5)

box_plot_button = tk.Button(root, text="Show Box Plot of Income by Age", command=show_box_plot)
box_plot_button.grid(row=10, columnspan=2, pady=5)

# Correlation matrix
# plt.figure(figsize=(10, 8))
# sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm')
# plt.title('Correlation Matrix')
# plt.show()

# # Regression coefficients
# coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
# plt.figure(figsize=(10, 6))
# coefficients.plot(kind='bar', x='Feature', y='Coefficient', legend=False)
# plt.title('Regression Coefficients')
# plt.show()

# # Actual vs. Predicted Matches
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, y_pred, alpha=0.5)
# plt.title('Actual vs. Predicted Matches')
# plt.xlabel('Actual Matches')
# plt.ylabel('Predicted Matches')
# plt.show()

# Print model performance metrics
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
print(f'R-squared: {r2_score(y_test, y_pred)}')

root.mainloop()