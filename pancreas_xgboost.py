# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_data(file_path):
    # Load data from CSV file
    data = pd.read_csv(file_path)
    # Assuming the last column contains the labels (1 for tumor, 0 for non-tumor)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def preprocess_data(X_train, X_test):
    # Standardize the feature vectors
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    return X_train_std, X_test_std

def train_xgboost(X_train_std, y_train, n_estimators=100, learning_rate=0.1, random_state=42):
    # Create and train the XGBoost classifier
    xgb_classifier = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
    xgb_classifier.fit(X_train_std, y_train)
    return xgb_classifier

def main():
    # Specify the path to your CSV file
    file_path = 'path/to/your/file.csv'

    # Load data
    X, y = load_data(file_path)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Preprocess data
    X_train_std, X_test_std = preprocess_data(X_train, X_test)

    # Train XGBoost classifier
    xgb_classifier = train_xgboost(X_train_std, y_train)

    return xgb_classifier

if __name__ == "__main__":
    main()
