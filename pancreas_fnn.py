# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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

def build_neural_network(input_dim):
    # Build a simple feedforward neural network
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))  # Hidden layer with 64 neurons and ReLU activation
    model.add(Dense(1, activation='sigmoid'))  # Output layer with sigmoid activation for binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_neural_network(model, X_train_std, y_train, epochs=10, batch_size=32):
    # Train the neural network
    model.fit(X_train_std, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

def main():
    # Specify the path to your CSV file
    file_path = 'path/to/your/file.csv'

    # Load data
    X, y = load_data(file_path)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Preprocess data
    X_train_std, X_test_std = preprocess_data(X_train, X_test)

    # Build the neural network
    input_dim = X_train_std.shape[1]
    model = build_neural_network(input_dim)

    # Train the neural network
    train_neural_network(model, X_train_std, y_train)

    return model

if __name__ == "__main__":
    main()
