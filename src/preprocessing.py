# src/preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath="/Users/kalu/Desktop/mlops/labs/data/creditcard.csv"):
    """
    Load the credit card dataset from the specified filepath
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: The loaded DataFrame
    """
    return pd.read_csv(filepath)

def preprocess_data(data):
    """
    Preprocess the credit card data by separating target and features
    
    Args:
        data (pd.DataFrame): The credit card DataFrame
        
    Returns:
        tuple: (X, y) where X is the features DataFrame and y is the target Series
    """
    y = data["Amount"]
    X = data.drop(columns=["Amount"])
    return X, y

def split_data(X, y, test_size=0.3, val_size=0.5, random_state=42):
    """
    Split the dataset into training, validation, and test sets
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        test_size (float): Size of test+validation set as a proportion of the full dataset
        val_size (float): Size of validation set as a proportion of the test+validation set
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split: training and temp (validation + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: validation and test from temp
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    """
    Main function to run the preprocessing pipeline
    """
    # Load data
    data = load_data()
    
    # Preprocess data
    X, y = preprocess_data(data)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    main()