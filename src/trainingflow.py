from metaflow import FlowSpec, step, Parameter, current
import sys
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Add src directory to path to import preprocessing
sys.path.append('/Users/kalu/Desktop/mlops/src')
import preprocessing

class ModelTrainingFlow(FlowSpec):
    """
    A flow to train a regression model for credit card Amount prediction.
    """
    
    # Model parameters
    n_estimators = Parameter('n_estimators',
                         default=10,
                         type=int,
                         help='Number of trees in the forest')
    
    max_depth = Parameter('max_depth',
                       default=5,
                       type=int,
                       help='Maximum depth of the trees')
    
    min_samples_split = Parameter('min_samples_split',
                              default=10,
                              type=int,
                              help='Minimum samples required to split an internal node')
    
    random_state = Parameter('random_state',
                         default=42,
                         type=int,
                         help='Random state for reproducibility')
    
    @step
    def start(self):
        """
        Start the flow and load/preprocess data.
        """
        print("Starting model training flow")
        
        # Load and preprocess data
        data = preprocessing.load_data()
        X, y = preprocessing.preprocess_data(data)
        
        # Only split into train and validation, leave test for scoring flow
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Store the data
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        
        # Print data shapes
        print(f"Training data shapes: X={X_train.shape}, y={y_train.shape}")
        print(f"Validation data shapes: X={X_val.shape}, y={y_val.shape}")
        
        self.next(self.train_model)
    
    @step
    def train_model(self):
        """
        Train the model with the specified parameters.
        """
        print(f"Training model with parameters:")
        print(f"  n_estimators: {self.n_estimators}")
        print(f"  max_depth: {self.max_depth}")
        print(f"  min_samples_split: {self.min_samples_split}")
        print(f"  random_state: {self.random_state}")
        
        # Create the model - max_depth is already an integer from the Parameter
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth if self.max_depth > 0 else None,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state
        )
        
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        
        # Store model parameters for later use
        self.model_params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth if self.max_depth > 0 else None,
            'min_samples_split': self.min_samples_split,
            'random_state': self.random_state
        }
        
        # Store feature column names
        self.feature_columns = self.X_train.columns.tolist()
        
        print("Model trained successfully")
        
        self.next(self.evaluate)
    
    @step
    def evaluate(self):
        """
        Evaluate the model on validation data only.
        """
        # Evaluate on validation data
        from sklearn.metrics import mean_squared_error, r2_score
        val_preds = self.model.predict(self.X_val)
        self.val_mse = mean_squared_error(self.y_val, val_preds)
        self.val_rmse = np.sqrt(self.val_mse)
        self.val_r2 = r2_score(self.y_val, val_preds)
        
        print(f"Validation metrics:")
        print(f"  MSE: {self.val_mse:.4f}")
        print(f"  RMSE: {self.val_rmse:.4f}")
        print(f"  R²: {self.val_r2:.4f}")
        
        self.next(self.end)
    
    @step
    def end(self):
        """
        End the flow.
        """
        print("Model training flow completed successfully")
        print(f"Model registered in Metaflow with run ID: {current.run_id}")
        print(f"Validation RMSE: {self.val_rmse:.4f}")
        
        # Print instructions for using the scoring flow
        print("\nTo score new data with this model, run:")
        print(f"python src/scoringflow.py run --model_path {current.run_id}")

if __name__ == "__main__":
    ModelTrainingFlow()