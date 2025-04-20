from metaflow import FlowSpec, step, Parameter, namespace, Flow

import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Add src directory to path to import preprocessing
sys.path.append('/Users/kalu/Desktop/mlops/src')
import preprocessing

class ModelScoringFlow(FlowSpec):
    """
    A flow to score new data using a trained model.
    """
    
    # Parameters
    model_path = Parameter('model_path',
                       default=None,
                       help='Metaflow run ID of the model to use')
    
    input_data_path = Parameter('input_data_path',
                            default='/Users/kalu/Desktop/mlops/labs/data/creditcard.csv',
                            help='Path to new data for prediction')
    
    output_path = Parameter('output_path',
                        default='predictions.csv',
                        help='Path to save predictions')
    
    @step
    def start(self):
        """
        Start the flow and validate parameters.
        """
        print("Starting model scoring flow")
        
        # Find the latest model if none specified
        if self.model_path is None:
            try:
                latest_run = Flow('ModelTrainingFlow').latest_successful_run
                self.model_path = latest_run.id
                print(f"Using latest ModelTrainingFlow run: {self.model_path}")
            except Exception as e:
                raise ValueError(f"Error finding latest model run: {str(e)}")
        
        print(f"Model run ID: {self.model_path}")
        print(f"Input data path: {self.input_data_path}")
        print(f"Output path: {self.output_path}")
        
        self.next(self.load_data)
    
    @step
    def load_data(self):
        """
        Load and preprocess the input data.
        """
        print("Loading and preprocessing data")
        
        # Load raw data
        data = preprocessing.load_data(self.input_data_path)
        print(f"Loaded data with shape: {data.shape}")
        
        # Preprocess data (same preprocessing as during training)
        X, y = preprocessing.preprocess_data(data)
        print(f"Preprocessed features shape: {X.shape}")
        
        # Store for next step
        self.X = X
        self.y = y  # Original target values for comparison
        
        self.next(self.load_model)
    

    # Then in the load_model step
    @step
    def load_model(self):
        """
        Load the model from Metaflow.
        """
        print(f"Loading model from Metaflow run: {self.model_path}")
        try:
            # Use the correct method to access a specific run
            namespace(None)  # Use default namespace
            run = Flow('ModelTrainingFlow')[self.model_path]

            
            # Get model parameters
            model_params = run.data.model_params
            print(f"Model parameters: {model_params}")
            
            # Recreate the model
            self.model = RandomForestRegressor(
                n_estimators=model_params['n_estimators'],
                max_depth=model_params['max_depth'],
                min_samples_split=model_params['min_samples_split'],
                random_state=model_params['random_state']
            )
            
            # Retrain the model with stored data
            X_train = run.data.X_train
            y_train = run.data.y_train
            
            print(f"Retraining model with stored training data: {X_train.shape}")
            self.model.fit(X_train, y_train)
            
            print("Model loaded and retrained successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
        
            
        self.next(self.make_predictions)
    
    @step
    def make_predictions(self):
        """
        Make predictions using the loaded model.
        """
        print("Making predictions")
        
        # Make predictions
        self.predictions = self.model.predict(self.X)
        
        # Calculate metrics
        mse = mean_squared_error(self.y, self.predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y, self.predictions)
        
        print(f"Evaluation metrics:")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {r2:.4f}")
        
        self.metrics = {
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
        
        self.next(self.save_predictions)
    
    @step
    def save_predictions(self):
        """
        Save the predictions to a CSV file.
        """
        print(f"Saving predictions to {self.output_path}")
        
        # Create a DataFrame with predictions
        results = pd.DataFrame({
            'actual': self.y,
            'predicted': self.predictions,
            'error': self.y - self.predictions
        })
        
        # Save to CSV
        results.to_csv(self.output_path, index=False)
        
        # Save a sample of predictions for display
        self.prediction_sample = results.head(10)
        
        self.next(self.end)
    
    @step
    def end(self):
        """
        End the flow.
        """
        print("\nModel scoring completed successfully")
        print(f"Predictions saved to: {self.output_path}")
        
        # Print sample predictions
        print("\nSample predictions:")
        print(self.prediction_sample)
        
        # Print metrics
        print("\nEvaluation metrics:")
        print(f"  RMSE: {self.metrics['rmse']:.4f}")
        print(f"  R²: {self.metrics['r2']:.4f}")

if __name__ == "__main__":
    ModelScoringFlow()