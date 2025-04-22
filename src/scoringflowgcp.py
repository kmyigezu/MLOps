from metaflow import Flow, FlowSpec, step, Parameter, current, resources, kubernetes, conda_base, retry, catch, timeout, namespace
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import preprocessing

# @conda_base(libraries={'scikit-learn': '1.0.2', 'pandas': '1.3.5', 'numpy': '1.21.6'})
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
    
    @retry(times=3)
    @timeout(minutes=5)
    @kubernetes(image='gcr.io/mlops603/metaflow-custom:v1.1', image_pull_policy='Always')
    @resources(cpu=4, memory=16384)
    @step
    def start(self):
        """
        Start the flow and validate parameters.
        """
        print("Starting model scoring flow")
        
        # Find the latest model if none specified
        if self.model_path is None:
            latest_run = Flow('ModelTrainingFlow').latest_successful_run
            self.resolved_model_path = latest_run.id
            print(f"Using latest ModelTrainingFlow run: {self.resolved_model_path}")
        else:
            self.resolved_model_path = self.model_path

        print(f"Model run ID: {self.resolved_model_path}")
        print(f"Input data path: {self.input_data_path}")
        print(f"Output path: {self.output_path}")
        
        self.next(self.load_data)
    
    @retry(times=3)
    @timeout(minutes=10)
    @kubernetes(image='gcr.io/mlops603/metaflow-custom:v1.1', image_pull_policy='Always')
    @resources(cpu=4, memory=16384)
    @step
    def load_data(self):
        """
        Load and preprocess the input data.
        """
        print("Loading and preprocessing data")
        
        # Load raw data
        data = preprocessing.load_data()
        print(f"Loaded data with shape: {data.shape}")
        
        # Preprocess data (same preprocessing as during training)
        X, y = preprocessing.preprocess_data(data)

        # Store for next step
        self.X = X
        self.y = y  # Original target values for comparison
        
        self.next(self.load_model)
    
    @retry(times=2)
    @catch(var='error')
    @kubernetes(image='gcr.io/mlops603/metaflow-custom:v1.1', image_pull_policy='Always')
    @resources(cpu=4, memory=16384)
    @step
    def load_model(self):
        """
        Load the model from Metaflow.
        """
        print(f"Loading model from Metaflow run: {self.resolved_model_path}")

        # Use the correct method to access a specific run
        namespace(None)  # Use default namespace
        run = Flow('ModelTrainingFlow')[self.resolved_model_path]
        
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

        
        self.next(self.make_predictions)
    
    @timeout(minutes=15)
    @kubernetes(image='gcr.io/mlops603/metaflow-custom:v1.1', image_pull_policy='Always')
    @resources(cpu=4, memory=16384)
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
        print(f"  R2: {r2:.4f}")
        
        self.metrics = {
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
        
        self.next(self.save_predictions)
    
    @retry(times=2)
    @kubernetes(image='gcr.io/mlops603/metaflow-custom:v1.1', image_pull_policy='Always')
    @resources(cpu=4, memory=16384)
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