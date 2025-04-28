import requests
import json

BASE_URL = "http://localhost:8000"  

def test_single_prediction():
    """Test prediction for a single input with 13 features."""
    data = {
        "feature1": 1.0, "feature2": 2.0, "feature3": 3.0, "feature4": 4.0,
        "feature5": 5.0, "feature6": 6.0, "feature7": 7.0, "feature8": 8.0,
        "feature9": 9.0, "feature10": 10.0, "feature11": 11.0, "feature12": 12.0,
        "feature13": 13.0
    }
    
    response = requests.post(f"{BASE_URL}/predict/", json=data)
    
    result = response.json()
    print("Single prediction result:")
    print(f"Prediction: {result['prediction']}")
    print(f"Class: {result.get('class', 'N/A')}")


if __name__ == "__main__":
    test_single_prediction()