import argparse
import warnings
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Suppress warnings for cleaner CLI output during evaluation
warnings.filterwarnings("ignore")

def train_model():
    """Loads a self-contained dataset and trains the Decision Tree."""
    # Using scikit-learn's built-in toy dataset to avoid external CSV dependencies
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Initialize and train the Decision Tree Classifier
    # random_state is set for reproducible results
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)
    
    return clf, iris.target_names

def predict_species(clf, target_names, features):
    """Predicts the target class based on terminal inputs."""
    prediction = clf.predict([features])
    
    # Calculate confidence probability
    probabilities = clf.predict_proba([features])[0]
    confidence = max(probabilities) * 100
    
    predicted_class = target_names[prediction[0]]
    return predicted_class, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI Decision Tree Classifier for Botanical Prediction")
    
    # Define the command-line arguments required from the user/evaluator
    parser.add_argument("--sepal-length", type=float, required=True, help="Sepal length in cm")
    parser.add_argument("--sepal-width", type=float, required=True, help="Sepal width in cm")
    parser.add_argument("--petal-length", type=float, required=True, help="Petal length in cm")
    parser.add_argument("--petal-width", type=float, required=True, help="Petal width in cm")
    
    args = parser.parse_args()
    
    # Train the model instantly on execution
    model, class_names = train_model()
    
    # Prepare the input array from parsed arguments
    input_features = [
        args.sepal_length, 
        args.sepal_width, 
        args.petal_length, 
        args.petal_width
    ]
    
    # Execute prediction
    species, conf = predict_species(model, class_names, input_features)
    
    # Structured CLI Output
    print("\n" + "="*50)
    print("🌳 AI/ML DECISION TREE CLASSIFIER RESULT")
    print("="*50)
    print(f"Input Features : {input_features}")
    print(f"Predicted Class: {species.upper()}")
    print(f"Confidence     : {conf:.2f}%")
    print("="*50 + "\n")