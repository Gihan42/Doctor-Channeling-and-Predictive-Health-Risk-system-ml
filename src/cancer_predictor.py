# ml/cancer_predictor.py
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class CancerPredictor:
    def __init__(self, data_path=None, model_path='cancer_predictor_model.pkl'):
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.model = None

        if Path(model_path).exists():
            self.load_model()
        elif data_path:
            self.train_model(data_path)
        else:
            raise ValueError("Either provide data_path to train a new model or ensure model exists at specified path")

    def train_model(self, data_path):
        """Train the model using the provided dataset"""
        print("Loading and preprocessing data...")
        data = pd.read_csv(data_path)

        # Data preprocessing
        X = data.drop(columns='Diagnosis')
        y = data['Diagnosis']

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )

        print("Training Logistic Regression model...")
        # Train logistic regression
        self.model = LogisticRegression(solver='lbfgs', max_iter=1000)
        self.model.fit(X_train, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained with accuracy: {accuracy:.2f}")

        # Save model
        self.save_model()

    def predict(self, input_features):
        """
        Make a prediction based on input features.

        Args:
            input_features: Dictionary containing patient features

        Returns:
            Dictionary containing prediction results
        """
        if not self.model:
            raise ValueError("Model not loaded or trained")

        # Convert input to dataframe in correct order
        input_df = pd.DataFrame([[
            input_features['Age'],
            input_features['Gender'],
            input_features['BMI'],
            input_features['Smoking'],
            input_features['GeneticRisk'],
            input_features['PhysicalActivity'],
            input_features['AlcoholIntake'],
            input_features['CancerHistory']
        ]], columns=[
            'Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk',
            'PhysicalActivity', 'AlcoholIntake', 'CancerHistory'
        ])

        # Scale features
        scaled_input = self.scaler.transform(input_df)

        # Make prediction
        probability = self.model.predict_proba(scaled_input)[0][1]
        prediction = self.model.predict(scaled_input)[0]

        # Create interpretation
        if prediction == 1:
            interpretation = "High risk of cancer detected"
        else:
            interpretation = "Low risk of cancer detected"
            if probability > 0.3:
                interpretation += " (moderate risk - consider further screening)"

        return {
            'prediction': int(prediction),
            'probability': float(probability),
            'interpretation': interpretation
        }

    def save_model(self):
        """Save the trained model and scaler to disk"""
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler
            }, f)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        """Load the trained model and scaler from disk"""
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
        print(f"Model loaded from {self.model_path}")