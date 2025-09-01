import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class HeartAttackPredictor:
    def __init__(self, data_path=None, model_path='heart_attack_predictor_model.pkl'):
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
        print("Dataset columns:", data.columns.tolist())  # Debug feature order
        print("Class distribution:", data['output'].value_counts().to_dict())  # Debug class balance

        # Rename columns to match our input format
        data = data.rename(columns={
            'trestbps': 'trtbps',
            'thalach': 'thalachh',
            'exang': 'exng',
            'slope': 'slp',
            'ca': 'caa',
            'thal': 'thall'
        })

        # Data preprocessing
        X = data.drop(columns='output')
        y = data['output']

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        print("Training Logistic Regression model...")
        # Train logistic regression with class_weight='balanced' to handle imbalance
        self.model = LogisticRegression(solver='liblinear', max_iter=1000, class_weight='balanced')
        self.model.fit(X_train, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained with accuracy: {accuracy:.2f}")

        # Generate evaluation reports
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # Plot confusion matrix
        self._plot_confusion_matrix(y_test, y_pred)

        # Save model
        self.save_model()

    def predict(self, input_features):
        if not self.model:
            raise ValueError("Model not loaded or trained")

        # Define the exact feature order used during training
        feature_order = [
            'ge', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg',
            'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall'
        ]

        # Create input DataFrame with correct feature names
        input_df = pd.DataFrame([[
            input_features['ge'],
            input_features['sex'],
            input_features['cp'],
            input_features['trtbps'],
            input_features['chol'],
            input_features['fbs'],
            input_features['restecg'],
            input_features['thalachh'],
            input_features['exng'],
            input_features['oldpeak'],
            input_features['slp'],
            input_features['caa'],
            input_features['thall']
        ]], columns=feature_order)

        scaled_input = self.scaler.transform(input_df)
        probability = self.model.predict_proba(scaled_input)[0][1]
        prediction = self.model.predict(scaled_input)[0]

        if prediction == 1:
            interpretation = "Low risk of heart attack detected"
        else:
            interpretation = "High risk of heart attack detected "
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

    def _plot_confusion_matrix(self, y_true, y_pred):
        """Helper method to plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('confusion_matrix.png')
        plt.close()