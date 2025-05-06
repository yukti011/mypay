import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from data_processor import DataProcessor
from gan_model import GANFraudDetector

class FraudDetector:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.gan_model = None
        self.scaler = StandardScaler()
        self.normalizer = MinMaxScaler(feature_range=(-1, 1))
        
    def detect(self, data):
        """
        Detect fraud in the given dataset
        
        Args:
            data (DataFrame): Input dataset with transaction data
            
        Returns:
            DataFrame: Original data with fraud predictions and scores
        """
        # Preprocess data
        processed_data = self.data_processor.preprocess(data)
        
        # Get numerical features for the model
        features = processed_data.select_dtypes(include=['float64', 'int64']).copy()
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Normalize for GAN input
        normalized_features = self.normalizer.fit_transform(scaled_features)
        
        # If model not already trained, train with this data
        if self.gan_model is None:
            self.train(normalized_features)
        
        # Detect anomalies
        fraud_scores = self.gan_model.detect_anomalies(normalized_features)
        
        # Determine threshold for fraud detection (can be tuned)
        threshold = np.percentile(fraud_scores, 95)  # Top 5% as fraud
        
        # Add predictions to original data
        processed_data['fraud_score'] = fraud_scores
        processed_data['is_fraud'] = (fraud_scores > threshold).astype(int)
        
        return processed_data
    
    def train(self, normalized_data):
        """
        Train the GAN model on the given data
        
        Args:
            normalized_data (ndarray): Normalized numerical features
        """
        # Initialize GAN model with input dimension
        input_dim = normalized_data.shape[1]
        self.gan_model = GANFraudDetector(input_dim)
        
        # Train the model
        self.gan_model.train(normalized_data)
        
        return self.gan_model
