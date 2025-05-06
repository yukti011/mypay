import numpy as np
from models import GAN, AutoEncoder

class GANFraudDetector:
    def __init__(self, input_dim):
        """
        Initialize the fraud detector model
        
        Args:
            input_dim (int): Number of features in the input data
        """
        self.input_dim = input_dim
        self.model = GAN(input_dim)  # Using our ensemble model
        self.is_trained = False
        
    def train(self, normal_data, epochs_gan=None, epochs_ae=None, batch_size=None):
        """
        Train the ensemble models for fraud detection
        
        Args:
            normal_data (ndarray): Normalized data for training
            epochs_gan (int): Not used in this implementation
            epochs_ae (int): Not used in this implementation
            batch_size (int): Not used in this implementation
        """
        # Train the model
        self.model.train(normal_data)
        self.is_trained = True
        
    def detect_anomalies(self, data):
        """
        Detect anomalies in the given data
        
        Args:
            data (ndarray): Input data for anomaly detection
            
        Returns:
            ndarray: Anomaly scores for each data point
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Get anomaly scores from the model
        anomaly_scores = self.model.decision_function(data)
        
        return anomaly_scores
