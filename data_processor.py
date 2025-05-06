import pandas as pd
import numpy as np
from datetime import datetime

class DataProcessor:
    def __init__(self):
        self.categorical_columns = []
        self.numerical_columns = []
        self.date_columns = []
        
    def preprocess(self, data):
        """
        Preprocess the input data for fraud detection
        
        Args:
            data (DataFrame): Input transaction data
            
        Returns:
            DataFrame: Processed data
        """
        # Create a copy to avoid modifying the original
        df = data.copy()
        
        # Detect column types
        self._detect_column_types(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Process date/time features
        df = self._process_date_features(df)
        
        # Encode categorical features
        df = self._encode_categorical_features(df)
        
        # Feature engineering
        df = self._feature_engineering(df)
        
        return df
    
    def _detect_column_types(self, df):
        """Detect types of columns in the dataframe"""
        for col in df.columns:
            # Check if column is a date
            if 'date' in col.lower() or 'time' in col.lower():
                self.date_columns.append(col)
            # Check if column is categorical
            elif df[col].dtype == 'object' or df[col].nunique() < 20:
                self.categorical_columns.append(col)
            # Otherwise, assume numerical
            else:
                self.numerical_columns.append(col)
    
    def _handle_missing_values(self, df):
        """Handle missing values in the dataframe"""
        # For numerical columns, fill with median
        for col in self.numerical_columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        # For categorical columns, fill with mode
        for col in self.categorical_columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])
        
        # For date columns, forward fill
        for col in self.date_columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(method='ffill')
        
        return df
    
    def _process_date_features(self, df):
        """Extract useful features from date columns"""
        for col in self.date_columns:
            try:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                
                # Extract useful components
                df[f'{col}_hour'] = df[col].dt.hour
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                
                # Add these new columns to numerical columns
                self.numerical_columns.extend([
                    f'{col}_hour', f'{col}_day', f'{col}_month', 
                    f'{col}_year', f'{col}_dayofweek'
                ])
            except:
                # If conversion fails, keep column as is
                pass
        
        return df
    
    def _encode_categorical_features(self, df):
        """Encode categorical features using one-hot encoding"""
        for col in self.categorical_columns:
            # One-hot encode if few unique values, otherwise label encode
            if df[col].nunique() < 10:
                one_hot = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, one_hot], axis=1)
                self.numerical_columns.extend(one_hot.columns)
            else:
                df[f'{col}_encoded'] = df[col].astype('category').cat.codes
                self.numerical_columns.append(f'{col}_encoded')
        
        return df
    
    def _feature_engineering(self, df):
        """Create new features that might help in fraud detection"""
        # Example: Transaction amounts statistics (if amount column exists)
        amount_cols = [col for col in df.columns if 'amount' in col.lower() or 'price' in col.lower() or 'value' in col.lower()]
        
        if amount_cols:
            # Use the first amount column found
            amount_col = amount_cols[0]
            
            # Create amount-related features
            df['amount_log'] = np.log1p(df[amount_col])
            df['amount_bin'] = pd.qcut(df[amount_col], q=5, labels=False, duplicates='drop')
            
            # Add to numerical columns
            self.numerical_columns.extend(['amount_log'])
            self.categorical_columns.append('amount_bin')
        
        return df
