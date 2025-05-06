from datetime import datetime
from app import db

class User(db.Model):
    """User model for storing user information"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # UPI specific fields
    upi_id = db.Column(db.String(64), unique=True)
    phone_number = db.Column(db.String(15))
    device_id = db.Column(db.String(64))
    last_login = db.Column(db.DateTime)
    is_verified = db.Column(db.Boolean, default=False)
    
    # Relationships
    transactions = db.relationship('Transaction', backref='user', lazy='dynamic')
    upi_transactions = db.relationship('UPITransaction', backref='user', lazy='dynamic')
    
    def __repr__(self):
        return f'<User {self.username}>'

class Transaction(db.Model):
    """Transaction model for storing transaction data"""
    id = db.Column(db.Integer, primary_key=True)
    transaction_id = db.Column(db.String(64), unique=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    merchant_id = db.Column(db.String(64))
    merchant_category = db.Column(db.String(64))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    ip_address = db.Column(db.String(64))
    device_id = db.Column(db.String(64))
    
    # Fraud detection results
    fraud_score = db.Column(db.Float, default=0.0)
    is_fraud = db.Column(db.Boolean, default=False)
    
    # Audit fields
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<Transaction {self.transaction_id}>'

class FraudDetectionResult(db.Model):
    """Model for storing the results of fraud detection analysis"""
    id = db.Column(db.Integer, primary_key=True)
    file_name = db.Column(db.String(256), nullable=False)
    total_transactions = db.Column(db.Integer, default=0)
    fraud_count = db.Column(db.Integer, default=0)
    fraud_percentage = db.Column(db.Float, default=0.0)
    avg_transaction_amount = db.Column(db.Float, default=0.0)
    total_amount = db.Column(db.Float, default=0.0)
    processing_time = db.Column(db.Float, default=0.0)
    avg_fraud_score = db.Column(db.Float, default=0.0)
    risk_level = db.Column(db.String(64), default='Low')
    date_range = db.Column(db.String(128))
    score_distribution = db.Column(db.String(256))  # Stored as JSON string
    
    # Audit fields
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    
    def __repr__(self):
        return f'<FraudDetectionResult {self.id} - {self.file_name}>'

class FraudDetectionModel(db.Model):
    """Model for storing information about trained fraud detection models"""
    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(128), nullable=False)
    model_type = db.Column(db.String(64), nullable=False)
    accuracy = db.Column(db.Float, default=0.0)
    false_positive_rate = db.Column(db.Float, default=0.0)
    false_negative_rate = db.Column(db.Float, default=0.0)
    training_date = db.Column(db.DateTime, default=datetime.utcnow)
    file_path = db.Column(db.String(256), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    
    def __repr__(self):
        return f'<FraudDetectionModel {self.model_name}>'

class UPITransaction(db.Model):
    """Model for storing UPI-specific transaction data"""
    id = db.Column(db.Integer, primary_key=True)
    transaction_id = db.Column(db.String(64), unique=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # UPI specific fields
    upi_id = db.Column(db.String(64), nullable=False)
    recipient_upi_id = db.Column(db.String(64), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    location = db.Column(db.String(128))
    ip_address = db.Column(db.String(64))
    device_id = db.Column(db.String(64))
    app_version = db.Column(db.String(32))
    os_version = db.Column(db.String(64))
    
    # Transaction specific 
    transaction_type = db.Column(db.String(32))  # P2P, merchant payment, bill payment, etc.
    payment_method = db.Column(db.String(32))    # Bank account, credit card, etc.
    merchant_id = db.Column(db.String(64))
    merchant_category = db.Column(db.String(64))
    payment_description = db.Column(db.String(256))
    
    # Security and verification
    auth_method = db.Column(db.String(32))       # PIN, OTP, biometric
    device_fingerprint = db.Column(db.String(128))
    is_trusted_device = db.Column(db.Boolean, default=False)
    is_trusted_recipient = db.Column(db.Boolean, default=False)
    
    # Fraud detection results
    fraud_score = db.Column(db.Float, default=0.0)
    is_fraud = db.Column(db.Boolean, default=False)
    fraud_flags = db.Column(db.String(512))  # JSON string of flags
    blocking_reason = db.Column(db.String(128))
    
    # Status tracking
    status = db.Column(db.String(32), default='completed')  # completed, pending, failed, blocked
    response_code = db.Column(db.String(16))
    error_message = db.Column(db.String(256))
    
    # Audit fields
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<UPITransaction {self.transaction_id}>'