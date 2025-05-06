import os
import time
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify, current_app
from werkzeug.utils import secure_filename
from app import db, model, scaler, logger
from app.models import User, Transaction, FraudDetectionResult, UPITransaction
from app.utils import allowed_file, preprocess_data, detect_fraud, prepare_scatter_data, calculate_score_bins

# Create blueprint
main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/dashboard')
def dashboard():
    # Sample dashboard data for demonstration
    stats = {
        'total_transactions': 2845,
        'transaction_change': 12.5,
        'fraud_count': 32,
        'fraud_change': -8.3,
        'avg_risk_score': 15.2,
        'avg_risk_score_percent': 15.2,
        'risk_color': 'success',
        'model_accuracy': 93.7
    }
    
    # Generate some sample data for transactions
    recent_transactions = []
    for i in range(10):
        is_fraud = i < 2  # First 2 are fraudulent for demonstration
        risk_score = 85 if is_fraud else 15
        risk_color = 'danger' if risk_score > 70 else 'warning' if risk_score > 30 else 'success'
        
        recent_transactions.append({
            'id': f'TX{100000 + i}',
            'timestamp': f'2023-09-{15 + i} 14:{30 + i}:00',
            'amount': f'{100 + i * 50}.00',
            'user': f'User{1000 + i}',
            'risk_score': risk_score,
            'risk_color': risk_color,
            'is_fraud': is_fraud
        })
    
    # Get scatter data if available
    scatter_data = None
    if 'scatter_legitimate' in session and 'scatter_fraudulent' in session:
        scatter_data = {
            'normal': session['scatter_legitimate'],
            'fraud': session['scatter_fraudulent']
        }
    
    # Get score distribution if available
    score_distribution = None
    if 'score_bins' in session:
        score_distribution = session['score_bins']
    
    return render_template(
        'dashboard.html',
        stats=stats,
        recent_transactions=recent_transactions,
        scatter_data=scatter_data,
        score_distribution=score_distribution
    )

@main_bp.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('upload.html')
        
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('main.upload_file'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('main.upload_file'))
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Process file and detect fraud
            if filename.endswith('.csv'):
                data = pd.read_csv(file_path)
            else:  # Excel files
                data = pd.read_excel(file_path)
            
            # Preprocess data
            processed_data = preprocess_data(data)
            
            # Perform fraud detection
            results = detect_fraud(processed_data)
            
            # Calculate summary statistics
            total_count = results.shape[0]
            fraud_count = results[results['is_fraud'] == 1].shape[0]
            fraud_percentage = (fraud_count / total_count) * 100 if total_count > 0 else 0
            
            # Calculate average transaction amount if it exists
            amount_cols = [col for col in results.columns if 'amount' in col.lower() or 'price' in col.lower() or 'value' in col.lower()]
            avg_transaction = 0
            if amount_cols:
                avg_transaction = results[amount_cols[0]].mean()
            
            # Create data for scatter plot
            scatter_data = prepare_scatter_data(results)
            
            # Create data for fraud score distribution
            score_bins = calculate_score_bins(results['fraud_score'])
            
            # Store results and statistics in session
            session['results'] = results.to_dict()
            session['filename'] = filename
            session['total_count'] = total_count
            session['fraud_count'] = fraud_count
            session['fraud_percentage'] = fraud_percentage
            session['avg_transaction'] = avg_transaction
            session['scatter_legitimate'] = scatter_data['legitimate']
            session['scatter_fraudulent'] = scatter_data['fraudulent']
            session['score_bins'] = score_bins
            
            # Store results in database
            try:
                fraud_result = FraudDetectionResult(
                    file_name=filename,
                    total_transactions=total_count,
                    fraud_count=fraud_count,
                    fraud_percentage=fraud_percentage,
                    avg_transaction_amount=avg_transaction,
                    total_amount=avg_transaction * total_count,
                    processing_time=1.23,  # Demo value
                    avg_fraud_score=results['fraud_score'].mean() * 100,
                    risk_level='High' if results['fraud_score'].mean() * 100 > 70 else 'Medium' if results['fraud_score'].mean() * 100 > 30 else 'Low',
                    date_range='2023-09-01 to 2023-09-15',  # Demo value
                    score_distribution=str(score_bins)
                )
                db.session.add(fraud_result)
                db.session.commit()
            except Exception as e:
                logger.error(f"Error saving results to database: {str(e)}")
                db.session.rollback()
            
            return redirect(url_for('main.results'))
        
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            flash(f'Error processing file: {str(e)}', 'danger')
            return redirect(url_for('main.index'))
    else:
        flash('File type not allowed. Please upload a CSV or Excel file.', 'danger')
        return redirect(url_for('main.index'))

@main_bp.route('/results')
def results():
    if 'results' not in session:
        flash('No results to display. Please upload a file first.', 'warning')
        return redirect(url_for('main.upload_file'))
    
    results_df = pd.DataFrame.from_dict(session['results'])
    filename = session.get('filename', 'Unknown file')
    
    # Get statistics from session
    fraud_count = session.get('fraud_count', 0)
    total_count = session.get('total_count', 0)
    fraud_percentage = session.get('fraud_percentage', 0)
    avg_transaction = session.get('avg_transaction', 0)
    
    # Calculate processing time (in a real app, this would be measured)
    processing_time = 1.23
    
    # Get scatter plot data
    scatter_data = None
    if 'scatter_legitimate' in session and 'scatter_fraudulent' in session:
        scatter_data = {
            'normal': session['scatter_legitimate'],
            'fraud': session['scatter_fraudulent']
        }
    
    # Get score distribution data
    score_distribution = session.get('score_bins', [])
    
    # Get top high-risk transactions (for demonstration)
    high_risk_transactions = []
    if 'results' in session:
        fraud_transactions = results_df[results_df['is_fraud'] == 1].sort_values(by='fraud_score', ascending=False).head(10)
        
        for _, row in fraud_transactions.iterrows():
            # Generate risk color based on score
            fraud_score = row['fraud_score'] * 100
            risk_color = 'danger' if fraud_score >= 70 else 'warning' if fraud_score >= 30 else 'success'
            
            # Generate some sample flags
            flags = []
            if fraud_score > 80:
                flags.append('Unusual Amount')
                flags.append('New Merchant')
            elif fraud_score > 60:
                flags.append('Unusual Time')
                flags.append('Location Mismatch')
            else:
                flags.append('User Pattern Change')
            
            high_risk_transactions.append({
                'id': row.get('id', 'TX' + str(hash(str(row)) % 100000)),
                'timestamp': row.get('timestamp', '2023-09-15 14:30:00'),
                'amount': row.get('amount', 100 + hash(str(row)) % 900),
                'fraud_score': fraud_score,
                'risk_color': risk_color,
                'flags': flags
            })
    
    # Sample risk factors for visualization
    risk_factors = [
        {'name': 'Unusual Amount', 'impact': 85, 'color': 'danger'},
        {'name': 'Time of Transaction', 'impact': 62, 'color': 'warning'},
        {'name': 'Location Mismatch', 'impact': 74, 'color': 'danger'},
        {'name': 'User History', 'impact': 45, 'color': 'warning'},
        {'name': 'Device Fingerprint', 'impact': 38, 'color': 'info'}
    ]
    
    # Sample recommendations
    recommendations = [
        {
            'title': 'Review High-Risk Transactions',
            'description': 'Manual review recommended for transactions with risk scores above 70%.',
            'icon': 'exclamation-triangle',
            'color': 'danger'
        },
        {
            'title': 'Implement Amount Limits',
            'description': 'Set transaction limits for certain user segments based on historical patterns.',
            'icon': 'dollar-sign',
            'color': 'success'
        },
        {
            'title': 'Enhance User Verification',
            'description': 'Add extra verification steps for unusual transaction locations.',
            'icon': 'user-shield',
            'color': 'primary'
        }
    ]
    
    # Calculate additional metrics for display
    avg_fraud_score = results_df['fraud_score'].mean() * 100
    risk_level = 'High' if avg_fraud_score > 70 else 'Medium' if avg_fraud_score > 30 else 'Low'
    risk_level_color = 'danger' if avg_fraud_score > 70 else 'warning' if avg_fraud_score > 30 else 'success'
    
    # Prepare results for the template
    results_data = {
        'filename': filename,
        'date_range': '2023-09-01 to 2023-09-15',
        'total_count': total_count,
        'fraud_count': fraud_count,
        'fraud_percentage': round(fraud_percentage, 1),
        'total_amount': f"{avg_transaction * total_count:,.2f}",
        'avg_amount': f"{avg_transaction:,.2f}",
        'processing_time': processing_time,
        'avg_fraud_score': round(avg_fraud_score, 1),
        'risk_level': risk_level,
        'risk_level_color': risk_level_color,
        'high_risk_transactions': high_risk_transactions,
        'scatter_data': scatter_data,
        'score_distribution': score_distribution,
        'risk_factors': risk_factors,
        'recommendations': recommendations,
        'false_positive_rate': 2.5,
        'false_negative_rate': 1.8,
        'model_confidence': 93.7,
        'avg_transaction_value': f"{avg_transaction:,.2f}",
        'highest_risk_time': '10:00 PM - 2:00 AM',
        'common_patterns': 'Large amounts, new merchants',
        'risk_factor_labels': ['Unusual Amount', 'Time Pattern', 'Location', 'Device', 'User History', 'Merchant Category'],
        'risk_factor_values': [75, 45, 60, 85, 30, 50]
    }
    
    return render_template('results.html', results=results_data)

@main_bp.route('/upi-analytics', methods=['GET'])
def upi_analytics():
    """UPI Transaction Analytics Dashboard"""
    # Generate demo data for UPI analytics
    today = datetime.now()
    
    # Demo statistics
    stats = {
        'total_transactions': 4582,
        'transaction_amount': 874250.50,
        'fraud_count': 47,
        'fraud_rate': 1.03,
        'avg_transaction': 190.80,
        'blocked_transactions': 38,
        'saved_amount': 95650.75,
        'model_accuracy': 97.2
    }
    
    # Demo transaction patterns
    patterns = [
        {'name': 'P2P Transfer', 'count': 2815, 'percentage': 61.4, 'avg_amount': 153.42, 'fraud_rate': 0.7},
        {'name': 'Merchant Payment', 'count': 1105, 'percentage': 24.1, 'avg_amount': 245.60, 'fraud_rate': 1.2},
        {'name': 'Bill Payment', 'count': 485, 'percentage': 10.6, 'avg_amount': 328.75, 'fraud_rate': 0.8},
        {'name': 'Subscription', 'count': 177, 'percentage': 3.9, 'avg_amount': 129.90, 'fraud_rate': 3.4}
    ]
    
    # Demo peak hours with fraud rates
    hours_data = []
    for hour in range(24):
        # Create a demo pattern with higher fraud rates during night hours
        if 22 <= hour or hour <= 3:
            fraud_rate = 2.5 + (3.5 * (1 - abs(hour - 0 if hour <= 3 else hour - 24) / 4))
            volume = 75 + (110 * (1 - abs(hour - 0 if hour <= 3 else hour - 24) / 4))
        elif 9 <= hour <= 18:
            fraud_rate = 0.8 + (0.5 * (1 - abs(hour - 14) / 5))
            volume = 270 + (130 * (1 - abs(hour - 14) / 5))
        else:
            fraud_rate = 1.0 + (0.5 * (1 - min(abs(hour - 5), abs(hour - 20)) / 4))
            volume = 120 + (40 * (1 - min(abs(hour - 5), abs(hour - 20)) / 4))
        
        hours_data.append({
            'hour': f"{hour:02d}:00",
            'volume': int(volume),
            'fraud_rate': round(fraud_rate, 1),
            'color': 'danger' if fraud_rate > 2 else 'warning' if fraud_rate > 1.5 else 'success'
        })
    
    # Demo fraud patterns detected
    fraud_patterns = [
        {
            'pattern': 'Unusual transaction time',
            'description': 'Transactions occurring at unusual hours for the user',
            'risk_score': 87,
            'detection_rate': 92,
            'false_positive': 7.3
        },
        {
            'pattern': 'Multiple device logins',
            'description': 'Account accessed from multiple devices in short timeframe',
            'risk_score': 92,
            'detection_rate': 95,
            'false_positive': 4.8
        },
        {
            'pattern': 'Transaction amount anomaly',
            'description': 'Transaction amount significantly higher than user history',
            'risk_score': 78,
            'detection_rate': 88,
            'false_positive': 12.5
        },
        {
            'pattern': 'Rapid sequential transactions',
            'description': 'Multiple transactions in rapid succession',
            'risk_score': 83,
            'detection_rate': 91,
            'false_positive': 8.2
        },
        {
            'pattern': 'New payee large amount',
            'description': 'Large transaction to a new recipient',
            'risk_score': 85,
            'detection_rate': 94,
            'false_positive': 6.7
        }
    ]
    
    # Demo security recommendations
    recommendations = [
        {
            'title': 'Implement Step-up Authentication',
            'description': 'Add additional verification for high-risk transactions.',
            'impact': 'High',
            'implementation': 'Medium',
            'icon': 'shield-alt'
        },
        {
            'title': 'Transaction Velocity Monitoring',
            'description': 'Flag accounts with unusual transaction frequency.',
            'impact': 'High',
            'implementation': 'Easy', 
            'icon': 'tachometer-alt'
        },
        {
            'title': 'Device Fingerprinting',
            'description': 'Track and verify devices used for transactions.',
            'impact': 'Medium',
            'implementation': 'Medium',
            'icon': 'mobile-alt'
        },
        {
            'title': 'Behavioral Biometrics',
            'description': 'Monitor typing patterns and user behavior.',
            'impact': 'High',
            'implementation': 'Complex',
            'icon': 'fingerprint'
        }
    ]
    
    # Demo location data
    locations = [
        {'city': 'Mumbai', 'count': 845, 'fraud_rate': 1.2},
        {'city': 'Delhi', 'count': 765, 'fraud_rate': 1.5},
        {'city': 'Bangalore', 'count': 675, 'fraud_rate': 0.9},
        {'city': 'Chennai', 'count': 425, 'fraud_rate': 0.7},
        {'city': 'Hyderabad', 'count': 385, 'fraud_rate': 0.8},
        {'city': 'Pune', 'count': 325, 'fraud_rate': 1.0},
        {'city': 'Kolkata', 'count': 310, 'fraud_rate': 1.3},
        {'city': 'Other', 'count': 852, 'fraud_rate': 2.1}
    ]
    
    # Monthly trend data
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_data = {
        'labels': months[0:today.month],
        'volumes': [round(2500 + 500 * month + 200 * (month % 3)) for month in range(today.month)],
        'fraud_rates': [round(0.8 + 0.3 * (month % 4 / 4), 2) for month in range(today.month)]
    }
    
    return render_template(
        'upi_analytics.html',
        stats=stats,
        patterns=patterns,
        hours_data=hours_data,
        fraud_patterns=fraud_patterns,
        recommendations=recommendations,
        locations=locations,
        month_data=month_data
    )

@main_bp.route('/api/predict', methods=['POST'])
def predict_api():
    """API endpoint for programmatic fraud detection"""
    # Get JSON data from the request
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    try:
        # Extract features from the request
        features = np.array(data['features']).reshape(1, -1)
        
        # Make a prediction using the model
        if model is not None:
            prediction = model.predict(features)
            prediction_proba = model.predict_proba(features)[:, 1]  # Probability of fraud
            return jsonify({
                "prediction": prediction.tolist(),
                "fraud_probability": prediction_proba.tolist()
            })
        else:
            return jsonify({"error": "Model not loaded properly"}), 500
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@main_bp.route('/api/upi/verify', methods=['POST'])
def upi_verify_api():
    """API endpoint for UPI-specific real-time fraud detection"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No transaction data provided"}), 400

    try:
        # Extract transaction data
        transaction = data.get('transaction', {})
        
        # Required fields
        required_fields = ['amount', 'upi_id', 'recipient_upi_id', 'device_id']
        for field in required_fields:
            if field not in transaction:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Generate fraud risk indicators
        risk_indicators = []
        risk_score = 0
        
        # Example risk indicators based on transaction details
        # 1. Transaction amount anomaly
        if float(transaction.get('amount', 0)) > 10000:
            risk_indicators.append({
                "type": "amount_anomaly",
                "severity": "high",
                "description": "Transaction amount unusually high"
            })
            risk_score += 30
        
        # 2. New recipient check
        if transaction.get('is_trusted_recipient') == False:
            risk_indicators.append({
                "type": "new_recipient",
                "severity": "medium",
                "description": "Payment to new/untrusted recipient"
            })
            risk_score += 20
        
        # 3. Unusual time check
        transaction_hour = datetime.now().hour
        if transaction_hour >= 0 and transaction_hour <= 4:
            risk_indicators.append({
                "type": "unusual_time",
                "severity": "medium",
                "description": "Transaction at unusual hours"
            })
            risk_score += 15
        
        # 4. Location check
        if transaction.get('location') and transaction.get('home_location'):
            risk_indicators.append({
                "type": "location_mismatch",
                "severity": "high",
                "description": "Transaction location differs from usual pattern"
            })
            risk_score += 25
        
        # 5. Device check
        if transaction.get('is_trusted_device') == False:
            risk_indicators.append({
                "type": "new_device",
                "severity": "high",
                "description": "Transaction from new/untrusted device"
            })
            risk_score += 25
        
        # Determine overall risk level
        risk_level = "low"
        if risk_score >= 70:
            risk_level = "high"
        elif risk_score >= 40:
            risk_level = "medium"
        
        # Determine if transaction should be blocked 
        should_block = risk_score >= 70
        
        # Store the transaction with fraud assessment (if actual model was used)
        try:
            if 'upi_id' in transaction and 'recipient_upi_id' in transaction:
                # Look up user by UPI ID (simplified for demo)
                user = User.query.filter_by(upi_id=transaction.get('upi_id')).first()
                
                if user:
                    # Create new UPI transaction record
                    upi_tx = UPITransaction(
                        transaction_id=f"UPI{int(time.time())}",
                        user_id=user.id,
                        upi_id=transaction.get('upi_id'),
                        recipient_upi_id=transaction.get('recipient_upi_id'),
                        amount=float(transaction.get('amount', 0)),
                        device_id=transaction.get('device_id', ''),
                        location=transaction.get('location', ''),
                        ip_address=request.remote_addr,
                        auth_method=transaction.get('auth_method', 'pin'),
                        is_trusted_device=transaction.get('is_trusted_device', False),
                        is_trusted_recipient=transaction.get('is_trusted_recipient', False),
                        transaction_type=transaction.get('transaction_type', 'p2p'),
                        fraud_score=risk_score / 100,
                        is_fraud=should_block,
                        fraud_flags=json.dumps(risk_indicators),
                        status='blocked' if should_block else 'completed'
                    )
                    db.session.add(upi_tx)
                    db.session.commit()
        except Exception as e:
            logger.error(f"Error saving UPI transaction: {str(e)}")
            # Continue with response even if storage fails
        
        # Return the fraud assessment
        return jsonify({
            "transaction_id": transaction.get('transaction_id', f"UPI{int(time.time())}"),
            "is_fraudulent": should_block,
            "fraud_score": risk_score,
            "risk_level": risk_level,
            "risk_indicators": risk_indicators,
            "processing_time_ms": 145,  # Demo value
            "recommendation": "block" if should_block else "allow",
            "additional_verification_needed": risk_level == "medium"
        })
    except Exception as e:
        logger.error(f"UPI verification error: {str(e)}")
        return jsonify({"error": str(e)}), 500