import os
import logging
import pickle
from pathlib import Path
from flask import Flask, render_template
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.middleware.proxy_fix import ProxyFix
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize SQLAlchemy
db = SQLAlchemy()

# Initialize the model and scaler (using the edited code's approach)
model = None
scaler = StandardScaler()

def create_app():
    # Create Flask app
    app = Flask(__name__, template_folder='../templates', static_folder='../static')
    app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key")
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)  # needed for url_for to generate with https
    CORS(app)  # Enable CORS for all routes

    # Register custom Jinja filters (from original code)
    @app.template_filter('number_format')
    def number_format(value):
        """Format a number with thousand separators"""
        return "{:,}".format(value) if value is not None else "0"

    @app.template_filter('format_currency')
    def format_currency(value):
        """Format a number as currency"""
        try:
            value = float(value)
            return f"${value:,.2f}"
        except (ValueError, TypeError):
            return value

    # Configure SQLAlchemy (from original code)
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_recycle": 300,
        "pool_pre_ping": True
    }

    # Initialize SQLAlchemy with app
    db.init_app(app)

    # Load the pre-trained fraud detection model (using edited code's approach)
    try:
        global model
        model_path = Path("best_rf_model.pkl")
        if model_path.exists():
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info("Model loaded successfully")
        else:
            logger.warning("Model file not found.  Using default scaler.")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")


    # Configure upload folder (from original code)
    upload_folder = 'uploads'
    os.makedirs(upload_folder, exist_ok=True)
    app.config['UPLOAD_FOLDER'] = upload_folder
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

    # Register error handlers (from original code)
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('404.html'), 404

    @app.errorhandler(500)
    def server_error(e):
        return render_template('500.html'), 500

    # Create database tables within app context (from original code)
    with app.app_context():
        # Import models to ensure they're registered with SQLAlchemy
        from app.models import User, Transaction, FraudDetectionResult
        db.create_all()

    # Register blueprints (from edited code)
    from app.routes import main_bp
    app.register_blueprint(main_bp)

    return app