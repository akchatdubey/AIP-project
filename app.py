from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import joblib
import os
import traceback
import json
import re
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Define common stopwords for text preprocessing (same as in training)
STOPWORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 
    'will', 'just', 'don', 'should', 'now'
])

# Text preprocessing function (same as used in training)
def preprocess_text(text):
    try:
        # Convert to string if not already
        if not isinstance(text, str):
            text = str(text)
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and keep only alphabetic characters and spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize (simple split by whitespace)
        tokens = text.split()
        
        # Remove stopwords and short words
        tokens = [word for word in tokens if word not in STOPWORDS and len(word) > 2]
        
        # Join tokens back into a string
        return ' '.join(tokens)
    except Exception as e:
        logging.error(f"Error preprocessing text: {str(e)}")
        return text  # Return original text on error

# Load the model and vectorizer
try:
    model = joblib.load(r"C:\Users\Acer\Downloads\AIP project\ProcurementPro\model.pkl")
    vectorizer = joblib.load(r"C:\Users\Acer\Downloads\AIP project\ProcurementPro\vectorizer.pkl")
    logging.debug("Model and vectorizer loaded successfully")
except Exception as e:
    logging.error(f"Error loading model or vectorizer: {str(e)}")
    traceback.print_exc()

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "fake-news-detector-secret")
CORS(app)

# Configure PostgreSQL database
database_url = os.environ.get("DATABASE_URL")
if database_url is None:
    logging.error("DATABASE_URL environment variable is not set")
    database_url = "sqlite:///fakenews.db"  # Fallback to SQLite
    logging.warning(f"Using fallback database: {database_url}")

app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Initialize the SQLAlchemy db
db = SQLAlchemy(app)

# Database availability flag - used as fallback for in-memory storage if needed
db_available = True

# Create in-memory storage as fallback
in_memory_predictions = []

# Import the Prediction model
try:
    # Using circular imports pattern for Flask-SQLAlchemy
    from models import Prediction
    logging.debug("Successfully imported Prediction model")
    
    # Create tables
    with app.app_context():
        db.create_all()
        logging.debug("Database tables created successfully")
except Exception as e:
    logging.error(f"Error with database setup: {str(e)}")
    traceback.print_exc()
    db_available = False

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        logging.debug(f"Received data: {data}")

        text = data.get("newsText", "")

        if not text:
            return jsonify({"error": "No news text provided"}), 400

        # Preprocess the text (same way as during training)
        processed_text = preprocess_text(text)
        logging.debug(f"Processed text: {processed_text}")

        # Transform and predict
        X = vectorizer.transform([processed_text])
        result = model.predict(X)[0]
        
        # Get prediction probability
        probability = model.predict_proba(X)[0]
        confidence = max(probability) * 100
        
        # Current timestamp
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")

        try:
            # Store in database if available
            if db_available:
                # Convert numpy types to native Python types if needed
                if hasattr(result, 'item'):
                    result = result.item()
                if hasattr(confidence, 'item'):
                    confidence = confidence.item()
                
                # Create new prediction record
                new_prediction = Prediction(
                    news_text=text,
                    prediction=str(result),
                    confidence=float(confidence),
                    timestamp=timestamp
                )
                # Add to database
                db.session.add(new_prediction)
                db.session.commit()
                logging.debug(f"Stored prediction in database with ID: {new_prediction.id}")
            else:
                # Fallback to in-memory storage
                prediction_record = {
                    "newsText": text,
                    "prediction": result,
                    "confidence": confidence,
                    "timestamp": timestamp_str
                }
                in_memory_predictions.append(prediction_record)
                logging.debug("Stored prediction in in-memory storage (database unavailable)")
        except Exception as storage_err:
            logging.error(f"Error storing prediction: {str(storage_err)}")
            traceback.print_exc()
            # Store in memory as fallback
            prediction_record = {
                "newsText": text,
                "prediction": result,
                "confidence": confidence,
                "timestamp": timestamp_str
            }
            in_memory_predictions.append(prediction_record)
            logging.debug("Stored prediction in in-memory storage (fallback)")
        
        return jsonify({
            "prediction": result,
            "confidence": confidence,
            "timestamp": timestamp_str
        })

    except Exception as e:
        logging.error(f"Error in /predict: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/history", methods=["GET"])
def get_history():
    try:
        if db_available:
            # Get predictions from database, ordered by timestamp descending
            predictions = Prediction.query.order_by(Prediction.timestamp.desc()).all()
            
            # Convert to list of dictionaries
            history = [prediction.to_dict() for prediction in predictions]
            logging.debug(f"Retrieved {len(history)} items from database")
        else:
            # Use in-memory storage as fallback
            # Sort by timestamp in descending order (newest first)
            history = sorted(in_memory_predictions, key=lambda x: x["timestamp"], reverse=True)
            logging.debug(f"Retrieved {len(history)} items from in-memory storage (database unavailable)")
                
        return jsonify(history)
    except Exception as e:
        logging.error(f"Error in /history: {str(e)}")
        traceback.print_exc()
        # Try to use in-memory as fallback
        try:
            history = sorted(in_memory_predictions, key=lambda x: x["timestamp"], reverse=True)
            logging.debug(f"Retrieved {len(history)} items from in-memory storage (fallback)")
            return jsonify(history)
        except:
            return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/history-page")
def history_page():
    return render_template("history.html")

@app.route("/education")
def education():
    return render_template("education.html")

@app.route("/about")
def about():
    return render_template("about.html")
        
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
