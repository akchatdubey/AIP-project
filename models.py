from datetime import datetime
from app import db

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    news_text = db.Column(db.Text, nullable=False)
    prediction = db.Column(db.String(10), nullable=False)  # 'true' for fake news, 'false' for real news
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Prediction {self.id}: {self.prediction} ({self.confidence:.2f}%)>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'newsText': self.news_text,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'timestamp': self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        }