from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime, timedelta
import json

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(20), nullable=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default='user') # 'user' or 'admin'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    history = db.relationship('MedicalHistory', backref='user', lazy=True)

class MedicalHistory(db.Model):
    __tablename__ = 'history'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    symptoms = db.Column(db.Text, nullable=False) # Store as JSON string
    predicted_disease = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    prediction_state = db.Column(db.String(50), nullable=True)
    recommended_doctor = db.Column(db.String(100), nullable=False)
    recommended_hospital = db.Column(db.String(255), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def set_symptoms(self, symptoms_list):
        self.symptoms = json.dumps(symptoms_list)
        
    def get_symptoms(self):
        return json.loads(self.symptoms)

    def get_local_timestamp(self):
        # Stored timestamps are UTC-like; display them in Asia/Kolkata time.
        return self.timestamp + timedelta(hours=5, minutes=30)

class Hospital(db.Model):
    __tablename__ = 'hospitals'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    location = db.Column(db.String(200), nullable=False)
    contact = db.Column(db.String(50), nullable=True)
    specialist_type = db.Column(db.String(100), nullable=False)
