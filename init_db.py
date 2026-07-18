from app import app
from models import db, Hospital

def seed_data():
    if Hospital.query.count() == 0:
        hospitals = [
            Hospital(name="City General Hospital", location="Downtown", contact="123-456-7890", specialist_type="General Physician"),
            Hospital(name="Heart Care Center", location="Westside", contact="123-456-7891", specialist_type="Cardiologist"),
            Hospital(name="Skin Health Clinic", location="Northside", contact="123-456-7892", specialist_type="Dermatologist"),
            Hospital(name="Neuro Medical Institute", location="Eastside", contact="123-456-7893", specialist_type="Neurologist"),
            Hospital(name="Allergy & Asthma Care", location="Southside", contact="123-456-7894", specialist_type="Allergist"),
            Hospital(name="Gastroenterology Associates", location="Downtown", contact="123-456-7895", specialist_type="Gastroenterologist")
        ]
        db.session.bulk_save_objects(hospitals)
        db.session.commit()
        print("Database seeded with hospital data.")

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        seed_data()
        print("Database initialized successfully.")
