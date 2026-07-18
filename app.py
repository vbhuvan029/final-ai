from flask import Flask, render_template
from flask_login import LoginManager
from sqlalchemy import inspect, text

from config import Config
from models import db, User
from routes.auth import auth_bp
from routes.prediction import prediction_bp
from routes.chatbot import chatbot_bp

app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)

login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.init_app(app)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


app.register_blueprint(auth_bp)
app.register_blueprint(prediction_bp)
app.register_blueprint(chatbot_bp)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


def _ensure_history_state_column():
    inspector = inspect(db.engine)
    columns = [col["name"] for col in inspector.get_columns("history")]
    if "prediction_state" not in columns:
        with db.engine.begin() as connection:
            connection.execute(text("ALTER TABLE history ADD COLUMN prediction_state VARCHAR(50)"))


with app.app_context():
    db.create_all()
    _ensure_history_state_column()


if __name__ == "__main__":
    app.run(debug=True)
