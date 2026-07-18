import random
from flask import Blueprint, render_template, redirect, url_for, flash, request, session
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User, MedicalHistory

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email address already exists')
            return redirect(url_for('auth.register'))
            
        new_user = User(
            name=name,
            email=email,
            phone=phone,
            password_hash=generate_password_hash(password, method='scrypt')
        )
        db.session.add(new_user)
        db.session.commit()
        
        return redirect(url_for('auth.login'))
        
    return render_template('register.html')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        
        if not user or not check_password_hash(user.password_hash, password):
            flash('Please check your login details and try again.')
            return redirect(url_for('auth.login'))
            
        login_user(user)
        return redirect(url_for('auth.dashboard'))
        
    return render_template('login.html')

@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@auth_bp.route('/dashboard')
@login_required
def dashboard():
    history_count = MedicalHistory.query.filter_by(user_id=current_user.id).count()
    recent_history = MedicalHistory.query.filter_by(user_id=current_user.id).order_by(MedicalHistory.timestamp.desc()).limit(5).all()
    return render_template('dashboard.html', history_count=history_count, recent_history=recent_history)

@auth_bp.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        phone = request.form.get('phone')
        user = User.query.filter_by(email=email).first()
        if user:
            if user.phone != phone:
                flash('The phone number does not match our records for this email.', 'danger')
                return redirect(url_for('auth.forgot_password'))
                
            otp = str(random.randint(100000, 999999))
            session['reset_email'] = email
            session['reset_otp'] = otp
            # Mocking SMS
            print(f"\n[MOCK SMS] To: {phone} | Your OTP for password reset is: {otp}\n")
            flash(f'An OTP has been sent to your phone. (Check server console or use {otp} for testing)')
            return redirect(url_for('auth.verify_otp'))
        else:
            flash('No account found with that email address.')
    return render_template('forgot_password.html')

@auth_bp.route('/verify_otp', methods=['GET', 'POST'])
def verify_otp():
    if 'reset_email' not in session:
        return redirect(url_for('auth.forgot_password'))
        
    if request.method == 'POST':
        entered_otp = request.form.get('otp')
        if entered_otp == session.get('reset_otp'):
            session['otp_verified'] = True
            return redirect(url_for('auth.reset_password'))
        else:
            flash('Invalid OTP. Please try again.')
    return render_template('verify_otp.html')

@auth_bp.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    if not session.get('otp_verified'):
        return redirect(url_for('auth.forgot_password'))
        
    if request.method == 'POST':
        password = request.form.get('password')
        email = session.get('reset_email')
        user = User.query.filter_by(email=email).first()
        if user:
            user.password_hash = generate_password_hash(password, method='scrypt')
            db.session.commit()
            session.pop('reset_email', None)
            session.pop('reset_otp', None)
            session.pop('otp_verified', None)
            flash('Your password has been updated! You can now log in.')
            return redirect(url_for('auth.login'))
    return render_template('reset_password.html')
