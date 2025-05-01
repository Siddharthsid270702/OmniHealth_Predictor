from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import uuid
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# ✅ *MySQL Configuration*
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:jarvis@localhost/hospital_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'supersecretkey'

# ✅ *Initialize Extensions*
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# ✅ *Load ML Model*
BASE_DIR = r'C:\Users\siddharth\Desktop\diseaseweb\tester\templates\newweb'
try:
    final_rf_model = joblib.load(os.path.join(BASE_DIR, 'model.joblib'))
    encoder = joblib.load(open(os.path.join(BASE_DIR, 'encoder.pkl'), 'rb'))
except FileNotFoundError:
    print("❌ Model files not found! Ensure 'model.joblib' and 'encoder.pkl' exist.")
    exit()

# ✅ *Load CSV Files*
try:
    testing_file = os.path.join(BASE_DIR, "Testing.csv")
    X = pd.read_csv(testing_file, encoding="utf-8") if os.path.exists(testing_file) else pd.DataFrame()
except Exception as e:
    print(f"❌ Error loading Testing.csv: {e}")
    X = pd.DataFrame()

symptoms = X.columns.values if not X.empty else []

try:
    description_df = pd.read_csv(r"C:\Users\siddharth\Desktop\sem4 capstone code\tester\Web page\Disease_Description.csv", encoding="utf-8")
    precaution_df = pd.read_csv(r"C:\Users\siddharth\Desktop\newweb - Copy\Disease_Precaution.csv", encoding="ISO-8859-1")

except FileNotFoundError:
    print("❌ One or more description/precaution CSVs are missing!")
    exit()

symptom_index = {symptom.replace("_", " ").title(): index for index, symptom in enumerate(symptoms)}
data_dict = {"symptom_index": symptom_index, "predictions_classes": encoder.classes_}

# ✅ *User Model*
class User(db.Model, UserMixin):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(8), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

class PatientHistory(db.Model):
    __tablename__ = 'patient_history'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(8), db.ForeignKey('users.patient_id'), nullable=False)
    symptom1 = db.Column(db.String(100))
    symptom2 = db.Column(db.String(100))
    symptom3 = db.Column(db.String(100))
    symptom4 = db.Column(db.String(100))
    predicted_disease = db.Column(db.String(100), nullable=False)
    prediction_time = db.Column(db.DateTime, default=db.func.current_timestamp())

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def generate_patient_id():
    return str(uuid.uuid4().hex)[:8]

def predict_disease(selected_symptoms):
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in selected_symptoms:
        if symptom in data_dict["symptom_index"]:
            input_data[data_dict["symptom_index"][symptom]] = 1
    input_data = np.array(input_data).reshape(1, -1)
    return final_rf_model.predict(input_data)[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
        patient_id = generate_patient_id()

        if User.query.filter_by(email=email).first():
            flash("❌ Email already registered. Please login.", "danger")
            return redirect(url_for('login'))

        new_user = User(patient_id=patient_id, name=name, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()

        return render_template('signup_success.html', patient_id=patient_id, name=name, email=email)

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        identifier = request.form['identifier']
        password = request.form['password']
        user = User.query.filter((User.patient_id == identifier) | (User.email == identifier)).first()

        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash("❌ Invalid credentials. Try again.", "danger")

    return render_template('login.html')

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    symptom_options = list(symptom_index.keys())

    if request.method == 'POST':
        selected_symptoms = [request.form.get(f'symptom{i}') for i in range(1, 5)]
        selected_symptoms = [s for s in selected_symptoms if s and s != "-select-"]

        if not selected_symptoms:
            flash("⚠ Please select at least one symptom.", "warning")
            return redirect(url_for('dashboard'))

        prediction = predict_disease(selected_symptoms)

        description = description_df.loc[description_df['Disease'] == prediction, 'Description'].values
        precaution_row = precaution_df.loc[precaution_df['Disease'] == prediction]

        precautions = []
        if not precaution_row.empty:
            precautions = precaution_row.iloc[0, 1:].dropna().tolist()

        # ✅ Save Prediction to Patient History
        new_entry = PatientHistory(
            patient_id=current_user.patient_id,
            symptom1=selected_symptoms[0] if len(selected_symptoms) > 0 else None,
            symptom2=selected_symptoms[1] if len(selected_symptoms) > 1 else None,
            symptom3=selected_symptoms[2] if len(selected_symptoms) > 2 else None,
            symptom4=selected_symptoms[3] if len(selected_symptoms) > 3 else None,
            predicted_disease=prediction
        )
        db.session.add(new_entry)
        db.session.commit()

        return render_template('dashboard.html', symptoms=symptom_options, prediction=prediction, 
                               description=description[0] if description.size > 0 else "No description available.",
                               precautions=precautions if precautions else ["No precautions available."])

    return render_template('dashboard.html', symptoms=symptom_options)




@app.route('/predict', methods=['GET', 'POST'])  # Ensure it allows POST
def predict():
    if request.method == 'POST':
        # Process symptoms here
        symptom1 = request.form.get('symptom1')
        symptom2 = request.form.get('symptom2')
        symptom3 = request.form.get('symptom3')
        symptom4 = request.form.get('symptom4')

        # Add prediction logic here...
        
        return render_template('dashboard.html', prediction="Sample Prediction", description="Sample description", precautions=["Precaution 1", "Precaution 2"])

    return render_template('dashboard.html')








@app.route('/patient_history')
@login_required
def patient_history():
    history = db.session.execute(
        db.select(PatientHistory).where(PatientHistory.patient_id == current_user.patient_id)
    ).scalars().all()

    return render_template('history.html', history=history)



@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
