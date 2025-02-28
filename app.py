# app.py
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'  # Change this to a secure secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    scans = db.relationship('Scan', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Scan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    prediction = db.Column(db.String(50), nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# Model Architecture
class BrainTumorEnsemble(nn.Module):
    def __init__(self, num_classes=2, print_shapes=False):
        super(BrainTumorEnsemble, self).__init__()
        # Load Pretrained Models
        self.vgg16 = models.vgg16(pretrained=True)
        self.resnet50 = models.resnet50(pretrained=True)
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        
        # Remove last layer (classifier) from each model
        self.vgg16.classifier = nn.Identity()
        self.resnet50.fc = nn.Identity()
        self.efficientnet.classifier = nn.Identity()
        
        # Define output feature sizes
        vgg16_out = 25088
        resnet50_out = 2048
        efficientnet_out = 1280
        
        # Fully connected layer for final classification
        self.fc = nn.Sequential(
            nn.Linear(vgg16_out + resnet50_out + efficientnet_out, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        self.print_shapes = print_shapes

    def forward(self, x):
        vgg_features = self.vgg16(x).view(x.size(0), -1)
        resnet_features = self.resnet50(x).view(x.size(0), -1)
        efficientnet_features = self.efficientnet(x).view(x.size(0), -1)
        combined_features = torch.cat([vgg_features, resnet_features, efficientnet_features], dim=1)
        output = self.fc(combined_features)
        return output

# Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def load_model(model_path):
    global model
    model = BrainTumorEnsemble(num_classes=2)
    state_dict = torch.load(model_path, map_location=device)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

def process_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

def get_prediction(image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(outputs, dim=1)
        confidence = probabilities[0][prediction].item()
    return prediction.item(), confidence

# Routes
@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('signup'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('signup'))
        
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    scans = Scan.query.filter_by(user_id=current_user.id).order_by(Scan.timestamp.desc()).all()
    return render_template('dashboard.html', scans=scans)

@app.route('/previous_predictions')
@login_required
def previous_predictions():
    scans = Scan.query.filter_by(user_id=current_user.id).order_by(Scan.timestamp.desc()).all()
    return render_template('previous_predictions.html', scans=scans)


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Save the file
        filename = f"{current_user.id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the image
        with open(filepath, 'rb') as f:
            image_bytes = f.read()
        image_tensor = process_image(image_bytes)
        
        # Get prediction
        class_names = ['No Tumor', 'Tumor']
        prediction, confidence = get_prediction(image_tensor)
        
        # Save scan record to database
        scan = Scan(
            filename=filename,
            prediction=class_names[prediction],
            confidence=confidence,
            user_id=current_user.id
        )
        db.session.add(scan)
        db.session.commit()
        
        # Prepare response
        response = {
            'prediction': class_names[prediction],
            'confidence': f"{confidence:.2%}",
            'filename': filename,
            'success': True
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    # Create database tables
    with app.app_context():
        db.create_all()
    
    # Load model
    model_path = 'C:/Users/sanke/OneDrive/Desktop/6630/Brain Tumour/Brain Tumour/best_model_epoch_3.pt'  # Update this path
    load_model(model_path)
    print("Model loaded successfully!")
    
    # Start Flask server
    app.run(debug=False, host='0.0.0.0', port=5000)