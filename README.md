# Brain Tumor Detection using Flask and Deep Learning

## 📌 Overview
This project is a web application for brain tumor detection using an **ensemble deep learning model**. Users can **upload MRI images**, and the system will **predict whether a tumor is present** along with a **confidence score**. The application also supports **user authentication**, stores previous scan records, and provides a user-friendly dashboard.

## 🚀 Features
- **User Authentication** (Signup/Login/Logout)
- **Upload MRI images** for analysis
- **Brain tumor detection** using an **ensemble deep learning model** (VGG16, ResNet50, EfficientNetB0)
- **Stores previous scan results** in a database
- **Dashboard for users** to view past predictions
- **REST API for predictions**

---
## 🏗️ Tech Stack
- **Backend:** Flask, Flask-SQLAlchemy, Flask-Login
- **Frontend:** HTML, CSS, JavaScript
- **Database:** SQLite
- **Deep Learning:** PyTorch, torchvision
- **Model Architecture:** Ensemble of VGG16, ResNet50, EfficientNetB0

---
## 📂 Project Structure
```
BrainTumorDetection/
│── static/                # Stores uploaded MRI images
│── templates/             # HTML templates for web pages
│── app.py                 # Main Flask application
│── models.py              # User and scan database models
│── requirements.txt       # Dependencies
│── README.md              # Project documentation
│── best_model_epoch_3.pt  # Pretrained model (stored via Git LFS)
│── .gitignore             # Files to exclude from version control
│── .gitattributes         # Configures Git LFS for large files
```

---
## 🔧 Installation & Setup
### 1️⃣ Clone the repository
```bash
git clone https://github.com/SankethaReddy/BrainTumour.git
cd BrainTumour
```

### 2️⃣ Create a virtual environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Set up the database
```bash
python
>>> from app import db
>>> db.create_all()
>>> exit()
```

### 5️⃣ Configure Git LFS (For large model file)
```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
```

### 6️⃣ Run the application
```bash
python app.py
```


---
## 🔍 How It Works
1. **User Signup/Login**
2. **Upload an MRI Image**
3. **Model Processes the Image**
4. **Displays Prediction & Confidence Score**
5. **Previous Scans are Stored for Future Reference**

---
## 🤖 Model Details
The model is an **ensemble of three pre-trained architectures**:
- **VGG16** (Feature extractor)
- **ResNet50** (Feature extractor)
- **EfficientNetB0** (Feature extractor)
- **Fully connected layer** for final classification (No Tumor/Tumor)

The model takes **224x224 RGB images** as input and outputs a **binary classification (No Tumor/Tumor)**.

---
## 📌 API Endpoints
| Endpoint           | Method | Description |
|-------------------|--------|-------------|
| `/`               | GET    | Home Page |
| `/signup`         | GET/POST | User Registration |
| `/login`          | GET/POST | User Login |
| `/logout`         | GET    | User Logout |
| `/dashboard`      | GET    | User Dashboard |
| `/predict`        | POST   | Upload an MRI scan and get a prediction |
| `/previous_predictions` | GET | View previous scan results |


---
## 🎯 Author
**Sanketha** 🚀

📌 **GitHub:** [SankethaReddy](https://github.com/SankethaReddy)

