# 🎓 Attendance System Using Face Recognition

A Django-based web application for automated attendance management using facial recognition. Originally developed as a semester project, this system leverages Haar Cascades for real-time face detection and matches faces against a stored database to mark attendance.

🌐 **Live Demo**: [https://saiky.pythonanywhere.com](https://saiky.pythonanywhere.com)

## 🚀 Features

- Real-time face detection and recognition using OpenCV
- Attendance records stored in a SQLite database
- Admin dashboard for managing sessions
- Simple web UI using Django templates and static files

## 🛠 Tech Stack

- **Backend**: Django, SQLite
- **Frontend**: HTML, CSS (Django Templates)
- **Computer Vision**: OpenCV, Haar Cascade Classifier
- **Language**: Python 3.6+

## 📦 Installation

> Ensure Python 3.6+ is installed. Use a virtual environment for better dependency management.

```bash
# Clone the repository
git clone https://github.com/saikirananumalla/Attendance-System-Using-Face-Recognition.git
cd Attendance-System-Using-Face-Recognition

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python manage.py runserver
