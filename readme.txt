How It Works
Users register by providing their name and facial image.
Facial encodings are generated and stored for future recognition.
The system continuously captures video frames from the webcam.
Detected faces are compared against stored facial encodings.
When a match is found, attendance is automatically recorded with the current date and time.
Duplicate entries for the same session are prevented.

Installation
Clone the Repository
git clone https://github.com/DebSaikia892/Attendance-System-2.0.git
cd FaceRecognitionAttendance
Create a Virtual Environment
python -m venv venv

Activate the environment

Install Dependencies
pip install -r requirements.txt

Running the Application
python main.py
