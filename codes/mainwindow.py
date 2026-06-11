import sys
import sqlite3
import os
import cv2
import csv
import torch
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QLineEdit
from PySide6.QtCore import QFile, QTimer
from PySide6.QtGui import QImage, QPixmap

class Mainwindow:
    def __init__(self):
        #initialize ui loader
        loader = QUiLoader()
        ui_path = r"E:\Projects\Attendance System 2.0\mainwindow.ui"
        ui_file = QFile(ui_path)
        ui_file.open(QFile.ReadOnly)

        #load ui
        self.ui = loader.load(ui_file)
        ui_file.close()
        #label
        self.camera_label = self.ui.findChild(QLabel, "cameralabel")
        self.name_label = self.ui.findChild(QLabel, "namelabel")
        self.date_label = self.ui.findChild(QLabel, "datelabel")
        self.time_label = self.ui.findChild(QLabel, "timelabel")
        self.status_label = self.ui.findChild(QLabel, "statuslabel")
        self.status_label2 = self.ui.findChild(QLabel, "statuslabel2")
        self.id_label = self.ui.findChild(QLabel, "idlabel")
        self.name_input = self.ui.findChild(QLineEdit, "nameinput")
        #buttons
        self.ui.signinbutton.clicked.connect(self.handle_signin)
        self.ui.signoutbutton.clicked.connect(self.handle_signout)
        self.ui.capturebutton.clicked.connect(self.handle_capture)
        self.ui.retakebutton.clicked.connect(self.handle_retake)
        self.ui.entrybutton.clicked.connect(self.handle_entry)
        

        #date and time
        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self.update_datetime)
        self.clock_timer.start(1000)
        now = datetime.now()
        year = str(now.year)
        month = now.strftime("%b")
        #attendance file
        base_folder = r"E:\Projects\Attendance System 2.0\Attendance"
        year_folder = os.path.join(base_folder, year)
        os.makedirs(year_folder, exist_ok=True)
        self.file_path = os.path.join(year_folder, f"{now.month:02d}_{month}_{year}.csv")
        if not os.path.exists(self.file_path):
            with open(self.file_path, mode = 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Name", "Date", "Sign In", "Sign Out"])

        #database
        self.conn = sqlite3.connect("E:\\Projects\\Attendance System 2.0\\faces.db")
        self.cursor = self.conn.cursor()
        self.cursor.execute('CREATE TABLE IF NOT EXISTS faces(ID INTEGER PRIMARY KEY AUTOINCREMENT, Name TEXT NOT NULL, Image BLOB NOT NULL, Encoding BLOB NOT NULL)')
        #load faces
        self.ids, self.known_names, self.known_encodings = self.load_faces()
        self.current_id = ""
        self.current_name = "Unknown"
        self.previous_name = self.current_name
        self.image = None

        #CUDA
        device = "cuda" if torch.cuda.is_available() else "cpu"
        #YOLOv8n
        self.detector = YOLO("yolov8n-face.pt")
        self.detector.to(device)
        #ArcFace
        self.app = FaceAnalysis(name="buffalo_l")
        # app.prepare(ctx_id=0) # GPU
        self.app.prepare(ctx_id=-1) # CPU

        self.frame_count = 0
        #load camera
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        #timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30) #update every 30ms, so 30 fps


    def update_frame(self):
        ret, self.frame = self.cap.read()
        self.display_frame = self.frame.copy()
        # if not ret:
        #     print("Failed to capture frame.")
        #     return
        self.recognize_faces()    
    
    #face recognition function
    def recognize_faces(self):
        if len(self.known_names) == 0:
            self.status_label.setText("No registered faces found.")
            return
        frame_skip = 10
        self.frame_count += 1
        
        #YOLOv8n 
        results = self.detector(self.display_frame, verbose=False)
        faces = []
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                faces.append((x1, y1, x2, y2))
                #Bounding Box
                padding = 7
                cv2.rectangle(self.display_frame, (x1 - padding, y1 - padding), (x2 + padding, y2 + padding), (0, 255, 0), 2)
        if len(faces) > 1: 
            self.name_label.setText("Multiple faces detected")
            self.id_label.setText("")
            self.display_frame = cv2.cvtColor(self.display_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = self.display_frame.shape
            bytes_per_line = ch * w
            image = QImage(self.display_frame, w, h, bytes_per_line, QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(image))
            
        elif len(faces) == 0:
            self.name_label.setText("No face detected")
            self.id_label.setText("")
            self.display_frame = cv2.cvtColor(self.display_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = self.display_frame.shape
            bytes_per_line = ch * w
            image = QImage(self.display_frame, w, h, bytes_per_line, QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(image))
        else:
            if self.frame_count % frame_skip == 0:
                x1, y1, x2, y2 = faces[0]
                faces_insight = self.app.get(self.display_frame)
                if len(faces_insight) == 1:
                    # if previous_name != current_name: 
                    encoding = faces_insight[0].embedding
                    #Normalize
                    encoding = encoding / np.linalg.norm(encoding)
                    if len(self.known_encodings) == 0:
                        self.current_name = "Unknown"
                        self.current_id = ""
                    else:    
                    #Cosine similarity
                        similarities = [np.dot(known_encoding, encoding) for known_encoding in self.known_encodings]

                        best_index = np.argmax(similarities)
                        best_score = similarities[best_index]
                        threshold = 0.5
                        if best_score > threshold:
                            self.current_name = self.known_names[best_index]
                            self.current_id = self.ids[best_index]
                        else:
                            self.current_name = "Unknown"
                            self.current_id = ""
            #Display name
            self.name_label.setText(self.current_name)
            self.id_label.setText(self.current_id)
            #convert BGR to RGB 
            self.display_frame = cv2.cvtColor(self.display_frame, cv2.COLOR_BGR2RGB)
            #height, width, channel of the frame
            h, w, ch = self.display_frame.shape
            #size per row, cause Qt needs to know after which memory slot it needs to look for the next row of the image 
            bytes_per_line = ch * w
            image = QImage(self.display_frame, w, h, bytes_per_line, QImage.Format_RGB888)
            #display image in label
            self.camera_label.setPixmap(QPixmap.fromImage(image))

    #load registered faces from database
    def load_faces(self):
        cursor = self.conn.cursor()
        cursor.execute("select ID, Name, Encoding from faces")
        rows = cursor.fetchall()
        ids = []
        known_names = []
        known_encodings = []
        for id, name, enc_blob in rows:
            ids.append(str(id))
            known_names.append(name)
            encoding = blob_encode(enc_blob)
            known_encodings.append(encoding)
        known_encodings = [known_encoding / np.linalg.norm(known_encoding) for known_encoding in known_encodings]
        return ids, known_names, known_encodings

    # signin signout handlers
    def handle_signin(self):
        name = self.current_name
        if name != "Unknown":
            self.sign_in(name)
    def handle_signout(self):
        name = self.current_name
        if name != "Unknown":
            self.sign_out(name)
    #signin
    def sign_in(self, name):
        date = datetime.now().strftime("%d-%m-%Y")
        time1 = datetime.now().strftime("%H:%M:%S")
        rows = []
        with open(self.file_path, mode = 'r', newline = '') as f:
            reader = csv.reader(f)
            rows = list(reader)

        for row in rows[1:]:
            if row[0] == name and row[1] == date:
                self.status_label.setText("Already Signed IN for the day!")
                QTimer.singleShot(3000, lambda: self.status_label.setText(""))
                return
        
        with open(self.file_path, mode = 'a', newline = '') as f:
            writer = csv.writer(f)
            writer.writerow([name, date, time1, ""])
        
        self.status_label.setText("Signed In")
        QTimer.singleShot(3000, lambda: self.status_label.setText(""))
    #signout
    def sign_out(self, name):
        date = datetime.now().strftime("%d-%m-%Y")
        time2 = datetime.now().strftime("%H:%M:%S")
        rows = []
        with open(self.file_path, mode = 'r', newline = '') as f:
            reader = csv.reader(f)
            rows = list(reader)
            signed_in = False
            for row in rows[1:]:
                if row[0] == name and row[1] == date and row[3] != "":
                    self.status_label.setText("Already Signed OUT for the day!")
                    QTimer.singleShot(3000, lambda: self.status_label.setText(""))
                    return
                if row[0] == name and row[1] == date and row[2] != "":
                    signed_in = True

                if row[0] == name and row[1] == date and row[3] == "":
                    row[3] = time2
            if signed_in == False:
                self.status_label.setText("No Sign In found!")
                QTimer.singleShot(3000, lambda: self.status_label.setText(""))
                return
        
        with open(self.file_path, mode = 'w', newline = "") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        
        self.status_label.setText("Signed Out")
        QTimer.singleShot(3000, lambda: self.status_label.setText(""))

    #capture image for registration
    def handle_capture(self):
        self.timer.stop()
        captured_frame = self.frame.copy()
        self.image = captured_frame
        frame_rgb = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        display_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(display_image))
    #retake captured image
    def handle_retake(self):
        self.timer.start(30) #resume camera feed

    def handle_entry(self):
        if self.image is None:
            self.status_label2.setText("No image captured.")
            QTimer.singleShot(3000, lambda: self.status_label2.setText(""))
            return
        image_blob = img_blob(self.image)
        encoding_blob = encode_blob(self.face_encoding())
        name = self.name_input.text().strip()
        if name == "" :
            self.status_label2.setText("Please enter a name.")
            QTimer.singleShot(3000, lambda: self.status_label2.setText(""))
            return
        self.cursor.execute("INSERT INTO faces (Name, Image, Encoding) VALUES (?, ?, ?)", (name, image_blob, encoding_blob))
        self.conn.commit()
        self.status_label2.setText("Entry added.")
        QTimer.singleShot(3000, lambda: self.status_label2.setText(""))
        self.ids, self.known_names, self.known_encodings = self.load_faces() #reload faces after entry
        self.timer.start(30) #resume camera feed

    def face_encoding(self):
        faces = self.app.get(self.image)
        if len(faces) == 0:
            self.status_label2.setText("No face detected.")
            return None
        if len(faces) > 1:
            self.status_label2.setText("Multiple faces detected.")
            return None
        encoding = faces[0].embedding # numpy array (512 dim)
        return encoding
        
    def update_datetime(self):
        now = datetime.now()

        current_date = now.strftime("%A, %d %B %Y")
        current_time = now.strftime("%I:%M:%S %p")  

        self.date_label.setText(current_date)
        self.time_label.setText(current_time)

    def show(self):
        self.ui.show()

#utility functions
def encode_blob(encoding):
    return encoding.tobytes()
#convert images to blob
def img_blob(image):  
    success, buffer = cv2.imencode('.jpg', image)
    if not success:
        print('Encoding failed for image.')
    return buffer.tobytes()  
def blob_image(blob):
    nparr = np.frombuffer(blob, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image
def blob_encode(enc_blob):
    encoding = np.frombuffer(enc_blob, dtype=np.float32)
    return encoding
   

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Mainwindow()
    window.show()
    sys.exit(app.exec())
