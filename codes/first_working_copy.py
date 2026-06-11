import sqlite3
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis

device = "cuda" if torch.cuda.is_available() else "cpu"
#YOLOv8n
detector = YOLO("yolov8n-face.pt")
detector.to(device)
#ArcFace
app = FaceAnalysis(name="buffalo_l")
# app.prepare(ctx_id=0) # GPU
app.prepare(ctx_id=-1) # CPU

##database function##
def database_entry():
    conn = sqlite3.connect("E:\\Projects\\Attendance System 2.0\\faces.db")
    print('DB Connected.')

    cursor = conn.cursor()

    cursor.execute('CREATE TABLE IF NOT EXISTS faces(ID INTEGER PRIMARY KEY AUTOINCREMENT, Name TEXT NOT NULL, Image BLOB NOT NULL, Encoding BLOB NOT NULL)')

    print('Table created/verified.')

    #enter code below here .....

    choice = input('Have you previously made an entry into the system? (y/n)').lower()

    if choice == 'n':
        name = input("Please enter your name: ")
        print('''Please capture your photo for entry.
        Press 's' to capture image
        Press 'q' to close window''')
        image = camera_input()
        if image is None:
            print("No image captured")
        else:
            print("Image captured successfully")
            encoding = face_encoding(image)
            if encoding is None:
                print("Encoding failed. Try Again.")
            else:
                print("Encoding successfull")
                #entry into the table        
                entries(conn, image, encoding, name)
    elif choice == 'y':
        print('Skipping entry.')
    
    recognize_faces(conn)

    conn.commit()
    conn.close()


##functions for entries##
def entries(conn, image, encoding, name):
    img_b = image_blob(image)
    enc_b = encode_blob(encoding)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO faces(Name, Image, Encoding) VALUES (?, ?, ?)", (name, img_b, enc_b))
    #confirm entry
    cursor.execute("SELECT * FROM faces WHERE ID = ?", (cursor.lastrowid,))
    row = cursor.fetchone()
    if row:
        print("Entry confirmed in database")
    else:
        print("Entry Unsuccessfull")
    print()

##helper functions##    
#convert encodings to blob
def encode_blob(encoding):
    return encoding.tobytes()
#convert images to blob
def image_blob(image):  
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

def camera_input():
    # index = input('Select camera(use index): ')
    
    cap = cv2.VideoCapture(int(0), cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Could not open camera.")
        return None
    
    while True:
        #original frame
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame.")
            break
            
        #copying the original frame to display
        display_frame = frame.copy()
        
        #YOLOv8n 
        results = detector(display_frame, verbose=False)
        faces = []
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                faces.append((x1, y1, x2, y2))
                #Bounding Box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if len(faces) > 1:
            cv2.putText(display_frame, "Multiple faces detected", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        elif len(faces) == 1:
            x1, y1, x2, y2 = faces[0]
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 
        else:
            cv2.putText(display_frame, "No faces detected", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #output
        cv2.imshow('Camera Feed', display_frame)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            captured = frame.copy()
            if len(faces) != 1:
                print("Capture failed. Ensure exactly ONE face.")
                continue

            cv2.imshow("Captured Image", captured)
            print("Press 'y' to accept and 'r' to retake")

            while True:
                key2 = cv2.waitKey(0) & 0xFF

                if key2 == ord('y'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return captured
                elif key2 == ord('r'):
                    cv2.destroyWindow("Captured Image")
                    break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return None

#image encoding function
def face_encoding(image):
    faces = app.get(image)
    if len(faces) == 0:
        print("No faces detected during encoding.")
        return None
    if len(faces) > 1:
        print("Multiple faces detected during encoding.")
        return None
    encoding = faces[0].embedding # numpy array (512 dim)
    return encoding

#load faces and encodings from database
def load_faces(conn):
    cursor = conn.cursor()
    cursor.execute("select Name, Encoding from faces")
    rows = cursor.fetchall()
    known_names = []
    known_encodings = []
    for name, enc_blob in rows:
        known_names.append(name)
        encoding = blob_encode(enc_blob)
        known_encodings.append(encoding)
    return known_names, known_encodings

#face recognition function
def recognize_faces(conn):
    known_names, known_encodings = load_faces(conn)
    #Normalize encodings
    known_encodings = [known_encoding / np.linalg.norm(known_encoding) for known_encoding in known_encodings]
    if len(known_names) == 0:
        print("No registered faces found.")
        return
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Could not open camera.")
        return
    print("Press 'q' to quit.")
    frame_count = 0
    frame_skip = 5
    current_name = "Unknown"
    while True:
        previous_name = current_name
        ret, frame = cap.read()
        if not ret:
            break
        #copying the original frame to display
        display_frame = frame.copy()
        frame_count += 1
        
        #YOLOv8n 
        results = detector(display_frame, verbose=False)
        faces = []
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                faces.append((x1, y1, x2, y2))
                #Bounding Box
                padding = 7
                cv2.rectangle(display_frame, (x1 - padding, y1 - padding), (x2 + padding, y2 + padding), (0, 255, 0), 2)
        if len(faces) > 1:
            cv2.putText(display_frame, "Multiple faces detected", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) 
        elif len(faces) == 0:
            cv2.putText(display_frame, "No faces detected", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            if frame_count % frame_skip == 0:
                x1, y1, x2, y2 = faces[0]
                # face_crop = frame[y1:y2, x1:x2]
                # faces_insight = app.get(face_crop)
                faces_insight = app.get(display_frame)
                if len(faces_insight) == 1:
                    # if previous_name != current_name: 
                    encoding = faces_insight[0].embedding
                    #Normalize
                    encoding = encoding / np.linalg.norm(encoding)
                    #Cosine similarity
                    similarities = [np.dot(known_encoding, encoding) for known_encoding in known_encodings]

                    best_index = np.argmax(similarities)
                    best_score = similarities[best_index]
                    threshold = 0.5
                    if best_score > threshold:
                        current_name = known_names[best_index]
                    else:
                        current_name = "Unknown"
        #Display name
        cv2.putText(display_frame, current_name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Face Recognition', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
        

database_entry()
