import cv2
import sqlite3
import face_recognition
from pygrabber.dshow_graph import FilterGraph

#detect cameras in the system
def camera_names():
    graph = FilterGraph()
    devices = graph.get_input_devices()
    return devices
cams = camera_names()
for i, cam in enumerate(cams):
    print(f"{i}: {cam}")

##camera function to take input image##
def camera_input():
    index = input('Select camera(use index): ')
    
    cap = cv2.VideoCapture(int(index), cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Could not open camera.")
        return
    
    #Haar-Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    while True:
        #original frame
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame.")
            break
            
        #copying the original frame to display
        display_frame = frame.copy()
        
        #face detection (haar-cascade)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x,y), (x+w, y+h), (0,255,0), 2)

        #output
        cv2.imshow('Camera Feed', display_frame)

        key = cv2.waitKey(1)



        if key == ord('s'):
            captured = frame.copy()

            cv2.imshow("Captured Image", captured)
            print("Press 'y' to accept and 'r' to retake")

            while True:
                key2 = cv2.waitKey(0)

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

# image = camera_input()
# if image is None:
#     print("No image captured")
# else:
#     print("Image captured successfully")
#     # print("Press any key to close window")
#     # cv2.imshow("Returned Image", image)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()

def display_table():
    conn = sqlite3.connect("E:\\Projects\\Attendance System 2.0\\faces.db")
    cursor = conn.cursor()
    cursor.execute("select ID,Name from faces")
    rows = cursor.fetchall()
    for row in rows:
        print(row)

display_table()
