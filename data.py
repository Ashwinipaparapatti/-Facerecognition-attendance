import cv2
import face_recognition
import numpy as np
import os
import mysql.connector
from mysql.connector import Error
from datetime import datetime, timedelta
import shutil
import time

# Connect to MySQL database
def connect_to_mysql():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            database='attendence',  # Replace with your database name
            user='root',            # Replace with your MySQL username
            password='mf90zk@ash123'  # Replace with your MySQL password
        )
        if connection.is_connected():
            print('Connected to MySQL database')
            return connection
    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
        return None

# Function to check if the face is already recognized today
def is_already_recognized_today(connection, name):
    try:
        cursor = connection.cursor()
        today_date = datetime.now().date()
        cursor.execute("SELECT COUNT(*) FROM information1 WHERE name=%s AND DATE(entry_time)=%s", (name, today_date))
        count = cursor.fetchone()[0]
        return count > 0
    except Error as e:
        print(f"Error checking recognized face in MySQL: {e}")
        return False

# Function to store recognized face in the database
def store_recognized_face_in_mysql(connection, name, roll_number):
    try:
        cursor = connection.cursor()
        entry_time = datetime.now()
        cursor.execute("INSERT INTO information1 (name, roll_number, entry_time) VALUES (%s, %s, %s)", (name, roll_number, entry_time))
        connection.commit()
        print(f"{name} ({roll_number}) recognized and stored in MySQL database.")
    except Error as e:
        print(f"Error storing recognized face in MySQL: {e}")

# Path to your image folder
image_folder_path = r'c:\Users\ashwi\OneDrive\Desktop\Face_recog\Face_recog\faces'

known_face_encodings = []
known_face_names = []

# Define roll numbers for each person
roll_numbers = {
    "Ashwini": "22211a6793",
    "Gayatri": "22211a6794",
    "Dhoni":"22211a6795",
    "Honey":"22211a6796",
    "Nani":"22211a6797",
    "Mrunal":"22211a6798"
}

# Load each folder in the image folder
for folder_name in os.listdir(image_folder_path):
    folder_path = os.path.join(image_folder_path, folder_name)
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith((".png", ".jpg", ".jpeg")):
                # Load an image
                image_path = os.path.join(folder_path, filename)
                image = face_recognition.load_image_file(image_path)

                # Encode the face into a vector
                face_encoding = face_recognition.face_encodings(image)[0]

                # Store the face encoding and name
                known_face_encodings.append(face_encoding)
                known_face_names.append(folder_name)  # Assuming folder name is the person's name

                # Move the main photo to the respective folder
                destination_folder = os.path.join(image_folder_path, folder_name)
                if not os.path.exists(destination_folder):
                    os.makedirs(destination_folder)
                shutil.move(image_path, os.path.join(destination_folder, filename))

print("Faces loaded and encoded")

# Function to capture and store 5 snapshots
def capture_snapshots(video_capture, name):
    # Create a folder for the recognized face if it doesn't exist
    face_folder_path = os.path.join(image_folder_path, name)
    if not os.path.exists(face_folder_path):
        os.makedirs(face_folder_path)
    
    # Check if the folder already contains 5 snapshots
    if len(os.listdir(face_folder_path)) >= 5:
        
        return
    
    snapshots = 0
    while snapshots < 5:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Detect faces in the frame
        face_locations = face_recognition.face_locations(frame)
        
        if len(face_locations) == 0:
            print("No face detected in the image")
            continue  # Skip to the next iteration of the loop if no face is detected
        
        # Display the notification to change angle
        frame_with_message = frame.copy()
        cv2.putText(frame_with_message, "Change your angle!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("My Face Recognition Project", frame_with_message)
        cv2.waitKey(1000)  # Display message for 1 second

        # Capture the next frame without the message
        ret, frame = video_capture.read()
        if not ret:
            break

        # Save the snapshot in the person's folder
        snapshot_filename = os.path.join(face_folder_path, f"{name}_{snapshots + 1}.jpg")
        cv2.imwrite(snapshot_filename, frame)
        snapshots += 1

        time.sleep(1)  # Wait for 1 more second before next snapshot

    print(f"{name} has now {len(os.listdir(face_folder_path))} snapshots.")

# Connect to MySQL database
connection = connect_to_mysql()
if connection:
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    video_capture = cv2.VideoCapture(0)

    def detect_bounding_box(frame):
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
        return faces

    while True:
        ret, frame = video_capture.read()

        if not ret:
            break

        faces = detect_bounding_box(frame)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

            # Extract the face encoding from the detected face
            face_encoding = face_recognition.face_encodings(frame, [(y, x+w, y+h, x)])[0]

            # Compare the face encoding with known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            # If a match is found, get the name
            if True in matches:
                match_index = matches.index(True)
                name = known_face_names[match_index]
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

                # Check if the face has fewer than 5 snapshots
                capture_snapshots(video_capture, name)

                # Check if the face has already been recognized today
                if not is_already_recognized_today(connection, name):
                    # Store recognized face in MySQL database
                    store_recognized_face_in_mysql(connection, name, roll_numbers.get(name, "Unknown"))

        cv2.imshow("My Face Recognition Project", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
