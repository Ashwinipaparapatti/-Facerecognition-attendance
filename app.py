
# from flask import Flask, render_template, jsonify, request, redirect, url_for, send_file
# import threading
# import time
# import cv2
# import face_recognition
# import numpy as np
# import os
# import mysql.connector
# from mysql.connector import Error
# from datetime import datetime
# import shutil
# from werkzeug.utils import secure_filename
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import pandas as pd
# import matplotlib.pyplot as plt
# import io
# import base64

# app = Flask(__name__)

# app.config['UPLOAD_FOLDER'] = 'uploads'
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# # Global variable to control the face recognition process
# running = False
# # Path to your image folder
# image_folder_path = r'c:\Users\ashwi\OneDrive\Desktop\Face_recog\Face_recog\faces'
# known_face_encodings = []
# known_face_names = []

# # Define roll numbers for each person
# roll_numbers = {
#     "Ashwini": "22211a6793",
#     "Gayatri": "22211a6794",
#     "Dhoni": "22211a6795",
#     "Honey": "22211a6796",
#     "Nani": "22211a6797",
#     "Mrunal": "22211a6798",
#     "Saketh": "22211a6799"
# }

# def load_faces():
#     global known_face_encodings, known_face_names
#     for folder_name in os.listdir(image_folder_path):
#         folder_path = os.path.join(image_folder_path, folder_name)
#         if os.path.isdir(folder_path):
#             for filename in os.listdir(folder_path):
#                 if filename.endswith((".png", ".jpg", ".jpeg")):
#                     image_path = os.path.join(folder_path, filename)
#                     image = face_recognition.load_image_file(image_path)
#                     face_encodings = face_recognition.face_encodings(image)
#                     if face_encodings:
#                         face_encoding = face_encodings[0]
#                         known_face_encodings.append(face_encoding)
#                         known_face_names.append(folder_name)
#                         destination_folder = os.path.join(image_folder_path, folder_name)
#                         if not os.path.exists(destination_folder):
#                             os.makedirs(destination_folder)
#                         shutil.move(image_path, os.path.join(destination_folder, filename))
#                     else:
#                         print("")
#     print("Faces loaded and encoded")

# def connect_to_mysql():
#     try:
#         connection = mysql.connector.connect(
#             host='localhost',
#             database='attendence',
#             user='root',
#             password='mf90zk@ash123'
#         )
#         if connection.is_connected():
#             print('Connected to MySQL database')
#             return connection
#     except Error as e:
#         print(f"Error while connecting to MySQL: {e}")
#         return None

# def is_already_recognized_today(connection, name):
#     try:
#         cursor = connection.cursor()
#         today_date = datetime.now().date()
#         cursor.execute("SELECT COUNT(*) FROM information1 WHERE name=%s AND DATE(entry_time)=%s", (name, today_date))
#         count = cursor.fetchone()[0]
#         return count > 0
#     except Error as e:
#         print(f"Error checking recognized face in MySQL: {e}")
#         return False

# def store_recognized_face_in_mysql(connection, name, roll_number):
#     try:
#         cursor = connection.cursor()
#         entry_time = datetime.now()
#         cursor.execute("INSERT INTO information1 (name, roll_number, entry_time) VALUES (%s, %s, %s)", (name, roll_number, entry_time))
#         connection.commit()
#         print(f"{name} ({roll_number}) recognized and stored in MySQL database.")
#     except Error as e:
#         print(f"Error storing recognized face in MySQL: {e}")

# def capture_snapshots(video_capture, name):
#     face_folder_path = os.path.join(image_folder_path, name)
#     if not os.path.exists(face_folder_path):
#         os.makedirs(face_folder_path)
#     if len(os.listdir(face_folder_path)) >= 5:
#         return
#     snapshots = 0
#     while snapshots < 5:
#         ret, frame = video_capture.read()
#         if not ret:
#             break
#         face_locations = face_recognition.face_locations(frame)
#         if len(face_locations) == 0:
#             print("No face detected in the image")
#             continue
#         frame_with_message = frame.copy()
#         cv2.putText(frame_with_message, "Change your angle!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
#         cv2.imshow("My Face Recognition Project", frame_with_message)
#         cv2.waitKey(1500)
#         ret, frame = video_capture.read()
#         if not ret:
#             break
#         snapshot_filename = os.path.join(face_folder_path, f"{name}_{snapshots + 1}.jpg")
#         cv2.imwrite(snapshot_filename, frame)
#         snapshots += 1
#         time.sleep(1)
#     print(f"{name} has now {len(os.listdir(face_folder_path))} snapshots.")

# def generate_student_pie_chart(connection, name):
#     try:
#         cursor = connection.cursor()
#         cursor.execute("SELECT COUNT(*) FROM information1")
#         total_days = cursor.fetchone()[0]

#         cursor.execute("SELECT COUNT(*) FROM information1 WHERE name=%s", (name,))
#         present_days = cursor.fetchone()[0]

#         absent_days = total_days - present_days

#         fig, ax = plt.subplots(figsize=(3, 3))  # Set a moderate size for the pie chart
#         ax.pie([present_days, absent_days], labels=["Present", "Absent"], colors=['green', 'red'], autopct='%1.1f%%')
#         ax.set_title(f"Attendance for {name}")

#         img = io.BytesIO()
#         plt.savefig(img, format='png')
#         img.seek(0)
#         img_b64 = base64.b64encode(img.getvalue()).decode()
#         return img_b64
#     except Error as e:
#         print(f"Error generating pie chart: {e}")
#         return None

# def process_frame(frame, connection):
#     face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#     def detect_bounding_box(frame):
#         gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
#         return faces
#     faces = detect_bounding_box(frame)
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
#         face_encoding = face_recognition.face_encodings(frame, [(y, x+w, y+h, x)])[0]
#         matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
#         if True in matches:
#             match_index = matches.index(True)
#             name = known_face_names[match_index]
#             cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
#             if not is_already_recognized_today(connection, name):
#                 store_recognized_face_in_mysql(connection, name, roll_numbers.get(name, "Unknown"))

#             face_folder_path = os.path.join(image_folder_path, name)
#             num_snapshots = len([f for f in os.listdir(face_folder_path) if f.endswith((".png", ".jpg", ".jpeg"))])
#             if num_snapshots < 5:
#                 capture_snapshots(video_capture, name)  # Capture snapshots if less than 5
            
#             # Generate and display pie chart
#             pie_chart_img = generate_student_pie_chart(connection, name)
#             if pie_chart_img:
#                 pie_chart = base64.b64decode(pie_chart_img)
#                 pie_chart_img_np = np.frombuffer(pie_chart, np.uint8)
#                 pie_chart_img_cv2 = cv2.imdecode(pie_chart_img_np, cv2.IMREAD_COLOR)
#                 if pie_chart_img_cv2 is not None:
#                     pie_chart_resized = cv2.resize(pie_chart_img_cv2, (150, 150))  # Set the size of the pie chart
#                     y_offset = max(0, y - 10 - pie_chart_resized.shape[0])
#                     x_offset = min(frame.shape[1] - pie_chart_resized.shape[1], x + w + 10)
#                     if x_offset + pie_chart_resized.shape[1] > frame.shape[1]:  # If not enough space on the right
#                         x_offset = x - pie_chart_resized.shape[1] - 10  # Position on the left of the face
#                     if y_offset + pie_chart_resized.shape[0] > frame.shape[0]:  # If not enough space above
#                         y_offset = y + h + 10  # Position below the face
#                     frame[y_offset:y_offset+pie_chart_resized.shape[0], x_offset:x_offset+pie_chart_resized.shape[1]] = pie_chart_resized
#     return frame

# def recognize_faces():
#     connection = connect_to_mysql()
#     if connection:
#         video_capture = cv2.VideoCapture(0)
#         with ThreadPoolExecutor(max_workers=4) as executor:
#             while running:
#                 ret, frame = video_capture.read()
#                 if not ret:
#                     break
#                 future = executor.submit(process_frame, frame, connection)
#                 for future in as_completed([future]):
#                     processed_frame = future.result()
#                     cv2.imshow("My Face Recognition Project", processed_frame)
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break
#         video_capture.release()
#         cv2.destroyAllWindows()

# def generate_excel():
#     try:
#         connection = connect_to_mysql()
#         if connection:
#             # Query attendance data
#             cursor = connection.cursor()
#             cursor.execute("SELECT name, roll_number, entry_time FROM information1 WHERE DATE(entry_time) = CURDATE()")
#             records = cursor.fetchall()

#             # Create a DataFrame
#             df = pd.DataFrame(records, columns=['Name', 'Roll Number', 'Entry Time'])

#             # Write DataFrame to Excel
#             excel_file_path =  r'C:\Users\ashwi\OneDrive\Desktop\Face_recog\Face_recog\attendance.xlsx'
#             df.to_excel(excel_file_path, index=False)

#             return excel_file_path
#     except Error as e:
#         print(f"Error generating Excel file: {e}")
#         return None
#     finally:
#         if connection:
#             connection.close()
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/start', methods=['POST'])
# def start_recognition():
#     global running
#     if not running:
#         running = True
#         threading.Thread(target=recognize_faces).start()
#     return jsonify(status="started")

# @app.route('/stop', methods=['POST'])
# def stop_recognition():
#     global running
#     running = False
#     return jsonify(status="stopped")

# @app.route('/upload', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return redirect(request.url)
#         file = request.files['file']
#         if file.filename == '':
#             return redirect(request.url)
#         if file:
#             filename = secure_filename(file.filename)
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#             return redirect(url_for('index'))
#     return render_template('upload.html')

# @app.route('/register', methods=['GET', 'POST'])
# def register():
#     if request.method == 'POST':
#         name = request.form['name']
#         roll_number = request.form['roll_number']
#         roll_numbers[name] = roll_number  # Store roll number for the new user
#         global running
#         running = False  # Stop recognition during registration
#         video_capture = cv2.VideoCapture(0)
#         capture_snapshots(video_capture, name)
#         video_capture.release()
#         load_faces()  # Reload faces to include the new user
#         return redirect(url_for('index'))
#     return render_template('register.html')

# @app.route('/download_excel')
# def download_excel():
#     excel_file_path = generate_excel()
#     if excel_file_path:
#         return send_file(excel_file_path, as_attachment=True)
#     else:
#         return "Error generating Excel file"

# if __name__ == '__main__':
#     load_faces()
#     app.run(debug=True)




from flask import Flask, render_template, jsonify, request, redirect, url_for, send_file
import threading
import time
import cv2
import face_recognition
import numpy as np
import os
import mysql.connector
from mysql.connector import Error
from datetime import datetime
import shutil
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variable to control the face recognition process
running = False
# Path to your image folder
image_folder_path = r'c:\Users\ashwi\OneDrive\Desktop\Face_recog\Face_recog\faces'
known_face_encodings = []
known_face_names = []

# Define roll numbers for each person
roll_numbers = {
    "Ashwini": "22211a6793",
    "Gayatri": "22211a6794",
    "Dhoni": "22211a6795",
    "Honey": "22211a6796",
    "Nani": "22211a6797",
    "Mrunal": "22211a6798",
    "Saketh": "22211a6799",
    "Reshmi":"22211a0550",
    "Eshwari":"22211a05d6"
}

def load_faces():
    global known_face_encodings, known_face_names
    for folder_name in os.listdir(image_folder_path):
        folder_path = os.path.join(image_folder_path, folder_name)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith((".png", ".jpg", ".jpeg")):
                    image_path = os.path.join(folder_path, filename)
                    image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(image)
                    if face_encodings:
                        face_encoding = face_encodings[0]
                        known_face_encodings.append(face_encoding)
                        known_face_names.append(folder_name)
                        destination_folder = os.path.join(image_folder_path, folder_name)
                        if not os.path.exists(destination_folder):
                            os.makedirs(destination_folder)
                        shutil.move(image_path, os.path.join(destination_folder, filename))
                    else:
                        print("")
    print("Faces loaded and encoded")

def connect_to_mysql():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            database='attendence',
            user='root',
            password='mf90zk@ash123'
        )
        if connection.is_connected():
            print('Connected to MySQL database')
            return connection
    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
        return None

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

def store_recognized_face_in_mysql(connection, name, roll_number):
    try:
        cursor = connection.cursor()
        entry_time = datetime.now()
        cursor.execute("INSERT INTO information1 (name, roll_number, entry_time) VALUES (%s, %s, %s)", (name, roll_number, entry_time))
        connection.commit()
        print(f"{name} ({roll_number}) recognized and stored in MySQL database.")
    except Error as e:
        print(f"Error storing recognized face in MySQL: {e}")

def capture_snapshots(video_capture, name):
    face_folder_path = os.path.join(image_folder_path, name)
    if not os.path.exists(face_folder_path):
        os.makedirs(face_folder_path)
    if len(os.listdir(face_folder_path)) >= 5:
        return
    snapshots = 0
    while snapshots < 5:
        ret, frame = video_capture.read()
        if not ret:
            break
        face_locations = face_recognition.face_locations(frame)
        if len(face_locations) == 0:
            print("No face detected in the image")
            continue
        frame_with_message = frame.copy()
        cv2.putText(frame_with_message, "Change your angle!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("My Face Recognition Project", frame_with_message)
        cv2.waitKey(1500)
        ret, frame = video_capture.read()
        if not ret:
            break
        snapshot_filename = os.path.join(face_folder_path, f"{name}_{snapshots + 1}.jpg")
        cv2.imwrite(snapshot_filename, frame)
        snapshots += 1
        time.sleep(1)
    print(f"{name} has now {len(os.listdir(face_folder_path))} snapshots.")



def generate_attendance_summary(connection):
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT DATE(entry_time) as date, COUNT(*) as present_count FROM information1 GROUP BY DATE(entry_time)")
        data = cursor.fetchall()
        dates, present_counts = zip(*data)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(dates, present_counts, color='blue')
        ax.set_title('Daily Attendance Summary')
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Students Present')
        plt.xticks(rotation=45)

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        img_b64 = base64.b64encode(img.getvalue()).decode()
        return img_b64
    except Error as e:
        print(f"Error generating attendance summary: {e}")
        return None

def generate_individual_attendance_report(connection, name):
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT DATE(entry_time) as date, COUNT(*) as present FROM information1 WHERE name=%s GROUP BY DATE(entry_time)", (name,))
        data = cursor.fetchall()

        if not data:
            return None

        dates, present = zip(*data)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(dates, present, marker='o', linestyle='-', color='green')
        ax.set_title(f'Attendance Report for {name}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Attendance')
        plt.xticks(rotation=45)
        plt.yticks([0, 1], ["Absent", "Present"])

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        img_b64 = base64.b64encode(img.getvalue()).decode()
        return img_b64
    except Error as e:
        print(f"Error generating individual attendance report: {e}")
        return None
    
def generate_student_pie_chart(connection, name):
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM information1")
        total_days = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM information1 WHERE name=%s", (name,))
        present_days = cursor.fetchone()[0]

        absent_days = total_days - present_days

        fig, ax = plt.subplots(figsize=(3, 3))  # Set a moderate size for the pie chart
        ax.pie([present_days, absent_days], labels=["Present", "Absent"], colors=['green', 'red'], autopct='%1.1f%%')
        ax.set_title(f"Attendance for {name}")

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        img_b64 = base64.b64encode(img.getvalue()).decode()
        return img_b64
    except Error as e:
        print(f"Error generating pie chart: {e}")
        return None    
def process_frame(frame, connection):
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    def detect_bounding_box(frame):
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
        return faces
    faces = detect_bounding_box(frame)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
        face_encoding = face_recognition.face_encodings(frame, [(y, x+w, y+h, x)])[0]
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            if not is_already_recognized_today(connection, name):
                store_recognized_face_in_mysql(connection, name, roll_numbers.get(name, "Unknown"))

            face_folder_path = os.path.join(image_folder_path, name)
            num_snapshots = len([f for f in os.listdir(face_folder_path) if f.endswith((".png", ".jpg", ".jpeg"))])
            if num_snapshots < 5:
                capture_snapshots(video_capture, name)  # Capture snapshots if less than 5
            
            # Generate and display pie chart
            pie_chart_img = generate_student_pie_chart(connection, name)
            if pie_chart_img:
                pie_chart = base64.b64decode(pie_chart_img)
                pie_chart_img_np = np.frombuffer(pie_chart, np.uint8)
                pie_chart_img_cv2 = cv2.imdecode(pie_chart_img_np, cv2.IMREAD_COLOR)
                if pie_chart_img_cv2 is not None:
                    pie_chart_resized = cv2.resize(pie_chart_img_cv2, (150, 150))  # Set the size of the pie chart
                    y_offset = max(0, y - 10 - pie_chart_resized.shape[0])
                    x_offset = min(frame.shape[1] - pie_chart_resized.shape[1], x + w + 10)
                    if x_offset + pie_chart_resized.shape[1] > frame.shape[1]:  # If not enough space on the right
                        x_offset = x - pie_chart_resized.shape[1] - 10  # Position on the left of the face
                    if y_offset + pie_chart_resized.shape[0] > frame.shape[0]:  # If not enough space above
                        y_offset = y + h + 10  # Position below the face
                    frame[y_offset:y_offset+pie_chart_resized.shape[0], x_offset:x_offset+pie_chart_resized.shape[1]] = pie_chart_resized
    return frame


def recognize_faces():
    connection = connect_to_mysql()
    if connection:
        video_capture = cv2.VideoCapture(0)
        with ThreadPoolExecutor(max_workers=4) as executor:
            while running:
                ret, frame = video_capture.read()
                if not ret:
                    break
                future = executor.submit(process_frame, frame, connection)
                for future in as_completed([future]):
                    processed_frame = future.result()
                    cv2.imshow("My Face Recognition Project", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        video_capture.release()
        cv2.destroyAllWindows()

def generate_excel():
    try:
        connection = connect_to_mysql()
        if connection:
            # Query attendance data
            cursor = connection.cursor()
            cursor.execute("SELECT name, roll_number, entry_time FROM information1 WHERE DATE(entry_time) = CURDATE()")
            records = cursor.fetchall()

            # Create a DataFrame
            df = pd.DataFrame(records, columns=['Name', 'Roll Number', 'Entry Time'])

            # Write DataFrame to Excel
            excel_file_path =  r'C:\Users\ashwi\OneDrive\Desktop\Face_recog\Face_recog\attendance.xlsx'
            df.to_excel(excel_file_path, index=False)

            return excel_file_path
    except Error as e:
        print(f"Error generating Excel file: {e}")
        return None
    finally:
        if connection:
            connection.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_recognition():
    global running
    if not running:
        running = True
        threading.Thread(target=recognize_faces).start()
    return jsonify(status="started")

@app.route('/stop', methods=['POST'])
def stop_recognition():
    global running
    running = False
    return jsonify(status="stopped")

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('index'))
    return render_template('upload.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        roll_number = request.form['roll_number']
        roll_numbers[name] = roll_number  # Store roll number for the new user
        global running
        running = False  # Stop recognition during registration
        video_capture = cv2.VideoCapture(0)
        capture_snapshots(video_capture, name)
        video_capture.release()
        load_faces()  # Reload faces to include the new user
        return redirect(url_for('index'))
    return render_template('register.html')

@app.route('/download_excel')
def download_excel():
    excel_file_path = generate_excel()
    if excel_file_path:
        return send_file(excel_file_path, as_attachment=True)
    else:
        return "Error generating Excel file"

@app.route('/attendance_summary')
def attendance_summary():
    connection = connect_to_mysql()
    if connection:
        img_b64 = generate_attendance_summary(connection)
        if img_b64:
            return render_template('attendance_summary.html', img_b64=img_b64)
        else:
            return "Error generating attendance summary"
    else:
        return "Error connecting to database"

@app.route('/individual_attendance', methods=['GET', 'POST'])
def individual_attendance():
    if request.method == 'POST':
        name = request.form['name']
        connection = connect_to_mysql()
        if connection:
            img_b64 = generate_individual_attendance_report(connection, name)
            if img_b64:
                return render_template('individual_attendance.html', img_b64=img_b64, name=name)
            else:
                return render_template('individual_attendance.html', error=f"No attendance records found for {name}.")
        else:
            return "Error connecting to database"
    return render_template('individual_attendance.html')



if __name__ == '__main__':
    load_faces()
    app.run(debug=True)
