# import tkinter as tk
# from tkinter import messagebox
# import requests
# import threading
# import subprocess

# # Full path to app.py
# app_path = "C:\\Users\\ashwi\\OneDrive\\Desktop\\Face_recog\\Face_recog\\app.py"

# # Function to start the Flask server
# def start_flask_server():
#     subprocess.Popen(["python", app_path])

# # Function to start face recognition
# def start_recognition():
#     try:
#         response = requests.post('http://127.0.0.1:5000/start')
#         if response.json().get('status') == 'started':
#             messagebox.showinfo("Info", "Recognition started")
#         else:
#             messagebox.showerror("Error", "Failed to start recognition")
#     except requests.exceptions.RequestException as e:
#         messagebox.showerror("Error", f"Failed to connect to server: {e}")

# # Function to stop face recognition
# def stop_recognition():
#     try:
#         response = requests.post('http://127.0.0.1:5000/stop')
#         if response.json().get('status') == 'stopped':
#             messagebox.showinfo("Info", "Recognition stopped")
#         else:
#             messagebox.showerror("Error", "Failed to stop recognition")
#     except requests.exceptions.RequestException as e:
#         messagebox.showerror("Error", f"Failed to connect to server: {e}")

# # Function to start the server in a separate thread
# def start_server_thread():
#     server_thread = threading.Thread(target=start_flask_server)
#     server_thread.daemon = True
#     server_thread.start()

# # Create the main application window
# root = tk.Tk()
# root.title("Face Recognition Control GUI")

# # Create and place the start button
# start_button = tk.Button(root, text="Start Recognition", command=start_recognition)
# start_button.pack(pady=10)

# # Create and place the stop button
# stop_button = tk.Button(root, text="Stop Recognition", command=stop_recognition)
# stop_button.pack(pady=10)

# # Start the Flask server when the GUI starts
# start_server_thread()

# # Run the Tkinter event loop
# root.mainloop()

import tkinter as tk
from tkinter import simpledialog, messagebox
import requests
import threading
import subprocess

# Full path to app.py
app_path = "C:\\Users\\ashwi\\OneDrive\\Desktop\\Face_recog\\Face_recog\\app.py"

# Function to start the Flask server
def start_flask_server():
    subprocess.Popen(["python", app_path])

# Function to start face recognition
def start_recognition():
    try:
        response = requests.post('http://127.0.0.1:5000/start')
        if response.json().get('status') == 'started':
            messagebox.showinfo("Info", "Recognition started")
        else:
            messagebox.showerror("Error", "Failed to start recognition")
    except requests.exceptions.RequestException as e:
        messagebox.showerror("Error", f"Failed to connect to server: {e}")

# Function to stop face recognition
def stop_recognition():
    try:
        response = requests.post('http://127.0.0.1:5000/stop')
        if response.json().get('status') == 'stopped':
            messagebox.showinfo("Info", "Recognition stopped")
        else:
            messagebox.showerror("Error", "Failed to stop recognition")
    except requests.exceptions.RequestException as e:
        messagebox.showerror("Error", f"Failed to connect to server: {e}")

# Function to register a new face
def register_face():
    name = simpledialog.askstring("Input", "Enter your name:")
    if name:
        roll_number = simpledialog.askstring("Input", "Enter your roll number:")
        if roll_number:
            try:
                data = {'name': name, 'roll_number': roll_number}
                response = requests.post('http://127.0.0.1:5000/register', data=data)
                if response.status_code == 200:
                    messagebox.showinfo("Info", "Face registered successfully")
                else:
                    messagebox.showerror("Error", "Failed to register face")
            except requests.exceptions.RequestException as e:
                messagebox.showerror("Error", f"Failed to connect to server: {e}")

# Function to start the server in a separate thread
def start_server_thread():
    server_thread = threading.Thread(target=start_flask_server)
    server_thread.daemon = True
    server_thread.start()

# Create the main application window
root = tk.Tk()
root.title("Face Recognition Control GUI")

# Create and place the start button
start_button = tk.Button(root, text="Start Recognition", command=start_recognition)
start_button.pack(pady=10)

# Create and place the stop button
stop_button = tk.Button(root, text="Stop Recognition", command=stop_recognition)
stop_button.pack(pady=10)

# Create and place the register button
register_button = tk.Button(root, text="Register a Face", command=register_face)
register_button.pack(pady=10)

# Start the Flask server when the GUI starts
start_server_thread()

# Run the Tkinter event loop
root.mainloop()

