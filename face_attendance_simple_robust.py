
import cv2
import numpy as np
import csv
import os
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import messagebox, ttk
import threading
import logging


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

try:
    from deepface import DeepFace
except ImportError as e:
    print(f"Error importing DeepFace: {e}")
    print("Ensure tensorflow, numpy, pandas, and deepface are installed correctly.")
    exit(1)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


known_face_names = []
img_dir = "imgs/"


def load_known_faces():
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
        logging.info(f"Created directory: {img_dir}")
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        if img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            known_face_names.append(os.path.splitext(img_name)[0])
            logging.info(f"Loaded image: {img_name}")
        else:
            logging.warning(f"Skipping {img_name}: Unsupported format")


csv_file = "Attendance.csv"
def init_csv():
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Time"])
        logging.info(f"Created CSV file: {csv_file}")


def show_popup(name):
    try:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Attendance", f"{name}'s attendance marked!")
        root.destroy()
    except Exception as e:
        logging.error(f"Error showing popup: {e}")


def show_attendance_table():
    try:
        def load_table():
            table.delete(*table.get_children())
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  
                for row in reader:
                    table.insert("", "end", values=row)

        root = tk.Tk()
        root.title("Attendance Records")
        table = ttk.Treeview(root, columns=("Name", "Time"), show="headings")
        table.heading("Name", text="Name")
        table.heading("Time", text="Time")
        table.pack(padx=10, pady=10)
        load_table()
        root.mainloop()
    except Exception as e:
        logging.error(f"Error showing attendance table: {e}")


def main():
    load_known_faces()
    init_csv()

    
    attendance_log = {}  
    COOLDOWN = timedelta(minutes=5)
    frame_count = 0
    PROCESS_EVERY_N_FRAMES = 5  

    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Could not open webcam")
        return

    logging.info("Webcam initialized")
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Could not read frame")
            break

        frame_count += 1
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            
            try:
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

                
                for img_name in os.listdir(img_dir):
                    img_path = os.path.join(img_dir, img_name)
                    if not img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue

                    try:
                        
                        result = DeepFace.verify(
                            small_frame,
                            img_path,
                            model_name="Facenet", 
                            enforce_detection=False,
                            distance_metric="cosine"
                        )
                        if result["verified"]:
                            name = os.path.splitext(img_name)[0]
                            current_time = datetime.now()
                            if name not in attendance_log or current_time - attendance_log[name] > COOLDOWN:
                                attendance_log[name] = current_time
                                with open(csv_file, 'a', newline='') as f:
                                    writer = csv.writer(f)
                                    writer.writerow([name, current_time.strftime("%Y-%m-%d %H:%M:%S")])
                                threading.Thread(target=show_popup, args=(name,), daemon=True).start()
                                logging.info(f"Attendance marked for {name}")
                            break
                    except Exception as e:
                        logging.warning(f"Error verifying {img_name}: {e}")
            except Exception as e:
                logging.error(f"Error processing frame: {e}")

        
        cv2.imshow("Attendance System", frame)

       
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('g'):
            threading.Thread(target=show_attendance_table, daemon=True).start()

    
    cap.release()
    cv2.destroyAllWindows()
    logging.info("Program terminated")

if __name__ == "__main__":
    main()
