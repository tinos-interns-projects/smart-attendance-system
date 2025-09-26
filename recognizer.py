import cv2
import numpy as np
from datetime import datetime
from db_config import get_db_connection

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")

# Dictionary to track the last recognition time for each employee
last_recognition_times = {}

def initialize_recognizer():
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("trainer.yml")
        return recognizer, None
    except cv2.error as e:
        return None, f"Error loading model: {str(e)}. Please train the model first."

def load_label_dict():
    try:
        label_dict = np.load("label_dict.npz")
        return {int(k): str(v) for k, v in label_dict.items()}, None
    except FileNotFoundError:
        return None, "Label dictionary not found. Please train the model first."
    except Exception as e:
        return None, f"Error loading label dictionary: {str(e)}"

def get_last_status_today(emp_id, today_date):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
        SELECT status
        FROM attendance_log
        WHERE emp_id = %s
        AND DATE(timestamp) = %s
        ORDER BY timestamp DESC
        LIMIT 1
    """
    cursor.execute(query, (emp_id, today_date))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result[0] if result else None

def determine_next_status(last_status):
    if last_status is None:
        return "IN"
    elif last_status == "IN":
        return "BREAK_START"
    elif last_status == "BREAK_START":
        return "BREAK_END"
    elif last_status == "BREAK_END":
        return "OUT"
    elif last_status == "OUT":
        return None  # Cannot log further events after OUT
    return None

def log_attendance(emp_id, status):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO attendance_log (emp_id, status, timestamp) VALUES (%s, %s, NOW())", (emp_id, status))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"Successfully logged attendance for emp_id={emp_id} with status={status}")
        return True
    except Exception as e:
        print(f"Error logging attendance: {e}")
        return False

def recognize_and_log():
    recognizer, error = initialize_recognizer()
    if error:
        return "error", error, None, 0

    label_dict, error = load_label_dict()
    if error:
        return "error", error, None, 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "error", "Failed to open webcam.", None, 0

    print("Starting real-time face recognition. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            cv2.destroyAllWindows()
            return "error", "Failed to capture frame from webcam.", None, 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = gray[y:y+h, x:x+w]

            label, confidence = recognizer.predict(face)
            if label in label_dict and confidence < 100:
                employee_name = label_dict[label]
                current_time = datetime.now()

                # Check cooldown period (5 minutes)
                if employee_name in last_recognition_times:
                    last_time = last_recognition_times[employee_name]
                    time_diff = (current_time - last_time).total_seconds() / 60
                    if time_diff < 5:
                        remaining_time = 5 - time_diff
                        cv2.putText(frame, f"Cooldown: {remaining_time:.1f} min", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        continue

                # Update last recognition time
                last_recognition_times[employee_name] = current_time

                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT emp_id FROM employees WHERE name = %s", (employee_name,))
                result = cursor.fetchone()
                if result:
                    emp_id = result[0]
                    today_date = datetime.now().strftime("%Y-%m-%d")
                    last_status = get_last_status_today(emp_id, today_date)
                    next_status = determine_next_status(last_status)
                    if next_status:
                        if log_attendance(emp_id, next_status):
                            message = f"Logged {next_status} for {employee_name} at {current_time.strftime('%Y-%m-%d %H:%M:%S')}"
                            print(message)
                            # Display name and confidence on the frame
                            cv2.putText(frame, f"{employee_name} ({confidence:.2f})", (x, y-10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        else:
                            print("Failed to log attendance.")
                    else:
                        print(f"Cannot log attendance for {employee_name}: Already logged OUT for today.")
                        cv2.putText(frame, "Already logged OUT", (x, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    print("Employee not found in database.")
                    cv2.putText(frame, "Unknown Employee", (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cursor.close()
                conn.close()
            else:
                # Display "Unknown" if confidence is too high or label not found
                cv2.putText(frame, "Unknown", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Display instructions on the frame
        cv2.putText(frame, "Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Face Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return "stopped", "Real-time recognition stopped by user.", None, 0

def manual_log(employee_name, status):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT emp_id FROM employees WHERE name = %s", (employee_name,))
    result = cursor.fetchone()
    if not result:
        cursor.close()
        conn.close()
        return False, f"Employee {employee_name} not found."

    emp_id = result[0]
    today_date = datetime.now().strftime("%Y-%m-%d")
    last_status = get_last_status_today(emp_id, today_date)
    expected_status = determine_next_status(last_status)
    
    if expected_status != status:
        cursor.close()
        conn.close()
        return False, f"Cannot log {status} for {employee_name}. Expected status: {expected_status or 'none'}."

    if log_attendance(emp_id, status):
        # Update last recognition time to prevent duplicate logs
        last_recognition_times[employee_name] = datetime.now()
        cursor.close()
        conn.close()
        return True, f"Manually logged {status} for {employee_name}."
    else:
        cursor.close()
        conn.close()
        return False, "Failed to log attendance."

if __name__ == "__main__":
    status, message, label, confidence = recognize_and_log()
    print(f"Status: {status}, Message: {message}, Confidence: {confidence}")