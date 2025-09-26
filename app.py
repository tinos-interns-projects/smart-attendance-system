import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import os
import base64
from io import BytesIO
from datetime import datetime, timedelta
from flask import Flask, render_template, request, session, redirect, url_for, jsonify, send_file
from db_config import get_db_connection
from recognizer import recognize_and_log, manual_log
from capture_face import capture_faces  # Import for potential future use

app = Flask(__name__)
app.secret_key = "admin123"  # Replace with a secure key

# Define the custom filter
@app.template_filter('datetimeformat')
def datetimeformat(value, format_string):
    if value == "now":
        return datetime.now().strftime(format_string)
    try:
        return datetime.strptime(value, '%Y-%m-%d %H:%M:%S').strftime(format_string)
    except (ValueError, TypeError):
        return value

def initialize_session():
    if "last_log" not in session:
        session["last_log"] = None
    if "angles_captured" not in session:
        session["angles_captured"] = {"front": 0, "left": 0, "right": 0}
    if "employee_name" not in session:
        session["employee_name"] = None
    if "current_angle" not in session:
        session["current_angle"] = None

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")

def train_model(employee_name=None):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    labels = []
    label_dict = {}
    current_id = 0

    if os.path.exists("label_dict.npz"):
        label_dict = np.load("label_dict.npz")
        label_dict = {int(k): str(v) for k, v in label_dict.items()}
        current_id = max(label_dict.keys()) + 1 if label_dict else 0

    if employee_name and employee_name not in label_dict.values():
        label_dict[current_id] = employee_name
        current_id += 1

    angles = ["front", "left", "right"]
    for emp_name in os.listdir(os.path.join("static", "face_data")):
        emp_id = None
        for k, v in label_dict.items():
            if v == emp_name:
                emp_id = k
                break
        if emp_id is None:
            emp_id = current_id
            label_dict[emp_id] = emp_name
            current_id += 1

        emp_base_dir = os.path.join("static", "face_data", emp_name)
        for angle in angles:
            angle_dir = os.path.join(emp_base_dir, angle)
            if not os.path.isdir(angle_dir):
                continue
            for image_name in os.listdir(angle_dir):
                image_path = os.path.join(angle_dir, image_name)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    faces.append(image)
                    labels.append(emp_id)

    if not faces:
        return False, "No face data available to train."

    recognizer.train(faces, np.array(labels))
    recognizer.save("trainer.yml")
    label_dict_str = {str(k): v for k, v in label_dict.items()}
    np.savez("label_dict.npz", **label_dict_str)
    return True, "Model trained successfully!"

def get_employees():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM employees")
    employees = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return employees

def check_break_timeout():
    """
    Check for employees with BREAK_START but no BREAK_END within 2 hours.
    If found, log an OUT and mark the break as exceeded.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    today_date = datetime.now().strftime("%Y-%m-%d")

    # Find employees with BREAK_START but no BREAK_END or OUT today
    query = """
        SELECT al.emp_id, al.timestamp
        FROM attendance_log al
        WHERE al.status = 'BREAK_START'
        AND DATE(al.timestamp) = %s
        AND NOT EXISTS (
            SELECT 1
            FROM attendance_log al2
            WHERE al2.emp_id = al.emp_id
            AND (al2.status = 'BREAK_END' OR al2.status = 'OUT')
            AND DATE(al2.timestamp) = %s
            AND al2.timestamp > al.timestamp
        )
    """
    cursor.execute(query, (today_date, today_date))
    break_starts = cursor.fetchall()

    current_time = datetime.now()
    for emp_id, break_start_time in break_starts:
        time_diff = (current_time - break_start_time).total_seconds() / 3600  # Time difference in hours
        if time_diff >= 2:
            # Log an OUT for this employee
            cursor.execute("""
                INSERT INTO attendance_log (emp_id, status, timestamp, is_break_exceeded)
                VALUES (%s, 'OUT', NOW(), TRUE)
            """, (emp_id,))
            print(f"Automatically logged OUT for emp_id={emp_id} due to break exceeding 2 hours.")
    conn.commit()
    cursor.close()
    conn.close()

def get_attendance_logs():
    check_break_timeout()  # Check for break timeouts before fetching logs
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT al.log_id, e.name, al.status, al.timestamp, al.is_break_exceeded
        FROM attendance_log al
        JOIN employees e ON al.emp_id = e.emp_id
        ORDER BY al.timestamp DESC
    """)
    logs = cursor.fetchall()
    cursor.close()
    conn.close()
    return logs

def get_filtered_attendance_logs(year=None, month=None, day=None, employee_name=None):
    check_break_timeout()
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
        SELECT al.log_id, e.name, al.status, al.timestamp, al.is_break_exceeded
        FROM attendance_log al
        JOIN employees e ON al.emp_id = e.emp_id
        WHERE 1=1
    """
    params = []
    if year:
        query += " AND YEAR(al.timestamp) = %s"
        params.append(year)
    if month:
        query += " AND MONTH(al.timestamp) = %s"
        params.append(month)
    if day:
        query += " AND DAY(al.timestamp) = %s"
        params.append(day)
    if employee_name:
        query += " AND e.name = %s"
        params.append(employee_name)
    query += " ORDER BY al.timestamp DESC"

    cursor.execute(query, params)
    logs = cursor.fetchall()
    cursor.close()
    conn.close()
    return logs

def get_attendance_summary():
    logs = get_attendance_logs()
    if not logs:
        return pd.DataFrame()
    df = pd.DataFrame(logs, columns=["Log ID", "Employee Name", "Status", "Timestamp", "Break Exceeded"])
    summary = df.groupby(["Employee Name", "Status"]).size().reset_index(name="Count")
    return summary

def get_attendance_timeline(employee_name):
    logs = get_attendance_logs()
    if not logs:
        return pd.DataFrame()
    df = pd.DataFrame(logs, columns=["Log ID", "Employee Name", "Status", "Timestamp", "Break Exceeded"])
    df = df[df["Employee Name"] == employee_name]
    return df

def get_todays_attendance_summary():
    check_break_timeout()
    today_date = datetime.now().strftime("%Y-%m-%d")
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
        SELECT e.name, al.status, al.timestamp
        FROM attendance_log al
        JOIN employees e ON al.emp_id = e.emp_id
        WHERE DATE(al.timestamp) = %s
        ORDER BY e.name, al.timestamp
    """
    cursor.execute(query, (today_date,))
    logs = cursor.fetchall()
    cursor.close()
    conn.close()

    summary = {}
    for name, status, timestamp in logs:
        if name not in summary:
            summary[name] = []
        summary[name].append({"status": status, "time": timestamp.strftime("%H:%M") if timestamp else "N/A"})

    return summary

def get_attendance_summary_by_date(date_str, employee_name=None):
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")  # Default to current date if empty

    check_break_timeout()
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
        SELECT e.name, al.status, al.timestamp
        FROM attendance_log al
        JOIN employees e ON al.emp_id = e.emp_id
        WHERE DATE(al.timestamp) = %s
    """
    params = [date_str]
    if employee_name:
        query += " AND e.name = %s"
        params.append(employee_name)
    query += " ORDER BY e.name, al.timestamp"
    
    cursor.execute(query, params)
    logs = cursor.fetchall()
    cursor.close()
    conn.close()

    summary = {}
    for name, status, timestamp in logs:
        if name:
            if name not in summary:
                summary[name] = {"IN": [], "BREAK_START": [], "BREAK_END": [], "OUT": []}
            summary[name][status].append(timestamp.strftime("%Y-%m-%d %H:%M:%S") if timestamp else "N/A")

    return summary

def get_filtered_attendance_metrics(year=None, month=None, day=None, employee_name=None):
    check_break_timeout()
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
        SELECT e.name, al.status, al.timestamp, al.is_break_exceeded
        FROM attendance_log al
        JOIN employees e ON al.emp_id = e.emp_id
        WHERE 1=1
    """
    params = []
    if year:
        query += " AND YEAR(al.timestamp) = %s"
        params.append(year)
    if month:
        query += " AND MONTH(al.timestamp) = %s"
        params.append(month)
    if day:
        query += " AND DAY(al.timestamp) = %s"
        params.append(day)
    if employee_name:
        query += " AND e.name = %s"
        params.append(employee_name)
    query += " ORDER BY e.name, al.timestamp"

    cursor.execute(query, params)
    logs = cursor.fetchall()
    cursor.close()
    conn.close()

    metrics = {}
    for name, status, timestamp, is_break_exceeded in logs:
        if name:
            if name not in metrics:
                metrics[name] = {"IN": [], "OUT": [], "BREAK_START": [], "BREAK_END": [], "is_break_exceeded": False}
            metrics[name][status].append(timestamp)
            if is_break_exceeded:
                metrics[name]["is_break_exceeded"] = True

    df_data = []
    for name, times in metrics.items():
        in_times = times["IN"]
        out_times = times["OUT"]
        break_starts = times["BREAK_START"]
        break_ends = times["BREAK_END"]
        is_break_exceeded = times["is_break_exceeded"]

        total_hours = 0
        break_duration = 0
        if in_times and out_times:
            total_hours = (out_times[-1] - in_times[0]).total_seconds() / 3600
        for i in range(min(len(break_starts), len(break_ends))):
            break_duration += (break_ends[i] - break_starts[i]).total_seconds() / 3600
        if len(break_starts) > len(break_ends) and is_break_exceeded:
            break_duration += 2  # Assume 2 hours for exceeded breaks

        worked_hours = total_hours - break_duration if total_hours > 0 else 0
        df_data.append({
            "Employee Name": name, 
            "Worked Hours": worked_hours, 
            "Break Hours": break_duration,
            "Break Exceeded": "Yes" if is_break_exceeded else "No"
        })

    return pd.DataFrame(df_data)

def get_employee_logs_for_today(employee_name):
    today_date = datetime.now().strftime("%Y-%m-%d")
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
        SELECT al.log_id, al.status, al.timestamp
        FROM attendance_log al
        JOIN employees e ON al.emp_id = e.emp_id
        WHERE e.name = %s AND DATE(al.timestamp) = %s
        ORDER BY al.timestamp
    """
    cursor.execute(query, (employee_name, today_date))
    logs = cursor.fetchall()
    cursor.close()
    conn.close()
    return logs

@app.route('/')
def home():
    initialize_session()
    return render_template('home.html')

@app.route('/add_employee', methods=['GET', 'POST'])
def add_employee():
    initialize_session()
    employees = get_employees()
    message = None
    error = None
    captured_images = []

    if request.method == 'POST':
        if 'new_employee' in request.form:
            new_employee = request.form['new_employee'].strip()
            if new_employee in employees:
                error = f"Employee '{new_employee}' already exists. Please choose a different name."
            elif not new_employee:
                error = "Employee name cannot be empty."
            else:
                session["employee_name"] = new_employee
                session["angles_captured"] = {"front": 0, "left": 0, "right": 0}
                session["current_angle"] = None
                session.modified = True

        elif 'angle' in request.form:
            session["current_angle"] = request.form['angle']
            session.modified = True
            employee_name = session.get("employee_name")
            angle = session.get("current_angle")
            if employee_name and angle:
                angle_dir = os.path.join("static", "face_data", employee_name, angle)
                if os.path.exists(angle_dir):
                    session["angles_captured"][angle] = len([f for f in os.listdir(angle_dir) if f.endswith('.jpg')])
                else:
                    session["angles_captured"][angle] = 0
                session.modified = True

        elif 'train_model' in request.form:
            new_employee = session.get("employee_name")
            all_captured = all(session["angles_captured"][angle] >= 10 for angle in ["front", "left", "right"])
            if all_captured and new_employee:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("INSERT INTO employees (name, folder_name) VALUES (%s, %s)", (new_employee, new_employee))
                conn.commit()
                cursor.close()
                conn.close()

                success, msg = train_model(new_employee)
                if success:
                    status, recog_message, label, confidence = recognize_and_log()
                    if status == "recognized":
                        message = f"{msg} {recog_message}"
                    else:
                        message = msg
                        error = recog_message
                    session["angles_captured"] = {"front": 0, "left": 0, "right": 0}
                    session["employee_name"] = None
                    session["current_angle"] = None
                else:
                    error = msg
            else:
                error = "Please capture 10 images for each angle before training the model."

    employee_name = session.get("employee_name")
    current_angle = session.get("current_angle")
    if employee_name and current_angle:
        angle_dir = os.path.join("static", "face_data", employee_name, current_angle)
        if os.path.exists(angle_dir):
            captured_images = [f for f in os.listdir(angle_dir) if f.endswith('.jpg')]
            session["angles_captured"][current_angle] = len(captured_images)
            session.modified = True

    return render_template('add_employee.html', 
                          employees=employees, 
                          employee_name=employee_name, 
                          angles_captured=session.get("angles_captured"), 
                          current_angle=current_angle,
                          captured_images=captured_images,
                          message=message, 
                          error=error)

@app.route('/capture_image', methods=['POST'])
def capture_image():
    employee_name = session.get("employee_name")
    angle = session.get("current_angle")
    
    if not employee_name or not angle:
        return jsonify({"status": "error", "message": "Employee name or angle not set."})

    data = request.json
    image_data = data.get('image')
    if not image_data:
        return jsonify({"status": "error", "message": "No image data provided."})

    try:
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error decoding image: {str(e)}"})

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=1, minSize=(20, 20))

    if len(faces) == 0:
        return jsonify({"status": "error", "message": "No face detected in the frame."})

    face_dir = os.path.join("static", "face_data", employee_name, angle)
    os.makedirs(face_dir, exist_ok=True)
    image_count = len([f for f in os.listdir(face_dir) if f.endswith('.jpg')])
    if image_count >= 10:
        return jsonify({"status": "error", "message": f"Maximum of 10 images already captured for {angle} angle."})

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        image_path = os.path.join(face_dir, f"{image_count}.jpg")
        print(f"Saving image to {image_path}")  # Debug logging
        cv2.imwrite(image_path, face)
        session["angles_captured"][angle] = image_count + 1
        session.modified = True
        return jsonify({"status": "success", "message": f"Captured image {image_count + 1} for {angle} angle.", "image_name": f"{image_count}.jpg"})

    return jsonify({"status": "error", "message": "Failed to capture face."})

@app.route('/delete_image', methods=['POST'])
def delete_image():
    employee_name = session.get("employee_name")
    angle = session.get("current_angle")
    image_name = request.form.get('image_name')

    if not employee_name or not angle or not image_name:
        return jsonify({"status": "error", "message": "Missing parameters."})

    image_path = os.path.join("static", "face_data", employee_name, angle, image_name)
    if os.path.exists(image_path):
        os.remove(image_path)
        angle_dir = os.path.join("static", "face_data", employee_name, angle)
        image_count = len([f for f in os.listdir(angle_dir) if f.endswith('.jpg')])
        session["angles_captured"][angle] = image_count
        session.modified = True
        return jsonify({"status": "success", "message": f"Deleted image {image_name}.", "image_count": image_count})
    else:
        return jsonify({"status": "error", "message": "Image not found."})

@app.route('/log_attendance', methods=['GET', 'POST'])
def log_attendance():
    initialize_session()
    employees = get_employees()
    todays_summary = get_todays_attendance_summary()
    message = None
    error = None
    selected_employee = None
    employee_logs = []

    if request.method == 'POST':
        if 'recognize' in request.form:
            status, recog_message, label, confidence = recognize_and_log()
            if status == "recognized" or status == "stopped":
                message = recog_message
                todays_summary = get_todays_attendance_summary()
            else:
                error = recog_message

        elif 'select_employee' in request.form:
            selected_employee = request.form.get('employee_name')
            if selected_employee:
                employee_logs = get_employee_logs_for_today(selected_employee)
            else:
                error = "Please select an employee."

        elif 'update_logs' in request.form:
            selected_employee = request.form.get('employee_name')
            updated_logs = []
            for key, value in request.form.items():
                if key.startswith('log_id_'):
                    log_id = int(value)
                    status = request.form.get(f'status_{log_id}')
                    timestamp_str = request.form.get(f'timestamp_{log_id}')
                    try:
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M')
                        updated_logs.append((log_id, status, timestamp))
                    except ValueError:
                        error = f"Invalid timestamp format for Log ID {log_id}. Use YYYY-MM-DDThh:mm (e.g., 2025-06-04T14:30)."
                        break

            if not error and updated_logs:
                try:
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    for log_id, status, timestamp in updated_logs:
                        cursor.execute("""
                            UPDATE attendance_log 
                            SET status = %s, timestamp = %s 
                            WHERE log_id = %s
                        """, (status, timestamp, log_id))
                    conn.commit()
                    message = "Attendance logs updated successfully!"
                    employee_logs = get_employee_logs_for_today(selected_employee)
                    todays_summary = get_todays_attendance_summary()
                except Exception as e:
                    conn.rollback()
                    error = f"Error updating logs: {str(e)}"
                finally:
                    cursor.close()
                    conn.close()

        elif 'add_break' in request.form:
            selected_employee = request.form.get('employee_name')
            break_start = request.form.get('new_break_start')
            break_end = request.form.get('new_break_end')

            if not selected_employee:
                error = "Please select an employee."
            else:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT emp_id FROM employees WHERE name = %s", (selected_employee,))
                result = cursor.fetchone()
                if not result:
                    error = f"Employee {selected_employee} not found."
                else:
                    emp_id = result[0]
                    today_date = datetime.now().strftime("%Y-%m-%d")
                    updates = []
                    for status, time in [('BREAK_START', break_start), ('BREAK_END', break_end)]:
                        if time:
                            try:
                                new_timestamp = datetime.strptime(f"{today_date} {time}", '%Y-%m-%d %H:%M')
                                cursor.execute("""
                                    INSERT INTO attendance_log (emp_id, status, timestamp)
                                    VALUES (%s, %s, %s)
                                """, (emp_id, status, new_timestamp))
                                updates.append(f"Added {status} at {time}")
                            except ValueError:
                                error = f"Invalid time format for {status}. Use HH:MM (e.g., 14:30)."
                                break

                    if updates and not error:
                        try:
                            conn.commit()
                            message = f"Added break for {selected_employee}: {', '.join(updates)}"
                            employee_logs = get_employee_logs_for_today(selected_employee)
                            todays_summary = get_todays_attendance_summary()
                        except Exception as e:
                            conn.rollback()
                            error = f"Error adding break: {str(e)}"
                    elif not updates and not error:
                        error = "Please provide at least one time to add a break."

                cursor.close()
                conn.close()

    return render_template('log_attendance.html',
                          employees=employees, 
                          todays_summary=todays_summary,
                          selected_employee=selected_employee,
                          employee_logs=employee_logs,
                          message=message, 
                          error=error)

@app.route('/admin_panel', methods=['GET', 'POST'])
def admin_panel():
    initialize_session()
    employees = get_employees()
    logs = get_attendance_logs()
    logs_df = pd.DataFrame(logs, columns=["Log ID", "Employee Name", "Status", "Timestamp", "Break Exceeded"]) if logs else pd.DataFrame()
    summary_df = get_attendance_summary()

    message = None
    error = None
    authorized = False
    selected_year = None
    selected_month = None
    selected_day = None
    selected_employee = None
    selected_date = datetime.now().strftime("%Y-%m-%d")

    if request.method == 'POST':
        if 'password' in request.form:
            if request.form['password'] == "admin123":
                session['admin_panel_authorized'] = True
                authorized = True
            else:
                error = "Incorrect password. Please try again."
        else:
            authorized = session.get('admin_panel_authorized', False)

        if authorized:
            if 'filter_logs' in request.form:
                selected_year = request.form.get('year')
                selected_month = request.form.get('month')
                selected_date = request.form.get('selected_date')
                selected_employee = request.form.get('selected_employee')

                if selected_date:
                    try:
                        date_obj = datetime.strptime(selected_date, "%Y-%m-%d")
                        selected_year = date_obj.year
                        selected_month = date_obj.month
                        selected_day = date_obj.day
                    except ValueError:
                        error = "Invalid date format. Please use YYYY-MM-DD."
                else:
                    today = datetime.now()
                    selected_year = today.year
                    selected_month = today.month
                    selected_day = today.day
                    selected_date = today.strftime("%Y-%m-%d")

                if selected_year and selected_year != "":
                    selected_year = int(selected_year)
                else:
                    selected_year = None
                if selected_month and selected_month != "":
                    selected_month = int(selected_month)
                else:
                    selected_month = None
                if selected_day and selected_day != "":
                    selected_day = int(selected_day)
                else:
                    selected_day = None
                if selected_employee == "all":
                    selected_employee = None

                logs = get_filtered_attendance_logs(selected_year, selected_month, selected_day, selected_employee)
                logs_df = pd.DataFrame(logs, columns=["Log ID", "Employee Name", "Status", "Timestamp", "Break Exceeded"]) if logs else pd.DataFrame()
                summary_df = get_attendance_summary()

            elif 'update_logs' in request.form:
                updated_logs = []
                for key, value in request.form.items():
                    if key.startswith('log_id_'):
                        log_id = int(value)
                        status = request.form.get(f'status_{log_id}')
                        timestamp_str = request.form.get(f'timestamp_{log_id}')
                        try:
                            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M')
                            updated_logs.append((log_id, status, timestamp))
                        except ValueError:
                            error = f"Invalid timestamp format for Log ID {log_id}. Use YYYY-MM-DDThh:mm (e.g., 2025-06-02T14:30)."
                            break

                if not error and updated_logs:
                    try:
                        conn = get_db_connection()
                        cursor = conn.cursor()
                        for log_id, status, timestamp in updated_logs:
                            cursor.execute("""
                                UPDATE attendance_log 
                                SET status = %s, timestamp = %s 
                                WHERE log_id = %s
                            """, (status, timestamp, log_id))
                        conn.commit()
                        message = "Attendance logs updated successfully!"
                        logs = get_attendance_logs()
                        logs_df = pd.DataFrame(logs, columns=["Log ID", "Employee Name", "Status", "Timestamp", "Break Exceeded"]) if logs else pd.DataFrame()
                        summary_df = get_attendance_summary()
                    except Exception as e:
                        conn.rollback()
                        error = f"Error updating logs: {str(e)}"
                    finally:
                        cursor.close()
                        conn.close()

    date_summary = get_attendance_summary_by_date(selected_date, selected_employee)

    metrics_df = get_filtered_attendance_metrics(selected_year, selected_month, selected_day, selected_employee)
    metrics_fig = None
    if not metrics_df.empty:
        fig = px.bar(metrics_df, x="Employee Name", y=["Worked Hours", "Break Hours"],
                     title="Attendance Metrics",
                     labels={"value": "Hours", "variable": "Metric"},
                     barmode="group")
        metrics_fig = fig.to_html(full_html=False, include_plotlyjs='cdn')

    summary_fig = None
    if not summary_df.empty:
        fig = px.bar(summary_df, x="Employee Name", y="Count", color="Status",
                     title="Attendance Events per Employee",
                     labels={"Count": "Number of Events"},
                     barmode="group")
        summary_fig = fig.to_html(full_html=False, include_plotlyjs='cdn')

    timeline_fig = None
    default_employee = employees[0] if employees else None
    timeline_employee = request.form.get('timeline_employee', default_employee)
    if timeline_employee:
        timeline_df = get_attendance_timeline(timeline_employee)
        if not timeline_df.empty:
            fig = px.line(timeline_df, x="Timestamp", y="Status",
                         title=f"Attendance Timeline for {timeline_employee}",
                         markers=True)
            timeline_fig = fig.to_html(full_html=False, include_plotlyjs='cdn')

    events = []
    if logs:
        for log in logs:
            employee_name, status, timestamp = log[1], log[2], log[3]
            if timestamp:
                events.append({
                    "title": f"{employee_name}: {status}",
                    "start": timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
                    "end": timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
                    "color": {"IN": "green", "OUT": "red", "BREAK_START": "orange", "BREAK_END": "blue"}.get(status, "gray")
                })

    years = list(range(2020, 2026))
    months = list(range(1, 13))

    return render_template('admin_panel.html',
                          authorized=authorized,
                          employees=employees,
                          logs=logs_df.to_dict('records'),
                          date_summary=date_summary,
                          summary_fig=summary_fig,
                          timeline_fig=timeline_fig,
                          metrics_fig=metrics_fig,
                          selected_employee=selected_employee,
                          timeline_employee=timeline_employee,
                          events=events,
                          years=years,
                          months=months,
                          selected_year=selected_year,
                          selected_month=selected_month,
                          selected_date=selected_date,
                          message=message,
                          error=error)

@app.route('/download_logs')
def download_logs():
    logs = get_attendance_logs()
    df = pd.DataFrame(logs, columns=["Log ID", "Employee Name", "Status", "Timestamp", "Break Exceeded"]) if logs else pd.DataFrame()
    csv = df.to_csv(index=False)
    return send_file(
        BytesIO(csv.encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='attendance_logs.csv'
    )

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)