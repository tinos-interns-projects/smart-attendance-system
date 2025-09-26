# Smart Attendance System Using Face Recognition  

Smart Attendance System is an **AI-powered application** that automates employee/student attendance using **real-time face recognition**.  
The system eliminates the need for manual or fingerprint-based methods, making attendance **fast, accurate, and contactless**.  

---

## 🚀 Features  

- Real-time **face detection and recognition** for attendance logging  
- Tracks **IN, BREAK START, BREAK END, and OUT** events  
- Automatic **duplicate prevention** (5-minute cooldown)  
- **Auto-logout** for long breaks (> 2.5 hours)  
- **Admin panel** for managing employees and logs  
- Export attendance reports as **CSV**  
- Interactive **visualizations** with timelines and charts  

---

## 📁 Project Structure  

```bash
Smart-Attendance-System/
├── app.py             # Flask backend
├── db_config.py       # Database configuration
├── recognizer.py      # Face detection + attendance logging
├── trainer.yml        # Trained LBPH model file
├── label_dict.npz     # Label mapping for faces
├── templates/         # Frontend (HTML, CSS, JS)
└── static/            # Static files (CSS, JS, images)


🧠 Face Recognition Workflow
Face Detection → HaarCascade Classifier

Recognition → OpenCV LBPH Algorithm

Logging → Stores attendance events in MySQL

Duplicate Prevention → 5-minute cooldown

Break Monitoring → Auto OUT if break exceeds 2.5 hrs

📊 Attendance Flow
First log → IN

Next → BREAK START → BREAK END

Final → OUT

Auto OUT if break exceeds 2.5 hrs

🛠️ Technologies Used
Backend: Python, Flask

Database: MySQL

Face Recognition: OpenCV, NumPy

Visualization: Pandas, Plotly, Chart.js

Frontend: HTML, CSS, JavaScript


📊 Admin Panel Features
Add or remove employees

View, edit, and filter logs (by date or employee)

Visualize attendance trends with charts

Export reports in CSV format

✅ Conclusion
The Smart Attendance System provides a safe, fast, and accurate way to track attendance using AI-powered face recognition.
It is ideal for schools, colleges, offices, and organizations looking for a modern, contactless solution.

👨‍💻 Developed By
Sabith & Arsal

