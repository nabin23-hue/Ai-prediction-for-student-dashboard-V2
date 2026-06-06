ITS320 Random Forest ML Upgrade - Organised Capstone Package

PROJECT TITLE
AI-Based Student Productivity and Performance Prediction System

WHAT THIS ZIP CONTAINS
1. app/                 Working Flask application with Random Forest ML upgrade
2. data/                Synthetic training dataset and dataset notes
3. docs/                Explanation, revision notes, presentation script, testing checklist

QUICK RUN STEPS
1. Open terminal inside the app folder.
2. Install dependencies:
   pip install -r requirements.txt
3. Run the app:
   python app.py
4. Open the local Flask link shown in the terminal, usually:
   http://127.0.0.1:5000

LOGIN DETAILS
Admin:   username admin    password 123
Teacher: username teacher  password 123
Student: username laxman   password 123
Student: username nabin    password 123
Student: username nishan   password 123

BEST DEMO ROUTES
/login or /               Login page
/teacher                  Teacher dashboard
/student/laxman           Student dashboard
/demo                     Combined presentation dashboard
/model                    Random Forest model evidence page
/admin                    Admin management dashboard
/privacy                  Ethics and privacy page
/tutorial                 User guidance page

MAIN DECISION
The system has been upgraded from a rule-based scoring prototype to a Random Forest ML prototype.
Random Forest is now used to predict:
- Student risk level
- Expected grade
- Productivity score
- Pass probability

IMPORTANT HONEST PRESENTATION LINE
"For the capstone prototype, Random Forest is trained using a synthetic education dataset based on realistic student indicators. In a production or research version, this training data should be replaced with real de-identified student data or public datasets such as UCI Student Performance Dataset or OULAD."

WHY THIS IS SAFE FOR PRESENTATION
- It keeps your previous design decisions.
- It does not break the existing Flask system.
- It adds real ML structure using scikit-learn Random Forest.
- It provides a model evidence page and combined demo dashboard.
- It clearly explains limitations and future work.
