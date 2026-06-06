from datetime import datetime
from io import StringIO
import csv
import random

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error

from flask import Flask, Response, redirect, render_template, request, url_for
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///student_ai.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(50), nullable=False, default="123")
    role = db.Column(db.String(20), nullable=False)  # admin, teacher, student
    full_name = db.Column(db.String(100), nullable=False)


class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    student_id = db.Column(db.String(30), nullable=False)
    course = db.Column(db.String(100), nullable=False)
    teacher_name = db.Column(db.String(100), nullable=False, default="Teacher")

    attendance = db.Column(db.Integer, default=70)
    study_hours = db.Column(db.Integer, default=2)
    assignments = db.Column(db.Integer, default=3)  # out of 5
    quiz = db.Column(db.Integer, default=60)
    lms_logins = db.Column(db.Integer, default=5)
    engagement = db.Column(db.Integer, default=60)
    missed_tasks = db.Column(db.Integer, default=1)

    productivity = db.Column(db.Integer, default=0)
    risk = db.Column(db.String(20), default="Medium")
    expected_grade = db.Column(db.String(20), default="B")
    pass_probability = db.Column(db.Integer, default=70)
    notification = db.Column(db.Text, default="")
    recommendation = db.Column(db.Text, default="")
    reason = db.Column(db.Text, default="")
    teacher_action = db.Column(db.Text, default="")

    streak = db.Column(db.Integer, default=3)
    daily_goal_done = db.Column(db.Integer, default=2)
    daily_goal_total = db.Column(db.Integer, default=4)
    weekly_goal_done = db.Column(db.Integer, default=8)
    weekly_goal_total = db.Column(db.Integer, default=12)

    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


def clamp(value, minimum=0, maximum=100):
    return max(minimum, min(maximum, int(round(value))))


FEATURE_NAMES = ["attendance", "study_hours", "assignments", "quiz", "lms_logins", "engagement", "missed_tasks"]


def baseline_score(attendance, study_hours, assignments, quiz, lms_logins, engagement, missed_tasks):
    """Creates labelled training data for the Random Forest prototype.
    In presentation, explain this as a synthetic training dataset based on education indicators.
    """
    assignment_percent = (assignments / 5) * 100
    study_score = min(study_hours * 20, 100)
    lms_score = min(lms_logins * 10, 100)
    productivity = clamp(
        (attendance * 0.25)
        + (study_score * 0.15)
        + (assignment_percent * 0.20)
        + (quiz * 0.20)
        + (lms_score * 0.10)
        + (engagement * 0.10)
        - (missed_tasks * 4)
    )
    pass_probability = clamp(productivity + ((quiz - 50) * 0.20) - (missed_tasks * 2), 20, 98)
    if productivity < 60:
        risk = "High"
    elif productivity < 80:
        risk = "Medium"
    else:
        risk = "Low"
    if productivity >= 90:
        expected_grade = "A"
    elif productivity >= 80:
        expected_grade = "B+"
    elif productivity >= 70:
        expected_grade = "B"
    elif productivity >= 60:
        expected_grade = "C"
    else:
        expected_grade = "At Risk"
    return productivity, pass_probability, risk, expected_grade


def build_random_forest_models():
    rng = random.Random(42)
    rows, productivity_y, pass_y, risk_y, grade_y = [], [], [], [], []
    for _ in range(700):
        attendance = rng.randint(40, 100)
        study_hours = rng.randint(0, 6)
        assignments = rng.randint(0, 5)
        quiz = rng.randint(25, 100)
        lms_logins = rng.randint(0, 12)
        engagement = rng.randint(20, 100)
        missed_tasks = rng.randint(0, 5)
        row = [attendance, study_hours, assignments, quiz, lms_logins, engagement, missed_tasks]
        productivity, pass_probability, risk, expected_grade = baseline_score(*row)
        rows.append(row)
        productivity_y.append(productivity)
        pass_y.append(pass_probability)
        risk_y.append(risk)
        grade_y.append(expected_grade)

    X_train, X_test, risk_train, risk_test = train_test_split(rows, risk_y, test_size=0.25, random_state=42, stratify=risk_y)
    _, _, grade_train, grade_test = train_test_split(rows, grade_y, test_size=0.25, random_state=42, stratify=grade_y)
    _, _, prod_train, prod_test = train_test_split(rows, productivity_y, test_size=0.25, random_state=42)
    _, _, pass_train, pass_test = train_test_split(rows, pass_y, test_size=0.25, random_state=42)

    risk_model = RandomForestClassifier(n_estimators=160, max_depth=7, random_state=42, class_weight="balanced")
    grade_model = RandomForestClassifier(n_estimators=160, max_depth=7, random_state=42, class_weight="balanced")
    productivity_model = RandomForestRegressor(n_estimators=160, max_depth=8, random_state=42)
    pass_model = RandomForestRegressor(n_estimators=160, max_depth=8, random_state=42)

    risk_model.fit(X_train, risk_train)
    grade_model.fit(X_train, grade_train)
    productivity_model.fit(X_train, prod_train)
    pass_model.fit(X_train, pass_train)

    evidence = {
        "risk_accuracy": round(accuracy_score(risk_test, risk_model.predict(X_test)) * 100, 1),
        "grade_accuracy": round(accuracy_score(grade_test, grade_model.predict(X_test)) * 100, 1),
        "productivity_mae": round(mean_absolute_error(prod_test, productivity_model.predict(X_test)), 2),
        "pass_mae": round(mean_absolute_error(pass_test, pass_model.predict(X_test)), 2),
        "training_rows": len(rows),
        "model_name": "Random Forest ML Prototype",
    }
    return risk_model, grade_model, productivity_model, pass_model, evidence


RISK_MODEL, GRADE_MODEL, PRODUCTIVITY_MODEL, PASS_MODEL, MODEL_EVIDENCE = build_random_forest_models()


def explain_prediction(inputs, risk, feature_importance):
    attendance, study_hours, assignments, quiz, lms_logins, engagement, missed_tasks = inputs
    reasons = []
    if attendance < 75:
        reasons.append("Attendance is one of the strongest risk indicators and is currently low.")
    elif attendance < 85:
        reasons.append("Attendance is acceptable but still below the ideal level.")
    if study_hours < 2:
        reasons.append("Daily study time is low, which reduces predicted productivity.")
    elif study_hours < 4:
        reasons.append("Study time is moderate and can be improved.")
    if assignments < 3:
        reasons.append("Assignment completion is low and increases predicted risk.")
    elif assignments < 5:
        reasons.append("Some assignments still need to be completed.")
    if quiz < 50:
        reasons.append("Quiz performance suggests the student needs extra academic support.")
    elif quiz < 75:
        reasons.append("Quiz performance is average, so revision is recommended.")
    if lms_logins < 4:
        reasons.append("LMS activity is low, showing reduced online learning engagement.")
    if engagement < 60:
        reasons.append("Engagement is below the expected level.")
    if missed_tasks > 0:
        reasons.append(f"The model also considers {missed_tasks} missed task(s) as a negative risk factor.")

    top_features = sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)[:3]
    important = ", ".join([name.replace("_", " ") for name, _ in top_features])
    model_text = f"Random Forest explanation: the model mainly considered {important}."
    if reasons:
        return model_text + " " + " ".join(reasons)
    return model_text + " The student is performing well across the main academic and productivity indicators."


def calculate_prediction(attendance, study_hours, assignments, quiz, lms_logins, engagement, missed_tasks):
    inputs = [attendance, study_hours, assignments, quiz, lms_logins, engagement, missed_tasks]

    risk = RISK_MODEL.predict([inputs])[0]
    expected_grade = GRADE_MODEL.predict([inputs])[0]
    productivity = clamp(PRODUCTIVITY_MODEL.predict([inputs])[0])
    pass_probability = clamp(PASS_MODEL.predict([inputs])[0], 20, 98)

    feature_importance = dict(zip(FEATURE_NAMES, RISK_MODEL.feature_importances_))

    if risk == "High":
        notification = "Important Alert: Random Forest model predicts high academic risk. Please improve attendance, complete pending tasks, and meet your lecturer for support."
        recommendation = "Complete one missing task today, attend support class, increase daily study time, and ask your teacher for feedback."
        teacher_action = "Contact the student this week, review missed tasks, and create a short improvement plan."
    elif risk == "Medium":
        notification = "Reminder: Random Forest model predicts medium risk. Improving consistency can reduce risk."
        recommendation = "Study at least 30 minutes daily, submit assignments on time, and keep attendance above 85%."
        teacher_action = "Send a reminder, monitor progress next week, and encourage revision practice."
    else:
        notification = "Good Progress: Random Forest model predicts low academic risk and strong study habits."
        recommendation = "Continue your current routine and attempt advanced practice tasks to maintain progress."
        teacher_action = "Give positive feedback and provide extension activities."

    reason = explain_prediction(inputs, risk, feature_importance)

    return {
        "productivity": productivity,
        "pass_probability": pass_probability,
        "risk": risk,
        "expected_grade": expected_grade,
        "notification": notification,
        "recommendation": recommendation,
        "teacher_action": teacher_action,
        "reason": reason,
    }


def apply_prediction(student):
    result = calculate_prediction(
        student.attendance,
        student.study_hours,
        student.assignments,
        student.quiz,
        student.lms_logins,
        student.engagement,
        student.missed_tasks,
    )
    for key, value in result.items():
        setattr(student, key, value)


def get_badges(student):
    badges = []
    if student.streak >= 5:
        badges.append("Consistent Learner")
    if student.assignments >= 4:
        badges.append("On-Time Submitter")
    if student.study_hours >= 4:
        badges.append("Focus Champion")
    if student.attendance >= 85:
        badges.append("Attendance Star")
    if student.quiz >= 75:
        badges.append("Quiz Improver")
    if not badges:
        badges.append("Starter Badge")
    return badges


def weekly_data(student):
    # Simple demo data for charts. It changes based on the student's current values.
    return {
        "labels": ["Week 1", "Week 2", "Week 3", "Week 4"],
        "attendance": [clamp(student.attendance - 10), clamp(student.attendance - 5), clamp(student.attendance - 2), student.attendance],
        "assignments": [clamp((student.assignments * 20) - 25), clamp((student.assignments * 20) - 15), clamp((student.assignments * 20) - 5), student.assignments * 20],
        "study": [clamp((student.study_hours * 20) - 20), clamp((student.study_hours * 20) - 10), clamp((student.study_hours * 20) - 5), min(student.study_hours * 20, 100)],
        "engagement": [clamp(student.engagement - 12), clamp(student.engagement - 8), clamp(student.engagement - 4), student.engagement],
    }


def get_students_list():
    return Student.query.order_by(Student.name).all()


def summary_data():
    students = get_students_list()
    if not students:
        return {
            "students": [],
            "total_students": 0,
            "high_risk": 0,
            "medium_risk": 0,
            "low_risk": 0,
            "avg_attendance": 0,
            "avg_score": 0,
            "avg_pass": 0,
        }

    return {
        "students": students,
        "total_students": len(students),
        "high_risk": len([s for s in students if s.risk == "High"]),
        "medium_risk": len([s for s in students if s.risk == "Medium"]),
        "low_risk": len([s for s in students if s.risk == "Low"]),
        "avg_attendance": round(sum(s.attendance for s in students) / len(students)),
        "avg_score": round(sum(s.productivity for s in students) / len(students)),
        "avg_pass": round(sum(s.pass_probability for s in students) / len(students)),
    }


def seed_data():
    if User.query.first():
        return

    users = [
        User(username="admin", password="123", role="admin", full_name="System Admin"),
        User(username="teacher", password="123", role="teacher", full_name="Teacher"),
        User(username="laxman", password="123", role="student", full_name="Laxman Shrestha"),
        User(username="nabin", password="123", role="student", full_name="Nabin Balami"),
        User(username="nishan", password="123", role="student", full_name="Nishan Singtan Tamang"),
    ]
    db.session.add_all(users)

    students = [
        Student(
            username="laxman",
            name="Laxman Shrestha",
            student_id="S2400285",
            course="Bachelor of IT",
            attendance=68,
            study_hours=2,
            assignments=2,
            quiz=55,
            lms_logins=3,
            engagement=50,
            missed_tasks=2,
            streak=3,
            daily_goal_done=2,
            daily_goal_total=4,
            weekly_goal_done=6,
            weekly_goal_total=12,
        ),
        Student(
            username="nabin",
            name="Nabin Balami",
            student_id="S2400290",
            course="Bachelor of IT",
            attendance=80,
            study_hours=3,
            assignments=3,
            quiz=70,
            lms_logins=6,
            engagement=68,
            missed_tasks=1,
            streak=5,
            daily_goal_done=3,
            daily_goal_total=4,
            weekly_goal_done=9,
            weekly_goal_total=12,
        ),
        Student(
            username="nishan",
            name="Nishan Singtan Tamang",
            student_id="S2400301",
            course="Bachelor of IT",
            attendance=92,
            study_hours=5,
            assignments=5,
            quiz=86,
            lms_logins=9,
            engagement=88,
            missed_tasks=0,
            streak=8,
            daily_goal_done=4,
            daily_goal_total=4,
            weekly_goal_done=12,
            weekly_goal_total=12,
        ),
    ]

    for student in students:
        apply_prediction(student)
        db.session.add(student)

    db.session.commit()


@app.route("/")
def home():
    return render_template("login.html")


@app.route("/login", methods=["POST"])
def login():
    username = request.form["username"].strip().lower()
    password = request.form["password"].strip()

    user = User.query.filter_by(username=username, password=password).first()

    if not user:
        return render_template("login.html", error="Invalid login details. Please try again.")

    if user.role == "admin":
        return redirect(url_for("admin"))
    if user.role == "teacher":
        return redirect(url_for("teacher"))
    if user.role == "student":
        return redirect(url_for("student_profile", username=user.username))

    return redirect(url_for("home"))


@app.route("/student/<username>")
def student_profile(username):
    student = Student.query.filter_by(username=username.lower()).first()
    if not student:
        return redirect(url_for("home"))
    return render_template(
        "student.html",
        student=student,
        badges=get_badges(student),
        chart=weekly_data(student),
    )


@app.route("/predict", methods=["POST"])
def predict():
    username = request.form["username"].strip().lower()
    student = Student.query.filter_by(username=username).first()
    if not student:
        return redirect(url_for("home"))

    student.name = request.form["name"].strip()
    student.student_id = request.form["student_id"].strip()
    student.course = request.form["course"].strip()
    student.attendance = int(request.form["attendance"])
    student.study_hours = int(request.form["study_hours"])
    student.assignments = int(request.form["assignments"])
    student.quiz = int(request.form["quiz"])
    student.lms_logins = int(request.form["lms_logins"])
    student.engagement = int(request.form["engagement"])
    student.missed_tasks = int(request.form["missed_tasks"])
    student.streak = int(request.form["streak"])
    student.daily_goal_done = int(request.form["daily_goal_done"])
    student.weekly_goal_done = int(request.form["weekly_goal_done"])

    apply_prediction(student)
    db.session.commit()

    return redirect(url_for("student_profile", username=username))


@app.route("/teacher")
def teacher():
    data = summary_data()
    chart = {
        "labels": [s.name for s in data["students"]],
        "attendance": [s.attendance for s in data["students"]],
        "productivity": [s.productivity for s in data["students"]],
        "engagement": [s.engagement for s in data["students"]],
    }
    return render_template("teacher.html", **data, chart=chart)


@app.route("/teacher/student/<username>")
def teacher_student_detail(username):
    student = Student.query.filter_by(username=username.lower()).first()
    if not student:
        return redirect(url_for("teacher"))
    return render_template(
        "student_detail.html",
        student=student,
        badges=get_badges(student),
        chart=weekly_data(student),
    )


@app.route("/admin")
def admin():
    data = summary_data()
    teachers = User.query.filter_by(role="teacher").all()
    return render_template("admin.html", **data, teachers=teachers)


@app.route("/admin/add_student", methods=["POST"])
def admin_add_student():
    username = request.form["username"].strip().lower()
    if not username:
        return redirect(url_for("admin"))

    existing_user = User.query.filter_by(username=username).first()
    existing_student = Student.query.filter_by(username=username).first()
    if existing_user or existing_student:
        return redirect(url_for("admin"))

    user = User(
        username=username,
        password=request.form.get("password", "123").strip() or "123",
        role="student",
        full_name=request.form["name"].strip(),
    )

    student = Student(
        username=username,
        name=request.form["name"].strip(),
        student_id=request.form["student_id"].strip(),
        course=request.form["course"].strip(),
        teacher_name=request.form.get("teacher_name", "Teacher").strip() or "Teacher",
        attendance=int(request.form.get("attendance", 75)),
        study_hours=int(request.form.get("study_hours", 2)),
        assignments=int(request.form.get("assignments", 3)),
        quiz=int(request.form.get("quiz", 65)),
        lms_logins=int(request.form.get("lms_logins", 5)),
        engagement=int(request.form.get("engagement", 60)),
        missed_tasks=int(request.form.get("missed_tasks", 1)),
    )
    apply_prediction(student)
    db.session.add(user)
    db.session.add(student)
    db.session.commit()
    return redirect(url_for("admin"))


@app.route("/admin/edit_student/<username>", methods=["GET", "POST"])
def admin_edit_student(username):
    student = Student.query.filter_by(username=username.lower()).first()
    if not student:
        return redirect(url_for("admin"))

    if request.method == "POST":
        student.name = request.form["name"].strip()
        student.student_id = request.form["student_id"].strip()
        student.course = request.form["course"].strip()
        student.teacher_name = request.form["teacher_name"].strip()
        student.attendance = int(request.form["attendance"])
        student.study_hours = int(request.form["study_hours"])
        student.assignments = int(request.form["assignments"])
        student.quiz = int(request.form["quiz"])
        student.lms_logins = int(request.form["lms_logins"])
        student.engagement = int(request.form["engagement"])
        student.missed_tasks = int(request.form["missed_tasks"])
        student.streak = int(request.form["streak"])
        student.daily_goal_done = int(request.form["daily_goal_done"])
        student.weekly_goal_done = int(request.form["weekly_goal_done"])
        apply_prediction(student)

        user = User.query.filter_by(username=student.username).first()
        if user:
            user.full_name = student.name

        db.session.commit()
        return redirect(url_for("admin"))

    return render_template("edit_student.html", student=student)


@app.route("/admin/delete_student/<username>", methods=["POST"])
def admin_delete_student(username):
    username = username.lower()
    student = Student.query.filter_by(username=username).first()
    user = User.query.filter_by(username=username, role="student").first()
    if student:
        db.session.delete(student)
    if user:
        db.session.delete(user)
    db.session.commit()
    return redirect(url_for("admin"))


@app.route("/download_report/<username>")
def download_report(username):
    student = Student.query.filter_by(username=username.lower()).first()
    if not student:
        return redirect(url_for("teacher"))

    report = f"""
AI-Based Student Productivity and Performance Prediction Report

Student Name: {student.name}
Student ID: {student.student_id}
Course: {student.course}
Teacher: {student.teacher_name}

Attendance: {student.attendance}%
Study Hours Per Day: {student.study_hours}
Assignments Completed: {student.assignments}/5
Quiz Score: {student.quiz}%
LMS Logins Per Week: {student.lms_logins}
Engagement Score: {student.engagement}%
Missed Tasks: {student.missed_tasks}

Productivity Score: {student.productivity}%
Risk Level: {student.risk}
Expected Grade: {student.expected_grade}
Pass Probability: {student.pass_probability}%
Learning Streak: {student.streak} days

Notification:
{student.notification}

Prediction Reason:
{student.reason}

AI Recommendation:
{student.recommendation}

Teacher Intervention Action:
{student.teacher_action}

Generated by AI-Based Student Productivity and Performance Prediction System Prototype.
""".strip()

    return Response(
        report,
        mimetype="text/plain",
        headers={"Content-Disposition": f"attachment;filename={student.username}_student_report.txt"},
    )


@app.route("/download_class_report")
def download_class_report():
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Name",
        "Student ID",
        "Course",
        "Teacher",
        "Attendance",
        "Study Hours",
        "Assignments",
        "Quiz",
        "LMS Logins",
        "Engagement",
        "Missed Tasks",
        "Productivity",
        "Risk",
        "Expected Grade",
        "Pass Probability",
        "Recommendation",
        "Teacher Action",
    ])

    for s in get_students_list():
        writer.writerow([
            s.name,
            s.student_id,
            s.course,
            s.teacher_name,
            f"{s.attendance}%",
            s.study_hours,
            f"{s.assignments}/5",
            f"{s.quiz}%",
            s.lms_logins,
            f"{s.engagement}%",
            s.missed_tasks,
            f"{s.productivity}%",
            s.risk,
            s.expected_grade,
            f"{s.pass_probability}%",
            s.recommendation,
            s.teacher_action,
        ])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=class_performance_report.csv"},
    )


@app.route("/model")
def model():
    importance = sorted(
        zip(FEATURE_NAMES, RISK_MODEL.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )
    return render_template("model.html", evidence=MODEL_EVIDENCE, importance=importance)


@app.route("/demo")
def demo():
    data = summary_data()
    sample_student = Student.query.order_by(Student.productivity).first()
    chart = {
        "labels": [s.name for s in data["students"]],
        "productivity": [s.productivity for s in data["students"]],
        "attendance": [s.attendance for s in data["students"]],
    }
    return render_template("demo.html", **data, student=sample_student, chart=chart, evidence=MODEL_EVIDENCE)


@app.route("/tutorial")
def tutorial():
    return render_template("tutorial.html")


@app.route("/privacy")
def privacy():
    return render_template("privacy.html")


@app.route("/logout")
def logout():
    return redirect(url_for("home"))


with app.app_context():
    db.create_all()
    seed_data()


if __name__ == "__main__":
    app.run(debug=True)
