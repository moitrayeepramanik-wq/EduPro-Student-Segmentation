"""
EduPro Realistic Sample Dataset Generator
Creates EduPro_Dataset.xlsx with 4 sheets: Users, Teachers, Courses, Transactions.
Matches the exact schema requirements for the EduPro MVP.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_edupro_dataset(filename="EduPro_Dataset.xlsx", seed=42):
    np.random.seed(seed)
    random.seed(seed)

    # 1. GENERATE USERS
    first_names = ["Aarav", "Ananya", "Rohan", "Priya", "Vikram", "Sneha", "Rahul", "Kavya", "Aditya", "Ishita",
                   "Arjun", "Diya", "Siddharth", "Meera", "Varun", "Riya", "Karan", "Neha", "Dev", "Pooja",
                   "Alex", "Jordan", "Taylor", "Morgan", "Sam", "Chris", "Pat", "Casey", "Riley", "Avery"]
    last_names = ["Sharma", "Verma", "Patel", "Gupta", "Singh", "Kumar", "Rao", "Joshi", "Nair", "Reddy",
                  "Deshmukh", "Chopra", "Mehta", "Bhat", "Iyer", "Smith", "Johnson", "Williams", "Brown", "Jones"]

    users_list = []
    genders = ["Male", "Female", "Other"]
    
    for i in range(1, 101):
        fn = random.choice(first_names)
        ln = random.choice(last_names)
        name = f"{fn} {ln}"
        age = random.randint(18, 55)
        gender = random.choices(genders, weights=[0.48, 0.48, 0.04])[0]
        email = f"{fn.lower()}.{ln.lower()}{i}@example.com"
        users_list.append({
            "UserID": f"USR{i:03d}",
            "UserName": name,
            "Age": age,
            "Gender": gender,
            "Email": email
        })
    users_df = pd.DataFrame(users_list)

    # 2. GENERATE TEACHERS
    expertises = ["Data Science & AI", "Web Development", "Cloud Computing", "Cyber Security", "UI/UX Design", "Business & Marketing"]
    teachers_list = []
    
    for i in range(1, 16):
        fn = random.choice(first_names)
        ln = random.choice(last_names)
        name = f"Prof. {fn} {ln}"
        age = random.randint(30, 62)
        gender = random.choice(["Male", "Female"])
        exp = random.choice(expertises)
        yoe = random.randint(3, 25)
        rating = round(random.uniform(4.0, 5.0), 1)
        teachers_list.append({
            "TeacherID": f"TCH{i:03d}",
            "TeacherName": name,
            "Age": age,
            "Gender": gender,
            "Expertise": exp,
            "YearsOfExperience": yoe,
            "TeacherRating": rating
        })
    teachers_df = pd.DataFrame(teachers_list)

    # 3. GENERATE COURSES
    courses_data = [
        # Data Science & AI
        ("Python for Beginners", "Data Science & AI", "Free", "Beginner", 0, 10, 4.6),
        ("Data Analysis with Pandas", "Data Science & AI", "Paid", "Beginner", 49, 15, 4.7),
        ("Machine Learning Fundamentals", "Data Science & AI", "Paid", "Intermediate", 99, 25, 4.8),
        ("Deep Learning & Neural Networks", "Data Science & AI", "Paid", "Advanced", 149, 40, 4.9),
        ("Natural Language Processing", "Data Science & AI", "Paid", "Advanced", 129, 35, 4.7),
        
        # Web Development
        ("HTML & CSS Mastery", "Web Development", "Free", "Beginner", 0, 8, 4.4),
        ("JavaScript Modern ES6+", "Web Development", "Paid", "Beginner", 39, 12, 4.6),
        ("Full-Stack React & Node.js", "Web Development", "Paid", "Intermediate", 89, 30, 4.8),
        ("Next.js & TypeScript Architecture", "Web Development", "Paid", "Advanced", 119, 32, 4.9),
        ("GraphQL & Backend API Design", "Web Development", "Paid", "Intermediate", 79, 20, 4.5),
        
        # Cloud Computing
        ("Cloud Fundamentals AWS & Azure", "Cloud Computing", "Free", "Beginner", 0, 10, 4.5),
        ("Docker & Kubernetes Specialist", "Cloud Computing", "Paid", "Intermediate", 99, 22, 4.7),
        ("Enterprise AWS Solutions Architect", "Cloud Computing", "Paid", "Advanced", 159, 45, 4.9),
        ("DevOps CI/CD Pipelines", "Cloud Computing", "Paid", "Intermediate", 89, 24, 4.6),

        # Cyber Security
        ("Introduction to Cyber Security", "Cyber Security", "Free", "Beginner", 0, 6, 4.3),
        ("Ethical Hacking & Penetration Testing", "Cyber Security", "Paid", "Intermediate", 109, 28, 4.8),
        ("Advanced Network Security & Defense", "Cyber Security", "Paid", "Advanced", 139, 38, 4.7),

        # UI/UX Design
        ("Figma UI Design for Beginners", "UI/UX Design", "Free", "Beginner", 0, 7, 4.5),
        ("UX Research & Prototyping", "UI/UX Design", "Paid", "Intermediate", 59, 16, 4.6),
        ("Design Systems & UI Architecture", "UI/UX Design", "Paid", "Advanced", 99, 25, 4.8),

        # Business & Marketing
        ("Digital Marketing Essentials", "Business & Marketing", "Free", "Beginner", 0, 8, 4.4),
        ("Product Management Foundations", "Business & Marketing", "Paid", "Intermediate", 69, 18, 4.7),
        ("Executive Leadership & Strategy", "Business & Marketing", "Paid", "Advanced", 129, 30, 4.8),
        ("Financial Analytics for Decision Making", "Business & Marketing", "Paid", "Intermediate", 89, 20, 4.6)
    ]

    courses_list = []
    for i, c in enumerate(courses_data, 1):
        courses_list.append({
            "CourseID": f"CRS{i:03d}",
            "CourseName": c[0],
            "CourseCategory": c[1],
            "CourseType": c[2],
            "CourseLevel": c[3],
            "CoursePrice": c[4],
            "CourseDuration": c[5],  # in hours
            "CourseRating": c[6]
        })
    courses_df = pd.DataFrame(courses_list)

    # Map Teacher Expertise to relevant Courses
    teacher_by_exp = {}
    for _, t in teachers_df.iterrows():
        teacher_by_exp.setdefault(t["Expertise"], []).append(t["TeacherID"])

    # 4. GENERATE TRANSACTIONS (Realistic learner behaviors)
    payment_methods = ["Credit Card", "Debit Card", "UPI", "PayPal", "Net Banking"]
    start_date = datetime(2025, 1, 1)
    
    transactions_list = []
    tx_id = 1

    # Create distinct behavioral archetypes for realistic clustering
    for _, user in users_df.iterrows():
        uid = user["UserID"]
        # Determine user archetype
        archetype = random.choices(["beginner_explorer", "developing_learner", "advanced_specialist", "power_learner"], weights=[0.3, 0.35, 0.2, 0.15])[0]

        if archetype == "beginner_explorer":
            n_courses = random.randint(1, 3)
            pref_cats = random.sample(list(courses_df["CourseCategory"].unique()), k=min(n_courses, 3))
            candidate_courses = courses_df[(courses_df["CourseLevel"] == "Beginner") & (courses_df["CourseCategory"].isin(pref_cats))]
        
        elif archetype == "developing_learner":
            n_courses = random.randint(3, 6)
            fav_cat = random.choice(list(courses_df["CourseCategory"].unique()))
            candidate_courses = courses_df[(courses_df["CourseCategory"] == fav_cat) | (courses_df["CourseLevel"].isin(["Beginner", "Intermediate"]))]

        elif archetype == "advanced_specialist":
            n_courses = random.randint(4, 7)
            fav_cat = random.choice(["Data Science & AI", "Cloud Computing", "Web Development"])
            candidate_courses = courses_df[(courses_df["CourseCategory"] == fav_cat) & (courses_df["CourseLevel"].isin(["Intermediate", "Advanced"]))]

        else: # power_learner
            n_courses = random.randint(6, 11)
            candidate_courses = courses_df[courses_df["CourseType"] == "Paid"]

        if candidate_courses.empty:
            candidate_courses = courses_df

        # Sample courses without replacement up to n_courses
        selected_courses = candidate_courses.sample(n=min(n_courses, len(candidate_courses)), replace=False)

        for _, course in selected_courses.iterrows():
            cid = course["CourseID"]
            cat = course["CourseCategory"]
            
            # Pick a teacher with matching expertise or fallback
            possible_teachers = teacher_by_exp.get(cat, teachers_df["TeacherID"].tolist())
            tid = random.choice(possible_teachers)
            
            # Transaction Date
            tx_date = start_date + timedelta(days=random.randint(0, 500))
            
            # Amount equal to CoursePrice (with occasional discount or full price)
            price = course["CoursePrice"]
            amount = price if course["CourseType"] == "Paid" else 0
            if amount > 0 and random.random() < 0.15: # 15% discount promo
                amount = round(amount * 0.85, 2)

            pm = random.choice(payment_methods) if amount > 0 else "Free Enrollment"

            transactions_list.append({
                "TransactionID": f"TXN{tx_id:05d}",
                "UserID": uid,
                "CourseID": cid,
                "TransactionDate": tx_date.strftime("%Y-%m-%d"),
                "Amount": amount,
                "PaymentMethod": pm,
                "TeacherID": tid
            })
            tx_id += 1

    tx_df = pd.DataFrame(transactions_list)

    # Save to Excel file with 4 sheets
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        users_df.to_excel(writer, sheet_name="Users", index=False)
        teachers_df.to_excel(writer, sheet_name="Teachers", index=False)
        courses_df.to_excel(writer, sheet_name="Courses", index=False)
        tx_df.to_excel(writer, sheet_name="Transactions", index=False)

    print(f"Dataset successfully created: {filename}")
    print(f"  - Users: {len(users_df)} rows")
    print(f"  - Teachers: {len(teachers_df)} rows")
    print(f"  - Courses: {len(courses_df)} rows")
    print(f"  - Transactions: {len(tx_df)} rows")

if __name__ == "__main__":
    generate_edupro_dataset()
