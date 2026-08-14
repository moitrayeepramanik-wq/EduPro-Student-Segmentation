# Master Prompt: EduPro Student Segmentation & Personalized Course Recommendation MVP

Build EduPro Student Segmentation & Personalized Course Recommendation MVP

You are an experienced Python data scientist and Streamlit developer.

I am a beginner in AI/ML, so prioritize a SIMPLE, WORKING, EASY-TO-UNDERSTAND implementation over a sophisticated implementation.

The project must actually run using the supplied Excel dataset and eventually be deployable on Streamlit Cloud.

Do not over-engineer the project.

---

## PROJECT

Build:
Student Segmentation and Personalized Course Recommendation System for EduPro

The purpose is:
1. Understand learner behavior.
2. Group similar learners.
3. Give each learner a meaningful segment.
4. Recommend courses based on their segment and preferences.
5. Display everything in a Streamlit web application.

---

## DATASET

The supplied Excel file contains:

### Users
- UserID
- UserName
- Age
- Gender
- Email

### Teachers
- TeacherID
- TeacherName
- Age
- Gender
- Expertise
- YearsOfExperience
- TeacherRating

### Courses
- CourseID
- CourseName
- CourseCategory
- CourseType
- CourseLevel
- CoursePrice
- CourseDuration
- CourseRating

### Transactions
- TransactionID
- UserID
- CourseID
- TransactionDate
- Amount
- PaymentMethod
- TeacherID

Use the actual dataset.
DO NOT assume exact row counts.

---

## IMPORTANT

I am a beginner in AI/ML.

Therefore:
- keep the code simple
- add comments explaining important logic
- avoid unnecessary libraries
- avoid deep learning
- avoid neural networks
- avoid TensorFlow/PyTorch
- avoid databases
- avoid authentication
- avoid APIs
- avoid complicated recommendation algorithms

The final system only needs to WORK correctly and look professional.

---

## IMPLEMENTATION

Build the following pipeline:

```
Excel
  ↓
Pandas
  ↓
Data Cleaning
  ↓
Merge Users + Transactions + Courses
  ↓
Create learner-level features
  ↓
StandardScaler
  ↓
K-Means
  ↓
Cluster Profiles
  ↓
Simple Personalized Recommendation
  ↓
Streamlit
```

---

### STEP 1 — DATA LOADING
Create a simple Python data loading system.
Load all four sheets.
Validate:
- required columns
- missing values
- duplicates
- invalid dates
Convert TransactionDate to datetime.
Do not modify the original Excel file.

---

### STEP 2 — MASTER DATASET
Merge:
Transactions + Users + Courses + Teachers
Use:
UserID, CourseID, TeacherID as the appropriate join keys.
Create one transaction-level dataframe.

---

### STEP 3 — LEARNER FEATURES
Create ONE ROW PER USER.
Create these features:

- **Engagement**: TotalCourses
- **Spending**: TotalSpending, AverageSpending
- **Preferences**: PreferredCategory, PreferredLevel
- **Exploration**: CategoryDiversity
- **Rating**: AverageCourseRating
- **Learning Depth**: BeginnerRatio, IntermediateRatio, AdvancedRatio, LearningDepthIndex (Beginner = 0, Intermediate = 0.5, Advanced = 1; weighted average of course levels)
- **Additional**: PaidCourseRatio, AverageCourseDuration

---

### STEP 4 — CLUSTERING
Use these numerical features initially:
- Age
- TotalCourses
- TotalSpending
- AverageSpending
- CategoryDiversity
- AverageCourseRating
- PaidCourseRatio
- AdvancedRatio
- LearningDepthIndex
- AverageCourseDuration

Do NOT use: UserID, UserName, Email, TransactionID, CourseID, TeacherID.
Standardize numerical features using StandardScaler.

---

### STEP 5 — FIND NUMBER OF CLUSTERS
Test K: 2, 3, 4, 5, 6
Calculate:
- inertia
- silhouette score
Create:
1. Elbow chart
2. Silhouette chart
Choose the best practical K.
Do not automatically assume K=4. However, if several values are reasonable, prefer 4 because it makes the learner segmentation easier to explain in the academic presentation.

---

### STEP 6 — K-MEANS
Train K-Means using the selected K (random_state=42).
Add the resulting cluster number to every learner.

---

### STEP 7 — CLUSTER INTERPRETATION
Calculate average values for every cluster.
Show:
- learner count
- average courses
- average spending
- category diversity
- average rating
- advanced ratio
- paid course ratio
- preferred category
- preferred level

Based on these statistics, create understandable names (e.g., Beginner Explorers, Developing Learners, Advanced Specialists, High-Value Power Learners).

---

### STEP 8 — PERSONALIZED RECOMMENDATION
Implement a clean, simple course recommendation algorithm:
1. Exclude courses already taken by the user.
2. Filter/rank remaining courses matching the user's preferred category or next level up, high course ratings, and popular courses within the user's segment.
3. Show course details along with teacher credentials and rating.

---

### STEP 9 — STREAMLIT WEB APP
Build a clean, modern, beginner-friendly Streamlit web application with dynamic dashboards, dataset uploader/selector, interactive cluster visualization, cluster profile cards, and course recommendation interface.
