"""
EduPro Core Machine Learning & Data Processing Pipeline
Steps 1 to 8 implementation cleanly structured for the Streamlit Web Application.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Numerical features required for K-Means clustering as specified in Step 4
NUMERICAL_FEATURES = [
    'Age',
    'TotalCourses',
    'TotalSpending',
    'AverageSpending',
    'CategoryDiversity',
    'AverageCourseRating',
    'PaidCourseRatio',
    'AdvancedRatio',
    'LearningDepthIndex',
    'AverageCourseDuration'
]

# Required columns for each Excel sheet validation
REQUIRED_COLUMNS = {
    "Users": ["UserID", "UserName", "Age", "Gender", "Email"],
    "Teachers": ["TeacherID", "TeacherName", "Age", "Gender", "Expertise", "YearsOfExperience", "TeacherRating"],
    "Courses": ["CourseID", "CourseName", "CourseCategory", "CourseType", "CourseLevel", "CoursePrice", "CourseDuration", "CourseRating"],
    "Transactions": ["TransactionID", "UserID", "CourseID", "TransactionDate", "Amount", "PaymentMethod", "TeacherID"]
}


# ==========================================
# STEP 1 — DATA LOADING & VALIDATION
# ==========================================

def load_data(file_source):
    """
    Load all four sheets from the Excel file (file path or file buffer).
    Returns a dictionary of dataframes for Users, Teachers, Courses, Transactions.
    """
    excel_file = pd.ExcelFile(file_source)
    sheets = {}
    for sheet_name in ["Users", "Teachers", "Courses", "Transactions"]:
        if sheet_name in excel_file.sheet_names:
            sheets[sheet_name] = pd.read_excel(excel_file, sheet_name=sheet_name)
        else:
            raise ValueError(f"Missing required sheet '{sheet_name}' in Excel file.")
    return sheets


def validate_data(sheets):
    """
    Validates:
    - Required columns in each sheet
    - Missing values count
    - Duplicate rows count
    - Invalid dates in Transactions
    Converts TransactionDate to datetime format.
    Returns validation summary report and cleaned sheets.
    """
    report = {}
    cleaned_sheets = {}

    for sheet_name, df in sheets.items():
        df_copy = df.copy()
        req_cols = REQUIRED_COLUMNS[sheet_name]
        
        # Check required columns
        missing_cols = [c for c in req_cols if c not in df_copy.columns]
        if missing_cols:
            raise ValueError(f"Sheet '{sheet_name}' is missing columns: {missing_cols}")
        
        # Count missing values & duplicates
        missing_count = df_copy[req_cols].isnull().sum().to_dict()
        duplicate_count = int(df_copy.duplicated(subset=[req_cols[0]]).sum())

        # Date conversion for Transactions
        invalid_dates_count = 0
        if sheet_name == "Transactions":
            df_copy["TransactionDate"] = pd.to_datetime(df_copy["TransactionDate"], errors="coerce")
            invalid_dates_count = int(df_copy["TransactionDate"].isnull().sum())

        report[sheet_name] = {
            "row_count": len(df_copy),
            "missing_values": missing_count,
            "duplicate_count": duplicate_count,
            "invalid_dates": invalid_dates_count
        }

        cleaned_sheets[sheet_name] = df_copy

    return report, cleaned_sheets


# ==========================================
# STEP 2 — MASTER DATASET MERGING
# ==========================================

def create_master_dataset(cleaned_sheets):
    """
    Merge Transactions + Users + Courses + Teachers
    Join keys: UserID, CourseID, TeacherID.
    Returns transaction-level master dataframe.
    """
    users_df = cleaned_sheets["Users"]
    teachers_df = cleaned_sheets["Teachers"]
    courses_df = cleaned_sheets["Courses"]
    tx_df = cleaned_sheets["Transactions"]

    # Step-by-step merge to avoid duplicate column name collisions
    master = tx_df.merge(
        users_df[["UserID", "UserName", "Age", "Gender", "Email"]],
        on="UserID",
        how="inner",
        suffixes=("", "_User")
    )

    master = master.merge(
        courses_df[["CourseID", "CourseName", "CourseCategory", "CourseType", "CourseLevel", "CoursePrice", "CourseDuration", "CourseRating"]],
        on="CourseID",
        how="inner"
    )

    master = master.merge(
        teachers_df[["TeacherID", "TeacherName", "Expertise", "YearsOfExperience", "TeacherRating"]],
        on="TeacherID",
        how="left"
    )

    return master


# ==========================================
# STEP 3 — LEARNER FEATURE ENGINEERING
# ==========================================

def create_learner_features(master_df, users_df):
    """
    Create ONE ROW PER USER with exact required features:
    - Engagement: TotalCourses
    - Spending: TotalSpending, AverageSpending
    - Preferences: PreferredCategory, PreferredLevel
    - Exploration: CategoryDiversity
    - Rating: AverageCourseRating
    - Learning Depth: BeginnerRatio, IntermediateRatio, AdvancedRatio, LearningDepthIndex
    - Additional: PaidCourseRatio, AverageCourseDuration
    """
    records = []

    # Helper function for safe mode
    def get_mode(series, default="Unknown"):
        mode_val = series.mode()
        return mode_val.iloc[0] if not mode_val.empty else default

    grouped = master_df.groupby("UserID")

    for user_id, group in grouped:
        total_courses = len(group)
        total_spending = group["Amount"].sum()
        avg_spending = group["Amount"].mean()
        
        pref_category = get_mode(group["CourseCategory"])
        pref_level = get_mode(group["CourseLevel"])
        cat_diversity = group["CourseCategory"].nunique()
        avg_rating = group["CourseRating"].mean()

        # Learning Depth Calculation
        beginner_count = (group["CourseLevel"] == "Beginner").sum()
        inter_count = (group["CourseLevel"] == "Intermediate").sum()
        adv_count = (group["CourseLevel"] == "Advanced").sum()

        beg_ratio = beginner_count / total_courses if total_courses > 0 else 0.0
        inter_ratio = inter_count / total_courses if total_courses > 0 else 0.0
        adv_ratio = adv_count / total_courses if total_courses > 0 else 0.0

        # Weighted average formula: Beginner=0, Intermediate=0.5, Advanced=1.0
        depth_index = (beg_ratio * 0.0) + (inter_ratio * 0.5) + (adv_ratio * 1.0)

        # Paid Course Ratio
        paid_count = ((group["CourseType"] == "Paid") | (group["Amount"] > 0)).sum()
        paid_ratio = paid_count / total_courses if total_courses > 0 else 0.0

        # Average Course Duration
        avg_duration = group["CourseDuration"].mean()

        records.append({
            "UserID": user_id,
            "TotalCourses": total_courses,
            "TotalSpending": round(total_spending, 2),
            "AverageSpending": round(avg_spending, 2),
            "PreferredCategory": pref_category,
            "PreferredLevel": pref_level,
            "CategoryDiversity": cat_diversity,
            "AverageCourseRating": round(avg_rating, 2),
            "BeginnerRatio": round(beg_ratio, 3),
            "IntermediateRatio": round(inter_ratio, 3),
            "AdvancedRatio": round(adv_ratio, 3),
            "LearningDepthIndex": round(depth_index, 3),
            "PaidCourseRatio": round(paid_ratio, 3),
            "AverageCourseDuration": round(avg_duration, 1)
        })

    features_df = pd.DataFrame(records)

    # Merge back user demographics (Age, Gender, UserName, Email)
    learner_df = users_df[["UserID", "UserName", "Age", "Gender", "Email"]].merge(
        features_df, on="UserID", how="left"
    )

    # Fill NaN for users who might have 0 transactions
    learner_df["TotalCourses"] = learner_df["TotalCourses"].fillna(0).astype(int)
    learner_df["TotalSpending"] = learner_df["TotalSpending"].fillna(0.0)
    learner_df["AverageSpending"] = learner_df["AverageSpending"].fillna(0.0)
    learner_df["CategoryDiversity"] = learner_df["CategoryDiversity"].fillna(0).astype(int)
    learner_df["AverageCourseRating"] = learner_df["AverageCourseRating"].fillna(0.0)
    learner_df["BeginnerRatio"] = learner_df["BeginnerRatio"].fillna(0.0)
    learner_df["IntermediateRatio"] = learner_df["IntermediateRatio"].fillna(0.0)
    learner_df["AdvancedRatio"] = learner_df["AdvancedRatio"].fillna(0.0)
    learner_df["LearningDepthIndex"] = learner_df["LearningDepthIndex"].fillna(0.0)
    learner_df["PaidCourseRatio"] = learner_df["PaidCourseRatio"].fillna(0.0)
    learner_df["AverageCourseDuration"] = learner_df["AverageCourseDuration"].fillna(0.0)
    learner_df["PreferredCategory"] = learner_df["PreferredCategory"].fillna("General")
    learner_df["PreferredLevel"] = learner_df["PreferredLevel"].fillna("Beginner")

    return learner_df


# ==========================================
# STEP 4 & 5 — CLUSTERING & EVALUATION
# ==========================================

def scale_learner_features(learner_df):
    """
    Standardize the 10 specified numerical features using StandardScaler.
    Returns scaled numpy array and fitted scaler.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(learner_df[NUMERICAL_FEATURES])
    return X_scaled, scaler


def evaluate_kmeans_range(X_scaled, k_range=[2, 3, 4, 5, 6]):
    """
    Test K in k_range.
    Calculates Inertia (Elbow) and Silhouette Scores.
    """
    results = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertia = kmeans.inertia_
        sil_score = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0.0
        results.append({
            "K": k,
            "Inertia": round(inertia, 2),
            "SilhouetteScore": round(sil_score, 4)
        })
    return pd.DataFrame(results)


# ==========================================
# STEP 6 — K-MEANS TRAINING
# ==========================================

def train_kmeans_clustering(learner_df, X_scaled, k=4):
    """
    Train K-Means using selected K (random_state=42).
    Appends 'Cluster' column (e.g. Cluster 0, Cluster 1...) to learner_df.
    Returns updated learner_df, fitted model, and PCA 2D projections for plotting.
    """
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    learner_df_copy = learner_df.copy()
    learner_df_copy["Cluster"] = cluster_labels
    learner_df_copy["Cluster_Label"] = [f"Cluster {c}" for c in cluster_labels]

    # Compute PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    learner_df_copy["PCA1"] = pca_coords[:, 0]
    learner_df_copy["PCA2"] = pca_coords[:, 1]

    return learner_df_copy, kmeans, pca


# ==========================================
# STEP 7 — CLUSTER INTERPRETATION & NAMING
# ==========================================

def calculate_cluster_profiles(learner_df):
    """
    Calculate summary statistics for each cluster:
    - Learner count
    - Average courses, spending, category diversity, rating, advanced ratio, paid ratio
    - Preferred category & level
    Assigns human-understandable segment names.
    """
    grouped = learner_df.groupby("Cluster")

    profiles = []
    for cluster_id, group in grouped:
        def get_mode_val(series):
            m = series.mode()
            return m.iloc[0] if not m.empty else "N/A"

        prof = {
            "Cluster": cluster_id,
            "LearnerCount": len(group),
            "AvgCourses": round(group["TotalCourses"].mean(), 1),
            "AvgSpending": round(group["TotalSpending"].mean(), 2),
            "AvgCategoryDiversity": round(group["CategoryDiversity"].mean(), 1),
            "AvgCourseRating": round(group["AverageCourseRating"].mean(), 2),
            "AvgAdvancedRatio": round(group["AdvancedRatio"].mean(), 3),
            "AvgPaidCourseRatio": round(group["PaidCourseRatio"].mean(), 3),
            "AvgLearningDepth": round(group["LearningDepthIndex"].mean(), 3),
            "PreferredCategory": get_mode_val(group["PreferredCategory"]),
            "PreferredLevel": get_mode_val(group["PreferredLevel"])
        }
        profiles.append(prof)

    prof_df = pd.DataFrame(profiles)

    # Dynamic or heuristic mapping based on cluster characteristics
    segment_names = {}
    for idx, row in prof_df.iterrows():
        cid = row["Cluster"]
        adv = row["AvgAdvancedRatio"]
        spend = row["AvgSpending"]
        courses = row["AvgCourses"]
        depth = row["AvgLearningDepth"]
        div = row["AvgCategoryDiversity"]

        if spend > prof_df["AvgSpending"].median() and courses >= prof_df["AvgCourses"].median():
            name = "High-Value Power Learners"
        elif adv >= 0.35 or depth >= 0.6:
            name = "Advanced Specialists"
        elif div >= 2.0 and depth < 0.4:
            name = "Beginner Explorers"
        else:
            name = "Developing Learners"

        # Fallback deduplication if two clusters get same name
        if name in segment_names.values():
            name = f"{name} (Group {cid+1})"

        segment_names[cid] = name

    prof_df["SegmentName"] = prof_df["Cluster"].map(segment_names)
    return prof_df, segment_names


# ==========================================
# STEP 8 — PERSONALIZED RECOMMENDATION
# ==========================================

def recommend_courses_for_user(user_id, learner_df, master_df, courses_df, teachers_df, top_n=5):
    """
    Simple rule-based personalized recommendation:
    1. Filter out courses already taken by the user.
    2. Score remaining courses matching user's preferred category, level progression, and cluster popularity.
    3. Include course details and teacher info.
    """
    user_row = learner_df[learner_df["UserID"] == user_id]
    if user_row.empty:
        return pd.DataFrame()

    user_info = user_row.iloc[0]
    user_cluster = user_info["Cluster"]
    pref_cat = user_info["PreferredCategory"]
    pref_lvl = user_info["PreferredLevel"]

    # 1. Identify completed course IDs
    user_tx = master_df[master_df["UserID"] == user_id]
    taken_course_ids = set(user_tx["CourseID"].tolist())

    # 2. Filter untaken courses
    untaken_courses = courses_df[~courses_df["CourseID"].isin(taken_course_ids)].copy()
    if untaken_courses.empty:
        return pd.DataFrame()

    # 3. Calculate cluster popularity score per course
    cluster_users = set(learner_df[learner_df["Cluster"] == user_cluster]["UserID"])
    cluster_tx = master_df[master_df["UserID"].isin(cluster_users)]
    cluster_course_counts = cluster_tx["CourseID"].value_counts().to_dict()

    # 4. Scoring courses and building human-readable reasons
    scores = []
    reasons_list = []
    level_progression = {"Beginner": "Intermediate", "Intermediate": "Advanced", "Advanced": "Advanced"}

    for _, course in untaken_courses.iterrows():
        c_id = course["CourseID"]
        c_cat = course["CourseCategory"]
        c_lvl = course["CourseLevel"]
        c_rating = course["CourseRating"]

        score = float(c_rating) # Base score
        reasons = []

        # Preferred Category Match
        if c_cat == pref_cat:
            score += 3.0
            reasons.append(f"Matches your preferred category ({pref_cat})")

        # Preferred Level or Progression Match
        if c_lvl == pref_lvl:
            score += 2.0
            reasons.append(f"Matches your current level ({pref_lvl})")
        elif c_lvl == level_progression.get(pref_lvl):
            score += 1.5
            reasons.append(f"Next logical level progression ({c_lvl})")

        # Cluster Popularity Bonus
        pop_count = cluster_course_counts.get(c_id, 0)
        if pop_count > 0:
            score += min(pop_count * 0.5, 2.0)
            reasons.append(f"Popular among similar {user_info.get('SegmentName', 'segment')} learners ({pop_count} enrolled)")

        # High Rating Bonus
        if c_rating >= 4.5:
            reasons.append(f"Highly rated by students (⭐ {c_rating})")

        if not reasons:
            reasons.append("Recommended based on catalog quality score")

        scores.append(round(score, 2))
        reasons_list.append(" • ".join(reasons))

    untaken_courses["RecommendationScore"] = scores
    untaken_courses["Reason"] = reasons_list

    # Sort top N courses
    top_recs = untaken_courses.sort_values(by="RecommendationScore", ascending=False).head(top_n).copy()

    # Merge teacher expertise & teacher name recommendation matching expertise
    top_recs["MatchingTeacher"] = top_recs["CourseCategory"].apply(
        lambda cat: teachers_df[teachers_df["Expertise"] == cat]["TeacherName"].iloc[0]
        if not teachers_df[teachers_df["Expertise"] == cat].empty
        else teachers_df["TeacherName"].iloc[0]
    )

    top_recs["TeacherRating"] = top_recs["CourseCategory"].apply(
        lambda cat: teachers_df[teachers_df["Expertise"] == cat]["TeacherRating"].iloc[0]
        if not teachers_df[teachers_df["Expertise"] == cat].empty
        else teachers_df["TeacherRating"].iloc[0]
    )

    return top_recs
