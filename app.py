"""
EduPro Student Segmentation & Personalized Course Recommendation MVP
Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import textwrap
import io

from src.pipeline import (
    load_data,
    validate_data,
    create_master_dataset,
    create_learner_features,
    scale_learner_features,
    evaluate_kmeans_range,
    train_kmeans_clustering,
    calculate_cluster_profiles,
    recommend_courses_for_user,
    NUMERICAL_FEATURES
)

from src.utils import (
    apply_custom_css,
    plot_elbow_curve,
    plot_silhouette_curve,
    plot_pca_clusters,
    plot_cluster_feature_comparison,
    get_segment_badge_html
)

# Page configuration
st.set_page_config(
    page_title="EduPro | Student Segmentation & Recommendations",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom Dark Theme CSS
apply_custom_css()

# Header Banner
header_html = textwrap.dedent("""
<div class="header-box">
    <div class="header-title">🎓 EduPro Personalized Learning Engine</div>
    <div class="header-subtitle">
        Student Segmentation & Personalized Course Recommendation MVP
    </div>
</div>
""")
st.markdown(header_html, unsafe_allow_html=True)


# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/isometric/100/graduation-cap.png", width=70)
st.sidebar.title("🛠️ Settings")

dataset_option = st.sidebar.radio(
    "Dataset Source",
    ["Use Default EduPro Dataset", "Upload Custom Excel Dataset"],
    help="Select the Excel file containing Users, Teachers, Courses, and Transactions sheets."
)

uploaded_file = None
if dataset_option == "Upload Custom Excel Dataset":
    uploaded_file = st.sidebar.file_uploader("Upload .xlsx file", type=["xlsx"])
    file_to_load = uploaded_file
else:
    file_to_load = "EduPro_Dataset.xlsx"

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Clustering Settings")
selected_k = st.sidebar.slider(
    "Select Number of Clusters (K)",
    min_value=2,
    max_value=6,
    value=4,
    step=1,
    help="Default K=4 is recommended for intuitive academic presentation."
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**💡 Core Pipeline Architecture:**  
Student Data → StandardScaler → K-Means → Learner Segment → Course Recommendations
""")


# Data Pipeline Execution with Caching
@st.cache_data(show_spinner=False)
def run_full_pipeline(dataset_source, k_clusters):
    # Step 1: Load and Validate Data
    raw_sheets = load_data(dataset_source)
    validation_report, cleaned_sheets = validate_data(raw_sheets)
    
    # Step 2: Master Dataset
    master_df = create_master_dataset(cleaned_sheets)
    
    # Step 3: Learner Features
    learner_df = create_learner_features(master_df, cleaned_sheets["Users"])
    
    # Step 4: Scale Features
    X_scaled, scaler = scale_learner_features(learner_df)
    
    # Step 5: K Evaluation
    eval_df = evaluate_kmeans_range(X_scaled, k_range=[2, 3, 4, 5, 6])
    
    # Step 6: Train K-Means
    learner_clustered, kmeans_model, pca = train_kmeans_clustering(learner_df, X_scaled, k=k_clusters)
    
    # Step 7: Cluster Interpretation
    profiles_df, segment_map = calculate_cluster_profiles(learner_clustered)
    
    return {
        "raw_sheets": raw_sheets,
        "validation_report": validation_report,
        "cleaned_sheets": cleaned_sheets,
        "master_df": master_df,
        "learner_df": learner_clustered,
        "X_scaled": X_scaled,
        "eval_df": eval_df,
        "kmeans_model": kmeans_model,
        "profiles_df": profiles_df,
        "segment_map": segment_map
    }


# Main Pipeline Execution Block
if file_to_load is not None:
    try:
        pipeline_data = run_full_pipeline(file_to_load, selected_k)
        
        cleaned_sheets = pipeline_data["cleaned_sheets"]
        master_df = pipeline_data["master_df"]
        learner_df = pipeline_data["learner_df"]
        eval_df = pipeline_data["eval_df"]
        profiles_df = pipeline_data["profiles_df"]
        segment_map = pipeline_data["segment_map"]
        validation_report = pipeline_data["validation_report"]

        learner_df["SegmentName"] = learner_df["Cluster"].map(segment_map)

        # Tabs Layout
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🚀 Course Recommender",
            "🏷️ Learner Segments",
            "📊 Learner Profile Intelligence",
            "🎯 K-Means Evaluation",
            "📁 Master Transaction Data",
            "📥 Technical Exports & Q&A"
        ])

        # ==========================================
        # TOP DASHBOARD HIGHLIGHT METRICS
        # ==========================================
        total_users = len(cleaned_sheets["Users"])
        total_courses = len(cleaned_sheets["Courses"])
        total_enrollments = len(cleaned_sheets["Transactions"])
        total_categories = cleaned_sheets["Courses"]["CourseCategory"].nunique()

        # ==========================================
        # TAB 1: PERSONALIZED COURSE RECOMMENDER (MAIN VISIBLE PRODUCT)
        # ==========================================
        with tab1:
            st.markdown("### 🎓 Interactive Student Recommendation Engine")
            
            # Top 4 Overview Metric Cards
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Learners</div><div class="metric-value">{total_users:,}</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Courses</div><div class="metric-value">{total_courses:,}</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Enrollments</div><div class="metric-value">{total_enrollments:,}</div></div>', unsafe_allow_html=True)
            with m4:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Categories</div><div class="metric-value">{total_categories:,}</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Learner Selector
            user_options = learner_df.apply(lambda r: f"{r['UserID']} - {r['UserName']} ({r['SegmentName']})", axis=1).tolist()
            selected_user_str = st.selectbox("Select Student for Recommendation", options=user_options, index=0)
            
            selected_user_id = selected_user_str.split(" - ")[0]
            user_row = learner_df[learner_df["UserID"] == selected_user_id].iloc[0]
            user_segment = user_row["SegmentName"]

            col_prof, col_recs = st.columns([1, 1.4])

            # Output #2: Student Profile Box
            with col_prof:
                st.markdown("#### ──────────────────────────────")
                st.markdown("### 👤 LEARNER PROFILE")
                st.markdown("──────────────────────────────")
                
                profile_card_html = textwrap.dedent(f"""
                <div class="course-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h4 style="margin:0; color:#f8fafc;">Student ID: <b>{user_row['UserID']}</b></h4>
                        {get_segment_badge_html(user_segment)}
                    </div>
                    <p style="color:#cbd5e1; font-size:0.95rem; margin-top:12px;">
                        <b>Name:</b> {user_row['UserName']}<br>
                        <b>Age:</b> {user_row['Age']} &nbsp;|&nbsp; <b>Gender:</b> {user_row['Gender']}<br>
                        <b>Email:</b> {user_row['Email']}
                    </p>
                    <hr style="border-color:rgba(255,255,255,0.1);">
                    <div style="font-size:0.92rem; color:#e2e8f0; line-height:1.8;">
                        📚 <b>Courses Enrolled:</b> {user_row['TotalCourses']}<br>
                        💰 <b>Total Spending:</b> ₹{user_row['TotalSpending']:,.2f}<br>
                        ⭐ <b>Avg Course Rating:</b> {user_row['AverageCourseRating']} / 5.0<br>
                        🌐 <b>Category Diversity:</b> {user_row['CategoryDiversity']} categories<br>
                        🎯 <b>Preferred Category:</b> <span style="color:#38bdf8; font-weight:700;">{user_row['PreferredCategory']}</span><br>
                        📊 <b>Preferred Level:</b> <span style="color:#c084fc; font-weight:700;">{user_row['PreferredLevel']}</span><br>
                        🧠 <b>Learning Depth Index:</b> {user_row['LearningDepthIndex']}
                    </div>
                    <hr style="border-color:rgba(255,255,255,0.1);">
                    <div style="background:rgba(129, 140, 248, 0.1); border-left:3px solid #818cf8; padding:10px; border-radius:4px; font-size:0.88rem; color:#cbd5e1;">
                        <b>Segment Summary:</b><br>
                        This learner exhibits characteristics of <b>{user_segment}</b> based on engagement frequency, learning depth index, spending tier, and category diversity.
                    </div>
                </div>
                """)
                st.markdown(profile_card_html, unsafe_allow_html=True)

                st.markdown("#### 📚 Completed Courses History")
                user_tx = master_df[master_df["UserID"] == selected_user_id]
                if not user_tx.empty:
                    taken_display = user_tx[["CourseID", "CourseName", "CourseCategory", "CourseLevel", "Amount"]].drop_duplicates()
                    st.dataframe(taken_display, use_container_width=True, height=200)

            # Output #3: Personalized Course Recommendations
            with col_recs:
                st.markdown("#### ──────────────────────────────")
                st.markdown("### 🎯 RECOMMENDED COURSES")
                st.markdown("──────────────────────────────")
                
                rec_courses = recommend_courses_for_user(
                    selected_user_id,
                    learner_df,
                    master_df,
                    cleaned_sheets["Courses"],
                    cleaned_sheets["Teachers"],
                    top_n=5
                )

                if not rec_courses.empty:
                    for idx, (_, rec) in enumerate(rec_courses.iterrows(), 1):
                        reasons = rec["Reason"].split(" • ")
                        reasons_html = "".join([f'<div class="reason-item">✓ {r}</div>' for r in reasons])

                        rec_card_html = textwrap.dedent(f"""
                        <div class="course-card">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <div class="course-title">{idx}. 📖 {rec['CourseName']}</div>
                                <span style="background:rgba(56, 189, 248, 0.2); color:#38bdf8; border:1px solid #38bdf8; border-radius:12px; padding:3px 12px; font-size:0.82rem; font-weight:800;">
                                    Match Score: {rec['RecommendationScore']}
                                </span>
                            </div>
                            <div class="course-meta">
                                <span>🏷️ <b>Category:</b> {rec['CourseCategory']}</span>
                                <span>📊 <b>Level:</b> {rec['CourseLevel']}</span>
                                <span>⭐ <b>Rating:</b> {rec['CourseRating']}</span>
                                <span>💲 <b>Price:</b> ₹{rec['CoursePrice']}</span>
                            </div>
                            <div class="reason-box">
                                <div class="reason-title">WHY RECOMMENDED?</div>
                                {reasons_html}
                            </div>
                            <div class="teacher-box">
                                👨‍🏫 <b>Expert Instructor:</b> {rec['MatchingTeacher']} (⭐ Rating: {rec['TeacherRating']})
                            </div>
                        </div>
                        """)
                        st.markdown(rec_card_html, unsafe_allow_html=True)
                else:
                    st.warning("No untaken courses available for recommendation.")


        # ==========================================
        # TAB 2: LEARNER SEGMENTS (OUTPUT #1)
        # ==========================================
        with tab2:
            st.subheader("🏷️ Output #1 — Student Segmentation Profiles")
            st.markdown("Machine Learning outputs grouping learners into distinct behavioral segments:")

            st.dataframe(profiles_df, use_container_width=True)

            st.markdown("### 💡 Learner Segment Profiles Breakdown")
            seg_cols = st.columns(min(selected_k, 4))
            for idx, row in profiles_df.iterrows():
                with seg_cols[idx % 4]:
                    seg_card_html = textwrap.dedent(f"""
                    <div class="course-card">
                        <div>{get_segment_badge_html(row['SegmentName'])}</div>
                        <h4 style="margin-top:10px; color:#f8fafc;">Cluster {row['Cluster']}</h4>
                        <p style="color:#94a3b8; font-size:0.88rem;">
                            👥 <b>{row['LearnerCount']} Learners</b> ({round(row['LearnerCount']/len(learner_df)*100, 1)}%)<br>
                            📚 Avg Courses: <b>{row['AvgCourses']}</b><br>
                            💰 Avg Spending: <b>₹{row['AvgSpending']}</b><br>
                            🧠 Learning Depth: <b>{row['AvgLearningDepth']}</b><br>
                            ⭐ Pref Category: <b>{row['PreferredCategory']}</b><br>
                            🎯 Pref Level: <b>{row['PreferredLevel']}</b>
                        </p>
                    </div>
                    """)
                    st.markdown(seg_card_html, unsafe_allow_html=True)

            st.markdown("### 📊 Segment Feature Comparisons")
            comp_feat = st.selectbox(
                "Select Feature to Compare Across Segments",
                options=["AvgSpending", "AvgCourses", "AvgLearningDepth", "AvgCategoryDiversity", "AvgPaidCourseRatio", "AvgCourseRating"]
            )
            st.plotly_chart(plot_cluster_feature_comparison(profiles_df, feature=comp_feat), use_container_width=True)


        # ==========================================
        # TAB 3: LEARNER PROFILE INTELLIGENCE (OUTPUT #2 DETAILS)
        # ==========================================
        with tab3:
            st.subheader("📊 Output #2 — Student Learner Features Table")
            st.markdown("Single row per user feature table created from historical transactions:")

            st.dataframe(learner_df, use_container_width=True, height=400)

            st.markdown("### 📈 Feature Distribution Analysis")
            feature_to_plot = st.selectbox(
                "Select Feature to Visualize",
                options=NUMERICAL_FEATURES,
                key="tab3_feat"
            )
            
            import plotly.express as px
            fig_dist = px.histogram(
                learner_df,
                x=feature_to_plot,
                color="SegmentName",
                nbins=20,
                title=f"Distribution of {feature_to_plot} across Segments",
                template="plotly_dark"
            )
            fig_dist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,41,59,0.5)")
            st.plotly_chart(fig_dist, use_container_width=True)


        # ==========================================
        # TAB 4: K-MEANS EVALUATION (TECHNICAL METRICS)
        # ==========================================
        with tab4:
            st.subheader("🎯 Technical Output — K-Means Optimization & Cluster Plots")
            st.markdown("Validation curves proving optimal K selection:")

            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(plot_elbow_curve(eval_df, selected_k), use_container_width=True)
            with col_b:
                st.plotly_chart(plot_silhouette_curve(eval_df, selected_k), use_container_width=True)

            st.markdown("### 🗺️ 2D PCA Cluster Scatter Visualization")
            st.plotly_chart(plot_pca_clusters(learner_df, segment_map), use_container_width=True)


        # ==========================================
        # TAB 5: MASTER DATASET
        # ==========================================
        with tab5:
            st.subheader("📁 Transaction-Level Master Dataset")
            st.markdown("Fact-level join of Transactions + Users + Courses + Teachers:")
            st.dataframe(master_df, use_container_width=True, height=450)


        # ==========================================
        # TAB 6: TECHNICAL EXPORTS & ACADEMIC Q&A
        # ==========================================
        with tab6:
            st.subheader("📥 Technical Deliverables & Evaluator Q&A")

            col_ex1, col_ex2 = st.columns(2)
            with col_ex1:
                st.markdown("#### 1. Learner Feature Dataset (`learner_features.csv`)")
                export_cols = ["UserID", "Age", "TotalCourses", "TotalSpending", "AverageSpending", "CategoryDiversity", "AverageCourseRating", "PaidCourseRatio", "AdvancedRatio", "LearningDepthIndex", "Cluster", "SegmentName"]
                learner_export_df = learner_df[export_cols]
                
                csv_learner = learner_export_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download learner_features.csv",
                    data=csv_learner,
                    file_name="learner_features.csv",
                    mime="text/csv"
                )
                st.dataframe(learner_export_df.head(5), use_container_width=True)

            with col_ex2:
                st.markdown("#### 2. Sample Course Recommendations (`recommendations.csv`)")
                # Generate sample recommendations export for top 30 users
                all_recs = []
                for uid in learner_df["UserID"].head(30):
                    user_recs = recommend_courses_for_user(uid, learner_df, master_df, cleaned_sheets["Courses"], cleaned_sheets["Teachers"], top_n=2)
                    for _, r in user_recs.iterrows():
                        all_recs.append({
                            "UserID": uid,
                            "RecommendedCourse": r["CourseName"],
                            "RecommendationScore": r["RecommendationScore"],
                            "Reason": r["Reason"]
                        })
                recs_export_df = pd.DataFrame(all_recs)
                csv_recs = recs_export_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download recommendations.csv",
                    data=csv_recs,
                    file_name="recommendations.csv",
                    mime="text/csv"
                )
                st.dataframe(recs_export_df.head(5), use_container_width=True)

            st.markdown("---")
            st.markdown("### ❓ Core Evaluator Questions & System Answers")
            
            top_3_recs_str = ', '.join(rec_courses['CourseName'].head(3).tolist()) if not rec_courses.empty else 'N/A'
            
            qa_html = textwrap.dedent(f"""
            <div class="course-card">
                <p style="margin-bottom:12px;"><b style="font-size:1.02rem; color:#f8fafc;">Question 1: How many types of learners are there?</b><br>
                <span style="color:#34d399; font-weight:600; font-size:0.95rem;">👉 EduPro learners were segmented into {selected_k} behavioral groups based on standardization and K-Means clustering.</span></p>

                <p style="margin-bottom:12px;"><b style="font-size:1.02rem; color:#f8fafc;">Question 2: What are those learner types?</b><br>
                <span style="color:#34d399; font-weight:600; font-size:0.95rem;">👉 {', '.join(profiles_df['SegmentName'].tolist())}.</span></p>

                <p style="margin-bottom:12px;"><b style="font-size:1.02rem; color:#f8fafc;">Question 3: What type of learner is Student {selected_user_id}?</b><br>
                <span style="color:#34d399; font-weight:600; font-size:0.95rem;">👉 Student {selected_user_id} belongs to the <b>{user_segment}</b> segment.</span></p>

                <p style="margin-bottom:12px;"><b style="font-size:1.02rem; color:#f8fafc;">Question 4: What should Student {selected_user_id} learn next?</b><br>
                <span style="color:#34d399; font-weight:600; font-size:0.95rem;">👉 Recommended courses: <b>{top_3_recs_str}</b>.</span></p>

                <p style="margin-bottom:0px;"><b style="font-size:1.02rem; color:#f8fafc;">Question 5: Why were those courses recommended?</b><br>
                <span style="color:#34d399; font-weight:600; font-size:0.95rem;">👉 They match the learner's preferred category (<b>{user_row['PreferredCategory']}</b>) and level (<b>{user_row['PreferredLevel']}</b>), and are popular among similar learners in their segment.</span></p>
            </div>
            """)
            st.markdown(qa_html, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error executing EduPro pipeline: {e}")
        st.exception(e)
else:
    st.info("Please select or upload an Excel dataset to begin.")
