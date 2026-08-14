"""
Test pipeline script to verify EduPro_Dataset.xlsx loading and execution.
"""
import sys
import os

try:
    import pandas as pd
    from src.pipeline import load_data, validate_data, create_master_dataset, create_learner_features, scale_learner_features, evaluate_kmeans_range, train_kmeans_clustering, calculate_cluster_profiles, recommend_courses_for_user

    dataset_path = "EduPro_Dataset.xlsx"
    print(f"Loading dataset: {dataset_path}...")

    sheets = load_data(dataset_path)
    print("Sheets loaded:", list(sheets.keys()))

    report, cleaned_sheets = validate_data(sheets)
    print("\n--- Validation Report ---")
    for s_name, rep in report.items():
        print(f"Sheet {s_name}: {rep['row_count']} rows, Duplicates: {rep['duplicate_count']}, Invalid Dates: {rep['invalid_dates']}")

    master_df = create_master_dataset(cleaned_sheets)
    print(f"\nMaster Dataset created: {len(master_df)} transaction rows.")

    learner_df = create_learner_features(master_df, cleaned_sheets["Users"])
    print(f"Learner Features created: {len(learner_df)} user rows.")
    print("Sample learner features:")
    print(learner_df[['UserID', 'UserName', 'Age', 'TotalCourses', 'TotalSpending', 'PreferredCategory', 'LearningDepthIndex']].head())

    X_scaled, scaler = scale_learner_features(learner_df)
    print(f"\nFeature scaling complete. Matrix shape: {X_scaled.shape}")

    eval_df = evaluate_kmeans_range(X_scaled, k_range=[2, 3, 4, 5, 6])
    print("\n--- K-Means K Evaluation ---")
    print(eval_df)

    learner_clustered, kmeans_model, pca = train_kmeans_clustering(learner_df, X_scaled, k=4)
    print("\nK-Means K=4 clustering completed.")

    profiles_df, segment_map = calculate_cluster_profiles(learner_clustered)
    print("\n--- Cluster Segment Profiles ---")
    print(profiles_df[['Cluster', 'SegmentName', 'LearnerCount', 'AvgCourses', 'AvgSpending', 'AvgLearningDepth', 'PreferredCategory']])

    test_user_id = learner_clustered["UserID"].iloc[0]
    recs = recommend_courses_for_user(test_user_id, learner_clustered, master_df, cleaned_sheets["Courses"], cleaned_sheets["Teachers"], top_n=3)
    print(f"\n--- Top 3 Course Recommendations for {test_user_id} ---")
    print(recs[['CourseID', 'CourseName', 'CourseCategory', 'CourseLevel', 'CoursePrice', 'RecommendationScore', 'MatchingTeacher']])

    print("\nSUCCESS: All pipeline steps executed without error!")

except Exception as e:
    print(f"Pipeline Execution Error: {e}")
    import traceback
    traceback.print_exc()
