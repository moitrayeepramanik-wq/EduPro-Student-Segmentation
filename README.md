# 🎓 EduPro Student Segmentation & Personalized Course Recommendation MVP

An interactive, beginner-friendly **Machine Learning & Streamlit Web Application** designed to understand learner behavior, cluster students using K-Means, and deliver personalized course recommendations.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-K--Means-orange)

---

## 🌟 Key Features

1. **Transaction Master Dataset**: Merges `Transactions`, `Users`, `Courses`, and `Teachers` into a fact-level analytical dataset.
2. **Learner Feature Engineering**: Derives user-level metrics (`TotalCourses`, `TotalSpending`, `AverageSpending`, `PreferredCategory`, `PreferredLevel`, `CategoryDiversity`, `LearningDepthIndex`, `PaidCourseRatio`, `AverageCourseDuration`).
3. **K-Means Student Segmentation**: Standardizes numerical features using `StandardScaler` and clusters learners into intuitive behavioral groups (*Beginner Explorers*, *Developing Learners*, *Advanced Specialists*, *High-Value Power Learners*).
4. **Personalized Course Recommendation Engine**: Excludes completed courses and ranks candidate courses based on category match, level progression, segment popularity, and quality ratings. Includes explicit **"Why Recommended?"** reasoning.
5. **Interactive Dashboard**: Built with Streamlit, custom dark-mode glassmorphism styling, Plotly charts (Elbow Curve, Silhouette plot, 2D PCA Cluster Map), dynamic learner profile selector, and CSV data export capabilities.

---

## 📁 Repository Structure

```text
EduPro/
├── app.py                  # Main Streamlit Web Application
├── generate_dataset.py     # Script to generate sample EduPro dataset
├── test_pipeline.py        # End-to-end pipeline verification test
├── requirements.txt        # Python package dependencies
├── MASTER_PROMPT.md        # Detailed project specification master prompt
├── .gitignore              # Ignored files for Git
└── src/
    ├── pipeline.py         # Data processing, feature engineering, clustering, recommendations
    └── utils.py            # Streamlit UI themes, CSS styling, Plotly visualization helpers
```

---

## 🚀 Quick Start (Local Setup)

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR-USERNAME/EduPro.aspx.git
cd EduPro
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Sample Dataset (Optional)
```bash
python generate_dataset.py
```

### 4. Launch Streamlit Application
```bash
python -m streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## ☁️ Deploying to Streamlit Cloud

1. Push this repository to **GitHub**.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and sign in with your GitHub account.
3. Click **New App**, select your `EduPro` repository, set Main file path to `app.py`, and click **Deploy**!
