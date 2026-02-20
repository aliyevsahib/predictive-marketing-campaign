# Predictive Marketing Campaign & Customer Segmentation 🎯

## 📌 Project Overview
This project is an end-to-end Data Science pipeline designed to analyze customer behavior, segment them into distinct personality clusters, and build a highly optimized predictive model to maximize the ROI (Return on Investment) of targeted marketing campaigns. 

By transitioning from a "spray-and-pray" marketing approach to a data-driven predictive strategy, this project demonstrates how machine learning can significantly boost campaign revenue by effectively identifying hidden buyers.

## 🗄️ Dataset
* **Source:** Kaggle (Customer Personality Analysis)
* **Description:** The dataset contains structured data regarding customer demographics, historical spending habits across various product categories (wines, fruits, meat, etc.), and their response to previous marketing campaigns.

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`, `imblearn` (SMOTE)
* **Visualization:** `matplotlib`, `seaborn`

## 🚀 Methodology & Pipeline

### 1. Data Engineering & Cleaning
* Imputed missing values (e.g., using median for Income to avoid outlier distortion).
* Engineered powerful new features to strengthen the algorithmic signals: `Total_Spent`, `Total_Children`, `Customer_Days`, and simplified `Marital_Status`.
* Removed anomalies and zero-variance columns to reduce noise.

### 2. Exploratory Data Analysis (EDA) & Statistical Insights
* Conducted correlation analysis to identify the primary mathematical drivers of spending.
* **Key Insight:** Proved statistically that household spending on luxury items (like wine) drastically decreases as the number of children increases.

<img width="1180" height="684" alt="4" src="https://github.com/user-attachments/assets/b3872a3d-35ea-4048-a893-0c06294627a0" />


### 3. Unsupervised Learning (Customer Segmentation)
* Scaled features using `StandardScaler` to ensure distance-based algorithms worked correctly.
* Applied **K-Means Clustering** to segment the customer base into 4 distinct "Personalities" based on Income, Total Spent, Age, and Dependents. 
* This segmentation allows for highly personalized future marketing efforts.

### 4. Supervised Learning (Predictive Analytics)
* **The Goal:** Predict whether a customer will accept (`1`) or reject (`0`) the next marketing campaign.
* **The Challenge:** Highly imbalanced data (only 16% of customers previously accepted campaigns). An initial Random Forest model played it too safe, missing 76% of actual buyers.
* **The Optimization:** * Implemented **SMOTE** (Synthetic Minority Over-sampling Technique) to synthetically balance the training data.
    * Tuned the **Probability Threshold** (lowered to 0.35) to prioritize *Recall* over *Precision*.
* **Business Impact:** By tuning the algorithm to be more aggressive in hunting for "YES" customers, the model successfully captured a significantly higher percentage of buyers. In a real-world scenario, this specific optimization translates directly to capturing lost revenue, vastly outperforming random guessing or baseline models.

<img width="576" height="384" alt="8" src="https://github.com/user-attachments/assets/33aa2cf2-86dd-45d6-a536-08b050863beb" />


## 💡 Key Business Takeaways
1. **Targeted Spending:** High-income customers with zero children are the most lucrative demographic for premium products.
2. **Loyalty Matters:** Feature importance analysis revealed that `Customer_Days` (tenure) and `Total_Spent` are the strongest predictors of campaign acceptance.
3. **Imbalanced Data Strategy:** In digital marketing, the cost of a false positive (sending an email to a non-buyer) is practically zero, but the reward of a true positive is high. Optimizing models for **Recall** via SMOTE and threshold adjustment is critical for maximizing pure revenue.

## 💻 How to Run This Project
1. Clone this repository.
2. Ensure you have the required libraries installed (`pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn`).
3. Run the Jupyter Notebook `classification_of_customer_personality.ipynb` sequentially to view the analysis, visualizations, and model training.
