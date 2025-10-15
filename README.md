# OIBSIP_dataanalytics_05

# 🔒 Task 7: Credit Card Fraud Detection (Imbalanced Classification)

## **Internship**
**Oasis Infobyte** - Data Science Internship (Virtual)

## **Project Goal**
The primary objective of this high-stakes project was to build and optimize a machine learning classification system to accurately detect fraudulent credit card transactions. The core challenge involved tackling the severe **class imbalance** (very few fraud cases) inherent in financial data and maximizing metrics crucial for risk management, such as **Precision and Recall**.

## **Dataset**
The project utilized the `creditcard.csv` dataset, which contains anonymized transaction features (V1-V28), transaction `Time`, `Amount`, and the binary target variable, `Class` (0 = legitimate, 1 = fraud).

## **Key Steps & Methodology**

1.  **Exploration & Preprocessing:**
    * Analyzed the extreme **class imbalance**.
    * Created simple **Feature Engineering** examples (`Amount_log`, `V1_V2_interaction`).
    * Applied **Standard Scaling** to the `Time` and `Amount` features.
    * Performed a **Stratified Train/Test Split**.

2.  **Model Building & Comparison:**
    * Compared four distinct classification models, utilizing techniques like `class_weight='balanced'` to handle the imbalance:
        * **Logistic Regression** (Baseline)
        * **Random Forest Classifier** (Top Performer)
        * **Decision Tree Classifier**
        * **MLP Neural Network**

3.  **Advanced Evaluation:**
    * Used **ROC AUC Score** and the **Precision-Recall Curve (AP Score)** as primary metrics, given their suitability for imbalanced data.
    * Generated detailed **Classification Reports** to monitor Precision and Recall for the minority (fraud) class.

4.  **Optimization for Production:**
    * Implemented **Undersampling** as an additional technique to balance the training data for model comparison.
    * Performed **Threshold Tuning** on the Random Forest model to maximize the **F1 Score**, yielding an optimal threshold for real-time deployment.

5.  **Deployment Preparation:**
    * Saved the final optimized **Random Forest Model** and the **Standard Scaler** using the `joblib` library.

## **Tools and Libraries**
* **Python**
* **Pandas & NumPy:** Data manipulation.
* **Scikit-learn (sklearn):** `train_test_split`, `StandardScaler`, `LogisticRegression`, `RandomForestClassifier`, `MLPClassifier`, `precision_recall_curve`, `roc_auc_score`, `classification_report`.
* **Matplotlib & Seaborn:** Visualization (Distribution plots, ROC curves, PR curves).
* **Joblib:** For saving the final model and scaler.

## **Actionable Conclusions**
* **Optimal Model:** The **Random Forest Classifier** provided the best balance of Precision and Recall.
* **Risk Management Focus:** Through threshold tuning, we established a precise cut-off point that minimizes false negatives (undetected fraud) while controlling the number of false positives.

## **File Structure**
* `fraud_detection.py` - Primary Python script containing the ML pipeline, analysis, and evaluations.
* `creditcard.csv` - The original dataset file.
* `fraud_model_rf.joblib` - Saved Random Forest model (output).
* `cc_scaler.joblib` - Saved Standard Scaler object (output).
* `README.md` - This documentation file.
