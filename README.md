# Credit Card Fraud Detection

## Overview
This project aims to detect fraudulent credit card transactions using machine learning techniques. The dataset consists of real-world transactions, with highly imbalanced classes, where fraudulent transactions are significantly fewer than legitimate ones.

## Dataset
- **Source:** [Kaggle](https://www.kaggle.com/code/saumitratandon/credit-card-fraud-detection)
- **Features:**
  - Transaction details (e.g., amount, time)
  - Anonymized numerical features (e.g., PCA transformed values)
  - Target variable:class [ `0` (Legitimate), `1` (Fraudulent)]

## Project Workflow
### 1. Data Preprocessing
- Handling missing values (if any)
- Exploratory Data Analysis (EDA) for insights
- Normalization and scaling of numerical features
- Addressing class imbalance using techniques like SMOTE or undersampling

### 2. Feature Engineering
- Feature selection to improve model efficiency
- Generating new features if needed

### 3. Model Training
- Splitting dataset into training and testing sets
- Evaluating multiple machine learning models:
  - Logistic Regression
  - Random Forest
  - Decision Tree
- Hyperparameter tuning for optimal performance

### 4. Model Evaluation
- Metrics used:
  - Accuracy
  - Precision, Recall, F1-score
  - AUC-ROC Curve
  - Precision, Recall Trade-off
- Analyzing model performance for fraud detection


### **Results and Conclusion - Credit Card Fraud Detection (Cost-Sensitive Model)**  

#### **Results:**  
The cost-sensitive model has improved performance in detecting fraudulent transactions while maintaining high accuracy.  

- **Confusion Matrix:**  
  - **True Positives (Fraud Detected Correctly):** 76  
  - **False Negatives (Fraud Missed):** 22  
  - **True Negatives (Non-Fraud Detected Correctly):** 56,860  
  - **False Positives (Incorrectly Marked as Fraud):** 4  

- **Performance Metrics:**  
  - **Accuracy:** **99.95%**  
  - **Non-Fraudulent Transactions (Class 0):**  
    - Precision: **1.00**, Recall: **1.00**, F1-Score: **1.00**  
  - **Fraudulent Transactions (Class 1):**  
    - Precision: **0.95**, Recall: **0.78**, F1-Score: **0.85**  

- **Averages:**  
  - **Macro Avg F1-Score:** **0.93**  
  - **Weighted Avg F1-Score:** **1.00**  

#### **Conclusion:**  
The **cost-sensitive approach improved fraud detection precision (0.95)**, reducing false positives while still maintaining a good balance. However, the **recall for fraudulent transactions (0.78)** suggests that some fraud cases are still missed (**22 false negatives**). This trade-off is expected since the model is optimized to minimize both false positives and false negatives.  

To further enhance fraud detection, **techniques like anomaly detection, SMOTE (Synthetic Minority Oversampling), threshold tuning, or ensemble models** could be explored. Overall, the cost-sensitive model provides a strong balance between accuracy and fraud detection effectiveness, making it suitable for deployment in real-world applications. 🚀


## Dependencies
- Python 3.x
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- Imbalanced-learn (for handling imbalanced datasets)


## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebooks to preprocess data and train models.
4. (Optional) Deploy the model using Flask/Django.


---
I hope this documentation provides a clear and comprehensive guide for understanding and replicating the project. 

