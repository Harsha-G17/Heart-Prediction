****🫀 Heart Disease Prediction using Machine Learning**

This project uses a machine learning approach to predict the presence of heart disease in patients based on medical attributes. The model is trained on a dataset with various clinical features and aims to assist in early diagnosis.

📂 Project Structure
Heart Prediction.py: Main script for data loading, preprocessing, model training, evaluation, and prediction.

Dataset used: Heart Prediction Quantum Dataset.csv (expected at Downloads/ path).

🛠️ Features Used
The model uses features such as:

Age

Sex

Resting Blood Pressure

Cholesterol Level

Maximum Heart Rate Achieved

Other medical indicators (e.g., ST depression)

📌 Note: The script predicts "Cancer" in the print statements but is actually built for Heart Disease Prediction. You should change "Cancer" to "Heart Disease" for clarity.

🧪 ML Pipeline
Data Loading

CSV loaded via pandas.

Data Preprocessing

Handles missing values and duplicates.

Splits features (X) and labels (y).

Modeling

RandomForestClassifier used from sklearn.ensemble.

80-20 train-test split.

Features are optionally scaled using StandardScaler.

Evaluation

Accuracy Score

Confusion Matrix

Classification Report

Feature Importance visualization using matplotlib.

📊 Model Performance
The model provides:

High classification accuracy.

Feature importance insights via horizontal bar chart.

Ability to make predictions on new patient data.

🧠 Sample Prediction
python
Copy
Edit
new_data = np.array([[45,1,120,200,80,0.7]])
prediction = model.predict(new_data)
print("Predicted Heart Disease Status:", "Heart Disease" if prediction[0]==1 else "No Heart Disease")
📈 Visualization
Feature importance is visualized using bar charts to understand which features contribute most to prediction.

🧰 Requirements
Python 3.8+

Libraries:

numpy

pandas

matplotlib

scikit-learn

Install them using:

bash
Copy
Edit
pip install numpy pandas matplotlib scikit-learn
⚠️ To Do
Fix incorrect label in print statement from “Cancer” to “Heart Disease”.

Add more robust preprocessing (e.g., outlier removal, imputation).

Add cross-validation for improved reliability.

📬 Contact
For suggestions or queries, please reach out to the project author.

**
