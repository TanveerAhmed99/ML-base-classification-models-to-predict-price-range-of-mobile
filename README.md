📱 Mobile Price Range Classification
This project presents a classification model to predict the price range of mobile phones based on various features such as RAM, battery power, camera specs, and more. It leverages machine learning algorithms and data preprocessing techniques to train accurate classifiers.

🧠 Project Overview
Predicting the price category of a smartphone based on its specifications is a classic classification problem. This notebook walks through the end-to-end pipeline of loading the dataset, analyzing the data, preprocessing, applying models, and evaluating their performance.

💡 Objective
To classify mobile phones into 4 price categories:

0 = Low Cost

1 = Medium Cost

2 = High Cost

3 = Very High Cost

📁 Dataset
Source: Included as data.csv (assumed to be stored locally or on Colab)

Size: 2000 rows × 21 columns

Target variable: price_range (0 to 3)

🔧 Features Used
battery_power, ram, px_height, px_width, mobile_wt, talk_time, etc.

Binary features: dual_sim, four_g, three_g, wifi, touch_screen, etc.

📊 Exploratory Data Analysis (EDA)
Distribution of price_range

Feature correlation heatmap

Count plots and box plots for feature importance

🔎 Data Preprocessing
Checked for null values

Standardized features (if applicable)

Feature selection techniques may be applied

🧪 Classification Models Applied
Logistic Regression

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

Decision Tree

Random Forest

Naive Bayes

Each model is evaluated based on:

Accuracy Score

Confusion Matrix

Classification Report

🧾 Evaluation Metrics
Performance comparison includes:

Training vs Testing Accuracy

Precision, Recall, F1-score

Visualization of confusion matrices

📈 Results
Among the tested models, ensemble methods (like Random Forest) often deliver higher accuracy due to their robustness and ability to reduce overfitting.

🧰 Libraries Used
pandas, numpy

matplotlib, seaborn

sklearn for ML models and metrics

🚀 Getting Started
To run this notebook:

Clone the repo:

bash
Copy
Edit
git clone https://github.com/yourusername/mobile-price-classification.git
cd mobile-price-classification
Install dependencies:

bash
Copy
Edit
pip install pandas numpy matplotlib seaborn scikit-learn
Open and run the Jupyter notebook:

bash
Copy
Edit
jupyter notebook Classification_Models.ipynb
📜 License
This project is released under the MIT License.
