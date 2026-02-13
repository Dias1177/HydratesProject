# HydratesProject
Gas Hydrate Formation Prediction in Pipelines using Machine Learning

This project focuses on predicting gas hydrate formation in oil and gas pipelines using machine learning models.
The goal is to reduce аварийные риски, экономические потери и экологический ущерб за счёт раннего выявления опасных режимов.

📌 Problem Statement

Gas hydrates form under:

Low temperature

High pressure

Presence of water

Their formation can lead to:

Pipeline blockage

Pressure buildup

Mechanical failure

Oil/gas spills

Traditional prevention methods (methanol injection, heating, pressure regulation) are expensive and not always applied precisely.

This project explores whether machine learning can predict hydrate formation based on operational parameters.

🎯 Objective

Develop and compare machine learning models that predict hydrate formation using:

Temperature (°C)

Pressure (bar)

The task is formulated as a binary classification problem:

1 — Hydrate formation (dangerous regime)

0 — No hydrate formation

Special attention is given to minimizing false negatives, since missing a hydrate event can lead to аварии.

🧠 Models Used

Two models were implemented and compared:

1️⃣ Logistic Regression

Interpretable

Linear decision boundary

Allows coefficient analysis

2️⃣ Random Forest

Non-linear model

Robust to noise

Provides feature importance

⚙️ Pipeline

Data loading from Excel

Feature selection (Temperature, Pressure)

Train/Test split (80/20, stratified)

Model training

Performance evaluation:

Accuracy

Precision

Recall

F1-score

Confusion matrix

Decision boundary visualization

Feature importance analysis

📊 Evaluation Metrics

Since this is a safety-critical task, overall accuracy is not the only focus.

Important metrics:

Recall for hydrate class

False Negative Rate (FNR)

Confusion Matrix analysis

Missing a hydrate (FN) is considered more critical than a false alarm (FP).

📈 Results

Both models achieved high predictive performance (~90% accuracy).

Observations:

Logistic Regression showed better control over false negatives.

Random Forest captured non-linear patterns but sometimes missed rare dangerous states.

Temperature and Pressure were confirmed as dominant features.

📂 Repository Structure
├── data/
│   └── 2gas_hydrate_phase_based.xlsx
├── notebooks/
│   └── hydrate_prediction.ipynb
├── src/
│   └── model_training.py
├── README.md

🔬 Future Improvements

Add engineered physical features (distance to hydrate phase boundary)

Implement cost-sensitive learning

Optimize classification threshold

Extend to time-series prediction

Apply LSTM for dynamic behavior modeling

🌍 Practical Impact

This approach can:

Reduce unnecessary inhibitor usage

Lower operational costs

Improve environmental safety

Support real-time risk monitoring systems

📌 Author

Sanakul Salim
Nazarbayev Intellectual School (Astana)
2025
