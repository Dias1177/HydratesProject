Gas Hydrate Formation Prediction in Pipelines using Machine Learning

This project explores the use of machine learning to predict gas hydrate formation in oil and gas pipelines. Gas hydrates form under low temperatures and high pressures in the presence of water. Their formation can lead to pipeline blockages, pressure buildup, mechanical damage, production shutdowns, and environmental incidents.

Traditional prevention methods such as methanol injection, heating, and pressure regulation are often expensive and applied uniformly along the pipeline rather than precisely where risk is highest. This project investigates whether machine learning can help detect dangerous operating regimes in advance and support smarter prevention strategies.

The task is formulated as a binary classification problem. The model predicts whether hydrate formation will occur (1 — dangerous regime) or not (0 — safe regime) based on operational parameters. In this version of the project, temperature (°C) and pressure (bar) are used as input features.

Two machine learning models are implemented and compared:

Logistic Regression — chosen for its interpretability and ability to clearly show how each feature influences the prediction.

Random Forest — chosen for its ability to capture nonlinear relationships and interactions between features.

The workflow includes data loading, preprocessing, train-test splitting (80/20 with stratification), model training, evaluation, and visualization of decision boundaries. Model performance is evaluated using accuracy, precision, recall, F1-score, and confusion matrices.

Because this is a safety-related problem, minimizing false negatives (missing a dangerous hydrate event) is more important than simply maximizing accuracy. Special attention is given to recall for the hydrate class and analysis of the false negative rate.

Both models achieve high predictive performance (around 90% accuracy). Logistic Regression provides clearer interpretability and strong control over false negatives, while Random Forest captures more complex decision boundaries. Feature importance analysis confirms that temperature and pressure are the dominant factors influencing hydrate formation, which is consistent with physical theory.

This project demonstrates how machine learning can complement traditional engineering methods and potentially improve safety, reduce unnecessary inhibitor usage, and support risk-based operational decision-making in the oil and gas industry.

Future improvements may include adding physically derived features (such as distance to hydrate phase boundary), cost-sensitive learning to penalize dangerous errors more heavily, threshold optimization for safety-focused deployment, and extension to time-series models to account for dynamic operating conditions.

Author:
Sanakul Salim, Zhumatayev Dias
Nazarbayev Intellectual School, Astana
2025
