import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_excel("2gas_hydrate_phase_based.xlsx")

X = df[["Temperature_C", "Pressure_bar"]].values
y = df["Hydrate_formation"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_rf)
print("Random Forest accuracy:", acc_rf)
print("Random Forest classification report:\n", classification_report(y_test, y_rf))

# важности признаков RF
print("RF feature importances:", dict(zip(["Temperature_C","Pressure_bar"], rf.feature_importances_)))

#Logistic Regression
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)
y_lr = lr.predict(X_test_s)
acc_lr = accuracy_score(y_test, y_lr)
print("\nLogistic Regression accuracy:", acc_lr)
print("Logistic classification report:\n", classification_report(y_test, y_lr))

# коэффициенты логистической регрессии (для стандартизированных признаков)
print("LR coefficients (scaled features):", lr.coef_, " intercept:", lr.intercept_)

# Матрицы ошибок
cm_rf = confusion_matrix(y_test, y_rf)
cm_lr = confusion_matrix(y_test, y_lr)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
plt.title("RF Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.subplot(1,2,2)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens', xticklabels=[0,1], yticklabels=[0,1])
plt.title("LR Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Decision boundary (наглядно)

xx0, xx1 = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 300),
                       np.linspace(X[:,1].min()-5, X[:,1].max()+5, 300))
grid = np.c_[xx0.ravel(), xx1.ravel()]

Z_rf = rf.predict(grid).reshape(xx0.shape)

grid_s = scaler.transform(grid)
Z_lr = lr.predict(grid_s).reshape(xx0.shape)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.contourf(xx0, xx1, Z_rf, alpha=0.25, cmap="coolwarm")
plt.scatter(X[:,0], X[:,1], c=y, cmap="coolwarm", edgecolor='k', s=35)
plt.title("RF decision boundary")
plt.xlabel("Temperature_C")
plt.ylabel("Pressure_bar")

plt.subplot(1,2,2)
plt.contourf(xx0, xx1, Z_lr, alpha=0.25, cmap="coolwarm")
plt.scatter(X[:,0], X[:,1], c=y, cmap="coolwarm", edgecolor='k', s=35)
plt.title("Logistic decision boundary")
plt.xlabel("Temperature_C")
plt.ylabel("Pressure_bar")
plt.show()

pd.value_counts(y)
