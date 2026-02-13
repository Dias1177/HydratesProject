import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# new:
df = pd.read_excel("2gas_hydrate_phase_based.xlsx")

# X = [T_C, P_bar], y — как у вас
X = df[["Temperature_C", "Pressure_bar"]].values
y = df["Hydrate_formation"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Преобразование к физически осмысленным фичам: φ = [ln(P), 1/T_K]
def pt_to_features(X):
    T_C = X[:, 0]
    P   = X[:, 1]
    T_K = T_C + 273.15
    invT = 1.0 / T_K
    lnP = np.log(np.clip(P, 1e-3, None))  # защита от log(0)
    # Можно добавить взаимодействие lnP*invT для чуть большей гибкости:
    # return np.c_[lnP, invT, lnP*invT]
    return np.c_[lnP, invT]

pipe_exp = Pipeline([
    ("phi", FunctionTransformer(pt_to_features, validate=False)),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, random_state=42))
])

pipe_exp.fit(X_train, y_train)
y_pred = pipe_exp.predict(X_test)
print("Exp-Logistic accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Рисуем границу в осях (T_C, P_bar) — она получится изогнутой (экспоненциальной)
t_min, t_max = X[:,0].min()-1, X[:,0].max()+1
p_min, p_max = 0, max(300, X[:,1].max())  # ограничим до 300 bar при желании
tt, pp = np.meshgrid(np.linspace(t_min, t_max, 400),
                     np.linspace(p_min, min(300, p_max), 400))
grid = np.c_[tt.ravel(), pp.ravel()]

proba = pipe_exp.predict_proba(grid)[:,1].reshape(tt.shape)

plt.figure(figsize=(6,5))
# фоновая заливка вероятности
cs = plt.contourf(tt, pp, proba, levels=np.linspace(0,1,11), cmap="coolwarm", alpha=0.25)
# линия границы 0.5
plt.contour(tt, pp, proba, levels=[0.5], colors="k", linewidths=2)
plt.scatter(X[:,0], X[:,1], c=y, cmap="coolwarm", edgecolor="k", s=30)
plt.xlabel("Temperature (°C)")
plt.ylabel("Pressure (bar)")
plt.ylim(0, 300)  # если хотите видеть только до 300 bar
plt.title("Экспоненциальная граница: Logistic на [ln(P), 1/T]")
plt.colorbar(cs, label="P(hydrate=1)")
plt.show()
