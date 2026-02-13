import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, f1_score, brier_score_loss
)

RANDOM_STATE = 42

# 1) Загрузка
df = pd.read_excel("2gas_hydrate_phase_based.xlsx").copy()
assert set(["Temperature_C","Pressure_bar","Hydrate_formation"]).issubset(df.columns)

# Опционально: ограничим давление графиков до 300 bar (на метрики не влияет)
P_MAX_PLOT = 300

# 2) Признаки и цель
# Базовые сырые фичи
X_raw = df[["Temperature_C", "Pressure_bar"]].values
y = df["Hydrate_formation"].astype(int).values

# 3) Финальный тест-сплит (отложим 20% и больше к ним не прикасаемся до конца)
X_tr, X_te, y_tr, y_te = train_test_split(
    X_raw, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# 4) Преобразование к физически осмысленным фичам: φ = [ln(P), 1/T_K]
def pt_to_features(X):
    T_C = X[:, 0]
    P   = X[:, 1]
    T_K = T_C + 273.15
    invT = 1.0 / T_K
    lnP = np.log(np.clip(P, 1e-6, None))  # защита от log(0)
    # Можно добавить lnP*invT — слегка увеличит гибкость
    return np.c_[lnP, invT]

phi = FunctionTransformer(pt_to_features, validate=False)

# 5) Логистическая регрессия с кросс-валидацией гиперпараметров + калибровка вероятностей
logit_pipe = Pipeline([
    ("phi", phi),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced", random_state=RANDOM_STATE))
])

param_grid_lr = {
    "clf__C": [0.1, 0.3, 1.0, 3.0, 10.0]  # сила регуляризации
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
gs_lr = GridSearchCV(logit_pipe, param_grid_lr, cv=cv, scoring="roc_auc", n_jobs=-1)
gs_lr.fit(X_tr, y_tr)

best_lr = gs_lr.best_estimator_

# Калибровка вероятностей (Platt: sigmoid — устойчиво на маленьких выборках)
cal_lr = CalibratedClassifierCV(best_lr, cv=cv, method="sigmoid")
cal_lr.fit(X_tr, y_tr)

# 6) RandomForest с настройкой и калибровкой
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=3,
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    oob_score=True
)
param_grid_rf = {
    "n_estimators": [300, 500, 800],
    "min_samples_leaf": [1, 2, 3, 5],
    "max_depth": [None, 6, 10]
}
gs_rf = GridSearchCV(rf, param_grid_rf, cv=cv, scoring="roc_auc", n_jobs=-1)
gs_rf.fit(X_tr, y_tr)
best_rf = gs_rf.best_estimator_

cal_rf = CalibratedClassifierCV(best_rf, cv=cv, method="sigmoid")
cal_rf.fit(X_tr, y_tr)

# 7) Оценка на финальном тесте
def eval_model(name, model, X, y_true):
    y_prob = model.predict_proba(X)[:,1]
    y_pred = (y_prob >= 0.5).astype(int)
    print(f"\n{name}:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("ROC AUC:", roc_auc_score(y_true, y_prob))
    print("F1 (hydrate=1):", f1_score(y_true, y_pred, pos_label=1))
    print("Brier score (↓ лучше):", brier_score_loss(y_true, y_prob))
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} — Confusion Matrix (test)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.show()
    # Калибровка/кривые надежности
    CalibrationDisplay.from_estimator(model, X, y_true, n_bins=8)
    plt.title(f"{name} — Диаграмма надёжности (test)")
    plt.show()
    return y_prob, y_pred

yprob_lr, _ = eval_model("Logistic (φ=[lnP,1/T]) + calibration", cal_lr, X_te, y_te)
yprob_rf, _ = eval_model("RandomForest + calibration", cal_rf, X_te, y_te)

# 8) Визуализация границы для логистической модели в исходных осях (T_C, P_bar)
t_min, t_max = X_raw[:,0].min()-1, X_raw[:,0].max()+1
p_min, p_max = 0, max(P_MAX_PLOT, X_raw[:,1].max())

tt, pp = np.meshgrid(np.linspace(t_min, t_max, 400),
                     np.linspace(p_min, min(P_MAX_PLOT, p_max), 400))
grid = np.c_[tt.ravel(), pp.ravel()]
proba_lr = cal_lr.predict_proba(grid)[:,1].reshape(tt.shape)

plt.figure(figsize=(7,5))
cs = plt.contourf(tt, pp, proba_lr, levels=np.linspace(0,1,11), cmap="coolwarm", alpha=0.25)
plt.contour(tt, pp, proba_lr, levels=[0.5], colors="k", linewidths=2, linestyles="--")
plt.scatter(X_raw[:,0], X_raw[:,1], c=y, cmap="coolwarm", edgecolor="k", s=30)
plt.xlabel("Temperature (°C)"); plt.ylabel("Pressure (bar)")
plt.ylim(0, P_MAX_PLOT)
plt.title("Экспоненциальная граница: Logistic на [ln(P), 1/T_K] (калибровано)")
plt.colorbar(cs, label="P(hydrate=1)")
plt.show()

# 9) Для понимания: сколько итераций реально понадобилось логистической регрессии
print("Число итераций до сходимости у лучшей LR (по фолдам):",
      np.unique([est.named_steps["clf"].n_iter_ for est in gs_lr.cv_results_["param_clf__C"]]))
