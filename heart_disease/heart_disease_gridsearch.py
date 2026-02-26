"""
heart_disease_gridsearch.py
Быстрые модели с оптимизацией параметров (GridSearchCV)
Каждая модель сохраняет результаты в отдельный JSON файл
"""

import pandas as pd
import numpy as np
import time
import json
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🔍 БЫСТРЫЕ МОДЕЛИ С GRIDSEARCHCV")
print("=" * 80)

# ─────────────────────────────────────────────────────────────
# ЗАГРУЗКА ДАННЫХ
# ─────────────────────────────────────────────────────────────
df = pd.read_csv('train_fixed.csv')
X = df.drop('Heart Disease', axis=1)
y = df['Heart Disease'].map({'Absence': 0, 'Presence': 1})

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=10000, random_state=42, stratify=y
)

print(f"✅ Train: {len(X_train):,} | Test: {len(X_test):,}")
print("=" * 80)

# ─────────────────────────────────────────────────────────────
# МОДЕЛЬ 1: Logistic Regression + GridSearchCV
# ─────────────────────────────────────────────────────────────
print("\n1️⃣ Logistic Regression + GridSearchCV...")
start = time.time()

param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'class_weight': [None, 'balanced']
}

grid_lr = GridSearchCV(
    LogisticRegression(max_iter=500, random_state=42),
    param_grid_lr, cv=5, scoring='f1', n_jobs=-1, verbose=1
)
grid_lr.fit(X_train, y_train)

elapsed = time.time() - start
y_pred = grid_lr.predict(X_test)
y_proba = grid_lr.predict_proba(X_test)[:, 1]

results_lr = {
    'model': 'LogisticRegression_GridSearch',
    'time_minutes': elapsed / 60,
    'best_params': grid_lr.best_params_,
    'cv_best_score': grid_lr.best_score_,
    'test_accuracy': accuracy_score(y_test, y_pred),
    'test_f1': f1_score(y_test, y_pred),
    'test_roc_auc': roc_auc_score(y_test, y_proba)
}

with open('results_logistic_regression.json', 'w', encoding='utf-8') as f:
    json.dump(results_lr, f, indent=2, default=str)

print(f"   ⏱️ {elapsed/60:.2f} мин | F1: {results_lr['test_f1']:.4f}")
print(f"   📋 Параметры: {grid_lr.best_params_}")

# ─────────────────────────────────────────────────────────────
# МОДЕЛЬ 2: Random Forest + GridSearchCV
# ─────────────────────────────────────────────────────────────
print("\n2️⃣ Random Forest + GridSearchCV...")
start = time.time()

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'class_weight': [None, 'balanced']
}

grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_rf, cv=5, scoring='f1', n_jobs=-1, verbose=1
)
grid_rf.fit(X_train, y_train)

elapsed = time.time() - start
y_pred = grid_rf.predict(X_test)
y_proba = grid_rf.predict_proba(X_test)[:, 1]

results_rf = {
    'model': 'RandomForest_GridSearch',
    'time_minutes': elapsed / 60,
    'best_params': grid_rf.best_params_,
    'cv_best_score': grid_rf.best_score_,
    'test_accuracy': accuracy_score(y_test, y_pred),
    'test_f1': f1_score(y_test, y_pred),
    'test_roc_auc': roc_auc_score(y_test, y_proba)
}

with open('results_random_forest.json', 'w', encoding='utf-8') as f:
    json.dump(results_rf, f, indent=2, default=str)

print(f"   ⏱️ {elapsed/60:.2f} мин | F1: {results_rf['test_f1']:.4f}")
print(f"   📋 Параметры: {grid_rf.best_params_}")

# ─────────────────────────────────────────────────────────────
# МОДЕЛЬ 3: LinearSVC + GridSearchCV
# ─────────────────────────────────────────────────────────────
print("\n3️⃣ LinearSVC + GridSearchCV...")
start = time.time()

param_grid_svc = {
    'C': [0.01, 0.1, 1, 10],
    'class_weight': [None, 'balanced'],
    'max_iter': [1000, 2000]
}

grid_svc = GridSearchCV(
    LinearSVC(random_state=42),
    param_grid_svc, cv=5, scoring='f1', n_jobs=-1, verbose=1
)
grid_svc.fit(X_train, y_train)

elapsed = time.time() - start
y_pred = grid_svc.predict(X_test)

results_svc = {
    'model': 'LinearSVC_GridSearch',
    'time_minutes': elapsed / 60,
    'best_params': grid_svc.best_params_,
    'cv_best_score': grid_svc.best_score_,
    'test_accuracy': accuracy_score(y_test, y_pred),
    'test_f1': f1_score(y_test, y_pred),
    'test_roc_auc': 0
}

with open('results_linear_svc.json', 'w', encoding='utf-8') as f:
    json.dump(results_svc, f, indent=2, default=str)

print(f"   ⏱️ {elapsed/60:.2f} мин | F1: {results_svc['test_f1']:.4f}")
print(f"   📋 Параметры: {grid_svc.best_params_}")

# ─────────────────────────────────────────────────────────────
# МОДЕЛЬ 4: Voting Ensemble (из лучших параметров)
# ─────────────────────────────────────────────────────────────
print("\n4️⃣ Voting Ensemble...")
start = time.time()

lr_best = LogisticRegression(max_iter=500, random_state=42, **grid_lr.best_params_)
rf_best = RandomForestClassifier(random_state=42, **grid_rf.best_params_)

voting = VotingClassifier(
    estimators=[('lr', lr_best), ('rf', rf_best)],
    voting='soft', n_jobs=-1
)
voting.fit(X_train, y_train)

elapsed = time.time() - start
y_pred = voting.predict(X_test)
y_proba = voting.predict_proba(X_test)[:, 1]

results_voting = {
    'model': 'VotingEnsemble',
    'time_minutes': elapsed / 60,
    'test_accuracy': accuracy_score(y_test, y_pred),
    'test_f1': f1_score(y_test, y_pred),
    'test_roc_auc': roc_auc_score(y_test, y_proba)
}

with open('results_voting_ensemble.json', 'w', encoding='utf-8') as f:
    json.dump(results_voting, f, indent=2, default=str)

print(f"   ⏱️ {elapsed/60:.2f} мин | F1: {results_voting['test_f1']:.4f}")

# ─────────────────────────────────────────────────────────────
# ИТОГОВЫЙ ОТЧЁТ
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("📊 ИТОГОВЫЙ ОТЧЁТ")
print("=" * 80)

all_results = [results_lr, results_rf, results_svc, results_voting]
for res in sorted(all_results, key=lambda x: x['test_f1'], reverse=True):
    print(f"{res['model']:<35} F1: {res['test_f1']:.4f} | {res['time_minutes']:.2f} мин")

print("\n✅ Все результаты сохранены в files: results_*.json")
print("=" * 80)