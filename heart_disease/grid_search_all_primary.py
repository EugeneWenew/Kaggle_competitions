"""
primary_gridsearch.py
Первичный GridSearchCV для всех моделей
(будет еще второй, уточняющий прогон)
"""

import pandas as pd
import numpy as np
import time
import json
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🔍 ПЕРВИЧНЫЙ GRIDSEARCHCV — ВСЕ МОДЕЛИ")
print("=" * 80)
print(f"🕐 Начало: {time.strftime('%H:%M:%S')}")

total_start = time.time()

# ─────────────────────────────────────────────────────────────
# Загрузка данных
# ─────────────────────────────────────────────────────────────
df = pd.read_csv('train_fixed.csv')
X = df.drop('Heart Disease', axis=1)
y = df['Heart Disease'].map({'Absence': 0, 'Presence': 1})

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=10000, random_state=42, stratify=y
)

print(f"\n✅ Train: {len(X_train):,} | Test: {len(X_test):,}")
print("=" * 80)

# Базовые результаты (без GridSearch)
baseline = {
    'LightGBM': 0.8726,
    'XGBoost': 0.8720,
    'VotingEnsemble': 0.8693,
    'LogisticRegression': 0.8656,
    'LinearSVC': 0.8648
}

all_results = []

# ─────────────────────────────────────────────────────────────
# МОДЕЛЬ 1: LogisticRegression + GridSearchCV
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("1️⃣ LogisticRegression + GridSearchCV")
print("=" * 80)
start = time.time()

param_grid_lr = {
    'C': [0.1, 0.5, 1, 2, 5, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'class_weight': [None, 'balanced'],
    'max_iter': [500, 1000]
}

grid_lr = GridSearchCV(
    LogisticRegression(random_state=42, n_jobs=-1),
    param_grid_lr, cv=5, scoring='f1', n_jobs=-1, verbose=1
)
grid_lr.fit(X_train, y_train)

elapsed = time.time() - start
y_pred = grid_lr.predict(X_test)
y_proba = grid_lr.predict_proba(X_test)[:, 1]

results_lr = {
    'model': 'LogisticRegression_GridSearch',
    'time_min': elapsed / 60,
    'best_params': grid_lr.best_params_,
    'cv_best_score': float(grid_lr.best_score_),
    'test_accuracy': float(accuracy_score(y_test, y_pred)),
    'test_f1': float(f1_score(y_test, y_pred)),
    'test_roc_auc': float(roc_auc_score(y_test, y_proba)),
    'baseline_f1': baseline['LogisticRegression'],
    'improvement': float(f1_score(y_test, y_pred)) - baseline['LogisticRegression']
}

with open('results_lr_primary_grid.json', 'w', encoding='utf-8') as f:
    json.dump(results_lr, f, indent=2)

all_results.append(results_lr)

print(f"\n⏱️ {elapsed/60:.2f} мин | F1: {results_lr['test_f1']:.4f}")
print(f"📋 Параметры: {grid_lr.best_params_}")
print(f"📈 Улучшение: {results_lr['improvement']:+.4f}")

# ─────────────────────────────────────────────────────────────
# МОДЕЛЬ 2: LinearSVC + GridSearchCV
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("2️⃣ LinearSVC + GridSearchCV")
print("=" * 80)
start = time.time()

param_grid_svc = {
    'C': [0.01, 0.1, 0.5, 1, 2, 5, 10],
    'class_weight': [None, 'balanced'],
    'max_iter': [1000, 2000, 5000],
    'penalty': ['l1', 'l2']
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
    'time_min': elapsed / 60,
    'best_params': grid_svc.best_params_,
    'cv_best_score': float(grid_svc.best_score_),
    'test_accuracy': float(accuracy_score(y_test, y_pred)),
    'test_f1': float(f1_score(y_test, y_pred)),
    'test_roc_auc': 0,
    'baseline_f1': baseline['LinearSVC'],
    'improvement': float(f1_score(y_test, y_pred)) - baseline['LinearSVC']
}

with open('results_linearsvc_primary_grid.json', 'w', encoding='utf-8') as f:
    json.dump(results_svc, f, indent=2)

all_results.append(results_svc)

print(f"\n⏱️ {elapsed/60:.2f} мин | F1: {results_svc['test_f1']:.4f}")
print(f"📋 Параметры: {grid_svc.best_params_}")
print(f"📈 Улучшение: {results_svc['improvement']:+.4f}")

# ─────────────────────────────────────────────────────────────
# МОДЕЛЬ 3: LightGBM + GridSearchCV
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("3️⃣ LightGBM + GridSearchCV")
print("=" * 80)
start = time.time()

from lightgbm import LGBMClassifier

param_grid_lgb = {
    'n_estimators': [150, 200, 250],
    'max_depth': [6, 8, 10],
    'learning_rate': [0.05, 0.1, 0.15],
    'num_leaves': [31, 45, 63],
    'min_child_samples': [15, 20, 25],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

grid_lgb = GridSearchCV(
    LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1),
    param_grid_lgb, cv=5, scoring='f1', n_jobs=-1, verbose=1
)
grid_lgb.fit(X_train, y_train)

elapsed = time.time() - start
y_pred = grid_lgb.predict(X_test)
y_proba = grid_lgb.predict_proba(X_test)[:, 1]

results_lgb = {
    'model': 'LightGBM_GridSearch',
    'time_min': elapsed / 60,
    'best_params': grid_lgb.best_params_,
    'cv_best_score': float(grid_lgb.best_score_),
    'test_accuracy': float(accuracy_score(y_test, y_pred)),
    'test_f1': float(f1_score(y_test, y_pred)),
    'test_roc_auc': float(roc_auc_score(y_test, y_proba)),
    'baseline_f1': baseline['LightGBM'],
    'improvement': float(f1_score(y_test, y_pred)) - baseline['LightGBM']
}

with open('results_lightgbm_primary_grid.json', 'w', encoding='utf-8') as f:
    json.dump(results_lgb, f, indent=2)

all_results.append(results_lgb)

print(f"\n⏱️ {elapsed/60:.2f} мин | F1: {results_lgb['test_f1']:.4f}")
print(f"📋 Параметры: {grid_lgb.best_params_}")
print(f"📈 Улучшение: {results_lgb['improvement']:+.4f}")

# ─────────────────────────────────────────────────────────────
# МОДЕЛЬ 4: XGBoost + GridSearchCV
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("4️⃣ XGBoost + GridSearchCV")
print("=" * 80)
start = time.time()

from xgboost import XGBClassifier

param_grid_xgb = {
    'n_estimators': [150, 200, 250],
    'max_depth': [5, 7, 9],
    'learning_rate': [0.05, 0.1, 0.15],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [1, 3, 5]
}

grid_xgb = GridSearchCV(
    XGBClassifier(random_state=42, verbosity=0, n_jobs=-1),
    param_grid_xgb, cv=5, scoring='f1', n_jobs=-1, verbose=1
)
grid_xgb.fit(X_train, y_train)

elapsed = time.time() - start
y_pred = grid_xgb.predict(X_test)
y_proba = grid_xgb.predict_proba(X_test)[:, 1]

results_xgb = {
    'model': 'XGBoost_GridSearch',
    'time_min': elapsed / 60,
    'best_params': grid_xgb.best_params_,
    'cv_best_score': float(grid_xgb.best_score_),
    'test_accuracy': float(accuracy_score(y_test, y_pred)),
    'test_f1': float(f1_score(y_test, y_pred)),
    'test_roc_auc': float(roc_auc_score(y_test, y_proba)),
    'baseline_f1': baseline['XGBoost'],
    'improvement': float(f1_score(y_test, y_pred)) - baseline['XGBoost']
}

with open('results_xgboost_primary_grid.json', 'w', encoding='utf-8') as f:
    json.dump(results_xgb, f, indent=2)

all_results.append(results_xgb)

print(f"\n⏱️ {elapsed/60:.2f} мин | F1: {results_xgb['test_f1']:.4f}")
print(f"📋 Параметры: {grid_xgb.best_params_}")
print(f"📈 Улучшение: {results_xgb['improvement']:+.4f}")

# ─────────────────────────────────────────────────────────────
# МОДЕЛЬ 5: Voting Ensemble (LGB + XGB) + GridSearchCV
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("5️⃣ Voting Ensemble (LGB + XGB) + GridSearchCV")
print("=" * 80)
start = time.time()

# Используем лучшие параметры из предыдущих GridSearch
lgb_best = LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1, **grid_lgb.best_params_)
xgb_best = XGBClassifier(random_state=42, verbosity=0, n_jobs=-1, **grid_xgb.best_params_)

param_grid_voting = {
    'weights': [
        [1, 1], [1, 2], [2, 1], [1, 3], [3, 1],
        [2, 3], [3, 2], [1, 4], [4, 1]
    ]
}

grid_voting = GridSearchCV(
    VotingClassifier(
        estimators=[('lgb', lgb_best), ('xgb', xgb_best)],
        voting='soft', n_jobs=-1
    ),
    param_grid_voting, cv=5, scoring='f1', n_jobs=-1, verbose=1
)
grid_voting.fit(X_train, y_train)

elapsed = time.time() - start
y_pred = grid_voting.predict(X_test)
y_proba = grid_voting.predict_proba(X_test)[:, 1]

results_voting = {
    'model': 'VotingEnsemble_Boosts_GridSearch',
    'time_min': elapsed / 60,
    'best_params': grid_voting.best_params_,
    'cv_best_score': float(grid_voting.best_score_),
    'test_accuracy': float(accuracy_score(y_test, y_pred)),
    'test_f1': float(f1_score(y_test, y_pred)),
    'test_roc_auc': float(roc_auc_score(y_test, y_proba)),
    'baseline_f1': baseline['VotingEnsemble'],
    'improvement': float(f1_score(y_test, y_pred)) - baseline['VotingEnsemble']
}

with open('results_voting_primary_grid.json', 'w', encoding='utf-8') as f:
    json.dump(results_voting, f, indent=2)

all_results.append(results_voting)

print(f"\n⏱️ {elapsed/60:.2f} мин | F1: {results_voting['test_f1']:.4f}")
print(f"📋 Веса: {grid_voting.best_params_}")
print(f"📈 Улучшение: {results_voting['improvement']:+.4f}")

# ─────────────────────────────────────────────────────────────
# ИТОГОВЫЙ ОТЧЁТ
# ─────────────────────────────────────────────────────────────
total_elapsed = time.time() - total_start

print("\n" + "=" * 80)
print("📊 ИТОГОВЫЙ ОТЧЁТ")
print("=" * 80)

print(f"\n{'Модель':<35} {'База':<8} {'Grid':<8} {'Δ':<10} {'Время':<10}")
print("-" * 80)
for res in sorted(all_results, key=lambda x: x['test_f1'], reverse=True):
    name = res['model'].replace('_GridSearch', '').replace('_Boosts', '')
    print(f"{name:<35} {res['baseline_f1']:<8.4f} {res['test_f1']:<8.4f} {res['improvement']:+.4f} {res['time_min']:<10.2f} мин")

print(f"\n⏱️ ОБЩЕЕ ВРЕМЯ: {total_elapsed/60:.2f} минут")

# Лучшая модель
best = max(all_results, key=lambda x: x['test_f1'])
print(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ: {best['model']}")
print(f"   F1-Score:  {best['test_f1']:.4f}")
print(f"   Accuracy:  {best['test_accuracy']:.4f}")
print(f"   ROC-AUC:   {best['test_roc_auc']:.4f}")
print(f"   Улучшение: {best['improvement']:+.4f}")
print(f"   Параметры: {best['best_params']}")

print("\n✅ Результаты сохранены: results_*_primary_grid.json")
print(f"\n🕐 Конец: {time.strftime('%H:%M:%S')}")
print("=" * 80)