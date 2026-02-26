"""
gridsearch_boosts_150k.py
GridSearchCV для LightGBM и XGBoost
Обучение на 150k записей, ускоренная версия
"""

import pandas as pd
import numpy as np
import time
import json
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🔍 GRIDSEARCHCV — LIGHTGBM И XGBOOST (150K ВЫБОРКА)")
print("=" * 80)
print(f"🕐 Начало: {time.strftime('%H:%M:%S')}")

total_start = time.time()

# ─────────────────────────────────────────────────────────────
# Загрузка данных
# ─────────────────────────────────────────────────────────────
df = pd.read_csv('train_fixed.csv')
X = df.drop('Heart Disease', axis=1)
y = df['Heart Disease'].map({'Absence': 0, 'Presence': 1})

# 🔥 СОКРАЩАЕМ ВЫБОРКУ ДО 150,000
sample_size = 150000
if len(X) > sample_size:
    X_sample = X.sample(n=sample_size, random_state=42)
    y_sample = y.loc[X_sample.index]
    print(f"\n⚠️ Выборка сокращена: {len(X):,} → {len(X_sample):,}")
else:
    X_sample = X
    y_sample = y

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_sample, test_size=10000, random_state=42, stratify=y_sample
)

print(f"✅ Train: {len(X_train):,} | Test: {len(X_test):,}")
print("=" * 80)

# Базовые результаты (без GridSearch)
baseline = {
    'LightGBM': 0.8726,
    'XGBoost': 0.8720
}

all_results = []

# ─────────────────────────────────────────────────────────────
# МОДЕЛЬ 1: LightGBM + GridSearchCV
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("1️⃣ LightGBM + GridSearchCV")
print("=" * 80)
start = time.time()

from lightgbm import LGBMClassifier

param_grid_lgb = {
    'n_estimators': [50, 150],      # ← ДОБАВИЛ 50
    'max_depth': [3, 6, 8],               # ← ДОБАВИЛ 3
    'learning_rate': [0.05, 0.1, 0.15],
    'num_leaves': [31, 45, 63],
    'min_child_samples': [15, 20, 25],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

total_combinations = (len(param_grid_lgb['n_estimators']) * 
                      len(param_grid_lgb['max_depth']) * 
                      len(param_grid_lgb['learning_rate']) * 
                      len(param_grid_lgb['num_leaves']) * 
                      len(param_grid_lgb['min_child_samples']) * 
                      len(param_grid_lgb['subsample']) * 
                      len(param_grid_lgb['colsample_bytree']))

print(f"📊 Комбинаций параметров: {total_combinations:,}")
print(f"📊 Обучений (×5 фолдов): {total_combinations * 5:,}")

grid_lgb = GridSearchCV(
    LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1),
    param_grid_lgb, 
    cv=5, 
    scoring='f1', 
    n_jobs=-1, 
    verbose=1
)
grid_lgb.fit(X_train, y_train)

elapsed = time.time() - start
y_pred = grid_lgb.predict(X_test)
y_proba = grid_lgb.predict_proba(X_test)[:, 1]

test_f1 = f1_score(y_test, y_pred)
test_acc = accuracy_score(y_test, y_pred)
test_auc = roc_auc_score(y_test, y_proba)

results_lgb = {
    'model': 'LightGBM_GridSearch',
    'time_min': elapsed / 60,
    'best_params': grid_lgb.best_params_,
    'cv_best_score': float(grid_lgb.best_score_),
    'test_accuracy': float(test_acc),
    'test_f1': float(test_f1),
    'test_roc_auc': float(test_auc),
    'baseline_f1': baseline['LightGBM'],
    'improvement': float(test_f1 - baseline['LightGBM']),
    'sample_size': sample_size
}

# 🔥 СОХРАНЯЕМ СРАЗУ ПОСЛЕ ОБУЧЕНИЯ
with open('results_lightgbm_grid_150k.json', 'w', encoding='utf-8') as f:
    json.dump(results_lgb, f, indent=2)

all_results.append(results_lgb)

print(f"\n⏱️ {elapsed/60:.2f} мин | F1: {test_f1:.4f}")
print(f"📋 Параметры: {grid_lgb.best_params_}")
print(f"📈 Улучшение: {test_f1 - baseline['LightGBM']:+.4f}")

# ─────────────────────────────────────────────────────────────
# МОДЕЛЬ 2: XGBoost + GridSearchCV
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("2️⃣ XGBoost + GridSearchCV")
print("=" * 80)
start = time.time()

from xgboost import XGBClassifier

param_grid_xgb = {
    'n_estimators': [50, 150, 200, 250],      # ← ДОБАВИЛ 50
    'max_depth': [3, 5, 7, 9],                # ← ДОБАВИЛ 3
    'learning_rate': [0.05, 0.1, 0.15],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [1, 3, 5]
}

total_combinations = (len(param_grid_xgb['n_estimators']) * 
                      len(param_grid_xgb['max_depth']) * 
                      len(param_grid_xgb['learning_rate']) * 
                      len(param_grid_xgb['subsample']) * 
                      len(param_grid_xgb['colsample_bytree']) * 
                      len(param_grid_xgb['gamma']) * 
                      len(param_grid_xgb['min_child_weight']))

print(f"📊 Комбинаций параметров: {total_combinations:,}")
print(f"📊 Обучений (×5 фолдов): {total_combinations * 5:,}")

grid_xgb = GridSearchCV(
    XGBClassifier(random_state=42, verbosity=0, n_jobs=-1),
    param_grid_xgb, 
    cv=5, 
    scoring='f1', 
    n_jobs=-1, 
    verbose=1
)
grid_xgb.fit(X_train, y_train)

elapsed = time.time() - start
y_pred = grid_xgb.predict(X_test)
y_proba = grid_xgb.predict_proba(X_test)[:, 1]

test_f1 = f1_score(y_test, y_pred)
test_acc = accuracy_score(y_test, y_pred)
test_auc = roc_auc_score(y_test, y_proba)

results_xgb = {
    'model': 'XGBoost_GridSearch',
    'time_min': elapsed / 60,
    'best_params': grid_xgb.best_params_,
    'cv_best_score': float(grid_xgb.best_score_),
    'test_accuracy': float(test_acc),
    'test_f1': float(test_f1),
    'test_roc_auc': float(test_auc),
    'baseline_f1': baseline['XGBoost'],
    'improvement': float(test_f1 - baseline['XGBoost']),
    'sample_size': sample_size
}

# 🔥 СОХРАНЯЕМ СРАЗУ ПОСЛЕ ОБУЧЕНИЯ
with open('results_xgboost_grid_150k.json', 'w', encoding='utf-8') as f:
    json.dump(results_xgb, f, indent=2)

all_results.append(results_xgb)

print(f"\n⏱️ {elapsed/60:.2f} мин | F1: {test_f1:.4f}")
print(f"📋 Параметры: {grid_xgb.best_params_}")
print(f"📈 Улучшение: {test_f1 - baseline['XGBoost']:+.4f}")

# ─────────────────────────────────────────────────────────────
# ИТОГОВЫЙ ОТЧЁТ
# ─────────────────────────────────────────────────────────────
total_elapsed = time.time() - total_start

print("\n" + "=" * 80)
print("📊 ИТОГОВЫЙ ОТЧЁТ")
print("=" * 80)

print(f"\n{'Модель':<30} {'База':<8} {'Grid':<8} {'Δ':<10} {'Время':<10}")
print("-" * 80)
for res in sorted(all_results, key=lambda x: x['test_f1'], reverse=True):
    name = res['model'].replace('_GridSearch', '')
    print(f"{name:<30} {res['baseline_f1']:<8.4f} {res['test_f1']:<8.4f} {res['improvement']:+.4f} {res['time_min']:<10.2f} мин")

print(f"\n⏱️ ОБЩЕЕ ВРЕМЯ: {total_elapsed/60:.2f} минут")

# Лучшая модель
best = max(all_results, key=lambda x: x['test_f1'])
print(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ: {best['model']}")
print(f"   F1-Score:  {best['test_f1']:.4f}")
print(f"   Accuracy:  {best['test_accuracy']:.4f}")
print(f"   ROC-AUC:   {best['test_roc_auc']:.4f}")
print(f"   Улучшение: {best['improvement']:+.4f}")
print(f"   Параметры: {best['best_params']}")

# 🔥 СОХРАНЯЕМ ОБЩИЙ ОТЧЁТ
with open('results_boosts_grid_summary_150k.json', 'w', encoding='utf-8') as f:
    json.dump({
        'total_time_min': total_elapsed / 60,
        'sample_size': sample_size,
        'all_results': all_results,
        'best_model': best
    }, f, indent=2)

print("\n✅ Результаты сохранены:")
print("   • results_lightgbm_grid_150k.json")
print("   • results_xgboost_grid_150k.json")
print("   • results_boosts_grid_summary_150k.json")
print(f"\n🕐 Конец: {time.strftime('%H:%M:%S')}")
print("=" * 80)