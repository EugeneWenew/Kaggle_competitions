"""
final_gridsearch_regressions.py
Третий (финальный) GridSearchCV — уточнение параметров

┌─────────────────────────────────────────────────────────────────┐
│  ТЕКУЩИЕ ЛУЧШИЕ ПАРАМЕТРЫ (на момент 2026-02-25)               │
├─────────────────────────────────────────────────────────────────┤
│  LogisticRegression:                                            │
│    • C: 2                                                       │
│    • penalty: l2                                                │
│    • solver: liblinear                                          │
│    • class_weight: balanced                                     │
│    • max_iter: 500                                              │
│    • F1: 0.8679                                                 │
├─────────────────────────────────────────────────────────────────┤
│  LinearSVC:                                                     │
│    • C: 1                                                       │
│    • penalty: l1                                                │
│    • class_weight: balanced                                     │
│    • max_iter: 500-1000                                         │
│    • F1: 0.8664                                                 │
└─────────────────────────────────────────────────────────────────┘
"""

import pandas as pd
import numpy as np
import time
import json
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print(" ФИНАЛЬНЫЙ GRIDSEARCHCV — УТОЧНЕНИЕ ПАРАМЕТРОВ")
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

# Базовые результаты (лучшие на текущий момент)
baseline = {
    'LogisticRegression': 0.8679,
    'LinearSVC': 0.8664
}

all_results = []

# ─────────────────────────────────────────────────────────────
# МОДЕЛЬ 1: LogisticRegression + GridSearchCV (уточнение)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("1️⃣ LogisticRegression + GridSearchCV (уточнение)")
print("=" * 80)
start = time.time()

param_grid_lr = {
    'C': [1.8, 1.9, 2.1, 2.2],
    'penalty': ['l2'],
    'solver': ['liblinear'],
    'class_weight': ['balanced'],
    'max_iter': [400, 450, 550, 600]
}

total_combinations = (len(param_grid_lr['C']) * 
                      len(param_grid_lr['penalty']) * 
                      len(param_grid_lr['solver']) * 
                      len(param_grid_lr['class_weight']) * 
                      len(param_grid_lr['max_iter']))

print(f"📊 Комбинаций: {total_combinations} × 5 фолдов = {total_combinations * 5} обучений")

grid_lr = GridSearchCV(
    LogisticRegression(random_state=42, n_jobs=-1),
    param_grid_lr, 
    cv=5, 
    scoring='f1', 
    n_jobs=-1, 
    verbose=1
)
grid_lr.fit(X_train, y_train)

elapsed = time.time() - start
y_pred = grid_lr.predict(X_test)
y_proba = grid_lr.predict_proba(X_test)[:, 1]

test_f1 = f1_score(y_test, y_pred)
test_acc = accuracy_score(y_test, y_pred)
test_auc = roc_auc_score(y_test, y_proba)

results_lr = {
    'model': 'LogisticRegression_FinalGridSearch',
    'time_min': elapsed / 60,
    'best_params': grid_lr.best_params_,
    'cv_best_score': float(grid_lr.best_score_),
    'test_accuracy': float(test_acc),
    'test_f1': float(test_f1),
    'test_roc_auc': float(test_auc),
    'baseline_f1': baseline['LogisticRegression'],
    'improvement': float(test_f1 - baseline['LogisticRegression'])
}

# 🔥 Сохраняем сразу после обучения
with open('results_lr_final_grid.json', 'w', encoding='utf-8') as f:
    json.dump(results_lr, f, indent=2)

all_results.append(results_lr)

print(f"\n⏱️ {elapsed/60:.2f} мин | F1: {test_f1:.4f}")
print(f"📋 Параметры: {grid_lr.best_params_}")
print(f"📈 Улучшение: {test_f1 - baseline['LogisticRegression']:+.4f}")

# ─────────────────────────────────────────────────────────────
# МОДЕЛЬ 2: LinearSVC + GridSearchCV (уточнение)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("2️⃣ LinearSVC + GridSearchCV (уточнение)")
print("=" * 80)
start = time.time()

param_grid_svc = {
    'C': [0.8, 0.9, 1.1, 1.2],
    'penalty': ['l1'],
    'class_weight': ['balanced'],
    'max_iter': [600, 700, 800, 900]
}

total_combinations = (len(param_grid_svc['C']) * 
                      len(param_grid_svc['penalty']) * 
                      len(param_grid_svc['class_weight']) * 
                      len(param_grid_svc['max_iter']))

print(f"📊 Комбинаций: {total_combinations} × 5 фолдов = {total_combinations * 5} обучений")

grid_svc = GridSearchCV(
    LinearSVC(random_state=42),
    param_grid_svc, 
    cv=5, 
    scoring='f1', 
    n_jobs=-1, 
    verbose=1
)
grid_svc.fit(X_train, y_train)

elapsed = time.time() - start
y_pred = grid_svc.predict(X_test)

test_f1 = f1_score(y_test, y_pred)
test_acc = accuracy_score(y_test, y_pred)

results_svc = {
    'model': 'LinearSVC_FinalGridSearch',
    'time_min': elapsed / 60,
    'best_params': grid_svc.best_params_,
    'cv_best_score': float(grid_svc.best_score_),
    'test_accuracy': float(test_acc),
    'test_f1': float(test_f1),
    'test_roc_auc': 0,
    'baseline_f1': baseline['LinearSVC'],
    'improvement': float(test_f1 - baseline['LinearSVC'])
}

# 🔥 Сохраняем сразу после обучения
with open('results_linearsvc_final_grid.json', 'w', encoding='utf-8') as f:
    json.dump(results_svc, f, indent=2)

all_results.append(results_svc)

print(f"\n⏱️ {elapsed/60:.2f} мин | F1: {test_f1:.4f}")
print(f"📋 Параметры: {grid_svc.best_params_}")
print(f"📈 Улучшение: {test_f1 - baseline['LinearSVC']:+.4f}")

# ─────────────────────────────────────────────────────────────
# ИТОГОВЫЙ ОТЧЁТ
# ─────────────────────────────────────────────────────────────
total_elapsed = time.time() - total_start

print("\n" + "=" * 80)
print("📊 ИТОГОВЫЙ ОТЧЁТ")
print("=" * 80)

print(f"\n{'Модель':<35} {'База':<8} {'Финал':<8} {'Δ':<10} {'Время':<10}")
print("-" * 80)
for res in sorted(all_results, key=lambda x: x['test_f1'], reverse=True):
    name = res['model'].replace('_FinalGridSearch', '')
    print(f"{name:<35} {res['baseline_f1']:<8.4f} {res['test_f1']:<8.4f} {res['improvement']:+.4f} {res['time_min']:<10.2f} мин")

print(f"\n⏱️ ОБЩЕЕ ВРЕМЯ: {total_elapsed/60:.2f} минут")

# Лучшая модель
best = max(all_results, key=lambda x: x['test_f1'])
print(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ: {best['model']}")
print(f"   F1-Score:  {best['test_f1']:.4f}")
print(f"   Accuracy:  {best['test_accuracy']:.4f}")
print(f"   Улучшение: {best['improvement']:+.4f}")
print(f"   Параметры: {best['best_params']}")

# Сохраняем общий отчёт
with open('results_regressions_final_summary.json', 'w', encoding='utf-8') as f:
    json.dump({
        'total_time_min': total_elapsed / 60,
        'all_results': all_results,
        'best_model': best
    }, f, indent=2)

print("\n✅ Результаты сохранены:")
print("   • results_lr_final_grid.json")
print("   • results_linearsvc_final_grid.json")
print("   • results_regressions_final_summary.json")
print(f"\n🕐 Конец: {time.strftime('%H:%M:%S')}")
print("=" * 80)