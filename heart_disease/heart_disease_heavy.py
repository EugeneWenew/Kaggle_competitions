"""
heart_disease_heavy_local.py
Тяжёлые модели — быстрая оценка без GridSearchCV
"""

import pandas as pd
import numpy as np
import time
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🏋️ ТЯЖЁЛЫЕ МОДЕЛИ — БЫСТРАЯ ОЦЕНКА")
print("=" * 70)
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
print("=" * 70)

all_results = []

# ─────────────────────────────────────────────────────────────
# МОДЕЛЬ 1: XGBoost
# ─────────────────────────────────────────────────────────────
print("\n1️⃣ XGBoost...")
try:
    from xgboost import XGBClassifier
    start = time.time()
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)
    
    elapsed = time.time() - start
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    results = {
        'model': 'XGBoost',
        'time_min': elapsed / 60,
        'test_accuracy': float(accuracy_score(y_test, y_pred)),
        'test_f1': float(f1_score(y_test, y_pred)),
        'test_roc_auc': float(roc_auc_score(y_test, y_proba))
    }
    all_results.append(results)
    
    with open('results_xgboost_local.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"   ⏱️ {elapsed/60:.2f} мин | F1: {results['test_f1']:.4f} | Acc: {results['test_accuracy']:.4f}")
    
except ImportError:
    print("   ⚠️ XGBoost не установлен (pip install xgboost)")
    all_results.append({'model': 'XGBoost', 'time_min': 0, 'test_f1': 0, 'test_accuracy': 0, 'test_roc_auc': 0})

# ─────────────────────────────────────────────────────────────
# МОДЕЛЬ 2: LightGBM
# ─────────────────────────────────────────────────────────────
print("\n2️⃣ LightGBM...")
try:
    from lightgbm import LGBMClassifier
    start = time.time()
    
    model = LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    
    elapsed = time.time() - start
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    results = {
        'model': 'LightGBM',
        'time_min': elapsed / 60,
        'test_accuracy': float(accuracy_score(y_test, y_pred)),
        'test_f1': float(f1_score(y_test, y_pred)),
        'test_roc_auc': float(roc_auc_score(y_test, y_proba))
    }
    all_results.append(results)
    
    with open('results_lightgbm_local.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"   ⏱️ {elapsed/60:.2f} мин | F1: {results['test_f1']:.4f} | Acc: {results['test_accuracy']:.4f}")
    
except ImportError:
    print("   ⚠️ LightGBM не установлен (pip install lightgbm)")
    all_results.append({'model': 'LightGBM', 'time_min': 0, 'test_f1': 0, 'test_accuracy': 0, 'test_roc_auc': 0})

# ─────────────────────────────────────────────────────────────
# МОДЕЛЬ 3: SVC RBF (на подвыборке 50k)
# ─────────────────────────────────────────────────────────────
print("\n3️⃣ SVC RBF (на подвыборке 50k)...")
start = time.time()

# Берём подвыборку для скорости
X_train_sub, y_train_sub = resample(X_train, y_train, n_samples=50000, random_state=42)

model = SVC(kernel='rbf', C=10, gamma=0.01, probability=True, random_state=42)
model.fit(X_train_sub, y_train_sub)

elapsed = time.time() - start
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

results = {
    'model': 'SVC_RBF_50k',
    'time_min': elapsed / 60,
    'test_accuracy': float(accuracy_score(y_test, y_pred)),
    'test_f1': float(f1_score(y_test, y_pred)),
    'test_roc_auc': float(roc_auc_score(y_test, y_proba))
}
all_results.append(results)

with open('results_svc_rbf_local.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

print(f"   ⏱️ {elapsed/60:.2f} мин | F1: {results['test_f1']:.4f} | Acc: {results['test_accuracy']:.4f}")

# ─────────────────────────────────────────────────────────────
# ИТОГОВЫЙ ОТЧЁТ
# ─────────────────────────────────────────────────────────────
total_elapsed = time.time() - total_start

print("\n" + "=" * 70)
print("📊 ИТОГОВЫЙ ОТЧЁТ")
print("=" * 70)

print(f"\n{'Модель':<25} {'F1':<10} {'Accuracy':<10} {'Время':<10}")
print("-" * 70)
for res in sorted(all_results, key=lambda x: x['test_f1'], reverse=True):
    print(f"{res['model']:<25} {res['test_f1']:<10.4f} {res['test_accuracy']:<10.4f} {res['time_min']:<10.2f} мин")

print(f"\n⏱️ ОБЩЕЕ ВРЕМЯ: {total_elapsed/60:.2f} минут")

# Сравнение с базой
print("\n" + "=" * 70)
print("📈 СРАВНЕНИЕ С БАЗОВЫМИ МОДЕЛЯМИ")
print("=" * 70)

baseline = {
    'VotingEnsemble': 0.8693,
    'RandomForest': 0.8659,
    'LogisticRegression': 0.8656,
    'LinearSVC': 0.8648
}

print(f"\n{'Модель':<25} {'База':<10} {'Heavy':<10} {'Разница':<10}")
print("-" * 70)
for res in sorted(all_results, key=lambda x: x['test_f1'], reverse=True):
    name = res['model'].replace('_50k', '').replace('_RBF', '')
    base_f1 = baseline.get('RandomForest', 0.87)  # Для сравнения
    diff = res['test_f1'] - base_f1
    print(f"{name:<25} {base_f1:<10.4f} {res['test_f1']:<10.4f} {diff:+.4f}")

print("\n✅ Результаты сохранены: results_*_local.json")
print(f"\n🕐 Конец: {time.strftime('%H:%M:%S')}")
print("=" * 70)