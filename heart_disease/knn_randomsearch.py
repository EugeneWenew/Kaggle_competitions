"""
test_knn.py
Быстрая проверка K-Neighbors
"""

import pandas as pd
import numpy as np
import time
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🔍 БЫСТРАЯ ПРОВЕРКА K-NEIGHBORS")
print("=" * 70)

# Загрузка данных
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

# Базовые результаты для сравнения
baseline = {
    'LightGBM': 0.8726,
    'XGBoost': 0.8720,
    'VotingEnsemble': 0.8693,
    'RandomForest': 0.8659,
    'LogisticRegression': 0.8656,
    'LinearSVC': 0.8648
}

results = []

# ─────────────────────────────────────────────────────────────
# Тест разных значений K
# ─────────────────────────────────────────────────────────────
print("\n📊 Тест различных K:")
print(f"{'K':<10} {'F1':<10} {'Accuracy':<10} {'Время':<10}")
print("-" * 50)

for k in [3, 5, 7, 11, 15, 21, 31, 51]:
    start = time.time()
    
    model = KNeighborsClassifier(
        n_neighbors=k,
        weights='distance',  # Взвешенное голосование
        n_jobs=-1,
        metric='minkowski'
    )
    model.fit(X_train, y_train)
    
    elapsed = time.time() - start
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    results.append({
        'k': k,
        'f1': float(f1),
        'accuracy': float(acc),
        'time_min': elapsed / 60
    })
    
    print(f"{k:<10} {f1:<10.4f} {acc:<10.4f} {elapsed/60:<10.2f} мин")

# Лучший K
best = max(results, key=lambda x: x['f1'])

print("\n" + "=" * 70)
print("🏆 ЛУЧШИЙ РЕЗУЛЬТАТ KNN:")
print("=" * 70)
print(f"K = {best['k']}")
print(f"F1 = {best['f1']:.4f}")
print(f"Accuracy = {best['accuracy']:.4f}")
print(f"Время = {best['time_min']:.2f} мин")

# Сравнение с базой
print("\n" + "=" * 70)
print("📈 СРАВНЕНИЕ С ДРУГИМИ МОДЕЛЯМИ:")
print("=" * 70)
print(f"\n{'Модель':<25} {'F1':<10} {'Разница с KNN':<15}")
print("-" * 50)

all_models = [
    ('LightGBM', baseline['LightGBM']),
    ('XGBoost', baseline['XGBoost']),
    ('VotingEnsemble', baseline['VotingEnsemble']),
    ('RandomForest', baseline['RandomForest']),
    ('LogisticRegression', baseline['LogisticRegression']),
    ('LinearSVC', baseline['LinearSVC']),
    (f'KNN (K={best["k"]})', best['f1'])
]

for name, f1 in sorted(all_models, key=lambda x: x[1], reverse=True):
    diff = f1 - best['f1']
    if diff > 0:
        print(f"{name:<25} {f1:<10.4f} {diff:+.4f} (лучше)")
    elif diff < 0:
        print(f"{name:<25} {f1:<10.4f} {diff:+.4f} (хуже)")
    else:
        print(f"{name:<25} {f1:<10.4f} {diff:+.4f}")

# Сохранение
with open('results_knn.json', 'w', encoding='utf-8') as f:
    json.dump({
        'best_k': best['k'],
        'best_f1': best['f1'],
        'best_accuracy': best['accuracy'],
        'all_results': results
    }, f, indent=2)

print("\n✅ Результаты сохранены: results_knn.json")