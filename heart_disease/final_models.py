"""
final_models.py
Финализация: Threshold Tuning + Сохранение моделей
Выборка: 150k строк
"""

import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🏆 ФИНАЛИЗАЦИЯ МОДЕЛЕЙ")
print("=" * 80)

# ─────────────────────────────────────────────────────────────
# Загрузка данных (150k выборка)
# ─────────────────────────────────────────────────────────────
df = pd.read_csv('train_fixed.csv')
X = df.drop('Heart Disease', axis=1)
y = df['Heart Disease'].map({'Absence': 0, 'Presence': 1})

# 🔥 Ограничиваем выборку до 150k
sample_size = 150000
if len(X) > sample_size:
    X_sample = X.sample(n=sample_size, random_state=42)
    y_sample = y.loc[X_sample.index]
    print(f"✅ Выборка: {len(X_sample):,} строк")
else:
    X_sample, y_sample = X, y

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_sample, y_sample, test_size=10000, random_state=42, stratify=y_sample
)

# Скейлер для линейных моделей
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Train: {len(X_train):,} | Test: {len(X_test):,}")
print("=" * 80)

# ─────────────────────────────────────────────────────────────
# Функция поиска оптимального порога
# ─────────────────────────────────────────────────────────────
def find_best_threshold(y_true, y_proba):
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in np.arange(0.30, 0.70, 0.01):
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1

# ─────────────────────────────────────────────────────────────
# МОДЕЛИ И ПАРАМЕТРЫ (из лучших результатов)
# ─────────────────────────────────────────────────────────────
models = {
    'LightGBM': {
        'model': LGBMClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1
        ),
        'X_train': X_train,
        'X_test': X_test,
        'scale': False
    },
    'XGBoost': {
        'model': XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0, n_jobs=-1
        ),
        'X_train': X_train,
        'X_test': X_test,
        'scale': False
    },
    'LogisticRegression': {
        'model': LogisticRegression(
            C=2, penalty='l2', solver='liblinear',
            class_weight='balanced', max_iter=500, random_state=42, n_jobs=-1
        ),
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'scale': True
    },
    'LinearSVC': {
        'model': LinearSVC(
            C=1, penalty='l1', class_weight='balanced',
            max_iter=1000, random_state=42
        ),
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'scale': True
    }
}

results = []

# ─────────────────────────────────────────────────────────────
# Обучение + Threshold Tuning + Сохранение
# ─────────────────────────────────────────────────────────────
for name, config in models.items():
    print(f"\n🔹 {name}...")
    
    # Обучение
    config['model'].fit(config['X_train'], y_train)
    
    # Предсказания (вероятности)
    if hasattr(config['model'], 'predict_proba'):
        y_proba = config['model'].predict_proba(config['X_test'])[:, 1]
    else:
        # LinearSVC не имеет predict_proba по умолчанию
        y_proba = config['model'].decision_function(config['X_test'])
        # Нормализация для порога
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
    
    # Поиск оптимального порога
    threshold, best_f1 = find_best_threshold(y_test, y_proba)
    
    # Финальные метрики
    y_pred = (y_proba >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if hasattr(config['model'], 'predict_proba') else 0
    
    result = {
        'model': name,
        'threshold': round(threshold, 2),
        'f1': round(f1, 4),
        'accuracy': round(acc, 4),
        'roc_auc': round(auc, 4)
    }
    results.append(result)
    
    print(f"   Порог: {threshold:.2f} | F1: {f1:.4f} | Acc: {acc:.4f}")
    
    # Сохранение модели
    joblib.dump(config['model'], f'model_{name}.pkl')
    print(f"   ✅ Сохранено: model_{name}.pkl")

# Сохранение скейлера
joblib.dump(scaler, 'scaler.pkl')
print(f"\n✅ Сохранено: scaler.pkl")

# ─────────────────────────────────────────────────────────────
# Сохранение порогов
# ─────────────────────────────────────────────────────────────
thresholds = {r['model']: r['threshold'] for r in results}
with open('thresholds.json', 'w') as f:
    json.dump(thresholds, f, indent=2)
print(f"✅ Сохранено: thresholds.json")

# ─────────────────────────────────────────────────────────────
# Итоговая таблица
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("📊 ИТОГОВАЯ ТАБЛИЦА")
print("=" * 80)
print(f"\n{'Модель':<25} {'Порог':<10} {'F1':<10} {'Accuracy':<10} {'ROC-AUC':<10}")
print("-" * 80)
for r in sorted(results, key=lambda x: x['f1'], reverse=True):
    print(f"{r['model']:<25} {r['threshold']:<10.2f} {r['f1']:<10.4f} {r['accuracy']:<10.4f} {r['roc_auc']:<10.4f}")

# Лучшая модель
best = max(results, key=lambda x: x['f1'])
print(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ: {best['model']}")
print(f"   F1: {best['f1']:.4f} | Порог: {best['threshold']:.2f}")

# Сохранение итогового отчёта
with open('final_results.json', 'w', encoding='utf-8') as f:
    json.dump({
        'sample_size': sample_size,
        'test_size': len(X_test),
        'results': results,
        'best_model': best
    }, f, indent=2)
print(f"\n✅ Сохранено: final_results.json")
print("=" * 80)