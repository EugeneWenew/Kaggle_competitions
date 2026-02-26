"""
boosts_gridsearch_150k_progress.py
GridSearchCV для LightGBM и XGBoost
150k строк для обучения, увеличенный тест!
С ПОДРОБНЫМ ПРОГРЕССОМ для оценки времени!
"""

import pandas as pd
import numpy as np
import time
import json
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# ФУНКЦИЯ ДЛЯ СОХРАНЕНИЯ И СКАЧИВАНИЯ
# ─────────────────────────────────────────────────────────────
def save_and_download(data, filename):
    """Сохраняет в JSON и сразу скачивает"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    try:
        from google.colab import files
        files.download(filename)
        print(f"   📥 {filename} скачан!")
    except:
        print(f"   💾 {filename} сохранён локально")

print("=" * 80)
print("🚀 GRIDSEARCHCV ДЛЯ БУСТИНГОВ — 150K СТРОК")
print("=" * 80)
print(f"🕐 Начало: {time.strftime('%H:%M:%S')}")

total_start = time.time()

# ─────────────────────────────────────────────────────────────
# Загрузка данных
# ─────────────────────────────────────────────────────────────
df = pd.read_csv('train_fixed.csv')
print(f"\n✅ Всего данных: {len(df):,} строк")

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛЕНИЕ: 150k train, большой test
# ─────────────────────────────────────────────────────────────
TRAIN_SIZE = 150000
TEST_SIZE = 100000

df_train_full, df_test = train_test_split(
    df, 
    train_size=TRAIN_SIZE + TEST_SIZE, 
    stratify=df['Heart Disease'],
    random_state=42
)

df_train, _ = train_test_split(
    df_train_full,
    train_size=TRAIN_SIZE,
    stratify=df_train_full['Heart Disease'],
    random_state=42
)

print(f"✅ Train: {len(df_train):,} строк")
print(f"✅ Test: {len(df_test):,} строк")

# Подготовка
X_train = df_train.drop('Heart Disease', axis=1)
y_train = df_train['Heart Disease'].map({'Absence': 0, 'Presence': 1})

X_test = df_test.drop('Heart Disease', axis=1)
y_test = df_test['Heart Disease'].map({'Absence': 0, 'Presence': 1})

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("=" * 80)

# Базовые результаты (из предыдущих экспериментов на 630k)
baseline = {
    'LightGBM': 0.8725799189554255,
    'XGBoost': 0.8720027017899359,
    'VotingEnsemble': 0.869309838472834
}

all_results = []

# ─────────────────────────────────────────────────────────────
# ФУНКЦИЯ ДЛЯ ВЫВОДА РЕЗУЛЬТАТОВ GRIDSEARCH
# ─────────────────────────────────────────────────────────────
def print_gridsearch_results(grid_result, model_name, start_time, param_grid):
    """Выводит детальные результаты GridSearch"""
    elapsed = time.time() - start_time
    
    # Считаем комбинации
    total_combos = 1
    for key, values in param_grid.items():
        total_combos *= len(values)
    total_fits = total_combos * 5
    
    print(f"\n📊 ПРОГРЕСС {model_name}:")
    print(f"   Всего комбинаций: {total_combos}")
    print(f"   Всего обучений (5 фолдов): {total_fits}")
    print(f"   Прошло времени: {elapsed/60:.2f} мин")
    print(f"   Среднее время на обучение: {elapsed/total_fits:.2f} сек")
    
    # Таблица результатов
    print(f"\n📋 ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ ({model_name}):")
    print(f"{'Ранг':<6} {'F1 (cv)':<12} {'F1 (test)':<12} {'Параметры':<60}")
    print("-" * 100)
    
    cv_results = pd.DataFrame(grid_result.cv_results_)
    cv_results = cv_results.sort_values('rank_test_f1')
    
    for idx, row in cv_results.iterrows():
        params = row['params']
        param_str = str(params).replace('{', '').replace('}', '').replace("'", '')
        print(f"{int(row['rank_test_f1']):<6} {row['mean_test_score']:<12.4f} {row['rank_test_f1']:<12} {param_str:<60}")
    
    return elapsed

# ─────────────────────────────────────────────────────────────
# МОДЕЛЬ 1: LightGBM — GridSearchCV
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("1️⃣ LightGBM — GridSearchCV")
print("=" * 80)

start = time.time()

param_grid_lgb = {
    'n_estimators': [150, 200, 250],
    'max_depth': [6, 8, 10],
    'learning_rate': [0.05, 0.1, 0.15],
    'num_leaves': [31, 45, 63],
    'min_child_samples': [15, 20, 25],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# Подсчёт комбинаций
total_lgb = 1
for key, values in param_grid_lgb.items():
    total_lgb *= len(values)
print(f"\n📊 ВСЕГО КОМБИНАЦИЙ: {total_lgb} × 5 фолдов = {total_lgb * 5} обучений")
print(f"📋 Параметры:")
for key, values in param_grid_lgb.items():
    print(f"   {key}: {values}")
print("=" * 80)

from lightgbm import LGBMClassifier

grid_lgb = GridSearchCV(
    LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1),
    param_grid_lgb, cv=5, scoring='f1', n_jobs=-1, verbose=2
)
grid_lgb.fit(X_train_scaled, y_train)

elapsed = time.time() - start

# Вывод прогресса
print_gridsearch_results(grid_lgb, 'LightGBM', start, param_grid_lgb)

y_pred = grid_lgb.predict(X_test_scaled)
y_proba = grid_lgb.predict_proba(X_test_scaled)[:, 1]

results_lgb = {
    'model': 'LightGBM_GridSearch_150k',
    'time_min': elapsed / 60,
    'best_params': grid_lgb.best_params_,
    'cv_best_score': float(grid_lgb.best_score_),
    'test_accuracy': float(accuracy_score(y_test, y_pred)),
    'test_f1': float(f1_score(y_test, y_pred)),
    'test_roc_auc': float(roc_auc_score(y_test, y_proba)),
    'baseline_f1': baseline['LightGBM'],
    'improvement': float(f1_score(y_test, y_pred)) - baseline['LightGBM'],
    'train_size': len(df_train),
    'test_size': len(df_test)
}

save_and_download(results_lgb, 'results_lightgbm_150k_grid.json')
all_results.append(results_lgb)

print(f"\n⏱️ {elapsed/60:.2f} мин | F1: {results_lgb['test_f1']:.4f}")
print(f"📋 Лучшие параметры: {grid_lgb.best_params_}")
print(f"📈 Улучшение от базы (630k): {results_lgb['improvement']:+.4f}")

# ─────────────────────────────────────────────────────────────
# МОДЕЛЬ 2: XGBoost — GridSearchCV
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("2️⃣ XGBoost — GridSearchCV")
print("=" * 80)

start = time.time()

param_grid_xgb = {
    'n_estimators': [150, 200, 250],
    'max_depth': [5, 7, 9],
    'learning_rate': [0.05, 0.1, 0.15],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [1, 3, 5]
}

# Подсчёт комбинаций
total_xgb = 1
for key, values in param_grid_xgb.items():
    total_xgb *= len(values)
print(f"\n📊 ВСЕГО КОМБИНАЦИЙ: {total_xgb} × 5 фолдов = {total_xgb * 5} обучений")
print(f"📋 Параметры:")
for key, values in param_grid_xgb.items():
    print(f"   {key}: {values}")
print("=" * 80)

from xgboost import XGBClassifier

grid_xgb = GridSearchCV(
    XGBClassifier(random_state=42, verbosity=0, n_jobs=-1),
    param_grid_xgb, cv=5, scoring='f1', n_jobs=-1, verbose=2
)
grid_xgb.fit(X_train_scaled, y_train)

elapsed = time.time() - start

# Вывод прогресса
print_gridsearch_results(grid_xgb, 'XGBoost', start, param_grid_xgb)

y_pred = grid_xgb.predict(X_test_scaled)
y_proba = grid_xgb.predict_proba(X_test_scaled)[:, 1]

results_xgb = {
    'model': 'XGBoost_GridSearch_150k',
    'time_min': elapsed / 60,
    'best_params': grid_xgb.best_params_,
    'cv_best_score': float(grid_xgb.best_score_),
    'test_accuracy': float(accuracy_score(y_test, y_pred)),
    'test_f1': float(f1_score(y_test, y_pred)),
    'test_roc_auc': float(roc_auc_score(y_test, y_proba)),
    'baseline_f1': baseline['XGBoost'],
    'improvement': float(f1_score(y_test, y_pred)) - baseline['XGBoost'],
    'train_size': len(df_train),
    'test_size': len(df_test)
}

save_and_download(results_xgb, 'results_xgboost_150k_grid.json')
all_results.append(results_xgb)

print(f"\n⏱️ {elapsed/60:.2f} мин | F1: {results_xgb['test_f1']:.4f}")
print(f"📋 Лучшие параметры: {grid_xgb.best_params_}")
print(f"📈 Улучшение от базы (630k): {results_xgb['improvement']:+.4f}")

# ─────────────────────────────────────────────────────────────
# МОДЕЛЬ 3: VotingEnsemble (LGB + XGB) — GridSearch весов
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("3️⃣ VotingEnsemble (LGB + XGB) — GridSearch весов")
print("=" * 80)

start = time.time()

lgb_best = LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1, **grid_lgb.best_params_)
xgb_best = XGBClassifier(random_state=42, verbosity=0, n_jobs=-1, **grid_xgb.best_params_)

param_grid_voting = {
    'weights': [
        [1, 1], [1, 2], [2, 1], [1, 3], [3, 1],
        [2, 3], [3, 2], [1, 4], [4, 1]
    ]
}

total_voting = len(param_grid_voting['weights'])
print(f"\n📊 ВСЕГО КОМБИНАЦИЙ: {total_voting} × 5 фолдов = {total_voting * 5} обучений")
print(f"📋 Веса для проверки: {param_grid_voting['weights']}")
print("=" * 80)

grid_voting = GridSearchCV(
    VotingClassifier(
        estimators=[('lgb', lgb_best), ('xgb', xgb_best)],
        voting='soft', n_jobs=-1
    ),
    param_grid_voting, cv=5, scoring='f1', n_jobs=-1, verbose=2
)
grid_voting.fit(X_train_scaled, y_train)

elapsed = time.time() - start

# Вывод прогресса
print_gridsearch_results(grid_voting, 'VotingEnsemble', start, param_grid_voting)

y_pred = grid_voting.predict(X_test_scaled)
y_proba = grid_voting.predict_proba(X_test_scaled)[:, 1]

results_voting = {
    'model': 'VotingEnsemble_Boosts_150k',
    'time_min': elapsed / 60,
    'best_params': grid_voting.best_params_,
    'cv_best_score': float(grid_voting.best_score_),
    'test_accuracy': float(accuracy_score(y_test, y_pred)),
    'test_f1': float(f1_score(y_test, y_pred)),
    'test_roc_auc': float(roc_auc_score(y_test, y_proba)),
    'baseline_f1': baseline['VotingEnsemble'],
    'improvement': float(f1_score(y_test, y_pred)) - baseline['VotingEnsemble'],
    'train_size': len(df_train),
    'test_size': len(df_test)
}

save_and_download(results_voting, 'results_voting_150k_grid.json')
all_results.append(results_voting)

print(f"\n⏱️ {elapsed/60:.2f} мин | F1: {results_voting['test_f1']:.4f}")
print(f"📋 Лучшие веса: {grid_voting.best_params_}")
print(f"📈 Улучшение от базы (630k): {results_voting['improvement']:+.4f}")

# ─────────────────────────────────────────────────────────────
# ИТОГОВЫЙ ОТЧЁТ
# ─────────────────────────────────────────────────────────────
total_elapsed = time.time() - total_start

print("\n" + "=" * 80)
print("📊 ИТОГОВЫЙ ОТЧЁТ")
print("=" * 80)

print(f"\n{'Модель':<35} {'База (630k)':<12} {'150k Grid':<12} {'Δ':<10} {'Время':<10}")
print("-" * 80)
for res in sorted(all_results, key=lambda x: x['test_f1'], reverse=True):
    print(f"{res['model']:<35} {res['baseline_f1']:<12.4f} {res['test_f1']:<12.4f} {res['improvement']:+.4f} {res['time_min']:<10.2f} мин")

print(f"\n⏱️ ОБЩЕЕ ВРЕМЯ: {total_elapsed/60:.2f} минут")

# Лучшая модель
best = max(all_results, key=lambda x: x['test_f1'])
print(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ: {best['model']}")
print(f"   F1-Score:  {best['test_f1']:.4f}")
print(f"   Accuracy:  {best['test_accuracy']:.4f}")
print(f"   ROC-AUC:   {best['test_roc_auc']:.4f}")
print(f"   Улучшение: {best['improvement']:+.4f}")
print(f"   Параметры: {best['best_params']}")

# ─────────────────────────────────────────────────────────────
# ФИНАЛЬНОЕ СКАЧИВАНИЕ
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("📥 ФИНАЛЬНОЕ СКАЧИВАНИЕ")
print("=" * 80)

summary = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'total_time_min': total_elapsed / 60,
    'best_model': best['model'],
    'best_f1': best['test_f1'],
    'train_size': len(df_train),
    'test_size': len(df_test),
    'all_results': all_results
}

save_and_download(summary, 'results_boosts_150k_summary.json')

try:
    import shutil
    shutil.make_archive('results_boosts_150k', 'zip', '.', 'results_')
    save_and_download({}, 'results_boosts_150k.zip')
    print("✅ ZIP архив скачан!")
except Exception as e:
    print(f"⚠️ Не удалось создать ZIP: {e}")

print(f"\n🕐 Конец: {time.strftime('%H:%M:%S')}")
print("=" * 80)