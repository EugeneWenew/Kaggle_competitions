import pandas as pd
import numpy as np
import time
import json
import os
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================

print("=" * 70)
print("🔍 ГИБРИДНЫЙ ПОИСК: GridSearch + RandomizedSearch")
print("=" * 70)
print(f"🕐 Начало: {time.strftime('%H:%M:%S')}")

total_start = time.time()

df = pd.read_csv('train_fixed.csv')
print(f"✅ Загружено записей: {len(df):,}")

X = df.drop('Heart Disease', axis=1)
y = df['Heart Disease'].map({'Absence': 0, 'Presence': 1})

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=10000, random_state=42, stratify=y
)

print(f"📈 Train: {len(X_train):,}, Test: {len(X_test):,}")

# ============================================================
# 2. БАЗОВЫЕ РЕЗУЛЬТАТЫ (для сравнения)
# ============================================================

baseline_results = {
    'VotingEnsemble': 0.8693,
    'RandomForest': 0.8659,
    'LogisticRegression': 0.8656,
    'LinearSVC': 0.8648
}

# ============================================================
# 3. МОДЕЛИ С УЖЕ ИЗВЕСТНЫМИ РЕЗУЛЬТАТАМИ (GridSearch)
# ============================================================

gridsearch_models = {
    'LogisticRegression': {
        'model': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        'params': {
            'C': [0.1, 0.5, 1, 2, 5, 10],
            'class_weight': [None, 'balanced'],
            'solver': ['liblinear', 'saga'],
            'penalty': ['l1', 'l2']
        },
        'baseline_f1': baseline_results.get('LogisticRegression', None)
    },
    
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [100, 150, 200, 250],
            'max_depth': [8, 10, 12, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': [None, 'balanced']
        },
        'baseline_f1': baseline_results.get('RandomForest', None)
    },
    
    'LinearSVC': {
        'model': LinearSVC(max_iter=1000, random_state=42),
        'params': {
            'C': [0.5, 1, 2, 5, 10],
            'class_weight': [None, 'balanced']
        },
        'baseline_f1': baseline_results.get('LinearSVC', None)
    }
}

# ============================================================
# 4. НОВЫЕ МОДЕЛИ (RandomizedSearch)
# ============================================================

randomsearch_models = {
    'XGBoost': {
        'model': XGBClassifier(random_state=42, n_jobs=-1, verbosity=0),
        'params': {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.05, 0.1, 0.2, 0.3],
            'subsample': [0.8, 0.9, 1.0]
        },
        'n_iter': 20
    },
    
    'KNeighbors': {
        'model': KNeighborsClassifier(n_jobs=-1),
        'params': {
            'n_neighbors': [5, 10, 15, 20, 25],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'p': [1, 2]
        },
        'n_iter': 15
    },
    
    'AdaBoost': {
        'model': AdaBoostClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 150, 200],
            'learning_rate': [0.1, 0.3, 0.5, 0.7, 1.0],
            'algorithm': ['SAMME', 'SAMME.R']
        },
        'n_iter': 15
    }
}

# ============================================================
# 5. ФУНКЦИЯ СОХРАНЕНИЯ РЕЗУЛЬТАТОВ
# ============================================================

def save_results_incremental(results, model_name, filename='hybrid_search_results.json'):
    """Сохраняет результаты после каждой модели"""
    
    # Загружаем существующие результаты (если есть)
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
        except:
            all_results = []
    else:
        all_results = []
    
    # Добавляем новый результат
    all_results.append(results)
    
    # Сохраняем
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Сохраняем копию с именем модели
    safe_name = model_name.replace(' ', '_').replace('/', '_')
    individual_file = f'model_result_{safe_name}.json'
    with open(individual_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"💾 Сохранено: {individual_file}")
    
    return all_results

# ============================================================
# 6. ОБУЧЕНИЕ: GridSearch для проверенных моделей
# ============================================================

print("\n" + "=" * 70)
print("📌 ЭТАП 1: GridSearch (проверенные модели)")
print("=" * 70)

all_results = []
best_models = {}

for name, config in gridsearch_models.items():
    print(f"\n{'=' * 70}")
    print(f"🎯 {name} (GridSearch)")
    print(f"{'=' * 70}")
    
    if config['baseline_f1']:
        print(f"📊 Базовый F1: {config['baseline_f1']:.4f}")
    
    start = time.time()
    
    grid = GridSearchCV(
        estimator=config['model'],
        param_grid=config['params'],
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    
    grid.fit(X_train, y_train)
    elapsed = time.time() - start
    
    y_pred = grid.predict(X_test)
    y_proba = grid.predict_proba(X_test)[:, 1] if hasattr(grid, 'predict_proba') else None
    
    improvement = (grid.best_score_ - config['baseline_f1']) if config['baseline_f1'] else None
    
    result = {
        'model': name,
        'search_type': 'GridSearch',
        'best_params': grid.best_params_,
        'cv_f1': grid.best_score_,
        'test_f1': f1_score(y_test, y_pred),
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        'baseline_f1': config['baseline_f1'],
        'improvement': improvement,
        'time_min': elapsed / 60,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    best_models[name] = grid
    all_results = save_results_incremental(result, name)
    
    print(f"⏱️ Время: {elapsed/60:.1f} мин")
    print(f"📋 Параметры: {grid.best_params_}")
    print(f"📊 CV F1: {grid.best_score_:.4f}")
    print(f"📊 Test F1: {f1_score(y_test, y_pred):.4f}")
    if improvement:
        print(f"📈 Улучшение: {improvement:+.4f}")

# ============================================================
# 7. ОБУЧЕНИЕ: RandomizedSearch для новых моделей
# ============================================================

print("\n" + "=" * 70)
print("📌 ЭТАП 2: RandomizedSearch (новые модели)")
print("=" * 70)

for name, config in randomsearch_models.items():
    print(f"\n{'=' * 70}")
    print(f"🎯 {name} (RandomizedSearch)")
    print(f"{'=' * 70}")
    
    start = time.time()
    
    search = RandomizedSearchCV(
        estimator=config['model'],
        param_distributions=config['params'],
        n_iter=config['n_iter'],
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=0,
        random_state=42
    )
    
    search.fit(X_train, y_train)
    elapsed = time.time() - start
    
    y_pred = search.predict(X_test)
    y_proba = search.predict_proba(X_test)[:, 1] if hasattr(search, 'predict_proba') else None
    
    result = {
        'model': name,
        'search_type': 'RandomizedSearch',
        'best_params': search.best_params_,
        'cv_f1': search.best_score_,
        'test_f1': f1_score(y_test, y_pred),
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        'baseline_f1': None,
        'improvement': None,
        'time_min': elapsed / 60,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    best_models[name] = search
    all_results = save_results_incremental(result, name)
    
    print(f"⏱️ Время: {elapsed/60:.1f} мин")
    print(f"📋 Параметры: {search.best_params_}")
    print(f"📊 CV F1: {search.best_score_:.4f}")
    print(f"📊 Test F1: {f1_score(y_test, y_pred):.4f}")

# ============================================================
# 8. ФИНАЛЬНЫЙ АНСАМБЛЬ (ТОП-5 МОДЕЛЕЙ)
# ============================================================

print("\n" + "=" * 70)
print("📌 ЭТАП 3: Финальный ансамбль")
print("=" * 70)

results_sorted = sorted(all_results, key=lambda x: x['test_f1'], reverse=True)
top_5 = results_sorted[:5]

print("\n🏆 Топ-5 моделей для ансамбля:")
estimators = []

for i, res in enumerate(top_5, 1):
    print(f"{i}. {res['model']} (F1={res['test_f1']:.4f}, {res['search_type']})")
    estimators.append((res['model'], best_models[res['model']].best_estimator_))

voting = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
voting.fit(X_train, y_train)

y_pred_ensemble = voting.predict(X_test)
y_proba_ensemble = voting.predict_proba(X_test)[:, 1]

ensemble_result = {
    'model': 'VotingEnsemble_Final',
    'search_type': 'Ensemble',
    'test_f1': f1_score(y_test, y_pred_ensemble),
    'test_accuracy': accuracy_score(y_test, y_pred_ensemble),
    'test_roc_auc': roc_auc_score(y_test, y_proba_ensemble),
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
}

all_results = save_results_incremental(ensemble_result, 'VotingEnsemble_Final')

print(f"\n🎯 АНСАМБЛЬ:")
print(f"   Accuracy: {ensemble_result['test_accuracy']:.4f}")
print(f"   F1-Score: {ensemble_result['test_f1']:.4f}")
print(f"   ROC-AUC:  {ensemble_result['test_roc_auc']:.4f}")

# ============================================================
# 9. ФИНАЛЬНЫЙ ОТЧЁТ
# ============================================================

total_elapsed = time.time() - total_start

print("\n" + "=" * 70)
print("📊 ФИНАЛЬНЫЙ ОТЧЁТ")
print("=" * 70)

print(f"\n{'#':<3} {'Модель':<20} {'Тип':<15} {'Test F1':<10} {'Улучшение':<10} {'Время':<10}")
print("-" * 70)

for i, res in enumerate(results_sorted, 1):
    imp_str = f"{res['improvement']:+.4f}" if res['improvement'] is not None else "N/A"
    print(f"{i:<3} {res['model']:<20} {res['search_type']:<15} {res['test_f1']:<10.4f} {imp_str:<10} {res['time_min']:<10.1f} мин")

print(f"\n⏱️ ОБЩЕЕ ВРЕМЯ: {total_elapsed/60:.1f} минут ({total_elapsed/3600:.2f} часов)")

print("\n✅ Результаты сохранены в:")
print("   • hybrid_search_results.json (все модели)")
print("   • model_result_*.json (каждая модель отдельно)")

print(f"\n🕐 Конец: {time.strftime('%H:%M:%S')}")
print("=" * 70)