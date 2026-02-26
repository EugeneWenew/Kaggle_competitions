# ============================================================================
# HOUSE PRICES PREDICTION - ИСПРАВЛЕННАЯ ВЕРСИЯ
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# НАСТРОЙКА ПУТЕЙ
# ============================================================================
SCRIPT_DIR = Path(__file__).parent
print("="*80)
print("HOUSE PRICES PREDICTION")
print("="*80)

# ============================================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================================
print("\nЗагрузка данных...")
train = pd.read_csv(SCRIPT_DIR / 'train.csv')
test = pd.read_csv(SCRIPT_DIR / 'test.csv')
sample_submission = pd.read_csv(SCRIPT_DIR / 'sample_submission.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

test_ids = test['Id'].copy()

# ============================================================================
# ПОДГОТОВКА ДАННЫХ - ПОЛНЫЙ ПАЙПЛАЙН
# ============================================================================
print("\n" + "="*80)
print("ПОДГОТОВКА ДАННЫХ")
print("="*80)

# Разделяем признаки и целевую переменную
X_train = train.drop(['Id', 'SalePrice'], axis=1).copy()
y_train = train['SalePrice'].copy()
X_test = test.drop('Id', axis=1).copy()

# 1. Обработка пропусков
print("\n1. Обработка пропусков...")

# Удаляем признаки с >80% пропусков
missing_pct = X_train.isnull().sum() / len(X_train)
cols_to_drop = missing_pct[missing_pct > 0.8].index.tolist()
print(f"Удалено признаков с >80% пропусков: {cols_to_drop}")

X_train.drop(columns=cols_to_drop, inplace=True, errors='ignore')
X_test.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# Числовые - заполняем медианой
numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_features:
    median_val = X_train[col].median()
    X_train[col].fillna(median_val, inplace=True)
    X_test[col].fillna(median_val, inplace=True)

# Категориальные - заполняем модой
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
for col in categorical_features:
    mode_val = X_train[col].mode()[0]
    X_train[col].fillna(mode_val, inplace=True)
    X_test[col].fillna(mode_val, inplace=True)

# 2. Кодирование категориальных признаков
print("\n2. Кодирование категориальных признаков...")

label_encoders = {}
for col in categorical_features:
    if col in X_train.columns:
        le = LabelEncoder()
        # Объединяем train и test для корректного кодирования
        all_values = pd.concat([X_train[col], X_test[col]], axis=0)
        le.fit(all_values)
        X_train[col] = le.transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        label_encoders[col] = le

print(f"Всего признаков после обработки: {X_train.shape[1]}")

# 3. Логарифмирование целевой переменной
print("\n3. Логарифмирование SalePrice...")
y_train_log = np.log1p(y_train)

# 4. Проверка на неконечные значения
print("\n4. Проверка данных на NaN и Inf...")
print(f"NaN в X_train: {X_train.isnull().sum().sum()}")
print(f"NaN в X_test: {X_test.isnull().sum().sum()}")
print(f"Inf в X_train: {np.isinf(X_train.values).sum()}")
print(f"Inf в X_test: {np.isinf(X_test.values).sum()}")

# Заменяем бесконечные значения на NaN, затем на медиану
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

for col in X_train.columns:
    if X_train[col].isnull().any():
        median_val = X_train[col].median()
        X_train[col].fillna(median_val, inplace=True)
        X_test[col].fillna(median_val, inplace=True)

print(f"\nПосле очистки:")
print(f"NaN в X_train: {X_train.isnull().sum().sum()}")
print(f"NaN в X_test: {X_test.isnull().sum().sum()}")

# ============================================================================
# МАСШТАБИРОВАНИЕ
# ============================================================================
print("\n" + "="*80)
print("МАСШТАБИРОВАНИЕ")
print("="*80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_test_scaled shape: {X_test_scaled.shape}")

# ============================================================================
# ФУНКЦИЯ ДЛЯ ОБУЧЕНИЯ И ОЦЕНКИ МОДЕЛЕЙ
# ============================================================================
def train_and_evaluate(X_train, y_train, X_test, models_dict, cv_folds=5, use_log=False):
    """Обучение и оценка моделей с кросс-валидацией"""
    results = []
    predictions = {}
    
    y_target = np.log1p(y_train) if use_log else y_train
    
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    for name, model in models_dict.items():
        print(f"\nОбучение модели: {name}")
        
        try:
            # Кросс-валидация
            cv_scores = cross_val_score(
                model, X_train, y_target, 
                cv=kf, 
                scoring='neg_root_mean_squared_error',
                n_jobs=-1,
                error_score='raise'
            )
            
            rmse_mean = -cv_scores.mean()
            rmse_std = cv_scores.std()
            
            print(f"  CV RMSE: {rmse_mean:.2f} (+/- {rmse_std:.2f})")
            
            # Обучение на всех данных
            model.fit(X_train, y_target)
            
            # Предсказания для теста
            if use_log:
                predictions[name] = np.expm1(model.predict(X_test))
            else:
                predictions[name] = model.predict(X_test)
            
            results.append({
                'name': name,
                'rmse_mean': rmse_mean,
                'rmse_std': rmse_std
            })
            
        except Exception as e:
            print(f"  ❌ Ошибка в модели {name}: {str(e)}")
            continue
    
    return results, predictions

# ============================================================================
# ОБУЧЕНИЕ МОДЕЛЕЙ
# ============================================================================
print("\n" + "="*80)
print("ОБУЧЕНИЕ МОДЕЛЕЙ")
print("="*80)

models = {
    'Ridge': Ridge(alpha=100, random_state=42),
    'Lasso': Lasso(alpha=0.001, random_state=42, max_iter=10000),
    'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42, max_iter=10000),
    'RandomForest': RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1),
    'LightGBM': LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1)
}

results, predictions = train_and_evaluate(
    X_train_scaled, y_train, X_test_scaled,
    models, cv_folds=5, use_log=True
)

# Сортировка по RMSE
results_sorted = sorted(results, key=lambda x: x['rmse_mean'])

print("\n" + "="*80)
print("РЕЗУЛЬТАТЫ (ТОП-3 МОДЕЛИ)")
print("="*80)
for i, res in enumerate(results_sorted[:3], 1):
    print(f"{i}. {res['name']}: RMSE = {res['rmse_mean']:.2f} (+/- {res['rmse_std']:.2f})")

# ============================================================================
# АНСАМБЛЬ ЛУЧШИХ МОДЕЛЕЙ
# ============================================================================
print("\n" + "="*80)
print("АНСАМБЛЬ ЛУЧШИХ МОДЕЛЕЙ")
print("="*80)

if len(results_sorted) >= 3:
    top3_models = results_sorted[:3]
    top3_names = [m['name'] for m in top3_models]
    
    weights = [1.0 / m['rmse_mean'] for m in top3_models]
    weights = [w / sum(weights) for w in weights]
    
    ensemble_pred = np.zeros(len(test_ids))
    for name, weight in zip(top3_names, weights):
        ensemble_pred += predictions[name] * weight
        print(f"{name}: вес = {weight:.3f}")
else:
    print("Недостаточно моделей для ансамбля")
    ensemble_pred = predictions[results_sorted[0]['name']]

# ============================================================================
# СОЗДАНИЕ ФАЙЛА САБМИШНА
# ============================================================================
print("\n" + "="*80)
print("СОЗДАНИЕ ФАЙЛА САБМИШНА")
print("="*80)

best_model_name = results_sorted[0]['name']
best_predictions = predictions[best_model_name]

submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': best_predictions
})

submission.to_csv(SCRIPT_DIR / 'submission_best_model.csv', index=False)
print(f"\nФайл сабмишна сохранён: submission_best_model.csv")
print(f"Лучшая модель: {best_model_name}")
print(f"CV RMSE: {results_sorted[0]['rmse_mean']:.2f}")

submission_ensemble = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': ensemble_pred
})

submission_ensemble.to_csv(SCRIPT_DIR / 'submission_ensemble.csv', index=False)
print(f"Файл сабмишна (ансамбль) сохранён: submission_ensemble.csv")

# ============================================================================
# ВИЗУАЛИЗАЦИЯ
# ============================================================================
print("\n" + "="*80)
print("ВИЗУАЛИЗАЦИЯ")
print("="*80)

# Важность признаков (Random Forest)
rf_full = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
rf_full.fit(X_train, y_train_log)

feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_full.feature_importances_
}).sort_values('importance', ascending=False)

print("\nТоп-20 важных признаков:")
print(feature_importance.head(20))

plt.figure(figsize=(12, 10))
plt.barh(range(20), feature_importance['importance'].head(20).values)
plt.yticks(range(20), feature_importance['feature'].head(20).values)
plt.xlabel('Importance')
plt.title('Top 20 Feature Importances (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(SCRIPT_DIR / 'feature_importance.png', dpi=300)
plt.close()
print("\nГрафик сохранён: feature_importance.png")

print("\n" + "="*80)
print("ГОТОВО!")
print("="*80)