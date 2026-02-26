# ============================================================================
# AMES HOUSING PRICE PREDICTION - KAGGLE COMPETITION
# С визуализацией корреляции и важности признаков
# ============================================================================

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import os

# Получаем путь к текущему скрипту
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Визуализация
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Моделирование
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import mean_squared_error, make_scorer

# Загрузка данных с правильными путями
print("Загрузка данных...")
train = pd.read_csv(os.path.join(SCRIPT_DIR, 'train.csv'))
test = pd.read_csv(os.path.join(SCRIPT_DIR, 'test.csv'))
sample_submission = pd.read_csv(os.path.join(SCRIPT_DIR, 'sample_submission.csv'))

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Сохраняем Id для submission
train_ID = train['Id']
test_ID = test['Id']

# Удаляем Id из данных
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)

# ============================================================================
# EDA и визуализация
# ============================================================================

print("\n" + "="*80)
print("EXPLORATORY DATA ANALYSIS")
print("="*80)

# Основная статистика
print("\nОсновные статистики целевой переменной:")
print(train['SalePrice'].describe())

# Проверка на пропуски
print("\nПропущенные значения в train (топ-15):")
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum() / train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(15))

# Визуализация распределения SalePrice
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(train['SalePrice'], bins=50, edgecolor='black', alpha=0.7)
axes[0].set_title('Distribution of SalePrice', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Price')
axes[0].set_ylabel('Frequency')

# Логарифмическое преобразование
train['SalePrice'] = np.log1p(train['SalePrice'])

axes[1].hist(train['SalePrice'], bins=50, edgecolor='black', alpha=0.7, color='green')
axes[1].set_title('Distribution of Log(SalePrice)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Log(Price)')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'eda_visualization.png'), dpi=300, bbox_inches='tight')
print("\n✓ Визуализация сохранена: eda_visualization.png")
plt.show()

# ============================================================================
# Корреляционная матрица
# ============================================================================

print("\n" + "="*80)
print("CORRELATION ANALYSIS")
print("="*80)

# Числовые признаки для корреляции
numeric_features = train.select_dtypes(include=[np.number]).columns.tolist()

# Корреляция с SalePrice
corr_matrix = train[numeric_features].corr()
saleprice_corr = corr_matrix['SalePrice'].sort_values(ascending=False)

print("\nТоп-15 признаков с наибольшей корреляцией с SalePrice:")
print(saleprice_corr.head(16))  # включая саму SalePrice

# Визуализация корреляционной матрицы (топ-20 признаков)
top_corr_features = saleprice_corr.head(20).index.tolist()
top_corr_matrix = train[top_corr_features].corr()

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(top_corr_matrix, dtype=bool))
sns.heatmap(top_corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix - Top 20 Features', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
print("\n✓ Корреляционная матрица сохранена: correlation_matrix.png")
plt.show()

# ============================================================================
# Предобработка данных
# ============================================================================

print("\n" + "="*80)
print("DATA PREPROCESSING")
print("="*80)

# Объединяем train и test для совместной обработки
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values

all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)

print(f"all_data size: {all_data.shape}")

# Заполнение пропусков
# PoolQC - заполняем None
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")

# MiscFeature
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")

# Alley
all_data["Alley"] = all_data["Alley"].fillna("None")

# Fence
all_data["Fence"] = all_data["Fence"].fillna("None")

# FireplaceQu
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

# LotFrontage - заполняем медианой по Neighborhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median())
)

# Garage features
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    all_data[col] = all_data[col].fillna('None')
    
all_data['GarageYrBlt'] = all_data['GarageYrBlt'].fillna(0)
all_data['GarageArea'] = all_data['GarageArea'].fillna(0)
all_data['GarageCars'] = all_data['GarageCars'].fillna(0)

# Basement features
for col in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']:
    all_data[col] = all_data[col].fillna('None')
    
all_data['BsmtFinSF1'] = all_data['BsmtFinSF1'].fillna(0)
all_data['BsmtFinSF2'] = all_data['BsmtFinSF2'].fillna(0)
all_data['BsmtUnfSF'] = all_data['BsmtUnfSF'].fillna(0)
all_data['TotalBsmtSF'] = all_data['TotalBsmtSF'].fillna(0)
all_data['BsmtFullBath'] = all_data['BsmtFullBath'].fillna(0)
all_data['BsmtHalfBath'] = all_data['BsmtHalfBath'].fillna(0)

# MasVnrType and MasVnrArea
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

# MSZoning
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

# Utilities
all_data = all_data.drop(['Utilities'], axis=1)

# Functional
all_data["Functional"] = all_data["Functional"].fillna("Typ")

# Electrical
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

# KitchenQual
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

# Exterior1st and Exterior2nd
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

# SaleType
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

# MSSubClass как категориальный
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

# Проверка оставшихся пропусков
print(f"\nОставшиеся пропуски: {all_data.isnull().sum().sum()}")

# Кодирование категориальных признаков
from sklearn.preprocessing import LabelEncoder

categorical_cols = all_data.select_dtypes(include=['object']).columns.tolist()

print(f"\nКодирование {len(categorical_cols)} категориальных признаков...")

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    all_data[col] = le.fit_transform(all_data[col].astype(str))
    label_encoders[col] = le

print("✓ Кодирование завершено")

# Разделение обратно на train и test
train_processed = all_data[:ntrain]
test_processed = all_data[ntrain:]

print(f"\nTrain processed shape: {train_processed.shape}")
print(f"Test processed shape: {test_processed.shape}")

# ============================================================================
# Моделирование
# ============================================================================

print("\n" + "="*80)
print("MODELING")
print("="*80)

# Функция для оценки моделей
def rmsle_cv(model):
    kf = KFold(5, shuffle=True, random_state=42).get_n_splits(train_processed.values)
    rmse = np.sqrt(-cross_val_score(model, train_processed.values, y_train, 
                                     scoring="neg_mean_squared_error", cv=kf))
    return rmse

# Импортируем модели
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import cross_val_score, KFold
import xgboost as xgb
import lightgbm as lgb

print("\nОбучение моделей...")

# Lasso
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
score = rmsle_cv(lasso)
print(f"\nLasso: {score.mean():.4f} ({score.std():.4f})")

# ElasticNet
elastic_net = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=3))
score = rmsle_cv(elastic_net)
print(f"ElasticNet: {score.mean():.4f} ({score.std():.4f})")

# Kernel Ridge
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
score = rmsle_cv(KRR)
print(f"Kernel Ridge: {score.mean():.4f} ({score.std():.4f})")

# Gradient Boosting
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state=5)
score = rmsle_cv(GBoost)
print(f"Gradient Boosting: {score.mean():.4f} ({score.std():.4f})")

# XGBoost
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1, random_state=7, 
                             nthread=-1)
score = rmsle_cv(model_xgb)
print(f"XGBoost: {score.mean():.4f} ({score.std():.4f})")

# LightGBM
model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin=55, bagging_fraction=0.8,
                              bagging_freq=5, feature_fraction=0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf=6, min_sum_hessian_in_leaf=11)
score = rmsle_cv(model_lgb)
print(f"LightGBM: {score.mean():.4f} ({score.std():.4f})")

# ============================================================================
# Stacking
# ============================================================================

print("\n" + "="*80)
print("STACKING ENSEMBLE")
print("="*80)

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        for model in self.models_:
            model.fit(X, y)
            
        return self
    
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)

# Усреднение моделей
averaged_models = AveragingModels(models=(GBoost, model_xgb, model_lgb))
score = rmsle_cv(averaged_models)
print(f"\nAveraged models score: {score.mean():.4f} ({score.std():.4f})")

# Обучаем финальную модель
print("\nОбучение финальной модели...")
averaged_models.fit(train_processed.values, y_train)

# Предсказания
print("Создание предсказаний...")
stacked_pred = np.expm1(averaged_models.predict(test_processed.values))

# Создание submission
submission = pd.DataFrame({
    'Id': test_ID,
    'SalePrice': stacked_pred
})

# Сохранение
submission_path = os.path.join(SCRIPT_DIR, 'house_price_submission.csv')
submission.to_csv(submission_path, index=False)
print(f"\n✓ Submission сохранен: {submission_path}")

# Показываем первые строки
print("\nПервые 10 строк submission:")
print(submission.head(10))

print("\n" + "="*80)
print("ГОТОВО!")
print("="*80)