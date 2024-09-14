"""
This module contains functions related to behavioral patterns analysis.
"""
# pylint: disable=C0301

# Third-party imports
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# 读取 CSV 文件
df = pd.read_csv('data/user_data.csv')
# ---- 数据预处理 ---- #
df = df.dropna(subset=['Predicted MBTI', 'Predicted Zodiac Sign'])

df['Total Sales'] = df[['90-day Cat1 Actions', '90-day Cat2 Actions', '90-day Cat3 Actions']].sum(axis=1)

# 标准化类目行为次数
scaler = StandardScaler()
df[['Cat1 Actions', 'Cat2 Actions', 'Cat3 Actions']] = scaler.fit_transform(df[['90-day Cat1 Actions', '90-day Cat2 Actions', '90-day Cat3 Actions']])

# ---- 散点图1：用星座颜色、销量大小表示 ---- #
# plt.figure(figsize=(10, 8))
# scatter = plt.scatter(df['Cat1 Actions'], df['Cat2 Actions'], s=df['Total Sales'], c=pd.Categorical(df['Predicted Zodiac Sign']).codes, cmap='tab20', alpha=0.6)
# plt.title('Clustering of Users by Zodiac Sign')
# plt.xlabel('Category 1 Actions (Normalized)')
# plt.ylabel('Category 2 Actions (Normalized)')
# plt.colorbar(scatter, label='Zodiac Sign')
# plt.show()

# ---- 散点图2：用MBTI颜色、销量大小表示 ---- #
plt.figure(figsize=(10, 8))
scatter = plt.scatter(df['Cat2 Actions'], df['Cat3 Actions'], s=df['Total Sales'], c=pd.Categorical(df['Predicted MBTI']).codes, cmap='tab20', alpha=0.6)
plt.title('Clustering of Users by MBTI')
plt.xlabel('Category 2 Actions (Normalized)')
plt.ylabel('Category 3 Actions (Normalized)')
plt.colorbar(scatter, label='MBTI Type')
plt.show()


# ---- 数据预处理 ---- #
df = df.dropna(subset=['Predicted MBTI', 'Predicted Zodiac Sign'])
features = [
    '10-day Cat1 Actions', '10-day Cat2 Actions', '10-day Cat3 Actions',
    '30-day Cat1 Actions', '30-day Cat2 Actions', '30-day Cat3 Actions',
    '90-day Cat1 Actions', '90-day Cat2 Actions', '90-day Cat3 Actions'
]
X = df[features]

# ---- 消费行为对 MBTI 的预测 ---- #
y_mbti = df['Predicted MBTI']
X_train_mbti, X_test_mbti, y_train_mbti, y_test_mbti = train_test_split(X, y_mbti, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_mbti_scaled = scaler.fit_transform(X_train_mbti)
X_test_mbti_scaled = scaler.transform(X_test_mbti)
knn_mbti = KNeighborsClassifier(n_neighbors=5)
knn_mbti.fit(X_train_mbti_scaled, y_train_mbti)
y_pred_mbti = knn_mbti.predict(X_test_mbti_scaled)
mbti_accuracy = accuracy_score(y_test_mbti, y_pred_mbti)
cv_mbti_scores = cross_val_score(knn_mbti, X_train_mbti_scaled, y_train_mbti, cv=5)

# ---- 消费行为对 星座 的预测 ---- #
y_zodiac = df['Predicted Zodiac Sign']
X_train_zodiac, X_test_zodiac, y_train_zodiac, y_test_zodiac = train_test_split(X, y_zodiac, test_size=0.3, random_state=42)
X_train_zodiac_scaled = scaler.fit_transform(X_train_zodiac)
X_test_zodiac_scaled = scaler.transform(X_test_zodiac)
knn_zodiac = KNeighborsClassifier(n_neighbors=5)
knn_zodiac.fit(X_train_zodiac_scaled, y_train_zodiac)
y_pred_zodiac = knn_zodiac.predict(X_test_zodiac_scaled)
zodiac_accuracy = accuracy_score(y_test_zodiac, y_pred_zodiac)
cv_zodiac_scores = cross_val_score(knn_zodiac, X_train_zodiac_scaled, y_train_zodiac, cv=5)

# ---- 创建一个包含两个图表的面板 (Panel) ---- #
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# ---- 图表1: 对比MBTI和星座的准确率 (条形图) ---- #
accuracies = [mbti_accuracy, zodiac_accuracy]
categories = ['MBTI', 'Zodiac Sign']

sns.barplot(x=categories, y=accuracies, palette='Set2', ax=axes[0])
axes[0].set_title('Comparison of Prediction Accuracy: MBTI vs Zodiac Sign')
axes[0].set_xlabel('Category')
axes[0].set_ylabel('Accuracy')
axes[0].set_ylim(0, 1)
axes[0].text(0, mbti_accuracy + 0.02, f'{mbti_accuracy:.2f}', ha='center', fontsize=12)
axes[0].text(1, zodiac_accuracy + 0.02, f'{zodiac_accuracy:.2f}', ha='center', fontsize=12)

# ---- 图表2: 交叉验证准确率分布 (箱线图) ---- #
cv_scores_df = pd.DataFrame({
    'MBTI': cv_mbti_scores,
    'Zodiac Sign': cv_zodiac_scores
})

sns.boxplot(data=cv_scores_df, palette='Set3', ax=axes[1])
axes[1].set_title('Cross-validation Score Distribution: MBTI vs Zodiac Sign')
axes[1].set_ylabel('Accuracy')
axes[1].set_ylim(0, 1)

# 调整布局
plt.tight_layout()

# 显示图表
plt.show()
