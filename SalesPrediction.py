import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap
# 读取 CSV 文件
df = pd.read_csv('user_data.csv')

# 定义权重
weights = {
    'yoy': 0.2,  # 去年同期
    'mom': 0.2,  # 上月同期
    '90_days': 0.1,  # 近90天行为次数
    '60_days': 0.1,  # 近60天行为次数
    '10_days': 0.1,  # 近10天行为次数
    'random': 0.3   # 随机因素
}

# 随机因素
random_factor = np.random.uniform(0.9, 1.1, size=len(df))

# 基础预测值的计算（不考虑每日随机系数）
category_sales = {
    'Category 1': (
        weights['yoy'] * df['Last Year 7-day Cat1 Purchase Count'] +
        weights['mom'] * df['Last Month 7-day Cat1 Purchase Count'] +
        weights['90_days'] * df['90-day Cat1 Actions'] +
        weights['60_days'] * df['30-day Cat1 Actions'] +  # 假设近60天用30天行为次数代替
        weights['10_days'] * df['10-day Cat1 Actions'] +
        weights['random'] * random_factor * df['Last 7-day Cat1 Purchase Count']
    ) / 30,  # 每天的基础销量预测
    'Category 2': (
        weights['yoy'] * df['Last Year 7-day Cat2 Purchase Count'] +
        weights['mom'] * df['Last Month 7-day Cat2 Purchase Count'] +
        weights['90_days'] * df['90-day Cat2 Actions'] +
        weights['60_days'] * df['30-day Cat2 Actions'] +
        weights['10_days'] * df['10-day Cat2 Actions'] +
        weights['random'] * random_factor * df['Last 7-day Cat2 Purchase Count']
    ) / 30,
    'Category 3': (
        weights['yoy'] * df['Last Year 7-day Cat3 Purchase Count'] +
        weights['mom'] * df['Last Month 7-day Cat3 Purchase Count'] +
        weights['90_days'] * df['90-day Cat3 Actions'] +
        weights['60_days'] * df['30-day Cat3 Actions'] +
        weights['10_days'] * df['10-day Cat3 Actions'] +
        weights['random'] * random_factor * df['Last 7-day Cat3 Purchase Count']
    ) / 30
}

# 为未来30天每一天生成一个随机系数 (0.5 到 2.1 之间)
daily_random_factors = np.random.uniform(0.8, 1.2, size=(30, 3))

# 创建未来30天的销量预测数据，应用每日的随机系数
future_sales = pd.DataFrame({
    f'Day {i+1}': [
        category_sales['Category 1'].sum() * daily_random_factors[i, 0],
        category_sales['Category 2'].sum() * daily_random_factors[i, 1],
        category_sales['Category 3'].sum() * daily_random_factors[i, 2]
    ] for i in range(30)
}, index=['Category 1', 'Category 2', 'Category 3'])

# 计算每日销量总和
total_daily_sales = future_sales.sum(axis=0)

# 创建Panel
fig, axes = plt.subplots(1, 2, figsize=(16, 8))


# ---- 图表1: 三个类目未来30天销量趋势图 ---- #
future_sales.T.plot(ax=axes[0], marker='o')
axes[0].set_title('Future 30-day Sales Trend by Category')
axes[0].set_xlabel('Day')
axes[0].set_ylabel('Sales')
axes[0].legend(title='Category') 

# ---- 图表2: 每日销量占比的累积占比图 ---- #
future_sales.div(total_daily_sales, axis=1).T.plot.area(ax=axes[1], stacked=True, cmap='Set2', alpha=0.7)
axes[1].set_title('Daily Sales Percentage by Category')
axes[1].set_xlabel('Day')
axes[1].set_ylabel('Percentage of Total Sales')
axes[1].legend(title='Category')



conclusion_text = "For illustration purposes: In practice, sales predictions are based on long-term and short-term historical data, current traffic levels, competitor actions, and risk assessments."
wrapped_text = "\n".join(textwrap.wrap(conclusion_text, width=80))
fig.text(0.5, 0.9, wrapped_text, ha='center', fontsize=12)
# Adjust layout to ensure the text fits well
plt.tight_layout(rect=[0, 0.03, 1, 0.95])




plt.tight_layout()
plt.show()
