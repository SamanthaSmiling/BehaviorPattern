

import random  # Standard library first
import string  # Standard library
import pandas as pd  # Third-party libraries
import numpy as np  # Third-party libraries


# 设置随机种子
np.random.seed(42)

# 定义常量
NUM_USERS = 10000  
ZODIAC_SIGNS = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']
MBTI_TYPES = ['INTJ', 'INTP', 'ENTJ', 'ENTP', 'INFJ', 'INFP', 'ENFJ', 'ENFP', 'ISTJ', 'ISFJ', 'ESTJ', 'ESFJ', 'ISTP', 'ISFP', 'ESTP', 'ESFP']

# 12位的随机数字，唯一
user_ids = [''.join(random.choices(string.digits, k=12)) for _ in range(NUM_USERS)]

# 18到90岁，其中25-45岁的概率更高(方便做分析展示)
age_groups = [range(18, 26), range(26, 46), range(46, 91)]
age_probs = [0.2, 0.6, 0.2]  
ages = np.random.choice(np.concatenate([np.array(list(group)) for group in age_groups]), 
                        size=NUM_USERS, 
                        p=np.concatenate([np.full(len(group), prob/len(group)) for group, prob in zip(age_groups, age_probs)]))

# 12星座中任一个，5%的数据为空用以模拟实际情况中无法获取标签值的情况
zodiac_signs = np.random.choice(ZODIAC_SIGNS + [None], size=NUM_USERS, p=[0.079]*12 + [1-0.079*12])

# 16种中的一个，20%数据为空用以模拟实际情况中无法获取标签值的情况
mbti_types = np.random.choice(MBTI_TYPES + [None], size=NUM_USERS, p=[0.05]*16 + [0.2])
# 88VIP and MBTI
def is_88vip(mbti):
    if mbti in ['INTP', 'ENTJ']:
        return np.random.choice([True, False], p=[0.9, 0.1])  # 90% chance for INTP, ENTJ
    else:
        return np.random.choice([True, False], p=[0.5, 0.5])  # Normal 50-50 chance for others

def generate_random_behavior(days, low, high):
    return np.random.randint(low, high + 1, size=NUM_USERS)

# 购买金额和单数
def generate_random_purchases(amount_range, count_range):
    scale = 3.0  # 指数分布的 scale 
    random_values = np.random.exponential(scale, size=NUM_USERS)  
    
    random_values = random_values * (amount_range[1] - amount_range[0]) + amount_range[0]
    
    # 返回购买金额和单数
    return random_values, np.random.randint(*count_range, size=NUM_USERS)

data = {
    'User ID': user_ids,
    'Predicted Age': ages,
    'Predicted Zodiac Sign': zodiac_signs,
    'Predicted MBTI': mbti_types,
    
    # 10天内类目行为次数
    '10-day Cat1 Actions': generate_random_behavior(10, 1, 30),
    '10-day Cat2 Actions': generate_random_behavior(10, 1, 30),
    '10-day Cat3 Actions': generate_random_behavior(10, 1, 30),
    
    # 30天内类目行为次数
    '30-day Cat1 Actions': generate_random_behavior(30, 1, 60),
    '30-day Cat2 Actions': generate_random_behavior(30, 1, 60),
    '30-day Cat3 Actions': generate_random_behavior(30, 1, 60),
    
    # 90天内类目行为次数
    '90-day Cat1 Actions': generate_random_behavior(90, 1, 150),
    '90-day Cat2 Actions': generate_random_behavior(90, 1, 150),
    '90-day Cat3 Actions': generate_random_behavior(90, 1, 150),
    
    # 是否88VIP
    # 'Is 88VIP': np.random.choice([True, False], size=NUM_USERS),
    'Is 88VIP' : np.array([is_88vip(mbti) for mbti in mbti_types]),
    
    # 近7天类目1购买金额和单数
    'Last 7-day Cat1 Purchase Amount': generate_random_purchases((0, 10000), (0, 100))[0],
    'Last 7-day Cat1 Purchase Count': generate_random_purchases((0, 10000), (0, 100))[1],
    
    # 去年同期7天类目1购买金额和单数
    'Last Year 7-day Cat1 Purchase Amount': generate_random_purchases((0, 10000), (0, 100))[0],
    'Last Year 7-day Cat1 Purchase Count': generate_random_purchases((0, 10000), (0, 100))[1],
    
    # 上月同期7天类目1购买金额和单数
    'Last Month 7-day Cat1 Purchase Amount': generate_random_purchases((0, 10000), (0, 100))[0],
    'Last Month 7-day Cat1 Purchase Count': generate_random_purchases((0, 10000), (0, 100))[1],
    
    # 近7天类目2购买金额和单数
    'Last 7-day Cat2 Purchase Amount': generate_random_purchases((0, 20000), (0, 300))[0],
    'Last 7-day Cat2 Purchase Count': generate_random_purchases((0, 10000), (0, 100))[1],
    
    # 去年同期7天类目2购买金额和单数
    'Last Year 7-day Cat2 Purchase Amount': generate_random_purchases((0, 20000), (0, 300))[0],
    'Last Year 7-day Cat2 Purchase Count': generate_random_purchases((0, 10000), (0, 100))[1],
    
    # 上月同期7天类目2购买金额和单数
    'Last Month 7-day Cat2 Purchase Amount': generate_random_purchases((0, 20000), (0, 300))[0],
    'Last Month 7-day Cat2 Purchase Count': generate_random_purchases((0, 10000), (0, 100))[1],
    
    # 近7天类目3购买金额和单数
    'Last 7-day Cat3 Purchase Amount': generate_random_purchases((0, 50000), (0, 100))[0],
    'Last 7-day Cat3 Purchase Count': generate_random_purchases((0, 10000), (0, 100))[1],
    
    # 去年同期7天类目3购买金额和单数
    'Last Year 7-day Cat3 Purchase Amount': generate_random_purchases((0, 50000), (0, 100))[0],
    'Last Year 7-day Cat3 Purchase Count': generate_random_purchases((0, 10000), (0, 100))[1],
    
    # 上月同期7天类目3购买金额和单数
    'Last Month 7-day Cat3 Purchase Amount': generate_random_purchases((0, 50000), (0, 100))[0],
    'Last Month 7-day Cat3 Purchase Count': generate_random_purchases((0, 10000), (0, 100))[1],
}


df = pd.DataFrame(data)
df.to_csv('data/user_data.csv', index=False)

print("CSV file created successfully.")
