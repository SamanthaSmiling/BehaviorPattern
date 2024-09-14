"""
This module explores correlations between various datasets.
"""

# pylint: disable=C0301

# Third-party imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency



# 读取 CSV 文件
user_data = pd.read_csv('data/user_data.csv')
print(user_data.columns)
# user_data.columns = user_data.columns.str.strip()  # Removes leading/trailing spaces


# Chi-Square test for Zodiac
zodiac_contingency_table = pd.crosstab(user_data['Predicted Zodiac Sign'], user_data['Is 88VIP'])
chi2_zodiac, p_zodiac, _, _ = chi2_contingency(zodiac_contingency_table)

# Chi-Square test for MBTI
mbti_contingency_table = pd.crosstab(user_data['Predicted MBTI'], user_data['Is 88VIP'])
chi2_mbti, p_mbti, _, _ = chi2_contingency(mbti_contingency_table)

# Output the results of Chi-Square Tests
chi2_results = {
    'Zodiac_Chi2': chi2_zodiac,
    'Zodiac_p_value': p_zodiac,
    'MBTI_Chi2': chi2_mbti,
    'MBTI_p_value': p_mbti
}

# chi2_results

# Step 3: Visualization with heatmaps for Zodiac and MBTI correlation with 88VIP status
# Conclusion based on the Chi-Square results:
if p_zodiac < 0.05 and p_mbti < 0.05:
    conclusion_text = (
        "Both Zodiac signs and MBTI types show statistically significant correlations with 88VIP status.\n"
        f"Zodiac Chi-Square: {chi2_zodiac:.2f}, p-value: {p_zodiac:.4f}.\n"
        f"MBTI Chi-Square: {chi2_mbti:.2f}, p-value: {p_mbti:.4f}.\n"
    )
elif p_mbti < 0.05:
    conclusion_text = (
        "MBTI types show a stronger correlation with 88VIP status than Zodiac signs.\n"
        f"MBTI Chi-Square: {chi2_mbti:.2f}, p-value: {p_mbti:.4f}.\n"
        f"Zodiac signs do not show significant correlation (p-value: {p_zodiac:.4f})."
    )
elif p_zodiac < 0.05:
    conclusion_text = (
        "Zodiac signs show a stronger correlation with 88VIP status than MBTI types.\n"
        f"Zodiac Chi-Square: {chi2_zodiac:.2f}, p-value: {p_zodiac:.4f}.\n"
        f"MBTI types do not show significant correlation (p-value: {p_mbti:.4f})."
    )
else:
    conclusion_text = (
        "Neither Zodiac signs nor MBTI types show a statistically significant correlation with 88VIP status.\n"
        f"Zodiac p-value: {p_zodiac:.4f}, MBTI p-value: {p_mbti:.4f}."
    )
# Create a figure with 2 subplots (side by side)
fig, axes = plt.subplots(1, 2, figsize=(18, 8))  # 1 row, 2 columns

# Heatmap for Zodiac correlation with 88VIP (left panel)
sns.heatmap(zodiac_contingency_table, annot=True, cmap="Blues", fmt="d", ax=axes[0])
axes[0].set_title('Correlation between Zodiac Signs and 88VIP Status')

# Heatmap for MBTI correlation with 88VIP (right panel)
sns.heatmap(mbti_contingency_table, annot=True, cmap="Greens", fmt="d", ax=axes[1])
axes[1].set_title('Correlation between MBTI Types and 88VIP Status')
# Add summary text below the plots
fig.text(0.5, 0.9, conclusion_text, ha='center', fontsize=12)
# Adjust layout to ensure the text fits well
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Adjust layout so the subplots don't overlap
plt.tight_layout()

# Show the figure
plt.show()

# Existing last line

# Add an extra line here
