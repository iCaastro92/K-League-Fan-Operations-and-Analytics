import pandas as pd
import numpy as np
from faker import Faker
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize Faker and seed for reproducibility
fake = Faker()
Faker.seed(42)
np.random.seed(42)

# 1. Define K-League parameters based on research
num_fans = 5000

teams = ['FC Seoul', 'Ulsan HD', 'Jeju SK', 'Jeonbuk Hyundai Motors', 'Gangwon FC']
team_weights = [0.30, 0.25, 0.15, 0.20, 0.10] # Market share approximation

age_segments = ['Under_18', '18_34', '35_54', '55_Plus']
age_weights = [0.15, 0.40, 0.35, 0.10] # Targeting young adults and families

membership_types = ['free_registered', 'paid_membership', 'season_ticket_partial', 'season_ticket_full']
membership_weights = [0.50, 0.30, 0.12, 0.08] # Pyramid of engagement

# 2. Generate Base Fan Data
fan_data = {
    'fan_id': [f"FAN-{str(i).zfill(6)}" for i in range(1, num_fans + 1)],
    'favorite_team': np.random.choice(teams, num_fans, p=team_weights),
    'age_segment': np.random.choice(age_segments, num_fans, p=age_weights),
    'membership_type': np.random.choice(membership_types, num_fans, p=membership_weights),
    'app_registered': np.random.choice([True, False], num_fans, p=[0.75, 0.25]),
    'push_opt_in': np.random.choice([True, False], num_fans, p=[0.60, 0.40])
}

fans_df = pd.DataFrame(fan_data)

# Force logic: If you have a season ticket, you MUST be app registered
fans_df.loc[fans_df['membership_type'].str.contains('season'), 'app_registered'] = True

# Display the first 5 rows to verify
print(fans_df.head())
print("\nDataset Info:")
print(fans_df.info())

# 3. Generate Transactional & Behavioral Data
# Define logic conditions based on membership tier
conditions = [
    fans_df['membership_type'] == 'season_ticket_full',
    fans_df['membership_type'] == 'season_ticket_partial',
    fans_df['membership_type'] == 'paid_membership',
    fans_df['membership_type'] == 'free_registered'
]

# Simulate tickets purchased per season based on tier
tickets_purchased_ranges = [
    np.random.randint(15, 20, size=num_fans),  # Full season
    np.random.randint(5, 12, size=num_fans),   # Partial
    np.random.randint(2, 6, size=num_fans),    # Paid mem (frequent single buyers)
    np.random.randint(1, 3, size=num_fans)     # Free (occasional)
]

fans_df['tickets_purchased_2026'] = np.select(conditions, tickets_purchased_ranges, default=1)

# Simulate attendance rates. 
# Insight: Season ticket holders often skip matches (lower rate), while single-ticket buyers rarely miss (high rate).
attendance_rates = [
    np.random.uniform(0.60, 0.90, size=num_fans),
    np.random.uniform(0.70, 0.95, size=num_fans),
    np.random.uniform(0.85, 1.00, size=num_fans),
    np.random.uniform(0.90, 1.00, size=num_fans)
]

fans_df['attendance_rate'] = np.select(conditions, attendance_rates, default=0.9)

# Calculate actual matches attended (rounded down to whole numbers)
fans_df['matches_attended_2026'] = (fans_df['tickets_purchased_2026'] * fans_df['attendance_rate']).astype(int)

# Calculate No-Show rate (Critical feature for churn prediction)
fans_df['no_show_rate'] = 1 - (fans_df['matches_attended_2026'] / fans_df['tickets_purchased_2026'])

# Assign base marginal ticket price (KRW)
fans_df['avg_ticket_price_krw'] = np.where(
    fans_df['membership_type'].str.contains('season'),
    0,  # Marginal cost per extra match is 0 for season ticket holders
    np.random.randint(16000, 25000, size=num_fans) # Based on Perplexity research
)

# Display the behavioral subset
print(fans_df[['fan_id', 'membership_type', 'tickets_purchased_2026', 'matches_attended_2026', 'no_show_rate']].head(10))

# 4. Generate Merchandise Order Data (Relational Table)
merch_categories = ['replica_jersey', 'player_jersey', 'scarf_flag', 'casual_apparel', 'mascot_item']
# Weights based on typical sports retail behavior
merch_weights = [0.30, 0.20, 0.25, 0.15, 0.10] 

# Determine how many items each fan bought based on their tier (using previous conditions)
merch_purchase_counts = np.select(
    conditions,
    [
        np.random.randint(2, 6, size=num_fans),  # Full season: high engagement
        np.random.randint(1, 4, size=num_fans),  # Partial
        np.random.randint(0, 3, size=num_fans),  # Paid mem
        np.random.randint(0, 2, size=num_fans)   # Free
    ],
    default=0
)

# Create the transactions list
merch_transactions = []
for i, fan_id in enumerate(fans_df['fan_id']):
    num_items = merch_purchase_counts[i]
    for _ in range(num_items):
        # Generate individual item transactions
        merch_transactions.append({
            'fan_id': fan_id,
            'product_category': np.random.choice(merch_categories, p=merch_weights),
            'purchase_amount_krw': np.random.randint(15000, 150000) # Broad price range for merch
        })

merch_df = pd.DataFrame(merch_transactions)

print(f"Total merchandise transactions generated: {len(merch_df)}")
print("\nMerchandise DataFrame Output:")
print(merch_df.head(10))

# 5. Data Aggregation and Merging
# Group merchandise transactions by fan_id to get total spend and item count
merch_summary = merch_df.groupby('fan_id').agg(
    total_merch_spend_krw=('purchase_amount_krw', 'sum'),
    total_merch_items=('product_category', 'count')
).reset_index()

# Merge the summary back into the main fans dataframe
crm_df = pd.merge(fans_df, merch_summary, on='fan_id', how='left')

# Fill NaN values (Not a Number) with 0 for fans who didn't buy any merchandise
crm_df['total_merch_spend_krw'] = crm_df['total_merch_spend_krw'].fillna(0)
crm_df['total_merch_items'] = crm_df['total_merch_items'].fillna(0)

# Display the merged dataset structure
print("Merged CRM Dataset - Key Columns:")
print(crm_df[['fan_id', 'membership_type', 'total_merch_spend_krw', 'total_merch_items']].head(10))

# 6. Calculating Customer Lifetime Value (CLV) for 2026
# Calculate total revenue from tickets
crm_df['total_ticket_spend_krw'] = crm_df['tickets_purchased_2026'] * crm_df['avg_ticket_price_krw']

# Calculate CLV (Total Ticket Spend + Total Merchandise Spend)
crm_df['clv_2026_krw'] = crm_df['total_ticket_spend_krw'] + crm_df['total_merch_spend_krw']

# Sort and display the top 5 fans by CLV to verify our VIPs
top_vips = crm_df.sort_values(by='clv_2026_krw', ascending=False).head(5)
print("Top 5 VIP Fans by CLV:")
print(top_vips[['fan_id', 'membership_type', 'total_ticket_spend_krw', 'total_merch_spend_krw', 'clv_2026_krw']])

# 7. Exploratory Data Analysis (EDA) - CLV by Membership Type
import matplotlib.pyplot as plt
import seaborn as sns

# Set the visual style
sns.set_theme(style="whitegrid")

# Create a figure
plt.figure(figsize=(10, 6))

# Boxplot for CLV based on Membership Type (Updated for Seaborn v0.14.0+)
sns.boxplot(
    data=crm_df, 
    x='membership_type', 
    y='clv_2026_krw', 
    hue='membership_type',  # <-- FIX: Asignado a la misma variable que 'x'
    palette='viridis',
    legend=False            # <-- FIX: Apagamos la leyenda redundante
)

plt.title('K-League Fans: CLV Distribution by Membership Type (2026)', fontsize=14)
plt.xlabel('Membership Type', fontsize=12)
plt.ylabel('Customer Lifetime Value (KRW)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()

# Save and show the plot
plt.savefig('clv_by_membership.png')
plt.show()

# 8. Correlation Analysis
# Select only numerical columns for the correlation matrix
numerical_cols = [
    'tickets_purchased_2026', 
    'matches_attended_2026', 
    'no_show_rate', 
    'total_merch_spend_krw', 
    'clv_2026_krw'
]
corr_matrix = crm_df[numerical_cols].corr()

plt.figure(figsize=(8, 6))
# Create a heatmap to visualize correlations
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Correlation Heatmap: Fan Behavior Metrics', fontsize=14)
plt.tight_layout()

# Save and show the plot
plt.savefig('correlation_heatmap.png')
plt.show()

# Export the dataset to a CSV file for Phase 3
crm_df.to_csv("k_league_crm_data.csv", index=False)