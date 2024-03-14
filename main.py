import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Reading the data and removing duplicates
df = pd.read_csv('teleco_market_basket.csv')
print(df.head(8))

# Drop rows where all elements are NaN
df_cleaned = df.dropna(how='all').copy()

# Save the cleaned DataFrame back to CSV if necessary
df_cleaned.to_csv('teleco_market_basket_cleaned.csv', index=False)

# Ensure there's a column that uniquely identifies each transaction
# If not, you can create one based on the DataFrame's index
df_cleaned['TransactionID'] = df_cleaned.index

# Melt the DataFrame
df_melted = df_cleaned.melt(id_vars='TransactionID', value_name='Item').dropna(subset=['Item'])

# Pivot the DataFrame to get the one-hot encoding
df_one_hot = df_melted.pivot_table(index='TransactionID', columns='Item', values='Item',
                                   aggfunc=lambda x: 1, fill_value=0)

# Reset the index to get the DataFrame in the correct shape
df_one_hot.reset_index(inplace=True)
df_one_hot = df_one_hot.drop(columns=['TransactionID'])

# Check the shape of the resulting DataFrame
print(df_one_hot.shape)
df_one_hot.to_csv('one_hot.csv')

# Apply the Apriori algorithm to find frequent itemsets
# min_support is a threshold that defines the minimum support level for an itemset to be considered frequent
frequent_itemsets = apriori(df_one_hot, min_support=0.01, use_colnames=True)

# Generate association rules
# You can adjust the metric and min_threshold according to your needs
# Common metrics include 'confidence', 'lift', and 'leverage'
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

# Display the rules
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
rules.to_csv("rules.csv")

# Assuming you have a one-hot encoded DataFrame 'df_one_hot'
frequent_itemsets = apriori(df_one_hot, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

# Sort the rules by the 'lift' metric in descending order
rules_sorted_by_lift = rules.sort_values('lift', ascending=False)

# Select the top 3 rules
top_3_rules = rules_sorted_by_lift.head(3)

print("Top 3 rules based on lift:")
print(top_3_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
top_3_rules.to_csv('top_three.csv')

