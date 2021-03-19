import pandas as pd

# For previewing all columns and rows
pd.options.display.max_columns = None
pd.options.display.max_rows = None

load_path = '/Users/shimi/Desktop/data science/mortages/data/retail_data.csv'
store_path = '/Users/shimi/Desktop/data science/mortages/data/potential_customers.csv'
save_mortgage_path = '/Users/shimi/Desktop/data science/mortages/data/test_potential_customers.csv'
save_confusion_matrix_plot = '/Users/shimi/Desktop/data science/mortages/plots/confusion_matrix.png'