##################################################
##### Import Libraries and Declare Constants #####
##################################################
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import calendar
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import timedelta, date
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from keras.optimizers import Adam
from keras.models import load_model




data_path = 'C:/Users/Andreas Panayiotou/Documents/Forecasting/Data/Franklin_Order_Data.csv'
window_size = 365                       # number of days used to predict
train_proportion = 0.7

#################################
##### Import and Clean Data #####
#################################
df = pd.read_csv(data_path)
df.rename(columns={"total_revenue": "daily_revenue", "total_orders": "order_volume"}, inplace=True)
print(df.columns)

df.drop('non_subscription_orders', axis=1, inplace=True)            # drop to prevent model shortcutting to order volume from umming sub and non-sub orders
df.dropna(inplace=True)                                             # drop na values
df['date'] = pd.to_datetime(df['date'])

target_variables = ['order_volume', 'subscription_orders']



#################################
##### Fill in Missing Dates #####
#################################
# Add in rows where there's no data (creates a row for each sku every day)
# Function to generate all dates within an interval
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

start_date = df.date.min()
end_date = df.date.max()

# Create a dataFrame with all possible combinations of SKUs and dates
all_skus = df.sku.unique()
all_dates = [single_date.strftime("%Y-%m-%d") for single_date in daterange(start_date, end_date)]

index = pd.MultiIndex.from_product([all_dates, all_skus], names=['date', 'sku'])
df_full = pd.DataFrame(index=index).reset_index()
df_full['date'] = pd.to_datetime(df_full['date'])

# Merge new dataframe with the original one
df_complete = pd.merge(df_full, df, on=['date', 'sku'], how='left')

# Fill NAs with 0 for total_orders, subscription_orders, avg_paid and revenue
df_complete[['order_volume', 'subscription_orders', 'avg_paid', 'daily_revenue', 'item_cost']] = df_complete[['order_volume', 'subscription_orders', 'avg_paid', 'daily_revenue', 'item_cost']].fillna(0)

# Create a DataFrame for SKU details
sku_details = df[['sku', 'sku_group', 'product_category']].drop_duplicates()

# Map sku_group and product_category to the new rows
df_complete['sku_group'] = df_complete['sku'].map(sku_details.set_index('sku')['sku_group'].to_dict())
df_complete['product_category'] = df_complete['sku'].map(sku_details.set_index('sku')['product_category'].to_dict())

# Fill the 'item_cost' column with the most recent non-zero value       - currently using max paid each day as item cost - replace with actual cost
df_complete.sort_values(by=['sku', 'date'], inplace=True)
df_complete['item_cost'].replace(to_replace=0, method='ffill', inplace=True)

df = df_complete
df = df.sort_values('date')



#########################
##### Wrangle Data ######
#########################
# Label encode sku and sku_group categorical data
df['sku'], sku_list = pd.factorize(df['sku'])
df['sku_group'], sku_group_list = pd.factorize(df['sku_group'])

sku_dict = {label: {value} for label, value in enumerate(sku_list)}
sku_group_dict = {label: {value} for label, value in enumerate(sku_group_list)}

#print(sku_dict)
#print(sku_group_dict)

# One-hot encode product_categories and drop original
onehot_encoding = pd.get_dummies(df['product_category'])
df = df.drop('product_category',axis = 1)
df = df.join(onehot_encoding)

#print(df.head(100))
print(df.columns)


# APPLY ONE-HOT ENCODING TO SKU AND SKU GROUP?? - slightly harder to associate datapoints with skus and increases dimensionality but removes artifical ordinality
# df = pd.get_dummies(df, columns=['item_sku', 'item_cat'])



#########################
##### Reshape Data  #####
#########################
# List of columns to reshape
columns_to_reshape = ['sku_group', 'item_cost', 'avg_paid', 'daily_revenue', 'order_volume', 'subscription_orders', 'accessories', 'dry', 'other', 'sample', 'supplements', 'treat', 'wet']

# Apply Pivot Table to reshape the DataFrame
reshaped_df = df.pivot_table(index='date', columns='sku', values=columns_to_reshape)

# Flatten Hierarchical Column Index
reshaped_df.columns = ['_'.join(map(str,col)).rstrip('_') for col in reshaped_df.columns.values]

# Reset Index
reshaped_df.reset_index(inplace=True)
df = reshaped_df

# Add year, week and day columns
df['year'] = df['date'].apply(lambda x: x.isocalendar()[0])
df['year'] = df['year'] - df['year'].min()                  # scale years to start from 0
df['week'] = df['date'].apply(lambda x: x.isocalendar()[1])
df['day'] = df['date'].apply(lambda x: x.isocalendar()[2])
df['date'] = df['date'].apply(lambda x: x.timestamp())      # convert date to unix timestamp


print("Column Names: " + df.columns)
print("Columns: " + str(df.shape[1]))
print("Samples: " + str(df.shape[0]))

#########################
##### Preprocessing #####
#########################
# Separate columns containing target variables
feature_columns = [column for column in df.columns if not any(target in column for target in target_variables)]
target_columns = [column for column in df.columns if any(target in column for target in target_variables)]
X = df[feature_columns]
y = df[target_columns]

# Load the standardizers
X_standardiser = joblib.load('C:/Users/Andreas Panayiotou/Documents/Forecasting/Model/X_standardiser.pkl')
y_standardiser = joblib.load('C:/Users/Andreas Panayiotou/Documents/Forecasting/Model/y_standardiser.pkl')

# Load the trained LSTM model
model = load_model('C:/Users/Andreas Panayiotou/Documents/Forecasting/Model/lstm_model_10_94982.h5')



X = X_standardiser.transform(X)
# y = y_standardiser.transform(y)

X = TimeseriesGenerator(X, X, length=window_size, batch_size=1)
# X = TimeseriesGenerator(X, y.values, length=window_size, batch_size=1)




# Make predictions
y_pred = model.predict(X)
y_true = y

# # Inverse standardization on y
# y_pred = y_standardiser.inverse_transform(y_pred)
# y_true = y_standardiser.inverse_transform(y)

# SKUs to forecast
skus_of_interest = [0, 1, 2]  # Replace with actual list of SKUs

# Plot actual vs predicted values for each SKU
for sku in skus_of_interest:
    plt.figure(figsize=(10, 6))
    
    # Extract the desired column from y_true and y_pred
    y_true_sku = y_true.iloc[:, sku]
    y_pred_sku = y_pred[:, sku]

    # Plot y_true and y_pred
    plt.plot(y_true_sku.values, label='True Target 1 for SKU {}'.format(sku))
    plt.plot(y_pred_sku, label='Predicted Target 1 for SKU {}'.format(sku))
    plt.title('Actual vs Predicted Target 1 Values for SKU {}'.format(sku))
    plt.legend()
    plt.show()

