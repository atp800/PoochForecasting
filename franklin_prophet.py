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
from fbprophet import Prophet
import os



data_path = 'C:/Users/Andreas Panayiotou/Documents/Forecasting/Data/Franklin_Order_Data.csv'
model_directory = 'C:/Users/Andreas Panayiotou/Documents/Forecasting/Model/Prophet'
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

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

# Create a dataframe with all possible combinations of SKUs and dates
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

# Pivot to reshape the df
reshaped_df = df.pivot_table(index='date', columns='sku', values=columns_to_reshape)

# Flatten hierarchical column index
reshaped_df.columns = ['_'.join(map(str,col)).rstrip('_') for col in reshaped_df.columns.values]

# Reset df index
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


# Separate columns containing target variables
feature_columns = [column for column in df.columns if not any(target in column for target in target_variables)]
target_columns = [column for column in df.columns if any(target in column for target in target_variables)]


# convert date back to regular timestamp
df['ds'] = pd.to_datetime(df['date'], origin='unix', unit='s')  

# Initialize a dictionary to hold all models
models = {}

# fit a model for each target
for target in target_variables:
    target_cols = [col for col in df.columns if target in col]  # get all columns for this target
    for col in target_cols:
        
        # prepare dataset for this target variable
        # data = df[['ds'] + [col] + feature_columns].rename(columns={col: 'y'})
        data = df[['ds'] + [col] + target_columns].rename(columns={col: 'y'})
        data.columns = pd.io.parsers.ParserBase({'names':data.columns})._maybe_dedup_names(data.columns)

        # fill forward (fill using last valid value)
        data.fillna(method='ffill', inplace=True)

        # create the model
        model = Prophet(holidays_prior_scale=8)
        model.add_country_holidays(country_name='France')

        # # use each column other than the current target as an additional regressor
        # for feature in feature_columns:
        #     if feature != 'ds':  # 'ds' column is not an additional regressor
        #         model.add_regressor(feature)

        # use each column from the target variables (excluding the current target) as an additional regressor
        for target_col in target_columns:
            if target_col != col:  # exclude the current target column
                model.add_regressor(target_col)

        # fit the model
        model.fit(data)

        # store the model
        models[col] = model

        # Save each model
        try:
            joblib.dump(model, f'C:/Users/Andreas Panayiotou/Documents/Forecasting/Model/Prophet/{col}_model.pkl')
            print("Model saved:")
        except Exception as e:
            print(f"Failed to save model {col}: {e}")

        
        print(col)

# initialize a dictionary to hold all forecasts
forecasts = {}

# make a forecast for each model and add to forecasts dictionary
for col, model in models.items():
    future = model.make_future_dataframe(periods=365)  # replace 365 with the number of periods to forecast
    future = pd.concat([future, df[feature_columns].iloc[-365:,:]], axis=1)  # append regressors for future dates
    forecast = model.predict(future)
    forecasts[col] = forecast


# # Save each model
# for col, model in models.items():
#     try:
#         joblib.dump(model, f'C:/Users/Andreas Panayiotou/Documents/Forecasting/Model/Prophet/{col}_model.pkl')
#     except Exception as e:
#         print(f"Failed to save model {col}: {e}")

########################
##### Plot Results #####
########################

# Mke metrics dictionary
metrics = {}

for col, forecast in forecasts.items():
    # Extracting predicted and true values
    y_true = df[col].values
    y_pred = forecast['yhat'].values

    # Calculating metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Store the metrics
    metrics[col] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}


# Plotting actual vs predicted values
for col, forecast in forecasts.items():
    plt.figure(figsize=(10, 6))
    plt.plot(df['ds'], df[col], label='Actual')
    plt.plot(forecast['ds'], forecast['yhat'], label='Predicted')
    plt.title(f'Actual vs Predicted Values for {col}')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.show()