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
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from scipy import stats



data_path = 'C:/Users/Andreas Panayiotou/Documents/Forecasting/Data/Franklin_Order_Data.csv'
window_size = 365                       # number of days used to predict
train_proportion = 0.82

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

# Reshape dataframe to be one row per date, with data from all skus as one datapoint
reshaped_df = df.pivot_table(index='date', columns='sku', values=columns_to_reshape)

# Flatten column index
reshaped_df.columns = ['_'.join(map(str,col)).rstrip('_') for col in reshaped_df.columns.values]

# Reset index
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


# Remove anomalies (doesn't remove datapoints, just caps values at upper and lower thresholds)
def cap_values(df, column_name, lower_threshold, upper_threshold):
    lower_bound = df[column_name].quantile(lower_threshold)
    upper_bound = df[column_name].quantile(upper_threshold)
    df[column_name] = df[column_name].clip(lower_bound, upper_bound)
    return df

lower_threshold = 0.01  # 1st percentile
upper_threshold = 0.99  # 99th percentile

for column in df.columns:
    if any(target in column for target in target_variables):
        df = cap_values(df, column, lower_threshold, upper_threshold)


#########################
##### Preprocessing #####
#########################
# Testtrain data split
train_size = int(len(df) * train_proportion)
df_train, df_test = df.iloc[:train_size], df.iloc[train_size:]

# Separate columns containing target variables
feature_columns = [column for column in df.columns if not any(target in column for target in target_variables)]
target_columns = [column for column in df.columns if any(target in column for target in target_variables)]

X_train = df_train[feature_columns]
X_test = df_test[feature_columns]
y_train = df_train[target_columns]
y_test = df_test[target_columns]

# Standardise data (fit scaler to training data)
X_standardiser = StandardScaler()
X_train = X_standardiser.fit_transform(X_train)
X_test = X_standardiser.transform(X_test)
# y_standardiser = StandardScaler()
# y_train = y_standardiser.fit_transform(y_train)
# y_test = y_standardiser.transform(y_test)

# Save standardiser for future data
joblib.dump(X_standardiser, 'C:/Users/Andreas Panayiotou/Documents/Forecasting/Model/X_standardiser.pkl')
# joblib.dump(y_standardiser, 'C:/Users/Andreas Panayiotou/Documents/Forecasting/Model/y_standardiser.pkl')
# SCALE TARGETS TOO?? - could improve performance but needs reversing to interpret results   -   cause nan loss

# Create time series generators
train_data = TimeseriesGenerator(X_train, y_train.values, length=window_size, batch_size=1)
test_data = TimeseriesGenerator(X_test, y_test.values, length=window_size, batch_size=1)
# # Create time series generators (numpy - without standardisation)
# train_data = TimeseriesGenerator(X_train.values, y_train.values, length=window_size, batch_size=1)
# test_data = TimeseriesGenerator(X_test.values, y_test.values, length=window_size, batch_size=1)

num_targets = y_train.shape[1]





# #################################
# ##### Build and Train Model #####
# #################################
learning_rate = 0.0002
adam = Adam(lr=learning_rate, clipvalue=1.0)

# Define model
early_stopping = EarlyStopping(monitor='val_loss', patience=2)  # patience is  number of epochs without improvement on val data

# Define model checkpoint
model_checkpoint = ModelCheckpoint('C:/Users/Andreas Panayiotou/Documents/Forecasting/Model/lstm_model_best.h5', 
                                   monitor='val_loss', 
                                   verbose=1, 
                                   save_best_only=True, 
                                   mode='min')

model = Sequential()
model.add(LSTM(7, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l2(0.01), input_shape=(window_size, X_train.shape[1])))
model.add(Dropout(0.4))                     # disables ndes with 0.4 chance to increase burden on others and reduce overfitting
model.add(LSTM(1, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.4))                     # disables ndes with 0.4 chance to increase burden on others and reduce overfitting
model.add(BatchNormalization())             # normalises output of previous layer
# model.add(Dense(5))                       # standard feed-forward layer to add complexity
model.add(Dense(num_targets))
model.compile(optimizer='adam', loss='mse') # mean squared error as loss function - could try alternatives? maybe lamb optimiser?

# Train model
history = model.fit(
    train_data, 
    epochs=20, 
    shuffle=False,                                  # preserve sample order
    validation_data=test_data,
    callbacks=[early_stopping, model_checkpoint]    # model saves each time validation loss improves, stops if it doesn't 3 epochs in a row
)

model.save('C:/Users/Andreas Panayiotou/Documents/Forecasting/Model/lstm_model.h5')

y_test_pred = model.predict(test_data)

########################
##### Plot Results #####
########################

# Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss progression during training and validation')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training loss', 'Validation loss'])
plt.show()

# Calculate metrics for the first target
mse1 = mean_squared_error(y_test[window_size:, 0], y_test_pred[:, 0])
mae1 = mean_absolute_error(y_test[window_size:, 0], y_test_pred[:, 0])
r21 = r2_score(y_test[window_size:, 0], y_test_pred[:, 0])

print('Test MSE for target1:', mse1)
print('Test MAE for target1:', mae1)
print('Test R^2 for target1:', r21)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(y_test[window_size:, 0], label='True Target 1')
plt.plot(y_test_pred[:, 0], label='Predicted Target 1')
plt.legend()
plt.show()

# Calculate metrics for the second target
mse2 = mean_squared_error(y_test[window_size:, 1], y_test_pred[:, 1])
mae2 = mean_absolute_error(y_test[window_size:, 1], y_test_pred[:, 1])
r22 = r2_score(y_test[window_size:, 1], y_test_pred[:, 1])

print('Test MSE for target2:', mse2)
print('Test MAE for target2:', mae2)
print('Test R^2 for target2:', r22)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(y_test[window_size:, 1], label='True Target 2')
plt.plot(y_test_pred[:, 1], label='Predicted Target 2')
plt.legend()
plt.show()