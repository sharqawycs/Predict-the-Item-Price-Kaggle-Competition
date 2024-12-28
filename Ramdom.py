from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from category_encoders import TargetEncoder, CatBoostEncoder
import seaborn as sns
from scipy.stats import mstats
# data normalization with sklearn
from sklearn.preprocessing import MinMaxScaler

# Assuming 'data' is your DataFrame and 'Item Price' is the target variable
data = pd.read_csv('train.csv')
Test=pd.read_csv('test.csv')
#print(data['X2'])
#rows_with_specific_id = data[data['X7'] == 'OUT010']

# Display the filtered rows
#print(rows_with_specific_id)
#print(data['X9'].mode())
def group_imputation(data):
    for idx, row in data[data['X2'].isnull()].iterrows():
      # Find rows with the same X1 value but non-null X2
        matching_rows = data[(data['X1'] == row['X1']) & (data['X2'].notnull())]
        
        if not matching_rows.empty:
          # Use the first matching row's X2 value to fill the missing X2
            data.at[idx, 'X2'] = matching_rows.iloc[0]['X2']
            # data['X2'] = np.log1p(data['X2'])  #################################
    
    for idx, row in data[data['X4'].isnull()].iterrows():
      # Find rows with the same X1 value but non-null X2
     matching_rows = data[(data['X1'] == row['X1']) & (data['X4'].notnull())]
     
     if not matching_rows.empty:
       # Use the first matching row's X2 value to fill the missing X2
         data.at[idx, 'X4'] = matching_rows.iloc[0]['X4']
    return data
def preprocess_data(data):
    data['X9'] = data['X9'].fillna('Small')
    
    data['X4'] = data['X4'].replace(0,np.nan)
    return data

train_data = data
test_data = Test
preprocess_data(train_data)
preprocess_data(test_data)
group_imputation(train_data)
group_imputation(test_data)
#print(train_data['X2'])

def fix_X3_mapping(data):
    data['X3'] = data['X3'].map({'Low Fat': 1, 'low fat': 1, 'LF': 1, 'Regular': 0, 'reg': 0}).astype(int)
    return data 

train_data = fix_X3_mapping(train_data)
test_data = fix_X3_mapping(test_data)
#print(train_data['X4'])
 #Map X9 to numerical values
train_data['X9'] = train_data['X9'].map({'Small': 1, 'Medium': 2, 'High': 3}).astype(int)
test_data['X9'] = test_data['X9'].map({'Small': 1, 'Medium': 2, 'High': 3}).astype(int)
# Map X10 with the tier number
train_data['X10'] = train_data['X10'].str[-1].astype(int) 
test_data['X10'] = test_data['X10'].str[-1].astype(int)
# target encoding
encdr = TargetEncoder()
train_data['X5'] = train_data['X5'].astype('category')
test_data['X5'] = test_data['X5'].astype('category')
train_data['X5'] = encdr.fit_transform(train_data['X5'], train_data['Y'])
test_data['X5'] = encdr.transform(test_data['X5'])

encdr = TargetEncoder()
train_data['X7'] = train_data['X7'].astype('category')
test_data['X7'] = test_data['X7'].astype('category')
train_data['X7'] = encdr.fit_transform(train_data['X7'], train_data['Y'])
test_data['X7'] = encdr.transform(test_data['X7'])
# X1 label encoding for all ids
encdr = LabelEncoder()
train_data['X1'] = encdr.fit_transform(train_data['X1'])
test_data['X1'] = encdr.fit_transform(test_data['X1'])
# we can maxAbs with X2, X6 
max_abs = MinMaxScaler()
train_data[['X2', 'X6']] = max_abs.fit_transform(train_data[['X2', 'X6']])
test_data[['X2', 'X6']] = max_abs.transform(test_data[['X2', 'X6']])
# log with X4
train_data['X4'] = np.log1p(train_data['X4'])
test_data['X4'] = np.log1p(test_data['X4'])

# encdr = LabelEncoder()
# train_data['X11'] = encdr.fit_transform(train_data['X11'])
# test_data['X11'] = encdr.fit_transform(test_data['X11'])




# Identify categorical and numerical columns
categorical_cols = ['X1', 'X3', 'X5', 'X7', 'X9', 'X10', 'X11']
numerical_cols = ['X4', 'X6', 'X8', 'X2']
def handle_outliers(data, cols):
    for col in cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[col] = data[col].clip(lower_bound, upper_bound)
    return data




train_data = handle_outliers(train_data, numerical_cols)
test_data = handle_outliers(test_data, numerical_cols)

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])




# Define the model
model = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=7, min_samples_split=9, bootstrap=True, random_state=42)

# Create and evaluate the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

X = train_data.drop('Y', axis=1)
y = train_data['Y']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)





# Fit the model
pipeline.fit(X_train, y_train)# Make predictions

y_pred = pipeline.predict(test_data)

train_score = pipeline.score(X_train, y_train)
test_score = pipeline.score(X_test, y_test)


#train_data.isnull().sum().sort_values(ascending=False)


# Print the training and test scores
print(f'Training Score (R-squared): {train_score}')
print(f'Test Score (R-squared): {test_score}')# Get rows with outliers
#train_data.info()

submission = pd.DataFrame({
    'row_id': test_data.index,  # Use the index of the test data as row_id
    'Y': y_pred
})
# Save the predictions to a CSV file for Kaggle submission
submission.to_csv('sample_submission.csv', index=False)