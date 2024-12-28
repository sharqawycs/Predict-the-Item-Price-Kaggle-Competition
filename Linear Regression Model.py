import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load datasets
train_file_path = 'train.csv'
test_file_path = 'test.csv'

train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Function to handle outliers using IQR method


def handle_outliers(data):
    for col in data.select_dtypes(include=[np.number]).columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Cap or floor the outliers
        data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
        data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])
    return data


# Handle outliers in the training data
train_data = handle_outliers(train_data)

# Data Analysis
print("Basic Information:")
print(train_data.info())
print("\nMissing Values:\n", train_data.isnull().sum())
print("\nDescriptive Statistics:\n", train_data.describe())

# Visualize distributions
sns.pairplot(train_data)
plt.show()

# Detect outliers using boxplot
for col in train_data.select_dtypes(include=[np.number]).columns:
    sns.boxplot(x=train_data[col])
    plt.title(f"Boxplot for {col}")
    plt.show()

# Function to fill missing X2 based on X1 grouping


def group_imputation(data):
    for idx, row in data[data['X2'].isnull()].iterrows():
        # Find rows with the same X1 value but non-null X2
        matching_rows = data[(data['X1'] == row['X1'])
                             & (data['X2'].notnull())]
        if not matching_rows.empty:
            # Use the first matching row's X2 value to fill the missing X2
            data.at[idx, 'X2'] = matching_rows.iloc[0]['X2']
    return data


# Apply group imputation to fill missing X2 values
train_data = group_imputation(train_data)

# Split features and target
target = 'Y'
X_train = train_data.drop(columns=[target])
y_train = train_data[target]
X_test = test_data.copy()

# Identify categorical and numerical columns
categorical_cols = X_train.select_dtypes(include=['object']).columns
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing
numerical_preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numerical_preprocessor, numerical_cols),
    ('cat', categorical_preprocessor, categorical_cols)
])

# Linear Regression Model
linear_model = LinearRegression()

# Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', linear_model)
])

# Training the model
pipeline.fit(X_train, y_train)

# Training Evaluation
y_train_pred = pipeline.predict(X_train)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print(f"Training MAE: {train_mae}")
print(f"Training MSE: {train_mse}")
print(f"Training R²: {train_r2}")

# Test Predictions
y_test_pred = pipeline.predict(X_test)

# Save the predictions
output = pd.DataFrame({
    'row_id': test_data.index,
    'Y': y_test_pred
})

output_file = os.path.join(os.path.dirname(
    train_file_path), 'price_prediction.csv')
output.to_csv(output_file, index=False)
print(f"Predictions saved to '{output_file}'")
