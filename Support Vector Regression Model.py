import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from scipy.stats import zscore

# Load data
def load_data(train_path, test_path, submission_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    sample_submission = pd.read_csv(submission_path)
    return train_data, test_data, sample_submission

# Preprocess data
def preprocess_data(train_data, test_data):
    X_train_full = train_data.drop('Y', axis=1)
    y_train_full = train_data['Y']
    X_test_full = test_data.copy()

    numerical_features = X_train_full.select_dtypes(include=[np.number]).columns
    categorical_features = X_train_full.select_dtypes(include=['object']).columns

    numerical_transformer = SimpleImputer(strategy='mean')
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    X_train_preprocessed = preprocessor.fit_transform(X_train_full)
    X_test_preprocessed = preprocessor.transform(X_test_full)

    new_columns = list(numerical_features) + list(preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features))

    X_train_preprocessed = pd.DataFrame(X_train_preprocessed, columns=new_columns)
    X_test_preprocessed = pd.DataFrame(X_test_preprocessed, columns=new_columns)

    return X_train_preprocessed, y_train_full, X_test_preprocessed

# Scale data
def scale_data(X_train, X_test):
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Handle outliers
def handle_outliers(X_train):
    z_scores = np.abs(zscore(X_train))
    z_score_threshold = 3
    X_train[np.where(z_scores > z_score_threshold)] = np.mean(X_train, axis=0)
    return X_train

# Feature selection
def select_features(X_train, y_train, X_test):
    lasso = Lasso(alpha=0.01, random_state=42)
    lasso.fit(X_train, y_train)

    selector = SelectFromModel(lasso, prefit=True)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    return X_train_selected, X_test_selected

# Dimensionality reduction
def reduce_dimensions(X_train, X_test, n_components=5):
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca

# Train and evaluate model
def train_and_evaluate(X_train, y_train, X_val, y_val):
    param_grid = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [0.1, 1, 10],
        'epsilon': [0.01, 0.1, 1]
    }

    grid = GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)

    metrics = {
        'train_mse': mean_squared_error(y_train, y_train_pred),
        'val_mse': mean_squared_error(y_val, y_val_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'val_r2': r2_score(y_val, y_val_pred),
        'best_params': grid.best_params_
    }

    return best_model, metrics

# Generate submission
def generate_submission(test_data, predictions, output_path):
    submission = pd.DataFrame({
        'row_id': test_data.index,
        'Y': predictions
    })
    submission.to_csv(output_path, index=False)

# Main workflow
def main():
    train_data, test_data, sample_submission = load_data('train.csv', 'test.csv', 'sample_submission.csv')

    X_train_preprocessed, y_train_full, X_test_preprocessed = preprocess_data(train_data, test_data)
    X_train_scaled, X_test_scaled = scale_data(X_train_preprocessed, X_test_preprocessed)
    X_train_scaled = handle_outliers(X_train_scaled)
    X_train_selected, X_test_selected = select_features(X_train_scaled, y_train_full, X_test_scaled)
    X_train_pca, X_test_pca = reduce_dimensions(X_train_selected, X_test_selected)

    X_train, X_val, y_train, y_val = train_test_split(X_train_pca, y_train_full, test_size=0.2, random_state=42)

    best_model, metrics = train_and_evaluate(X_train, y_train, X_val, y_val)

    print("Model Performance:", metrics)

    y_test_pred = best_model.predict(X_test_pca)
    generate_submission(test_data, y_test_pred, 'sample_submission.csv')

if __name__ == "__main__":
    main()
