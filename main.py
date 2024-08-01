from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Assuming the last column is the target variable
X = covtype_data.iloc[:, :-1]
y = covtype_data.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
linear_regression = LinearRegression()
ridge_regression = Ridge(alpha=1.0)  # L2 regularization
lasso_regression = Lasso(alpha=0.1)  # L1 regularization

# Fit models
linear_regression.fit(X_train_scaled, y_train)
ridge_regression.fit(X_train_scaled, y_train)
lasso_regression.fit(X_train_scaled, y_train)

# Predict using the models
y_pred_lr = linear_regression.predict(X_test_scaled)
y_pred_ridge = ridge_regression.predict(X_test_scaled)
y_pred_lasso = lasso_regression.predict(X_test_scaled)

# Evaluate models
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

metrics_lr = evaluate_model(y_test, y_pred_lr)
metrics_ridge = evaluate_model(y_test, y_pred_ridge)
metrics_lasso = evaluate_model(y_test, y_pred_lasso)

metrics_lr, metrics_ridge, metrics_lasso
