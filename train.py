import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Generate a synthetic dataset
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
data = pd.DataFrame(np.hstack((X, y)), columns=['X', 'y'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[['X']], data[['y']], test_size=0.2, random_state=42)

# Create an input example (use a sample from your training data)
input_example = X_train.head()

# Experiment 1: Default parameters
with mlflow.start_run(run_name="Experiment 1"):
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log parameters and metrics
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(model, "model", input_example=input_example) # Added input_example

    print(f"Experiment 1 - MSE: {mse}, R2: {r2}")

# Experiment 2: Different random state for data split
with mlflow.start_run(run_name="Experiment 2"):
    # Split data with a different random state
    X_train, X_test, y_train, y_test = train_test_split(data[['X']], data[['y']], test_size=0.2, random_state=100)

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log parameters and metrics
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_param("random_state", 100)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(model, "model", input_example=input_example) # Added input_example

    print(f"Experiment 2 - MSE: {mse}, R2: {r2}")

# Experiment 3: Different test size for data split
with mlflow.start_run(run_name="Experiment 3"):
    # Split data with a different test size
    X_train, X_test, y_train, y_test = train_test_split(data[['X']], data[['y']], test_size=0.3, random_state=42)

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log parameters and metrics
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_param("test_size", 0.3)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(model, "model", input_example=input_example) # Added input_example

    print(f"Experiment 3 - MSE: {mse}, R2: {r2}")