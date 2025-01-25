# MLOps Assignment: Experiment Tracking and Data Versioning

This repository contains the code and results for an MLOps assignment focused on experiment tracking using MLflow and data versioning using DVC.

## Objective

The goal of this assignment is to gain hands-on experience with popular MLOps tools and understand the processes they support. Specifically, the project involves:

1.  **Experiment Tracking:** Using MLflow to track and log experiments for a machine learning model.
2.  **Data Versioning:** Using DVC (Data Version Control) to version a dataset used in the project.


## Deliverables

The following deliverables demonstrate the completion of this assignment:

1.  **MLflow Experiment Logs:**
    *   Screenshots of the MLflow UI showing:
        *   The main experiment list.
        *   The table of runs within the "Default" experiment.
        *   The comparison view of two or more runs.
        *   The details page of one of the runs, including the parameters, metrics, and artifacts sections.
    *   Discussion of the different experiment runs, the parameters varied, and a comparison of the results (based on logged metrics).

2.  **DVC Repository:**
    *   Screenshots demonstrating:
        *   The output of `git log --oneline` to show the commit history, including commits related to adding and updating the dataset with DVC.
        *   The `data.csv` file open in a text editor *after* reverting to an older version of the dataset using `git checkout <commit hash>` and `dvc checkout`.
    *   Explanation of how DVC is used to track different versions of the dataset.

## Setup and Instructions

1.  **Environment Setup:**
    *   It is recommended to use a virtual environment for this project.
        ```bash
        # In the mlops_project directory
        python3 -m venv .venv  
        source .venv/bin/activate
        ```
    *   Install the required packages:
        ```bash
        pip install mlflow scikit-learn pandas numpy dvc
        ```

2.  **GitHub Repository:**
    *   Create a GitHub repository named `mlops_assignment`.
    *   Clone the repository to your local machine:
        ```bash
        git clone <repository URL>
        cd mlops_assignment
        ```

3.  **MLflow Experiment Tracking:**
    *   The `train.py` script contains the code for training a simple linear regression model using scikit-learn.
    *   MLflow is integrated into `train.py` to log parameters, metrics, and the trained model for each experiment run.
    *   Run the experiments:
        ```bash
        python train.py
        ```
    *   Launch the MLflow UI:
        ```bash
        mlflow ui
        ```
    *   Open your web browser and go to `http://localhost:5000` to view the experiment logs.

4.  **Data Versioning with DVC:**
    *   The `create_data.py` script generates a synthetic dataset (`data.csv`).
    *   Initialize DVC:
        ```bash
        dvc init
        ```
    *   Track the dataset with DVC:
        ```bash
        python create_data.py
        dvc add data.csv
        ```
    *   Commit the changes to Git:
        ```bash
        git add data.csv.dvc create_data.py .dvcignore .gitignore
        git commit -m "Add data creation and track with DVC"
        ```
    *   Make changes to the dataset (e.g., modify the random seed in `create_data.py` and regenerate the data).
    *   Track the updated dataset with DVC:
        ```bash
        dvc add data.csv
        ```
    *   Commit the changes to Git:
        ```bash
        git commit data.csv.dvc create_data.py -m "Update dataset"
        ```
    *   Demonstrate reverting to an older version:
        ```bash
        git log --oneline # Get the commit hash of the initial dataset version
        git checkout <commit hash>
        dvc checkout
        ```

## Experiments

Three different experiment runs were conducted as part of this assignment, varying specific parameters to observe their effects on the model's performance. Below is a summary of each experiment:

### Experiment 1: Default Parameters

- **Description**: This experiment uses the default parameters of the `LinearRegression` model from scikit-learn. It serves as the baseline for comparison with other experiments.
- **Parameters**:
  - `model`: LinearRegression (default parameters)
- **Logged Metrics**:
  - Mean Squared Error (MSE)
  - R-squared (R2)
- **Observations**: The results from this experiment provide a baseline understanding of the model's performance with its default settings.

### Experiment 2: Different Random State for Data Split

- **Description**: In this experiment, the `random_state` parameter for the `train_test_split` function is changed to `100`. This alters the way the dataset is split into training and testing sets, which can affect the model's training and evaluation.
- **Parameters**:
  - `model`: LinearRegression (default parameters)
  - `random_state`: 100 (for `train_test_split`)
- **Logged Metrics**:
  - Mean Squared Error (MSE)
  - R-squared (R2)
- **Observations**: Comparing these results with Experiment 1 helps in understanding how different data splits (due to different `random_state` values) can impact the model's performance metrics.

### Experiment 3: Different Test Size for Data Split

- **Description**: This experiment modifies the `test_size` parameter in the `train_test_split` function to `0.3`. This means that 30% of the dataset is used for testing, as opposed to the default 20% in the previous experiments.
- **Parameters**:
  - `model`: LinearRegression (default parameters)
  - `test_size`: 0.3 (for `train_test_split`)
- **Logged Metrics**:
  - Mean Squared Error (MSE)
  - R-squared (R2)
- **Observations**: This experiment helps in evaluating how the size of the testing set impacts the model's performance metrics, offering insights into the effect of having more or less data allocated for testing.

## Conclusion

This project demonstrates the use of MLflow for tracking machine learning experiments and DVC for versioning datasets. By logging parameters, metrics, and models with MLflow, we can easily compare different experiment runs and analyze the results. DVC enables us to manage and revert to different versions of the dataset, ensuring reproducibility and providing a clear history of data changes. These tools are essential for efficient and organized MLOps workflows.