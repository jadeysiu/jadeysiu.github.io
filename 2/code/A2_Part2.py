import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import KFold

# Function for creating the gaussian basis
def gaussian_basis(X, centres, sigma):

    # Difference between each observation and centre
    differences = X[:, None, :] - centres[None, :, :]

    # Squared Euclidean distance
    squared_distances = np.sum(differences**2, axis=2)

    # Gaussian basis values
    gaussian_values = np.exp(-squared_distances / (2 * sigma**2))

    # Add bias column
    phi = np.column_stack((np.ones(X.shape[0]), gaussian_values))

    return phi


file_path = Path(__file__).resolve().parent / "housing.data"
data = np.loadtxt(file_path)

# -------------------- Part E ------------------- #

# Separate features (x) and target (y)
x = data[:, :-1]
y = data[:, -1]

# Slices the first 100 rows and inputs everything but the MEDV (Training data)
x_train = x[:100, :]
y_train = y[:100]

# Slices from the 101 to the last row and the output (MEDV) (Testing data)
x_test = x[100:, :]
y_test = y[100:]

# Normalize using training statistics
mean_train = np.mean(x_train, axis=0)
std_train = np.std(x_train, axis=0)
std_train[std_train == 0] = 1
x_train_norm = (x_train - mean_train) / std_train
x_test_norm = (x_test - mean_train) / std_train

def gaussian_regression():
    # Stores the errors
    training_errors = []
    testing_errors = []

    # Number of Gaussian basis functions
    K_values = range(5, 100, 10)

    # Set random seed for reproducible centre selection
    rng = np.random.default_rng(1)
    # For testing isolated K
    # centre_indices = rng.permutation(x_train_norm.shape[0])

    # Calculate Gaussian width using normalized training observations
    differences = x_train_norm[:, None, :] - x_train_norm[None, :, :]
    distances = np.linalg.norm(differences, axis=2)

    # Exclude distances from observations to themselves
    unique_distances = distances[np.triu_indices(len(x_train_norm), k=1)]
    sigma = np.median(unique_distances)

    # Loop through each number of basis functions
    for K in K_values:

        # Randomly select K centres from the 100 training observations
        indices = rng.choice(x_train_norm.shape[0], size=K, replace=False)
        centres = x_train_norm[indices]
        # For testing isolated K
        # centres = x_train_norm[centre_indices[:K]]

        # Create Gaussian basis matrices
        phi_train = gaussian_basis(x_train_norm, centres, sigma)
        phi_test = gaussian_basis(x_test_norm, centres, sigma)

        # Calculate weights using ordinary least squares
        weights = np.linalg.lstsq(phi_train, y_train, rcond=None)[0]

        # Predict training and testing values
        y_train_pred = phi_train @ weights
        y_test_pred = phi_test @ weights

        # Calculate training and testing RMSE
        train_error = np.sqrt(np.mean((y_train - y_train_pred)**2))
        test_error = np.sqrt(np.mean((y_test - y_test_pred)**2))

        # Store errors for this K
        training_errors.append(train_error)
        testing_errors.append(test_error)

        # Print results
        print(f"K = {K:2d}: Training RMSE = {train_error:.3f}, "f"Testing RMSE = {test_error:.3f}")

    plt.figure(figsize=(9, 6))
    plt.plot(K_values, training_errors, marker='o', label='Training RMSE')
    plt.plot(K_values, testing_errors, marker='o', label='Testing RMSE')
    plt.xlabel('Number of Gaussian Basis Functions (K)')
    plt.ylabel('RMSE ($1000s)')
    plt.title('Gaussian Basis Regression: Training vs Testing RMSE')
    plt.xticks(list(K_values))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

gaussian_regression()

# -------------------- Part F ------------------- #
def gaussian_regression_reg():

    # Use 90 Gaussian basis functions
    K = 90

    # Regularization parameters
    lambda_values = [0, 0.01, 0.1, 1, 10, 100, 1000]

    # Store average validation RMSE for each lambda
    average_errors = []

    # Set up 10-fold cross-validation
    fold = KFold(n_splits=10, shuffle=True, random_state=1)

    for lam in lambda_values:

        fold_errors = []

        for train_index, val_index in fold.split(x_train):

            # Select training and validation data
            x_fold_train = x_train[train_index]
            y_fold_train = y_train[train_index]

            x_fold_val = x_train[val_index]
            y_fold_val = y_train[val_index]

            # Normalize using only fold-training observations
            fold_mean = np.mean(x_fold_train, axis=0)
            fold_std = np.std(x_fold_train, axis=0)
            fold_std[fold_std == 0] = 1
            x_fold_train_norm = (x_fold_train - fold_mean) / fold_std
            x_fold_val_norm = (x_fold_val - fold_mean) / fold_std

            # Randomly select K centres from fold-training data
            rng = np.random.default_rng(1)
            indices = rng.choice(x_fold_train_norm.shape[0], size=K, replace=False)
            centres = x_fold_train_norm[indices]

            # Calculate Gaussian width from fold-training data
            differences = (
                x_fold_train_norm[:, None, :]
                - x_fold_train_norm[None, :, :]
            )
            distances = np.linalg.norm(differences, axis=2)

            unique_distances = distances[np.triu_indices(len(x_fold_train_norm), k=1)]
            sigma = np.median(unique_distances)

            # Create Gaussian basis matrices
            phi_train = gaussian_basis(x_fold_train_norm, centres, sigma)
            phi_val = gaussian_basis(x_fold_val_norm, centres, sigma)

            # Calculate L2-regularized weights
            if lam == 0:
                weights = np.linalg.lstsq(phi_train, y_fold_train, rcond=None)[0]
            else:
                n_weights = phi_train.shape[1]
                I = np.eye(n_weights)

                weights = np.linalg.solve(
                    phi_train.T @ phi_train + lam * I,
                    phi_train.T @ y_fold_train
                )

            # Predict validation values
            y_val_prediction = phi_val @ weights

            # Calculate validation RMSE
            val_rmse = np.sqrt(np.mean((y_fold_val - y_val_prediction)**2))

            fold_errors.append(val_rmse)

        # Calculate and store average validation RMSE across the 10 folds
        average_rmse = np.mean(fold_errors)
        average_errors.append(average_rmse)

        print(f"Lambda {lam}: Average RMSE = {average_rmse:.4f}")

    # Select lambda with lowest validation RMSE
    best_index = np.argmin(average_errors)
    best_lambda = lambda_values[best_index]

    print(f"\nBest lambda: {best_lambda}")

    # Plot average validation RMSE versus lambda
    plt.figure(figsize=(9, 6))

    # Lambda = 0 cannot appear on logarithmic x-axis
    plt.semilogx(
        lambda_values[1:],
        average_errors[1:],
        marker='o',
        label='Average Validation RMSE'
    )

    # Show unregularized result as a reference
    plt.axhline(
        y=average_errors[0],
        color='orange',
        linestyle='--',
        label='Lambda = 0'
    )

    plt.xlabel('Regularization Parameter (Lambda)')
    plt.ylabel('Average Validation RMSE ($1000s)')
    plt.title('Gaussian Basis Regression: 10-Fold Cross-Validation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

gaussian_regression_reg()
