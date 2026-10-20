import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import KFold

# Function for creating the polynomial basis
def polynomial_basis(X, degree):

    # Number of observations and input features
    n_samples, n_features = X.shape

    # Start with the bias column (all 1s)
    phi = np.ones((n_samples, 1))

    # Generate polynomial basis functions
    for d in range(1, degree + 1):
        phi = np.column_stack((phi, X ** d))

    return phi

# Locate the dataset
file_path = Path(__file__).resolve().parent / "housing.data"

# Load the dataset into numpy array
data = np.loadtxt(file_path)

# ------------------ PART A & B ------------------------ #
# Separate features (x) and target (y)
x = data[:, :-1]
y = data[:, -1]

# Slices the first 100 rows and inputs everything but the MEDV (Training data)
x_train = x[:100, :]
y_train = y[:100]

# Slices from the 101 to the last row and the output (MEDV) (Testing data)
x_test = x[100:, :]
y_test = y[100:]

# Normalize features using the training mean and std (applied to both sets)
mean_train = np.mean(x_train, axis=0)
std_train = np.std(x_train, axis=0)
# CHAS (column 4) is constant in the first 100 rows, so std = 0; avoid dividing by zero
std_train[std_train == 0] = 1
x_train_norm = (x_train - mean_train) / std_train
x_test_norm = (x_test - mean_train) / std_train

def polynomial_regression(degrees):
    # Stores the errors
    training_errors = []
    testing_errors = []

    for degree in degrees:
        # Create polynomial basis function using the training data
        phi_train = polynomial_basis(x_train_norm, degree)
        # Estimate weight using training data
        weights, _, _, _ = np.linalg.lstsq(phi_train, y_train, rcond=None)
        # Predict the MEDV using the estimated weights 
        y_train_prediction = phi_train @ weights

        # Create polynomial basis function using the testing data
        phi_test = polynomial_basis(x_test_norm, degree)
        # Predict the MEDV using the previously estimated weights 
        y_test_prediction = phi_test @ weights

        # Calculate RMS error
        train_rmse = np.sqrt(np.mean((y_train - y_train_prediction)**2))
        test_rmse = np.sqrt(np.mean((y_test - y_test_prediction)**2))

        # Store errors
        training_errors.append(train_rmse)
        testing_errors.append(test_rmse)

        print(f"Degree {degree}: train RMSE = {train_rmse:.3f}, test RMSE = {test_rmse:.3f}")

    plt.figure()
    plt.plot(degrees, training_errors, marker='o', label='Training RMSE')
    plt.plot(degrees, testing_errors, marker='o', label='Testing RMSE')
    plt.yscale('log') # Helps with visualization
    plt.xlabel('Polynomial Degree')
    plt.ylabel('RMSE (House Value in $1000s)')
    plt.title('Polynomial Basis Regression: Training vs Testing Error')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Weights are from the last fitted degree
    return training_errors, testing_errors, weights

# Part A: degrees 1 to 7
polynomial_regression(range(1, 8))

# Part B: degree 1 only, then inspect the learned weights
_, _, weights_deg1 = polynomial_regression(range(1, 2))
print("All weights:", weights_deg1)
print("6th feature (RM):", weights_deg1[6])
print("8th feature (DIS):", weights_deg1[8])

# --------------- PART C ------------------------ #
def polynomial_regression_1d_vis(degree):
    # Using the second feature only (from normalized data calculated in Parts A and B)
    x_train_1d = x_train_norm[:, 1:2]
    x_test_1d = x_test_norm[:, 1:2]
    # Create polynomial basis function using the training data
    phi_train = polynomial_basis(x_train_1d, degree)
    # Estimate weight using training data
    weights = np.linalg.lstsq(phi_train, y_train, rcond=None)[0]
    # Create polynomial basis function using the testing data
    phi_test = polynomial_basis(x_test_1d, degree)

    # Generates x values to plot the learned curve
    x_curve = np.linspace(
        min(x_train_1d.min(), x_test_1d.min()),
        max(x_train_1d.max(), x_test_1d.max()),
        300
    ).reshape(-1, 1)

    phi_curve = polynomial_basis(x_curve, degree)
    y_curve = phi_curve @ weights

    # Plot training and testing points
    plt.figure(figsize=(9, 6))
    plt.scatter(x_train_1d[:, 0], y_train, color='blue', label='Training Data')
    plt.scatter(x_test_1d[:, 0], y_test, color='orange', label='Testing Data')
    # Plot learned polynomial
    plt.plot(x_curve[:, 0], y_curve, color='black', linewidth=2, label=f'Degree {degree} Polynomial')
    plt.xlabel('Normalized ZN')
    plt.ylabel('Median House Value ($1000s)')
    plt.title(f'Polynomial Basis Regression - Degree {degree}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # Zoom into the training region for high-degree polynomials
    # if degree >= 5:
    #     plt.xlim(x_train_1d.min(), x_train_1d.max())
    #     plt.ylim(0, 55)
    plt.tight_layout()
    plt.show()

# Generate the three plots
for degree in range(1, 8):
    polynomial_regression_1d_vis(degree)

# --------------- PART D ------------------------ #
def polynomial_regression_reg():

    # Use the second feature from the first 100 observations
    x_train_reg = x_train[:, 1:2]

    # Set polynomial degree
    degree = 8

    # Define different lambda values
    lambda_values = [0, 0.01, 0.1, 1, 10, 100, 1000]

    # Store one average validation RMSE per lambda
    average_errors = []

    # Create 10 shuffled folds and shuffle following reference tutorial
    fold = KFold(n_splits=10, shuffle=True, random_state=0) #Used random_state=0,1,2

    for lam in lambda_values:

        # Store the validation errors for current iteration
        fold_errors = []

        for train_index, val_index in fold.split(x_train_reg):

            # Select the 90 training observations
            x_fold_train = x_train_reg[train_index]
            y_fold_train = y_train[train_index]

            # Select the 10 validation observations
            x_fold_val = x_train_reg[val_index]
            y_fold_val = y_train[val_index]

            # Normalize the 90 training observations
            fold_mean = np.mean(x_fold_train, axis=0)
            fold_std = np.std(x_fold_train, axis=0)
            fold_std[fold_std == 0] = 1
            x_fold_train_norm = (x_fold_train - fold_mean) / fold_std
            x_fold_val_norm = (x_fold_val - fold_mean) / fold_std

            # Create the basis functions for training and validation
            phi_train = polynomial_basis(x_fold_train_norm, degree)
            phi_val = polynomial_basis(x_fold_val_norm, degree)

            # Calculate L2-regularized weights
            if lam == 0:
                weights = np.linalg.lstsq(phi_train, y_fold_train, rcond=None)[0] 
            else:
                K = phi_train.shape[1]
                I = np.eye(K)

                weights = np.linalg.solve(
                    phi_train.T @ phi_train + lam * I,
                    phi_train.T @ y_fold_train
                )

            # Predict validation values
            y_val_prediction = phi_val @ weights        

            # Calculate validation RMSE
            val_rmse = np.sqrt(np.mean((y_fold_val - y_val_prediction)**2))

            # Store this fold's validation error
            fold_errors.append(val_rmse)

        # Calculate and store average validation RMSE across 10 folds
        average_rmse = np.mean(fold_errors)
        average_errors.append(average_rmse)

        print(f"Lambda {lam}: Average RMSE = {average_rmse:.4f}")

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
    plt.title('Degree-8 Polynomial Regression: 10-Fold Cross-Validation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

polynomial_regression_reg()