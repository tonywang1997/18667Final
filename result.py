import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
from numpy.linalg import norm
from scipy.optimize import fmin_l_bfgs_b

class LinReg:
    """A class for the least-squares regression with
    Ridge penalization"""

    def __init__(self, X, y, lbda):
        self.X = X
        self.y = y
        self.n, self.d = X.shape
        self.lbda = lbda
    
    def grad(self, w):
        return self.X.T @ (self.X @ w - self.y) / self.n + self.lbda * w
    
    def f(self, w):
        return norm(self.X.dot(w) - self.y) ** 2 / (2. * self.n) + self.lbda * norm(w) ** 2 / 2.

    def f_i(self, i, w):
        return norm(self.X[i].dot(w) - self.y[i]) ** 2 / (2.) + self.lbda * norm(w) ** 2 / 2.
    
    def grad_i(self, i, w):
        return self.X[i] * (self.X[i] @ w - self.y[i]) + self.lbda * w

    def lipschitz_constant(self):
        """Return the Lipschitz constant of the gradient"""
        L = norm(self.X, ord=2) ** 2 / self.n + self.lbda
        return L

class LogReg:
    """A class for the logistic regression with L2 penalization"""

    def __init__(self, X, y, lbda):
        self.X = X
        self.y = y
        self.n, self.d = X.shape
        self.lbda = lbda
    
    def grad(self, w):
        bAx = self.y * np.dot(self.X, w)
        return -self.X.T @ (self.y * (1 + np.exp(bAx))**(-1)) / self.X.shape[0] + self.lbda * w

    def f(self, w):
        bAx = self.y * np.dot(self.X, w)
        return np.mean(np.log(1. + np.exp(- bAx))) + self.lbda * norm(w) ** 2 / 2.
    
    def f_i(self, i, w):
        bAx_i = self.y[i] * np.dot(self.X[i], w)
        return np.log(1. + np.exp(- bAx_i)) + self.lbda * norm(w) ** 2 / 2.
    
    def grad_i(self, i, w):
        a_i = self.X[i]
        b_i = self.y[i]
        return - a_i * b_i / (1. + np.exp(b_i * np.dot(a_i, w))) + self.lbda * w

    def lipschitz_constant(self):
        """Return the Lipschitz constant of the gradient"""
        L = norm(self.X, ord=2) ** 2 / (4. * self.n) + self.lbda
        return L
    
def adaptive_vr_sgd(X, y, lr=0.001, epochs=100, batch_size=32, regularization=0.01, task='classification'):
    n, d = X.shape
    weights = np.random.randn(d) * 0.01
    all_losses = []
    epsilon = 1e-8
    max_exp_input = 700

    for epoch in range(epochs):
        if task == 'classification':
            preds_full = X.dot(weights)
            preds_full = np.clip(preds_full, -max_exp_input, max_exp_input)
            preds_full = 1 / (1 + np.exp(-preds_full))
            full_grad = (X.T @ (preds_full - y)) / n + regularization * weights
        else:
            preds_full = X.dot(weights)
            full_grad = (X.T @ (preds_full - y)) / n + regularization * weights

        weights_snapshot = weights.copy()
        indices = np.random.permutation(n)
        batch_losses = []

        for i in range(0, n, batch_size):
            start = i
            end = min(i + batch_size, n)
            X_batch = X[indices[start:end]]
            y_batch = y[indices[start:end]]

            preds = X_batch.dot(weights)
            preds_snapshot = X_batch.dot(weights_snapshot)

            if task == 'classification':
                preds = np.clip(preds, -max_exp_input, max_exp_input)
                preds_snapshot = np.clip(preds_snapshot, -max_exp_input, max_exp_input)
                preds = 1 / (1 + np.exp(-preds))
                preds_snapshot = 1 / (1 + np.exp(-preds_snapshot))
                preds = np.clip(preds, epsilon, 1 - epsilon)
                preds_snapshot = np.clip(preds_snapshot, epsilon, 1 - epsilon)
                grad = (X_batch.T @ (preds - y_batch)) / (end - start) + regularization * weights
                grad_snapshot = (X_batch.T @ (preds_snapshot - y_batch)) / (end - start) + regularization * weights_snapshot
            else:
                grad = (X_batch.T @ (preds - y_batch)) / (end - start) + regularization * weights
                grad_snapshot = (X_batch.T @ (preds_snapshot - y_batch)) / (end - start) + regularization * weights_snapshot

            vr_grad = grad - grad_snapshot + full_grad
            clip_value = 5.0
            vr_grad = np.clip(vr_grad, -clip_value, clip_value)
            weights -= lr * vr_grad

            if task == 'classification':
                loss = -np.mean(y_batch * np.log(preds) + (1 - y_batch) * np.log(1 - preds))
            else:
                loss = mean_squared_error(y_batch, preds)
            loss += (regularization / 2) * np.sum(weights ** 2)
            batch_losses.append(loss)

        current_loss = np.mean(batch_losses)
        all_losses.append(current_loss)

    return weights, all_losses

def sgd(w0, model_class, steps, lbda=0, n_iter=100, cycling=False,
        averaging_on=False, start_late_averaging=0):
    """Stochastic gradient descent algorithm"""
    model = model_class
    w = w0.copy()
    w_new = w0.copy()
    
    n, d = model.X.shape
    if not cycling:
        indices = np.random.randint(0, n, n_iter)
        
    w_average = w0.copy()
    all_losses = []
    
    for k in range(n_iter):
        if cycling:
            i = k % n
        else:
            i = indices[k]
            
        w -= steps[k] * model.grad_i(i, w)
        w = w_new
        
        if k >= start_late_averaging:
            w_average = (1 - 1/(k - start_late_averaging + 1)) * w_average + 1/(k - start_late_averaging + 1) * w
            
        if averaging_on and k >= start_late_averaging:
            w_test = w_average.copy()
        else:
            w_test = w.copy()
            
        loss = model.f(w_test)
        all_losses.append(loss)
        
    if averaging_on:
        w_output = w_average.copy()
    else:
        w_output = w.copy()
        
    return w_output, all_losses

def sag(w_init, model_class, step=1., n_iter=100):
    """Stochastic average gradient algorithm."""
    w = w_init.copy()
    model = model_class
    n, d = model.X.shape
    
    gradient_memory = np.zeros((n, d))
    averaged_gradient = np.zeros(d)
    all_losses = []
    indices = np.random.permutation(np.arange(n_iter) % n)
    
    for idx in range(n_iter):
        i = indices[idx]
        
        averaged_gradient -= gradient_memory[i]/n
        gradient_memory[i] = model.grad_i(i, w)
        averaged_gradient += gradient_memory[i]/n
        w -= step * averaged_gradient
        
        loss = model.f(w)
        all_losses.append(loss)
        
    return w, all_losses

def saga(w_init, model_class, step=1., n_iter=100):
    """SAGA algorithm."""
    w = w_init.copy()
    model = model_class
    n, d = model.X.shape
    
    gradient_memory = np.zeros((n, d))
    averaged_gradient = np.zeros(d)
    all_losses = []
    indices = np.random.permutation(np.arange(n_iter) % n)
    
    for idx in range(n_iter):
        i = indices[idx]
        
        new_gradi = model.grad_i(i, w)
        w -= step * (new_gradi - gradient_memory[i] + averaged_gradient)
        averaged_gradient += (new_gradi - gradient_memory[i])/n
        gradient_memory[i] = new_gradi
        
        loss = model.f(w)
        all_losses.append(loss)
        
    return w, all_losses

# Load and preprocess data
mat_contents = sio.loadmat("data_orsay_2017.mat")
X_train_class = mat_contents["Xtrain"]
y_train_class = mat_contents["ytrain"].flatten()
X_test_class = mat_contents["Xtest"]
y_test_class = mat_contents["ytest"].flatten()

if np.array_equal(np.unique(y_train_class), [-1, 1]):
    y_train_class = (y_train_class + 1) / 2
    y_test_class = (y_test_class + 1) / 2

# Create regression targets
np.random.seed(42)
w_true = np.random.randn(X_train_class.shape[1])
y_train_reg = X_train_class @ w_true + np.random.randn(X_train_class.shape[0]) * 0.1
y_test_reg = X_test_class @ w_true + np.random.randn(X_test_class.shape[0]) * 0.1

# Standardize datasets
scaler_class = StandardScaler()
X_train_class = scaler_class.fit_transform(X_train_class)
X_test_class = scaler_class.transform(X_test_class)

# Training parameters
epochs = 100
regularization = 0.005
n, d = X_train_class.shape
w_init = np.zeros(d)

# Initialize models
log_reg_train = LogReg(X_train_class, y_train_class, regularization)
log_reg_test = LogReg(X_test_class, y_test_class, regularization)
lin_reg_train = LinReg(X_train_class, y_train_reg, regularization)
lin_reg_test = LinReg(X_test_class, y_test_reg, regularization)

# Set learning rates based on Lipschitz constants
steps_sgd = np.array([1.0/(log_reg_train.lipschitz_constant() * np.sqrt(k+1)) for k in range(epochs)])

# Train models for classification
weights_class_vrsgd, losses_class_vrsgd = adaptive_vr_sgd(X_train_class, y_train_class, lr=0.001, epochs=epochs, batch_size=32, regularization=regularization, task='classification')
weights_class_sgd, losses_class_sgd = sgd(w_init, log_reg_train, steps_sgd, regularization, epochs)
weights_class_saga, losses_class_saga = saga(w_init, log_reg_train, 1.0/log_reg_train.lipschitz_constant(), epochs)
weights_class_sag, losses_class_sag = sag(w_init, log_reg_train, 1.0/log_reg_train.lipschitz_constant(), epochs)

# Train models for regression
weights_reg_vrsgd, losses_reg_vrsgd = adaptive_vr_sgd(X_train_class, y_train_reg, lr=0.001, epochs=epochs, batch_size=32, regularization=regularization, task='regression')
weights_reg_sgd, losses_reg_sgd = sgd(w_init, lin_reg_train, steps_sgd, regularization, epochs)
weights_reg_saga, losses_reg_saga = saga(w_init, lin_reg_train, 1.0/lin_reg_train.lipschitz_constant(), epochs)
weights_reg_sag, losses_reg_sag = sag(w_init, lin_reg_train, 1.0/lin_reg_train.lipschitz_constant(), epochs)

# Calculate metrics
def calculate_classification_metrics(X_test, y_test, weights):
    preds = X_test.dot(weights)
    probs = 1 / (1 + np.exp(-preds))
    y_pred = (probs >= 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def calculate_regression_metrics(X_test, y_test, weights):
    preds = X_test.dot(weights)
    mse = mean_squared_error(y_test, preds)
    return mse

# Calculate metrics
accuracy_saga = calculate_classification_metrics(X_test_class, y_test_class, weights_class_saga)
accuracy_sgd = calculate_classification_metrics(X_test_class, y_test_class, weights_class_sgd)
accuracy_sag = calculate_classification_metrics(X_test_class, y_test_class, weights_class_sag)
accuracy_vrsgd = calculate_classification_metrics(X_test_class, y_test_class, weights_class_vrsgd)

mse_saga = calculate_regression_metrics(X_test_class, y_test_reg, weights_reg_saga)
mse_sgd = calculate_regression_metrics(X_test_class, y_test_reg, weights_reg_sgd)
mse_sag = calculate_regression_metrics(X_test_class, y_test_reg, weights_reg_sag)
mse_vrsgd = calculate_regression_metrics(X_test_class, y_test_reg, weights_reg_vrsgd)

# Compile results in a DataFrame
results_df = pd.DataFrame({
    'Algorithm': [
        'Adaptive VR-SGD (Classification)', 'SAGA (Classification)', 'SGD (Classification)', 'SAG (Classification)',
        'Adaptive VR-SGD (Regression)', 'SAGA (Regression)', 'SGD (Regression)', 'SAG (Regression)'
    ],
    'Final Training Loss': [
        losses_class_vrsgd[-1], losses_class_saga[-1], losses_class_sgd[-1], losses_class_sag[-1],
        losses_reg_vrsgd[-1], losses_reg_saga[-1], losses_reg_sgd[-1], losses_reg_sag[-1]
    ],
    'Test Accuracy': [
        accuracy_vrsgd, accuracy_saga, accuracy_sgd, accuracy_sag,
        None, None, None, None
    ],
    'Mean Squared Error': [
        None, None, None, None,
        mse_vrsgd, mse_saga, mse_sgd, mse_sag
    ]
})

print("\nSummary of Results:")
print(tabulate(results_df, headers='keys', tablefmt='pretty'))

# Plotting the losses
epochs_range = np.arange(1, epochs + 1)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

# Plot classification losses
ax1.plot(epochs_range, losses_class_vrsgd, label='Adaptive VR-SGD', linewidth=2)
ax1.plot(epochs_range, losses_class_saga, label='SAGA', linewidth=2)
ax1.plot(epochs_range, losses_class_sgd, label='SGD', linewidth=2)
ax1.plot(epochs_range, losses_class_sag, label='SAG', linewidth=2)
ax1.set_ylabel('Classification Loss')
ax1.set_title('Loss Convergence Comparison for Classification')
ax1.legend(loc='upper right')
ax1.grid(True)

# Plot regression losses
ax2.plot(epochs_range, losses_reg_vrsgd, label='Adaptive VR-SGD', linewidth=2)
ax2.plot(epochs_range, losses_reg_saga, label='SAGA', linewidth=2)
ax2.plot(epochs_range, losses_reg_sgd, label='SGD', linewidth=2)
ax2.plot(epochs_range, losses_reg_sag, label='SAG', linewidth=2)
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Regression Loss')
ax2.set_title('Loss Convergence Comparison for Regression')
ax2.legend(loc='upper right')
ax2.grid(True)

plt.tight_layout()
plt.show()
