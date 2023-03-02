import numpy as np

class LogReg:
  def __init__(self, learning_rate = 0.01, n_iter = 100, early_stopping = True, tol = 1e-10, schedule_rate = 0.01):
    self.learning_rate = learning_rate
    self.n_iter = n_iter
    self.early_stopping = early_stopping
    self.tol = tol
    self.schedule_rate = schedule_rate
  
  def sigmoid(self, z):
    return 1 / (1 + np.exp(-z))

  def compute_gradient(self, X, y, W):
    m, n = X.shape
    z = np.dot(X, W)
    y_pred = self.sigmoid(z)
    dj_dw = np.dot(X.T, (y_pred-y))/len(Xval_vect)

    return dj_dw


  def fit(self, X_train, y_train, X_val = None, y_val = None):
    m, n = X_train.shape
    self.W = np.zeros((n,))
    self.J_hist = []

    # perform gradient descent
    for i in tqdm(range(self.n_iter)):
      dj_dw = self.compute_gradient(X_train, y_train, self.W)
      
      self.W = self.W - self.learning_rate * dj_dw

      self.learning_rate *= (1 / (1 + self.schedule_rate * i))

      if self.early_stopping and X_val is not None and y_val is not None:
        y_pred = self.predict(X_val)
        val_cost = self.cost(y_pred, y_val)
        self.J_hist.append(val_cost)
        if i > 0 and val_cost > prev_val_cost - self.tol:
          print(f'Early stopping after {i} iterations')
          break
        prev_val_cost = val_cost

  def predict(self, X_test):
    z = np.dot(X_test, self.W)
    y_pred = np.round(self.sigmoid(z))
    return y_pred

  def cost(self, y_pred, y_actual, eps = 1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return np.mean(-(y_actual * np.log(y_pred) + (1 - y_actual) * np.log(1 - y_pred)))