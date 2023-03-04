import numpy as np

# creating the class for Lasso regression
class lasso_regression():
  # initing the hyperparameters
  def __init__(self, learning_rate , no_of_iteration , lambda_parmeter):
    self.learning_rate = learning_rate
    self.no_of_iteration=no_of_iteration
    self.lambda_parmeter=lambda_parmeter
  # fitting the dataset into lasso regrssion
  def fit(self,X,Y):
    # m -> number of datapoints  -> number of rows
    # n -> number of inputs feature -> number of columns
    self.m , self.n = X.shape
    self.w = np.zeros(self.n)
    self.b = 0 
    self.X=X
    self.Y=Y
    # implement of Gradient Descent algorithm for optimization
    for i in range(no_of_iteration):
      self.update_weights()

  # fuction for updating the weigths
  def update_weights(self, ):
    # linear equation of the model
    Y_pred = self.predict(self.X)

    # gradient (dw,db)


    # gradient for weight 
    dw = np.zeros(self.n)
    for i in range(self.n):
        if self.w[i]>0:
            dw[i] = (-(2(self.x[:,i]).dot(self.Y - Y_pred)) + self.lambda_parmeter) / self.m
        else:
            db[i] = (-(2(self.x[:,i]).dot(self.Y - Y_pred)) - self.lambda_parmeter) / self.m
    # gradient for bias
    db = - 2 * np.sum(self.Y -Y_pred)/self.m
    # updating the weigth and bias
    self.w = self.w - self.learning_rate * dw
    self.b = self.b - self.learning_rate * db
  #predicting the Target Variable
  def predict(self,X):
    return X.dot(self.w)+self.b