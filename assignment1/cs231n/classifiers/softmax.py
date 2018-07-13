import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  for i in range(num_train):
        score = X[i].dot(W)
        score -= np.max(score) #for numerical stability(since exponential value can be very large)
        p = np.exp(score) / np.sum(np.exp(score))
        loss += -np.log(p[y[i]])
        
        for j in range(num_class):
            dW[:,j] += (p[j] - (j==y[i])) * X[i]
  
  loss /= num_train
  loss += reg * np.sum(W*W)

  dW /= num_train
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  
  score = X.dot(W)
  score -= np.max(score) #for numerical stability(since exponential value can be very large)
  p = np.exp(score) / np.sum(np.exp(score), axis=1).reshape(-1,1)
  loss_i = - np.log(p[np.arange(p.shape[0]), y])
  loss = np.sum(loss_i)

  y_one_hot = np.zeros(p.shape)
  y_one_hot[np.arange(y_one_hot.shape[0]), y] = 1
  #y_one_hot 안 만들고 싶으면 p[np.arange(p.shape[0]),y] -= 1 해서 X.T와 p를 곱해도 됨
  dW = X.T.dot(p - y_one_hot)

    
  loss /= num_train
  loss += reg * np.sum(W*W)

  dW /= num_train
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

