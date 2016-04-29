import csv
import numpy as np
import tensorflow as tf

"""
Load data
"""
def load_adwords():
    fname = "approved_adwords_v3.csv"
    return load(fname)

def load_purchases():
    fname = "approved_data_purchase-v5.csv"
    return load(fname)

def load_ga():
    fname = "approved_ga_data_v2.csv"
    return load(fname)

def load(fname):
    reader = csv.DictReader(open(fname))
    res = {}
    for row in reader:
        for col, val in row.iteritems():
            res.setdefault(col, []).append(val)

    return res

def dict2array(datadict):
    """Convert data in dictionary into array

    Input: datadict dictionary
    Output: data numpy array

    """
    data = np.zeros(len(datadict.keys()), len(datadict[datadict.keys()[0]]))
    idx = 0
    for key in datadict.keys():
        data[i] = np.asarray(datadict[key], dtype=float32)

    return data

def reduce_logsumexp(input_tensor, reduction_indices=1, keep_dims=False):
  """Computes the sum of elements across dimensions of a tensor in log domain.
     
     It uses a similar API to tf.reduce_sum.

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    reduction_indices: The dimensions to reduce. 
    keep_dims: If true, retains reduced dimensions with length 1.
  Returns:
    The reduced tensor.
  """
  max_input_tensor1 = tf.reduce_max(input_tensor, 
                                    reduction_indices, keep_dims=keep_dims)
  max_input_tensor2 = max_input_tensor1
  if not keep_dims:
    max_input_tensor2 = tf.expand_dims(max_input_tensor2, 
                                       reduction_indices) 
  return tf.log(tf.reduce_sum(tf.exp(input_tensor - max_input_tensor2), 
                                reduction_indices, keep_dims=keep_dims)) + max_input_tensor1

def logsoftmax(input_tensor):
  """Computes normal softmax nonlinearity in log domain.

     It can be used to normalize log probability.
     The softmax is always computed along the second dimension of the input Tensor.     
 
  Args:
    input_tensor: Unnormalized log probability.
  Returns:
    normalized log probability.
  """
  return input_tensor - reduce_logsumexp(input_tensor, keep_dims=True)
