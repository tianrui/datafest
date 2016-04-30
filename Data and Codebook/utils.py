import csv
import numpy as np
import tensorflow as tf
import pdb

"""
Load data
"""
def load_adwords(endind):
    fname = "approved_adwords_v3.csv"
    return load(fname, endind)

def load_purchases(endind):
    fname = "approved_data_purchase-v5.csv"
    return load(fname, endind)

def load_ga(frac):
    fname = "approved_ga_data_v2.csv"
    return load(fname, endind)

def load(fname, endind):
    reader = csv.DictReader(open(fname))
    res = {}
    i = 0
    for row in reader:
        for col, val in row.iteritems():
            res.setdefault(col, []).append(val)
        i += 1
        #if i < endind:
        #    break

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

def str2num(strlist):
    """Convert string list to one hot encoded array

    Input: strlist list
    Output: data array
    """
    keys, ind = np.unique(strlist, return_inverse=True)
    data = np.zeros((len(strlist), len(keys)), dtype=int)
        
    data[np.arange(len(strlist)), ind] = 1
        
    return data

def preproc_purchases(frac=0.01):
    """ Preprocess the purchase csv, convert to 2D array
    with numerical values, convert categorical data to one hot
    encoding.

    Input: None
    Output: result array
    """
    rawdict = load_purchases(100)
    size = len(rawdict[rawdict.keys()[0]])
    endind = int(size*frac)
    converted_list = []
    converted_list.append(str2num(rawdict['event_id'][:endind]))
    converted_list.append(str2num(rawdict['primary_act_id'][:endind]))
    converted_list.append(str2num(rawdict['major_cat_name'][:endind]))
    converted_list.append(str2num(rawdict['minor_cat_name'][:endind]))
    converted_list.append(str2num(rawdict['venue_city'][:endind]))
    converted_list.append(str2num(rawdict['venue_state'][:endind]))
    converted_list.append(str2num(rawdict['venue_postal_cd_sgmt_1'][:endind]))
    converted_list.append(str2num(rawdict['la_event_type_cat'][:endind]))
    converted_list.append(str2num(rawdict['delivery_type_cd'][:endind]))
    converted_list.append(np.expand_dims(np.asarray(rawdict['tickets_purchased_qty'][:endind], dtype=int), axis=1))
    converted_list.append(np.expand_dims(np.asarray(rawdict['trans_face_val_amt'][:endind], dtype=float), axis=1))

    result = np.concatenate(converted_list, axis=1)
    return result


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
