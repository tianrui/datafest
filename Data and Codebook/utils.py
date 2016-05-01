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

def load_ga(endind):
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
    
    num_purchases, num_features = result.shape
    num_purchases, num_events = converted_list[0].shape
    aggr_result = np.zeros((num_events, num_features-num_events))
    
    for i in range(num_events):
        # Get sum of ticket price for the event
        aggr_result[i, -1] = np.sum(result[:,-1][(result[:,i] == 1)])
        # Get sum of ticket quantity purchased 
        aggr_result[i, -2] = np.sum(result[:,-2][(result[:,i] == 1)])
        # Get average for everything else
        for j in range(num_events, num_features-2):
            # Get the average for that data
            aggr_result[i,j - num_events] = np.mean(result[:,j][(result[:,i] == 1)])

        # Get the Attendance rate based on other concerts at the same venue
        # aggr_result[i,-1] = np.sum(result[:,-2][(result[:,i] == 1)])

    np.savetxt("purch_%f_aggr.csv" % (frac) ,aggr_result, delimiter=',')

    return result, aggr_result

def get_purchase_features(frac=0.01):
    rawdict = load_purchases(100)
    size = len(rawdict[rawdict.keys()[0]])
    endind = int(size*frac)

    features = []
    attributes = ['primary_act_id', 'major_cat_name', 'minor_cat_name', 'venue_city', 'venue_state', 'venue_postal_cd_sgmt_1', 'la_event_type_cat', 'delivery_type_cd']
    for attr in attributes:
        keys, indices = np.unique(rawdict[attr][:endind], return_inverse=True)
        features.append(np.expand_dims(np.asarray(keys), axis=1))
    features.append(np.expand_dims(np.array(['tickets_purchased']), axis=1))
    features.append(np.expand_dims(np.array(['trans_face_val_amt']), axis=1))

    result = np.concatenate(features, axis=0)
    num_features, blah = result.shape
    dict = np.concatenate([np.expand_dims(np.arange(num_features), axis=1),result], axis=1)
    list = dict.tolist()

    with open('purch_%f_dict.csv' % (frac), 'w') as file:
        file.writelines(','.join(i) + '\n' for i in list)

    return features

def preproc_purchases_categorical(frac=0.01):
    """ Aggregate the data

    Input: None
    Output: result array
    """
    rawdict = load_purchases(100)
    size = len(rawdict[rawdict.keys()[0]])
    endind = int(size*frac)
    converted_list = []

    converted_list.append(np.expand_dims(np.asarray(rawdict['event_id'][:endind]), axis=1))
    # converted_list.append(np.expand_dims(np.asarray(rawdict['primary_act_id'][:endind]), axis=1))
    converted_list.append(np.expand_dims(np.asarray(rawdict['primary_act_name'][:endind]), axis=1))
    converted_list.append(np.expand_dims(np.asarray(rawdict['major_cat_name'][:endind]), axis=1))
    converted_list.append(np.expand_dims(np.asarray(rawdict['minor_cat_name'][:endind]), axis=1))
    converted_list.append(np.expand_dims(np.asarray(rawdict['venue_city'][:endind]), axis=1))
    converted_list.append(np.expand_dims(np.asarray(rawdict['venue_state'][:endind]), axis=1))
    converted_list.append(np.expand_dims(np.asarray(rawdict['venue_postal_cd_sgmt_1'][:endind]), axis=1))
    converted_list.append(np.expand_dims(np.asarray(rawdict['la_event_type_cat'][:endind]), axis=1))
    converted_list.append(np.expand_dims(np.asarray(rawdict['delivery_type_cd'][:endind]), axis=1))
    converted_list.append(np.expand_dims(np.asarray(rawdict['tickets_purchased_qty'][:endind], dtype=int), axis=1))
    converted_list.append(np.expand_dims(np.asarray(rawdict['trans_face_val_amt'][:endind], dtype=float), axis=1))
    
    result = np.concatenate(converted_list, axis=1)

    num_purchases, num_features = result.shape
    events, ind_events = np.unique(converted_list[0], return_inverse=True)
    num_events = len(events)

    # aggr_result = np.zeros((num_events, num_features-1)) # Ignore by mail right now
    aggr_result = []
    for i in range(num_events):
        row = []
        # Get the corresponding features for other things
        for j in range(1,8):
            index = np.where(result[:,0] == events[i])[0][0]
            row.append(result[index, j])
        # Get sum of ticket quantity purchased 
        row.append(str(np.sum(np.asarray(result[:,-2],dtype=int)[(result[:,0] == events[i])])))
        # Get sum of ticket price for the event
        row.append(str(np.sum(np.asarray(result[:,-1],dtype=float)[(result[:,0] == events[i])])))
        aggr_result.append(row)

    with open('purch_%f_aggr_cate.csv' % (frac), 'w') as file:
        file.writelines(','.join(i) + '\n' for i in aggr_result)

    return result


def preprocess_ga(frac=0.01):
    """ Preprocess the ga csv, convert to 2D array
    with numerical values, convert categorical data to one hot
    encoding.

    Input: None
    Output: result array
    """
    
    rawdict = load_ga(100)
    size = len(rawdict[rawdict.keys()[0]])
    endind = int(size*frac)

    converted_list = []    
    converted_list.append(np.expand_dims(np.asarray(rawdict['fullvisitorid'][:endind]), axis=1))
    converted_list.append(np.expand_dims(np.asarray(rawdict['totals_visits'][:endind], dtype=int), axis=1))
    converted_list.append(np.expand_dims(np.asarray(rawdict['totals_hits'][:endind], dtype=int), axis=1))
    # converted_list.append(np.expand_dims(np.asarray(rawdict['totals_pageviews'][:endind], dtype=int), axis=1))
    converted_list.append(str2num(rawdict['device_devicecategory'][:endind]))
    converted_list.append(str2num(rawdict['geonetwork_subcontinent'][:endind]))
    converted_list.append(str2num(rawdict['geonetwork_region'][:endind]))
    converted_list.append(str2num(rawdict['geonetwork_metro'][:endind]))    
    converted_list.append(np.expand_dims(np.asarray(rawdict['hits_hour'][:endind], dtype=int), axis=1))
    converted_list.append(np.expand_dims(np.asarray(rawdict['hits_time'][:endind], dtype=int), axis=1))
    converted_list.append(str2num(rawdict['hits_type'][:endind]))

    result = np.concatenate(converted_list, axis=1)

    # with open('purch_1percent_aggr_cate.csv', 'w') as file:
        # file.writelines(','.join(i) + '\n' for i in aggr_result)

    # Average the data for ones with the same fullvisitorid
    num_data, num_features = result.shape
    ids, ind_ids = np.unique(converted_list[0], return_inverse=True)
    num_ids = len(ids)
    print "Number of unique ids: ", num_ids

    aggr_result = np.zeros((num_ids, num_features-1)) # Ignore by mail right now
    # with open('ga_1percent_aggr.csv', 'w') as file:
    #     for i in range(num_ids):
    #         row = []
    #         row.append(str(ids[i]))
    #         # Get the corresponding features for other things
    #         for j in range(1,num_features):
    #             row.append(str(np.mean(np.asarray(result[:,j],dtype=int)[(result[:,0] == ids[i])])))
    
    #         file.writelines(','.join(row) + '\n')

    for i in range(num_ids):
        for j in range(1,num_features):
            aggr_result[i, j-1] = np.mean(np.asarray(result[:,j],dtype=int)[(result[:,0] == ids[i])])

        print "Done processing for %d example" % i

    # np.savetext("ga_1percent_aggr_noid.csv", aggr_result, delimiter=',')

    return aggr_result

def get_ga_features(frac=0.01):
    rawdict = load_ga(100)
    size = len(rawdict[rawdict.keys()[0]])
    endind = int(size*frac)

    features = []
    features.append(np.expand_dims(np.array(['totals_visits']), axis=1))
    features.append(np.expand_dims(np.array(['totals_hits']), axis=1))
    attributes = ['device_devicecategory', 'geonetwork_subcontinent', 'geonetwork_region', 'geonetwork_metro']
    for attr in attributes:
        keys, indices = np.unique(rawdict[attr][:endind], return_inverse=True)
        features.append(np.expand_dims(np.asarray(keys), axis=1))

    features.append(np.expand_dims(np.array(['hits_hour']), axis=1))
    features.append(np.expand_dims(np.array(['hits_time']), axis=1))
    keys, indices = np.unique(rawdict['hits_type'][:endind], return_inverse=True)
    features.append(np.expand_dims(np.asarray(keys), axis=1))

    result = np.concatenate(features, axis=0)
    num_features, blah = result.shape
    dict = np.concatenate([np.expand_dims(np.arange(num_features), axis=1),result], axis=1)
    list = dict.tolist()

    with open('ga_1percent_dict.csv', 'w') as file:
        file.writelines(','.join(i) + '\n' for i in list)

    return features

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
