import csv
import numpy as np

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
