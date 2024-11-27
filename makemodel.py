# make model

"""

Modified by Daniel Kaplan
"""
import os
print(os.getcwd())
from periodictable import elements
import csv
import pandas as pd
import re
import numpy as np
import torch

from torch.utils.data import TensorDataset


dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using {dev} device")

def is_nan(x):
    return (x != x)

exit
def create_template():
    num_elem = len(elements._element)+1
    value_list = [0.0]*num_elem
    res = np.array(value_list)
    res = np.expand_dims(res, axis=1)
    return res
# check if string contain int or float number
# or it is a true string

def string_or_number(s):
    try:
        z = int(s)
        return z
    except ValueError:
        try:
            z = float(s)
            return z
        except ValueError:
            return s

#dataset = []

def split_elename_and_value(row_data):
    """input name with shape into name and shape."""

    """
        MATCHING NUMBER:
        1. matches first digit ('\d')
         - \d matches all digits
        2. if present, matches decimal point ('\.?')
         - \. matches '.'
         - '?' means 'match 1-or-0 times'
        3. matches subsequent decimal digits ('\d*')
         - \d matches all digits
         - '*' means 'match >=0 times'

        MATCHING ELEMENT NAME:
        1. matches letters
         - [a-z A-Z] matches all letters
         - + matches 1-or-more times
    """
    name_pattern = r"\d\.?\d*|[a-z A-Z]+"
    splits = re.findall(name_pattern, row_data)

    # splits is to in format of pair of element name, value (all in strng)
    # the number of pairs varies
    #print(elements.symbol)
    # locate all symbol positions

    symbol_pos = []
    index = 0
    while index < len(splits):
        ele_sym = splits[index]
        #print(type(ele_sym))
        if type(string_or_number(ele_sym)) == str:
            try:
                ele = elements.symbol(ele_sym)
                symbol_pos.append(index)
            except ValueError as msg:
                print(str(msg))

        index = index+1

    element_id = []
    element_value = []
    total_value = 0
    for pos in symbol_pos:
        init_pos = pos
        ele = elements.symbol(splits[init_pos])
        element_id.append(ele.number)
        init_pos = init_pos+1
        if init_pos < len(splits):
            value = string_or_number(splits[init_pos])
            if isinstance(value, int) or isinstance(value, float):
                element_value.append(max(0, value))
                total_value = total_value + value
            else:
                element_value.append(0)
        else:
            element_value.append(0)

    if  total_value >= 1:
        element_value[:] = [x / total_value for x in element_value]

    total_value = min(1, max(0, sum(element_value)))

    if element_value.count(0) > 0:
        split_value = (1-total_value)/element_value.count(0)
        element_value = [split_value if item == -1 else item for item in element_value]


    return list(zip(element_id, element_value))

def parse_file(file, sheet):
    data = pd.read_csv(open(file, 'rb'))

    np_arr = np.empty([1, 120, 1])
    label = []

    for index, row in data.iterrows():
        #print(row.Compound)

        if(is_nan(row.Compound)==False):
            data_point = create_template()
            ret = split_elename_and_value(row.Compound)

            if(is_nan(row.Tc)==False):
                if isinstance(row.Tc, int) or isinstance(row.Tc, float):
                    for item in ret:
                        data_point[item[0]] = item[1]

                    data_point = np.expand_dims(data_point, axis=0)
                    np_arr= np.concatenate((np_arr, data_point), axis=0)
                    if row.Tc > 0:
                        label.append([0.0, 1.0, row.Tc])
                    else:
                        label.append([1.0, 0.0, 0])

    return np_arr[1:, :], np.array(label)

x_train, y_train = parse_file('fail.csv', 'Fail')
x_train_neg, y_train_neg = parse_file('success.csv', 'Success')

print(f"Shape of tensor x: {x_train.shape}")
print(f"Shape of tensor y: {y_train.shape}")
print(f"Shape of tensor x: {x_train_neg.shape}")
print(f"Shape of tensor y: {y_train_neg.shape}")


x_train = np.concatenate((x_train, x_train_neg), axis=0)
y_train = np.concatenate((y_train, y_train_neg), axis=0)

print(f"Shape of tensor x: {x_train.shape}")
print(f"Shape of tensor y: {y_train.shape}")

print(f"Shape of tensor neg x: {x_train_neg.shape}")
print(f"Shape of tensor neg y: {y_train_neg.shape}")


x_train = x_train.reshape((-1, 1, 10, 12))*100
print(f"Shape of tensor x: {x_train.shape}")
print(f"Shape of tensor y: {y_train.shape}")

x_train = torch.tensor(x_train)
y_train = torch.tensor(y_train)
print(f"Shape of tensor x: {x_train.shape}")
print(f"Shape of tensor y: {y_train.shape}")


train_ds = TensorDataset(x_train, y_train)
print(train_ds)
train_features, train_labels = next(iter(train_ds))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")


train_ds, val_ds = torch.utils.data.random_split(train_ds, [0.8, 0.2])
torch.save(train_ds,'train_ds.pt')
torch.save(val_ds,'val_ds.pt')
