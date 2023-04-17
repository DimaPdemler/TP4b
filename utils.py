from numbers import Number
import numpy as np
from pandas import DataFrame

def isolate_int(string, separators):
    if type(separators) != list:
       separators = [separators]
    ints = []

    for i in range(1,len(separators)):
       string = string.replace(separators[i], separators[0])

    for z in string.split(separators[0]):
       if z.isdigit():
          ints.append(int(z))

    return ints

def normalize(dataframe, key, sum, weight_name='genWeight'):
    """
    Input :
        -dataframe : pandas dataframe or dictionary containing the weights to normalize, and the labels defining the classes in
                    which the weights will be normalized¨
        -key : key of the dataframe corresponding to the classes label
        -sum : defines the sum of the weights in each class after the normalization.
                sum can be a float in which case all classes will have an equal sum of weight
                or a dictionary/pd.Dataframe containing the sum as value and the class name as key
        -weight_name : name of the dataframe key corresponding to the weights
    Output : 
        Normalized dataframe with a type matching the input (pandas dataframe or dictionary)
    """
    dictionary = False
    if type(dataframe) == dict:
        dictionary = True
        dataframe = DataFrame(dataframe)
    classes = list(set(dataframe[key]))
    if isinstance(sum, Number):
        sum = dict(zip(classes, [sum]*len(classes)))
    if len(sum)!= len(classes):
        raise ValueError("The number of elements in sum doesn't match the number of classes in the dataframe")
    
    norm = np.zeros((len(dataframe[weight_name]),))
    for c in classes:
        sum_weights = dataframe.loc[dataframe[key] == c, weight_name].sum() # type: ignore
        norm += np.array(dataframe[key]==c)*sum[c]/sum_weights
    dataframe[weight_name] *= norm

    if dictionary:
        dataframe = dataframe.to_dict()
    
    return dataframe
   

def count_tauh(*args):
    """
    Input : 
        -channel : string of three characters corresponding to the three prompt leptons in the decay
        -L_genPartFlav : 3 (1 for each lepton) arguments describing the flavour of genParticle
    Output :
        -number of hadronic taus present in the event (either 0, 1 or 2) 
    """
    if len(args) == 1:
        if len(args[0]) != 4:
            raise TypeError("Wrong number of arguments")
        channel = args[0][0][0]
        genPartFlavs = args[0][1:]
    elif len(args) == 4:
        channel = args[0][0]
        genPartFlavs = args[1:]
    else:
        raise TypeError("Wrong number of arguments")
    
    is_list = False
    if type(genPartFlavs[0]) == list:
        is_list = True
        for lepton_flav in genPartFlavs:
            lepton_flav = np.array(lepton_flav)
    n_tauh = np.zeros_like(genPartFlavs[0]).astype('int64')
    for i, lepton_flav in enumerate(genPartFlavs):
        if channel[i] == 't':
            n_tauh += (lepton_flav==5).astype('int64')
    
    if is_list:
        n_tauh = n_tauh.tolist()
    
    return n_tauh 

    
def call_dict_with_list(dictionary, list_):
    """
    Input :
        -python dictionary
        -python list (potentially multidimensional) of entries
    Output :
        -list with the same structure as the input list, but with the keys replaced by the values of the dictionary at the corresponding keys 
    """
    if type(list_) != list:
        return dictionary[list_]
    else:
        sublist = []
        for el in list_:
            sublist.append(call_dict_with_list(dictionary, el))
        return sublist
    
def replace_prefix_in_list(list_, to_replace, replace_by):
    """
    Input :
        -list_ : python list of strings, potentially multidimensional
        -to_replace : list of characters or substrings that will be replaced in each element of the list
        -replace_by : list of characters or substrings that will replace the "to_replace" elements
    Output :
        -list with the same structure as the input list, with the replaced characters
    """
    if type(list_) != list:
        for i,s in enumerate(to_replace):
            if list_[:len(s)] == s:
                list_ = list_.replace(list_[:len(s)],replace_by[i])
        return list_
    else:
        sublist = []
        for el in list_:
            sublist.append(replace_prefix_in_list(el, to_replace, replace_by))
        return sublist
    

def split_dataset(data, ratio_train = 0.75, shuffle = True, print_sizes = True):
    """
    Input : 
        - data : dictionnary containing the variables of interest for each event
        - ratio_train : percentage of train + validation events going in the train dataset
        - shuffle : if True, the training and validation set are shuffled
    Output :
        - data_train : training dataset as pandas dataframe
        - data_val : validation dataset as pandas dataframe
        - data_test : test dataset as pandas dataframe
        - data_meas : measurement dataset as pandas dataframe
    """
    df = DataFrame.from_dict(data)

    data_tv = df.query("(event % 4 == 0) or (event % 4 == 1)")
    data_test = df.query("event % 4 == 2")
    data_meas = df.query("event % 4 == 3")

    if shuffle:
        data_tv = data_tv.sample(frac=1).reset_index(drop=True)

    data_train = data_tv.sample(frac = ratio_train)
    data_val = data_tv.drop(data_train.index)


    if print_sizes :
        N = len(df)
        print("Total number of events : ", N)
        print("Train set : {:.2f} %".format(100*len(data_train)/N))
        print("Validation set : {:.2f} %".format(100*len(data_val)/N))
        print("Test set : {:.2f} %".format(100*len(data_test)/N))
        print("Measurement set : {:.2f} %".format(100*len(data_meas)/N))

    return data_train, data_val, data_test, data_meas

def bucketize(dataframe, key, return_dict = True):
    """
    Input : 
        -dataframe : pandas dataframe or dictionary
        -key : key of the dataframe representing the classes names, that will be turned into indices
        -return_dict : if True, the function returns the dictionary linking the former class names to the corresponding integer indices
    Output : 
        -output : dataframe with integers replacing the values of dataframe[key] (one index per different value)  
        -class_names : dictionary linking the former class names to the corresponding integer indices    
    """
    dictionary = False
    if type(dataframe) == dict:
        dictionary = True
        dataframe = DataFrame(dataframe)

    class_names = {}
    for i,class_name in enumerate(dataframe[key]):
        if not class_name in class_names:
            class_names[class_name] = len(class_names)
    output = dataframe.copy()
    output[key].replace(list(class_names.keys()), list(class_names.values()), inplace=True)

    if dictionary:
        output = output.to_dict()
    
    if return_dict : 
        return output, class_names
    return output
    
