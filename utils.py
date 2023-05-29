from numbers import Number
import numpy as np
from pandas import DataFrame
from matplotlib import pyplot as plt
from functools import reduce
from operator import iconcat

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

#######################################################

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

#######################################################

def count_tauh(channel, genPartFlavs_1, genPartFlavs_2, genPartFlavs_3):
    """
    Input : 
        -channel : string of three characters corresponding to the three prompt leptons in the decay
        -genPartFlavs : 3 (1 for each lepton) arguments describing the flavour of genParticle
    Output :
        -number of hadronic taus present in the event (either 0, 1 or 2) 
    """
    # if len(args) == 1:
    #     if len(args[0]) != 4:
    #         raise TypeError("Wrong number of arguments")
    #     channel = args[0][0][0]
    #     genPartFlavs = args[0][1:]
    # elif len(args) == 4:
    #     channel = args[0][0]
    #     genPartFlavs = args[1:]
    # else:
    #     raise TypeError("Wrong number of arguments")
    channel = channel[0]
    is_list = False
    genPartFlavs = [genPartFlavs_1, genPartFlavs_2, genPartFlavs_3]
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

#######################################################
    
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

#######################################################
    
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
    
#######################################################

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
    data_test = df.query("event % 4 == 2").reset_index(drop=True)
    data_meas = df.query("event % 4 == 3").reset_index(drop=True)

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

#######################################################

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

#######################################################

def plot_hist(dataframe, keys, keys_label, bins_list, normalize = True, mode='n_tauh', return_counts = False, weights_name = 'weightNorm'):
    """
    Arguments:
        -dataframe: pandas dataframe of dictionary containing the data
        -keys: entries of the dataframe to be plotted
        -keys_label: name of x-axis of the histogram for plot (i.e. each key in keys)
        -bins_list: list of np.array representing the bin edges for each plot 
        -normalize: if True, every subhistogram of each plot will be normalized to 1
        -mode:  'simple' to plot only one type of data per plot
                'simple_signal_label' to have signal and background histogram on each plot
                'n_tauh' to separate the backgrounds with 0, 1 or 2 hadronic taus detected, on each plot
        -return_counts : if True, the function returns the counts for each bin as well as the uncertainty on the bin
        -weights _name : name of the column in which the weights are stored. If None or if the specified name is not
                         in the dataframe keys, all weights will be set to 1
    Output:
        -figs: list of pyplot figures, one for each key in keys
        -counts: (if renturn_counts) list of dictionaries containing the counts and the error on the bins, for each type of data (sub histograms) 
    """
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if type(dataframe) == dict:
        data_pd = DataFrame(dataframe)
    else:
        data_pd = dataframe.copy()

    sub_df = {}
    if mode == 'n_tauh':
        sub_df_keys =  ["signal", "bkg_0", "bkg_1", "bkg_2"]
        event_type_labels = ["signal", r"$background\ 0\times\tau_h$", r"$background\ 1\times\tau_h$", r"$background\ 2\times\tau_h$"]
        signal = data_pd.loc[data_pd['signal_label']==1]
        background = data_pd.loc[data_pd['signal_label']==0]
        background_0 = background.loc[background['n_tauh']==0]
        background_1 = background.loc[background['n_tauh']==1]
        background_2 = background.loc[background['n_tauh']==2]
        sub_df[sub_df_keys[0]] = signal
        sub_df[sub_df_keys[1]] = background_0
        sub_df[sub_df_keys[2]] = background_1
        sub_df[sub_df_keys[3]] = background_2
    elif mode == 'simple':
        sub_df_keys = ['Data']
        event_type_labels = [None]
        sub_df['Data'] = data_pd
    elif mode == 'simple_signal_label':
        sub_df_keys =  ["signal", "bkg"]
        event_type_labels = ['signal', 'background']
        signal = data_pd.loc[data_pd['signal_label'] == 1]
        background = data_pd.loc[data_pd['signal_label']==0]
        sub_df[sub_df_keys[0]] = signal
        sub_df[sub_df_keys[1]] = background
    else:
        raise ValueError(f"The mode {mode} is not valid")
    
    figs = []
    counts = []

    if type(keys) != list:
        keys = [keys]
        keys_label = [keys_label]
        bins_list = [bins_list]
    
    for i,key in enumerate(keys):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # dictionary for counts/errors
        sub_dicts = []
        for k in range(len(list(sub_df.keys()))):
            sub_dicts.append({'counts':[], 'errors':[]})
        counts_j = dict(zip(list(sub_df.keys()), sub_dicts))

        for j,event_type in enumerate(sub_df.keys()):
            
            df = sub_df[event_type]
            
            if (weights_name not in list(df.keys())) or (weights_name == None):
                if weights_name == None:
                    weights_name = 'genWeight'
                df[weights_name] = np.ones_like(df[list(df.keys())[0]])
            
            c,b = np.histogram(df[key], bins=bins_list[i], weights=df[weights_name])
            c2,_ = np.histogram(df[key], bins=bins_list[i], weights=df[weights_name]**2)
            if normalize:
                norm = np.sum(c)
                if norm != 0:
                    c /= norm
                    c2 /= norm**2
                else:
                    c2 = np.zeros_like(c)
            error = np.sqrt(c2)

            counts_j[event_type]['counts'] = c
            counts_j[event_type]['errors'] = error
            
            if event_type_labels[j] != None:
                ax.stairs(c, b, label=event_type_labels[j], linewidth=2)
            else:
                ax.stairs(c, b, linewidth=2)
            # ax.errorbar((b[1:]+b[:-1])/2, c, yerr = np.sqrt(c2), marker = '.',drawstyle = 'steps-mid', color=colors[j])
            ax.errorbar((b[1:]+b[:-1])/2, c, yerr = error,fmt='.', color='k', linewidth=1)
            ax.set_xlabel(keys_label[i])
            if mode != 'simple':
                ax.legend()
            ax.grid(True)
        figs.append(fig)
        counts.append(counts_j)
    if return_counts:
        return figs, counts
    return figs
    

#######################################################

def flatten_2D_list(multi_dim_list):
    new_list = []
    for ele in multi_dim_list:
        if type(ele) is list:
            new_list.append(ele)
        else:
            new_list.append([ele])
    return reduce(iconcat, new_list, [])
        

