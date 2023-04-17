from kinematic import *
from uproot import open
from os import listdir
from fnmatch import filter
from numpy import unique, array, empty, concatenate, ones
from numpy.random import choice
from utils import isolate_int, count_tauh, call_dict_with_list, replace_prefix_in_list

class Data_extractor():
    """
    A Data_extractor extracts data from a folder of root files containing the anatuples.
    It takes a channel as argument : channel = "tee" "tem" "tmm" "tte" or "ttm"
    When called, it returns the variables of interest for the DNN training
    """
    def __init__(self, channel, raw_vars_general, raw_vars_lepton1, raw_vars_lepton2, raw_vars_lepton3, output_vars, functions, input_vars):
        """
        -channel : flavour of the 3 prompt leptons present in the decay. channel = "tee" "tem" "tmm" "tte" or "ttm"
        -raw_vars_general : names of variables in the root files that will be loaded and which are present only once, and not for each lepton
        -raw_vars_lepton(1,2,3) : end of names of variables in the root files that will be loaded and which are defined for a specific lepton.
                                The naming convention for such variables is L_X where L = Electron(1,2), Muon(1,2), Tau(1,2). Only specify
                                _X, since L will be deduced from the channel
        -output_vars : names of variable of interest that will be created by the data extractor
        -functions : functions that will be used to compute the output_vars (one function for each output_vars in the right order). If the 
                     corresponding output variable is already present as raw variable, put None as a function.
        -input_vars : list of lists of variables that are passed to the functions to compute the output_vars. If the variable in question 
                      is specific to one lepton, then "(1,2,3)_X" will be converted to lepton(1,2,3)_X. 
                      For example, in tee channel "3_mass"->"Electron2_mass"

        """
        self.channel = channel
        if self.channel == "tee":
            self.n_taus = 1
            self.lepton1 = "Tau"
            self.lepton2 = "Electron1"
            self.lepton3 = "Electron2"
        elif self.channel == "tem":
            self.n_taus = 1
            self.lepton1 = "Tau"
            self.lepton2 = "Electron"
            self.lepton3 = "Muon"
        elif self.channel == "tmm":
            self.n_taus = 1
            self.lepton1 = "Tau"
            self.lepton2 = "Muon1"
            self.lepton3 = "Muon2"
        elif self.channel == "tte":
            self.n_taus = 2
            self.lepton1 = "Tau1"
            self.lepton2 = "Tau2"
            self.lepton3 = "Electron"
        elif self.channel == "ttm":
            self.n_taus = 2
            self.lepton1 = "Tau1"
            self.lepton2 = "Tau2"
            self.lepton3 = "Muon"
        else:
            raise ValueError("The channel name \""+channel+"\" is not valid")
        self.raw_vars = raw_vars_general
        for var in raw_vars_lepton1:
            self.raw_vars.append(self.lepton1+var)
        for var in raw_vars_lepton2:
            self.raw_vars.append(self.lepton2+var)
        for var in raw_vars_lepton3:
            self.raw_vars.append(self.lepton3+var)
        
        self.input_vars = replace_prefix_in_list(input_vars, to_replace=['1','2','3'], replace_by=[self.lepton1, self.lepton2, self.lepton3])

        self.functions = functions
        self.output_vars = output_vars


    def __call__(self, path, signal_prefix = ['HNL'], real_data_prefix = ['EGamma', 'SingleMuon', 'Tau'], data = None, file_list = None, with_mass_hyp = True):
        """
        Arguments :
            -path : the path to the root files
            -signal_prefix : beginning of names of the files containing the signal (here "HNL"). It can be a string or a list of strings
            -real_data_prefix : beginning of filenames that correspond to real data, and that will be ignored
            -data : dictionnary to which the extracted data will be appended (if None, the dictionary will be created)
            -file_list : list of root files from which data will be extracted (if None, all root files present in path will be used).
            -with_mass_hyp : if True, the data will contain , the HNL mass hypothesis in GeV for the signal events, and a random choice 
                             among the different hypothesis for background events
        Output : 
            -data : dictionary containing the event indices, the variables of interest, the label of the event, and the type of event.
                    By default, data will contain the entries "signal_label" (1 for signal, 0 for background), "channel" and "event_type" (name of the 
                    file in which the events were taken)
        """
        total_keys = self.output_vars.copy()
        total_keys.extend(['signal_label', 'channel', 'event_type'])
        if with_mass_hyp:
            total_keys.append('mass_hyp')
        if data == None:
            value_list = []
            for i in range(len(self.output_vars)):
                value_list.append(empty((0,)))
            data = dict(zip(self.output_vars, value_list))
        elif set(list(data.keys())) != set(total_keys):
            raise KeyError("The data keys don't match the names of the variable created by the data extractor : ", list(data.keys()), total_keys)

        if file_list == None:
            file_list = filter(listdir(path), '*.root')

        # Create a list of all considered HNL mass hypothesis
        if type(signal_prefix) != list:
                signal_prefix = [signal_prefix]

        mass_hyps = []
        if with_mass_hyp:
            for filename in file_list:
                for prefix in signal_prefix:
                    if filename[:len(prefix)] == prefix:
                        mass_hyps.append(isolate_int(filename, separators=['-', '_'])[0])
            mass_hyps = unique(array(mass_hyps))
        

        for filename in file_list:
            RealData = False
            for prefix in real_data_prefix:
                if filename[:len(prefix)] == prefix:
                    RealData = True
            if RealData:
                continue

            # Raw data loading
            anatuple = open(path+filename)['Event;1'].arrays(self.raw_vars, library="np") # type: ignore
            n = len(anatuple[list(anatuple.keys())[0]])
            anatuple['channel'] = [self.channel]*n


            # Creation of the data
            for i, var in enumerate(self.output_vars):
                if self.functions[i] == None:
                    data[var] = concatenate((data[var], anatuple[self.input_vars[i][0]]))
                else:
                    data[var] = concatenate((data[var], self.functions[i](call_dict_with_list(anatuple, self.input_vars[i]))))

            label = 0
            mass = ones((n,))
            for prefix in signal_prefix:
                if filename[:len(prefix)] == prefix:
                    label = 1
                    if with_mass_hyp:
                        mass *= isolate_int(filename,separators=['-', '_'])[0]
            if label == 0 and with_mass_hyp:
                mass = choice(mass_hyps, n)
            
            # Add mass hypothesis
            if with_mass_hyp:
                if 'mass_hyp' in data.keys():
                    data['mass_hyp'] = concatenate((data['mass_hyp'], mass))
                else:
                    data['mass_hyp'] = mass

            # Add signal label (by default)
            if 'signal_label' in data.keys():
                data['signal_label'] = concatenate((data['signal_label'], ones((n,))*label))
            else:
                data['signal_label'] = ones((n,))*label

            # Add channel (by default)
            if 'channel' in data.keys():
                data['channel'].extend([self.channel]*n)
            else:
                data['channel'] = [self.channel]*n

            # Add event type (by default)
            if 'event_type' in data.keys():
                data['event_type'].extend([filename.replace('.root','')]*n)
            else:
                data['event_type'] = [filename.replace('.root','')]*n

        return data



        
class Data_extractor_v1(Data_extractor):
    def __init__(self, channel):
        output_vars = ['event', 'genWeight', 'deltaR_12', 'deltaR_13', 'deltaR_23', 'pt_123', 'mt_12', 'mt_13', 'mt_23', 'Mt_tot', 'n_tauh']
        functions =[None, None, deltaR, deltaR, deltaR, sum_pt, transverse_mass, transverse_mass, transverse_mass, total_transverse_mass, count_tauh]
        raw_vars_general = ['event', 'genWeight', 'MET_pt', 'MET_phi']
        raw_vars_lepton1=['_eta', '_mass', '_phi', '_pt', '_genPartFlav']
        raw_vars_lepton2=['_eta', '_mass', '_phi', '_pt', '_genPartFlav']
        raw_vars_lepton3=['_eta', '_mass', '_phi', '_pt', '_genPartFlav']
        input_vars = [['event'], ['genWeight'], ['1_eta', '2_eta', '1_phi', '2_phi'], ['1_eta', '3_eta', '1_phi', '3_phi'],
                      ['2_eta', '3_eta', '2_phi', '3_phi'], [['1_pt', '2_pt', '3_pt'],['1_phi', '2_phi', '3_phi'],['1_eta', '2_eta', '3_eta'],
                       ['1_mass', '2_mass', '3_mass']], ['1_pt', '2_pt', '1_phi', '2_phi'], ['1_pt', '3_pt', '1_phi', '3_phi'], 
                       ['2_pt', '3_pt', '2_phi', '3_phi'], ['1_pt', '2_pt', '3_pt', 'MET_pt', '1_phi', '2_phi', '3_phi', 'MET_phi'], 
                       ['channel', '1_genPartFlav', '2_genPartFlav', '3_genPartFlav']]
        super().__init__(channel, raw_vars_general=raw_vars_general, raw_vars_lepton1=raw_vars_lepton1, raw_vars_lepton2=raw_vars_lepton2, 
                         raw_vars_lepton3=raw_vars_lepton3, output_vars=output_vars, functions=functions, input_vars=input_vars)
        