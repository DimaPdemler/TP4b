from kinematic import *
from uproot import open
from os import listdir
from fnmatch import filter
from numpy import ravel, unique, array, empty, concatenate, ones, logical_and
from numpy import abs as np_abs
from numpy.random import choice
from utils import isolate_int, count_tauh, call_dict_with_list, replace_prefix_in_list, flatten_2D_list
from copy import deepcopy

# Global variables
output_vars_v1 = ['event', 'genWeight', 'deltaR_12', 'deltaR_13', 'deltaR_23', 'pt_123', 'mt_12', 'mt_13', 'mt_23', 'Mt_tot', 'n_tauh']
output_vars_v2 = ['event', 'genWeight', 'deltaphi_12', 'deltaphi_13', 'deltaphi_23', 'deltaeta_12', 'deltaeta_13', 'deltaeta_23', 
                  'deltaR_12', 'deltaR_13', 'deltaR_23', 'pt_123', 'mt_12', 'mt_13', 'mt_23', 'Mt_tot', 'n_tauh']
output_vars_v3 = ['event', 'genWeight', 'deltaphi_12', 'deltaphi_13', 'deltaphi_23', 'deltaeta_12', 'deltaeta_13', 'deltaeta_23',
                   'deltaR_12', 'deltaR_13', 'deltaR_23', 'pt_123', 'mt_12', 'mt_13', 'mt_23', 'Mt_tot',
                    ['HNL_CM_angle_with_MET_1', 'HNL_CM_angle_with_MET_2'], ['W_CM_angle_HNL_1', 'W_CM_angle_HNL_2'], 
                    ['W_CM_angle_HNL_with_MET_1', 'W_CM_angle_HNL_with_MET_2'], ['HNL_CM_mass_1', 'HNL_CM_mass_2'],
                    ['HNL_CM_mass_with_MET_1', 'HNL_CM_mass_with_MET_2'], 'n_tauh']
output_vars_v4 = ['event', 'genWeight', 
                  'charge_1', 'charge_2', 'charge_3', 
                  'pt_1', 'pt_2', 'pt_3', 'pt_MET', 
                  'eta_1', 'eta_2', 'eta_3',
                  'mass_1', 'mass_2', 'mass_3', 
                  'deltaphi_12', 'deltaphi_13', 'deltaphi_23', 
                  'deltaphi_1MET', 'deltaphi_2MET', 'deltaphi_3MET',
                  ['deltaphi_1(23)', 'deltaphi_2(13)', 'deltaphi_3(12)', 
                  'deltaphi_MET(12)', 'deltaphi_MET(13)', 'deltaphi_MET(23)',
                  'deltaphi_1(2MET)', 'deltaphi_1(3MET)', 'deltaphi_2(1MET)', 'deltaphi_2(3MET)', 'deltaphi_3(1MET)', 'deltaphi_3(2MET)'],
                  'deltaeta_12', 'deltaeta_13', 'deltaeta_23', 
                  ['deltaeta_1(23)', 'deltaeta_2(13)', 'deltaeta_3(12)'],
                  'deltaR_12', 'deltaR_13', 'deltaR_23', 
                  ['deltaR_1(23)', 'deltaR_2(13)', 'deltaR_3(12)'],
                  'pt_123',
                  'mt_12', 'mt_13', 'mt_23', 
                  'mt_1MET', 'mt_2MET', 'mt_3MET',
                  ['mt_1(23)', 'mt_2(13)', 'mt_3(12)',
                  'mt_MET(12)', 'mt_MET(13)', 'mt_MET(23)',
                  'mt_1(2MET)', 'mt_1(3MET)', 'mt_2(1MET)', 'mt_2(3MET)', 'mt_3(1MET)', 'mt_3(2MET)'],
                  'mass_12', 'mass_13', 'mass_23',
                  'mass_123',
                  'Mt_tot',
                  ['HNL_CM_angle_with_MET_1', 'HNL_CM_angle_with_MET_2'], 
                  ['W_CM_angle_to_plane_1', 'W_CM_angle_to_plane_2'], ['W_CM_angle_to_plane_with_MET_1', 'W_CM_angle_to_plane_with_MET_2'],
                  ['HNL_CM_mass_1', 'HNL_CM_mass_2'], 
				  ['HNL_CM_mass_with_MET_1', 'HNL_CM_mass_with_MET_2'], 
                  ['W_CM_angle_12','W_CM_angle_13', 'W_CM_angle_23', 'W_CM_angle_1MET', 'W_CM_angle_2MET', 'W_CM_angle_3MET'],
                  'n_tauh']

"""deltaphi_1MET+..."""
"""deltaphi deltaR deltaeta : between 1(2+3)... and also with MET for deltaphi"""
"""deltaphi : (1,2,3)MET"""
"""mt : (1+2)MET, (1+2)3..."""
"""mass : (1+2), ... without MET"""
"""mass : (1+2+3)"""


#===================================================================================================

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
        self.flat_output_vars = flatten_2D_list(output_vars)


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
        total_keys = deepcopy(self.flat_output_vars)
        total_keys.extend(['signal_label', 'channel', 'event_type'])
        if with_mass_hyp:
            total_keys.append('mass_hyp')
        if data == None:
            value_list = []
            for i in range(len(self.flat_output_vars)):
                value_list.append(empty((0,)))
            data = dict(zip(self.flat_output_vars, value_list))
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
            print(self.channel)
            limit_charge = 3
            limit_tau_jet = 5
            limit_em_iso = 0.15

            cut = ''
            if self.channel == 'tte':
                # vars_to_load = deepcopy(self.raw_vars)
                # new_vars = ['Tau1_idDeepTau2018v2p5VSjet', 'Tau2_idDeepTau2018v2p5VSjet', 'Electron_pfRelIso03_all', 'Tau1_charge', 'Tau2_charge', 'Electron_charge']
                # for new_var in new_vars:
                #     if new_var not in vars_to_load:
                #         vars_to_load.append(new_var)
                # print(vars_to_load)
                # anatuple = open(path+filename)['Event;1'].arrays(vars_to_load, library='pd')
                # anatuple = anatuple[abs(anatuple['Tau1_charge'] + anatuple['Tau2_charge'] + anatuple['Electron_charge']) < limit_charge]
                # anatuple = anatuple[anatuple['Tau1_idDeepTau2018v2p5VSjet'] >= limit_tau_jet]
                # anatuple = anatuple[anatuple['Tau2_idDeepTau2018v2p5VSjet'] >= limit_tau_jet]
                # anatuple = anatuple[anatuple['Electron_pfRelIso03_all'] < limit_em_iso]
                # anatuple = anatuple[self.raw_vars]
                cut = '(abs(Tau1_charge + Tau2_charge + Electron_charge) < {}) & (Tau1_idDeepTau2018v2p5VSjet >= {}) & (Tau2_idDeepTau2018v2p5VSjet >= {}) & (Electron_pfRelIso03_all < {})'.format(limit_charge, limit_tau_jet, limit_tau_jet, limit_em_iso)

            if self.channel == 'tee':
                # vars_to_load = deepcopy(self.raw_vars)
                # new_vars = ['Tau_idDeepTau2018v2p5VSjet', 'Electron1_pfRelIso03_all', 'Electron2_pfRelIso03_all', 'Tau_charge', 'Electron1_charge', 'Electron2_charge']
                # for new_var in new_vars:
                #     if new_vars not in vars_to_load:
                #         vars_to_load.append(new_var)
                # anatuple = open(path+filename)['Event;1'].arrays(vars_to_load, library='pd')
                # anatuple = anatuple[abs(anatuple['Tau_charge'] + anatuple['Electron1_charge'] + anatuple['Electron2_charge']) < limit_charge]
                # anatuple = anatuple[anatuple['Tau_idDeepTau2018v2p5VSjet'] >= limit_tau_jet]
                # anatuple = anatuple[anatuple['Electron1_pfRelIso03_all'] < limit_em_iso]
                # anatuple = anatuple[anatuple['Electron2_pfRelIso03_all'] < limit_em_iso]
                cut = '(abs(Tau_charge + Electron1_charge + Electron2_charge) < {}) & (Tau_idDeepTau2018v2p5VSjet >= {}) & (Electron1_pfRelIso03_all < {}) & (Electron2_pfRelIso03_all < {})'.format(limit_charge, limit_tau_jet, limit_em_iso, limit_em_iso)

            if self.channel == 'tem':
                # vars_to_load = deepcopy(self.raw_vars)
                # new_vars = ['Tau_idDeepTau2018v2p5VSjet', 'Electron_pfRelIso03_all', 'Muon_pfRelIso03_all', 'Tau_charge', 'Electron_charge', 'Muon_charge']
                # for new_var in new_vars:
                #     if new_vars not in vars_to_load:
                #         vars_to_load.append(new_var)
                # anatuple = open(path+filename)['Event;1'].arrays(vars_to_load, library='pd')
                # anatuple = anatuple[abs(anatuple['Tau_charge'] + anatuple['Electron_charge'] + anatuple['Muon_charge']) < limit_charge]
                # anatuple = anatuple[anatuple['Tau_idDeepTau2018v2p5VSjet'] >= limit_tau_jet]
                # anatuple = anatuple[anatuple['Electron_pfRelIso03_all'] < limit_em_iso]
                # anatuple = anatuple[anatuple['Muon_pfRelIso03_all'] < limit_em_iso]
                cut = '(abs(Tau_charge + Electron_charge + Muon_charge) < {}) & (Tau_idDeepTau2018v2p5VSjet >= {}) & (Electron_pfRelIso03_all < {}) & (Muon_pfRelIso03_all < {})'.format(limit_charge, limit_tau_jet, limit_em_iso, limit_em_iso)

            if self.channel == 'tmm':
                # vars_to_load = deepcopy(self.raw_vars)
                # new_vars = ['Tau_idDeepTau2018v2p5VSjet', 'Muon1_pfRelIso03_all', 'Muon2_pfRelIso03_all', 'Tau_charge', 'Muon1_charge', 'Muon2_charge']
                # for new_var in new_vars:
                #     if new_vars not in vars_to_load:
                #         vars_to_load.append(new_var)
                # anatuple = open(path+filename)['Event;1'].arrays(vars_to_load, library='pd')
                # anatuple = anatuple[abs(anatuple['Tau_charge'] + anatuple['Muon1_charge'] + anatuple['Muon2_charge']) < limit_charge]
                # anatuple = anatuple[anatuple['Tau_idDeepTau2018v2p5VSjet'] >= limit_tau_jet]
                # anatuple = anatuple[anatuple['Muon1_pfRelIso03_all'] < limit_em_iso]
                # anatuple = anatuple[anatuple['Muon2_pfRelIso03_all'] < limit_em_iso]
                cut = '(abs(Tau_charge + Muon1_charge + Muon2_charge) < {}) & (Tau_idDeepTau2018v2p5VSjet >= {}) & (Muon1_pfRelIso03_all < {}) & (Muon2_pfRelIso03_all < {})'.format(limit_charge, limit_tau_jet, limit_em_iso, limit_em_iso)

            if self.channel == 'ttm':
                # vars_to_load = deepcopy(self.raw_vars)
                # new_vars = ['Tau1_idDeepTau2018v2p5VSjet', 'Tau2_idDeepTau2018v2p5VSjet', 'Muon_pfRelIso03_all', 'Tau1_charge', 'Tau2_charge', 'Muon_charge']
                # for new_var in new_vars:
                #     if new_vars not in vars_to_load:
                #         vars_to_load.append(new_var)
                # anatuple = open(path+filename)['Event;1'].arrays(vars_to_load, library='pd')
                # anatuple = anatuple[abs(anatuple['Tau1_charge'] + anatuple['Tau2_charge'] + anatuple['Muon_charge']) < limit_charge]
                # anatuple = anatuple[anatuple['Tau1_idDeepTau2018v2p5VSjet'] >= limit_tau_jet]
                # anatuple = anatuple[anatuple['Tau2_idDeepTau2018v2p5VSjet'] >= limit_tau_jet]
                # anatuple = anatuple[anatuple['Muon_pfRelIso03_all'] < limit_em_iso]
                cut = '(abs(Tau1_charge + Tau2_charge + Muon_charge) < {}) & (Tau1_idDeepTau2018v2p5VSjet >= {}) & (Tau2_idDeepTau2018v2p5VSjet >= {}) & (Muon_pfRelIso03_all < {})'.format(limit_charge, limit_tau_jet, limit_tau_jet, limit_em_iso)            
            # anatuple_pd = deepcopy(anatuple)
            # anatuple = anatuple.to_dict()
            # anatuple.pop(anatuple_pd.index.name, None)
            # for key in anatuple.keys():
            #     anatuple[key] = ravel(anatuple[key])

            anatuple = open(path+filename)['Event;1'].arrays(self.raw_vars, cut=cut, library='np') # type: ignore
            
            print("selection done")
            n = len(anatuple[list(anatuple.keys())[0]])
            print(n)

            if n==0:
                print("No event selected -> root file passed")
                continue

            anatuple['channel'] = [self.channel]*n


            # Creation of the data
            for i, var in enumerate(self.output_vars):
                if self.functions[i] == None:
                    print(var)
                    data[var] = concatenate((data[var], anatuple[self.input_vars[i][0]]))
                else:
                    print("reached var ", var)
                    outputs = self.functions[i](*call_dict_with_list(anatuple, self.input_vars[i]))
                    if type(var) == list:
                        print(len(var))
                        for j,v in enumerate(var):
                            data[v] = concatenate((data[v], outputs[j]))
                    else:
                        data[var] = concatenate((data[var], outputs))

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
    
#===================================================================================================

class Data_extractor_test(Data_extractor):
    def __init__(self):
        output_vars = ['test1', ['test_mix1', 'test_mix2'], 'test2']
        functions = [None, lambda a : (a[0]*a[1], a[0]+a[1]), lambda a : 2*a]
        raw_vars_general = ['test1', 'test2']
        raw_vars_lepton1 = []
        raw_vars_lepton2 = []
        raw_vars_lepton3 = []
        input_vars = [['test1'], ['test1', 'test2'], ['test2']]
        super().__init__(channel='tte', raw_vars_general=raw_vars_general, raw_vars_lepton1=raw_vars_lepton1, raw_vars_lepton2=raw_vars_lepton2, 
                         raw_vars_lepton3=raw_vars_lepton3, output_vars=output_vars, functions=functions, input_vars=input_vars, ) 

        
class Data_extractor_v1(Data_extractor):
    def __init__(self, channel):
        output_vars = deepcopy(output_vars_v1)
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
        
class Data_extractor_v2(Data_extractor):
    def __init__(self, channel):
        output_vars = deepcopy(output_vars_v2)
        functions =[None, None, deltaphi, deltaphi, deltaphi, deltaeta, deltaeta, deltaeta, deltaR, deltaR, deltaR, sum_pt, transverse_mass, transverse_mass, transverse_mass, total_transverse_mass, count_tauh]
        raw_vars_general = ['event', 'genWeight', 'MET_pt', 'MET_phi']
        raw_vars_lepton1=['_eta', '_mass', '_phi', '_pt', '_genPartFlav']
        raw_vars_lepton2=['_eta', '_mass', '_phi', '_pt', '_genPartFlav']
        raw_vars_lepton3=['_eta', '_mass', '_phi', '_pt', '_genPartFlav']
        input_vars = [['event'], ['genWeight'], ['1_phi', '2_phi'], ['1_phi', '3_phi'], ['2_phi', '3_phi'], ['1_eta', '2_eta'], 
                      ['1_eta', '3_eta'], ['2_eta', '3_eta'], ['1_eta', '2_eta', '1_phi', '2_phi'], ['1_eta', '3_eta', '1_phi', '3_phi'],
                      ['2_eta', '3_eta', '2_phi', '3_phi'], [['1_pt', '2_pt', '3_pt'],['1_phi', '2_phi', '3_phi'],['1_eta', '2_eta', '3_eta'],
                       ['1_mass', '2_mass', '3_mass']], ['1_pt', '2_pt', '1_phi', '2_phi'], ['1_pt', '3_pt', '1_phi', '3_phi'], 
                      ['2_pt', '3_pt', '2_phi', '3_phi'], ['1_pt', '2_pt', '3_pt', 'MET_pt', '1_phi', '2_phi', '3_phi', 'MET_phi'], 
                      ['channel', '1_genPartFlav', '2_genPartFlav', '3_genPartFlav']]
        super().__init__(channel, raw_vars_general=raw_vars_general, raw_vars_lepton1=raw_vars_lepton1, raw_vars_lepton2=raw_vars_lepton2, 
                         raw_vars_lepton3=raw_vars_lepton3, output_vars=output_vars, functions=functions, input_vars=input_vars)
        
class Data_extractor_v3(Data_extractor):
    def __init__(self, channel):
        output_vars = deepcopy(output_vars_v3)
        functions =[None, None, deltaphi, deltaphi, deltaphi, deltaeta, deltaeta, deltaeta, deltaR, deltaR, deltaR, sum_pt, transverse_mass,
                     transverse_mass, transverse_mass, total_transverse_mass, HNL_CM_angles_with_MET, W_CM_angles_to_plane, 
                     W_CM_angles_to_plane_with_MET, HNL_CM_masses, HNL_CM_masses_with_MET, count_tauh]
        raw_vars_general = ['event', 'genWeight', 'MET_pt', 'MET_phi']
        lepton_specific = ['_eta', '_mass', '_phi', '_pt', '_charge', '_genPartFlav']
        raw_vars_lepton1 = lepton_specific
        raw_vars_lepton2 = lepton_specific
        raw_vars_lepton3 = lepton_specific
        input_vars = [['event'], ['genWeight'], ['1_phi', '2_phi'], ['1_phi', '3_phi'], ['2_phi', '3_phi'], ['1_eta', '2_eta'], 
                      ['1_eta', '3_eta'], ['2_eta', '3_eta'], ['1_eta', '2_eta', '1_phi', '2_phi'], ['1_eta', '3_eta', '1_phi', '3_phi'],
                      ['2_eta', '3_eta', '2_phi', '3_phi'], [['1_pt', '2_pt', '3_pt'],['1_phi', '2_phi', '3_phi'],['1_eta', '2_eta', '3_eta'],
                       ['1_mass', '2_mass', '3_mass']], ['1_pt', '2_pt', '1_phi', '2_phi'], ['1_pt', '3_pt', '1_phi', '3_phi'], 
                      ['2_pt', '3_pt', '2_phi', '3_phi'], ['1_pt', '2_pt', '3_pt', 'MET_pt', '1_phi', '2_phi', '3_phi', 'MET_phi'],
                      ['1_charge', '2_charge', '3_charge', '1_pt', '2_pt', '3_pt', 'MET_pt', '1_phi', '2_phi', '3_phi', 'MET_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'],
                      ['1_charge', '2_charge', '3_charge', '1_pt', '2_pt', '3_pt', '1_phi', '2_phi', '3_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'], 
                      ['1_charge', '2_charge', '3_charge', '1_pt', '2_pt', '3_pt', 'MET_pt', '1_phi', '2_phi', '3_phi', 'MET_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'],
                      ['1_charge', '2_charge', '3_charge', '1_pt', '2_pt', '3_pt', '1_phi', '2_phi', '3_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'], 
                      ['1_charge', '2_charge', '3_charge', '1_pt', '2_pt', '3_pt', 'MET_pt', '1_phi', '2_phi', '3_phi', 'MET_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'],
                      ['channel', '1_genPartFlav', '2_genPartFlav', '3_genPartFlav']]
        super().__init__(channel, raw_vars_general=raw_vars_general, raw_vars_lepton1=raw_vars_lepton1, raw_vars_lepton2=raw_vars_lepton2, 
                         raw_vars_lepton3=raw_vars_lepton3, output_vars=output_vars, functions=functions, input_vars=input_vars)
        

class Data_extractor_v4(Data_extractor):
    def __init__(self, channel):
        output_vars = deepcopy(output_vars_v4)
        functions =[None, None,                 # event, genWeight
                    None, None, None,           # charges
                    None, None, None, None,     # pts
                    None, None, None,           # etas
                    None, None, None,           # masses
                    deltaphi, deltaphi, deltaphi, 
                    deltaphi, deltaphi, deltaphi,
                    deltaphi3,
                    deltaeta, deltaeta, deltaeta, 
                    deltaeta3,
                    deltaR, deltaR, deltaR, 
                    deltaR3,
                    sum_pt, 
                    transverse_mass, transverse_mass, transverse_mass, 
                    transverse_mass, transverse_mass, transverse_mass,
                    transverse_mass3,
                    invariant_mass, invariant_mass, invariant_mass,
                    invariant_mass,
                    total_transverse_mass, 
                    HNL_CM_angles_with_MET, 
                    W_CM_angles_to_plane, W_CM_angles_to_plane_with_MET,
			        HNL_CM_masses,
                    HNL_CM_masses_with_MET, 
                    W_CM_angles,
                    count_tauh]
        raw_vars_general = ['event', 'genWeight', 'MET_pt', 'MET_phi']
        lepton_specific = ['_eta', '_mass', '_phi', '_pt', '_charge', '_genPartFlav']
        raw_vars_lepton1 = lepton_specific
        raw_vars_lepton2 = lepton_specific
        raw_vars_lepton3 = lepton_specific
        input_vars = [['event'], ['genWeight'], 
			        ['1_charge'], ['2_charge'], ['3_charge'], 
			        ['1_pt'], ['2_pt'], ['3_pt'], ['MET_pt'],
			        ['1_eta'], ['2_eta'], ['3_eta'], 
			        ['1_mass'], ['2_mass'], ['3_mass'], 
			        ['1_phi', '2_phi'], ['1_phi', '3_phi'], ['2_phi', '3_phi'], 
			        ['1_phi', 'MET_phi'], ['2_phi', 'MET_phi'], ['3_phi', 'MET_phi'], 
			        ['1_pt', '2_pt', '3_pt', 'MET_pt', '1_phi', '2_phi', '3_phi', 'MET_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'],
			        ['1_eta', '2_eta'], ['1_eta', '3_eta'], ['2_eta', '3_eta'], 
			        ['1_pt', '2_pt', '3_pt', '1_phi', '2_phi', '3_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'],
			        ['1_eta', '2_eta', '1_phi', '2_phi'], ['1_eta', '3_eta', '1_phi', '3_phi'], ['2_eta', '3_eta', '2_phi', '3_phi'], 
			        ['1_pt', '2_pt', '3_pt', '1_phi', '2_phi', '3_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'],
			        [['1_pt', '2_pt', '3_pt'],['1_phi', '2_phi', '3_phi'],['1_eta', '2_eta', '3_eta'], ['1_mass', '2_mass', '3_mass']], 
			        ['1_pt', '2_pt', '1_phi', '2_phi'], ['1_pt', '3_pt', '1_phi', '3_phi'], ['2_pt', '3_pt', '2_phi', '3_phi'],
			        ['1_pt', 'MET_pt', '1_phi', 'MET_phi'], ['2_pt', 'MET_pt', '2_phi', 'MET_phi'], ['3_pt', 'MET_pt', '3_phi', 'MET_phi'],
			        ['1_pt', '2_pt', '3_pt', 'MET_pt', '1_phi', '2_phi', '3_phi', 'MET_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'],
			        [['1_pt', '2_pt'],['1_phi', '2_phi'],['1_eta', '2_eta'], ['1_mass', '2_mass']], [['1_pt', '3_pt'],['1_phi', '3_phi'],['1_eta', '3_eta'], ['1_mass', '3_mass']], [['2_pt', '3_pt'],['2_phi', '3_phi'],['2_eta', '3_eta'], ['2_mass', '3_mass']], 	
                    [['1_pt', '2_pt', '3_pt'],['1_phi', '2_phi', '3_phi'],['1_eta', '2_eta', '3_eta'], ['1_mass', '2_mass', '3_mass']], 
			        ['1_pt', '2_pt', '3_pt', 'MET_pt', '1_phi', '2_phi', '3_phi', 'MET_phi'],
			        ['1_charge', '2_charge', '3_charge', '1_pt', '2_pt', '3_pt', 'MET_pt', '1_phi', '2_phi', '3_phi', 'MET_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'],
			        ['1_charge', '2_charge', '3_charge', '1_pt', '2_pt', '3_pt', '1_phi', '2_phi', '3_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'], ['1_charge', '2_charge', '3_charge', '1_pt', '2_pt', '3_pt', 'MET_pt', '1_phi', '2_phi', '3_phi', 'MET_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'],
			        ['1_charge', '2_charge', '3_charge', '1_pt', '2_pt', '3_pt', '1_phi', '2_phi', '3_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'], 
			        ['1_charge', '2_charge', '3_charge', '1_pt', '2_pt', '3_pt', 'MET_pt', '1_phi', '2_phi', '3_phi', 'MET_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'],
			        ['1_pt', '2_pt', '3_pt', 'MET_pt', '1_phi', '2_phi', '3_phi', 'MET_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'],
			        ['channel', '1_genPartFlav', '2_genPartFlav', '3_genPartFlav']]
        super().__init__(channel, raw_vars_general=raw_vars_general, raw_vars_lepton1=raw_vars_lepton1, raw_vars_lepton2=raw_vars_lepton2, 
                         raw_vars_lepton3=raw_vars_lepton3, output_vars=output_vars, functions=functions, input_vars=input_vars)

