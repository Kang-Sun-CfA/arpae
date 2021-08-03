# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 22:11:37 2021

@author: kangsun
"""
import numpy as np
import pandas as pd
import logging
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import datetime as dt
import time
import sys, os
try:
    from hapi import ISO
except:
    logging.warning('hapi.py cannot be found. try to download to the current working directory')
    import urllib.request
    urllib.request.urlretrieve ('https://hitran.org/static/hapi/hapi.py', 'hapi.py')
from hapi import db_begin, fetch, fetch_by_ids, ISO, ISO_INDEX,\
    absorptionCoefficient_HT, absorptionCoefficient_Voigt, arange_
from collections import OrderedDict
from lmfit import Model

global_constants = {'kB':1.38065e-23,
                    'Pa_per_torr':133.322368,
                    'Pa_per_atm':101325,
                    'atm_per_torr':0.00131578947}

class HAPI_Molecule():
    '''
    a class for a molecule/isotope in hitran, wrapping around hapi's core functions
    it is a generic python class
    '''
    def __init__(self,nickname,mol_id,iso_id,
                 ref_mixing_ratio=None,IntensityThreshold=0.):
        '''
        nickname:
            however you want to call the molecule/isotopologue. it will be the 
            ID of this molecule and the hitran line list file. Use 'H2O' or 'H2O_11'
            as nickname of water vapor
        mol_id:
            molecular id in hitran convention. h2o is 1, co2 is 2, n2o is 4
        iso_d:
            local isotopologue id in hitran convention. see https://hitran.org/docs/iso-meta/
        ref_mixing_ratio:
            I don't think it matters for now
        IntensityThreshold:
            absolute value of minimum intensity in cm-1/ (molec x cm-2) to consider.   
            NOTE: default value is 0.   
            NOTE2: Setting this parameter to a value above zero is only recommended for very experienced users. 
        '''
        self.logger = logging.getLogger(__name__)
        self.nickname = nickname
        self.mol_id = mol_id
        self.iso_id = iso_id
        if np.isscalar(iso_id):
            self.global_iso_id = ISO[(mol_id,iso_id)][ISO_INDEX['id']]
            self.abundance = ISO[(mol_id,iso_id)][ISO_INDEX['abundance']]
            self.molecular_weight = ISO[(mol_id,iso_id)][ISO_INDEX['mass']]
            self.iso_name = ISO[(mol_id,iso_id)][ISO_INDEX['iso_name']]
            self.mol_name = ISO[(mol_id,iso_id)][ISO_INDEX['mol_name']]
        else:
            self.global_iso_id = [ISO[(mol_id,iid)][ISO_INDEX['id']] for iid in iso_id]
            self.abundance = [ISO[(mol_id,iid)][ISO_INDEX['abundance']] for iid in iso_id]
            self.molecular_weight = [ISO[(mol_id,iid)][ISO_INDEX['mass']] for iid in iso_id]
            self.iso_name = [ISO[(mol_id,iid)][ISO_INDEX['iso_name']] for iid in iso_id]
            self.mol_name = [ISO[(mol_id,iid)][ISO_INDEX['mol_name']] for iid in iso_id]
        if ref_mixing_ratio is None:
            if mol_id == 1:
                ref_mixing_ratio = 0.01
            elif mol_id == 2:
                ref_mixing_ratio = 420e-6
            elif mol_id == 4:
                ref_mixing_ratio = 330e-9
        self.ref_mixing_ratio = ref_mixing_ratio
        self.mixing_ratio = ref_mixing_ratio
        self.IntensityThreshold = IntensityThreshold
    
    def fetch_table(self,hapi_database_dir,
                    min_wavenumber=2242,max_wavenumber=2243.5,if_fetch=False):
        ''' 
        hapi_database_dir:
            directory to save or that contains the hitran line list for this molecule
        min/max_wavenumber:
            boundaries of wavenumbers to fetch line lists, if if_fetch is True
        if_fetch:
            download/overwrite line list if True
        '''
        if not os.path.exists(hapi_database_dir):
            os.makedirs(hapi_database_dir)
        self.hapi_database_dir = hapi_database_dir
        db_begin(hapi_database_dir)
        if if_fetch:
            if np.isscalar(self.iso_id):
                fetch(self.nickname,self.mol_id,self.iso_id,min_wavenumber,max_wavenumber)
            else:
                fetch_by_ids(self.nickname,self.global_iso_id,min_wavenumber,max_wavenumber)
        
    def get_optical_depth(self,nu,mixing_ratio,T_K,p_torr,L_cm,
                          air_density=None,profile='Voigt',H2O_mixing_ratio=0.,
                          return_OD=False):
        '''
        IMPORTANT:H2O_mixing_ratio should be related to DRY AIR, not total air
        nu:
            wavenumber grid, should be sorted
        mixing_ratio:
            mixing_ratio of this molecule relative to dry air
        T_K, p_torr, L_cm:
            obviously
        air_density:
            dry air density in SI unit. inferred from T/p if is None
        profile:
            line shape type. voigt and HT are supported for now
        H2O_mixing_ratio:
            mixing_ratio of water vapor
        return_OD:
            if True, return optical depth as an array. otherwise attached the 
            ODs as an attribute to the object
        '''
        if air_density is None:
            # note pressure should be converted to dry air partial pressure
            air_density = p_torr/(1+H2O_mixing_ratio)*global_constants['Pa_per_torr']/T_K/global_constants['kB']*1e-6
        self.mixing_ratio = mixing_ratio
        self.T_K = T_K
        self.p_torr = p_torr
        self.L_cm = L_cm
        if H2O_mixing_ratio == 0:
            diluent_dict = {'air':1}
        else:
            diluent_dict = {'air':1/(1+H2O_mixing_ratio),'H2O':H2O_mixing_ratio/(1+H2O_mixing_ratio)}
        if profile.lower() == 'voigt':
            _,sigma = absorptionCoefficient_Voigt(SourceTables=self.nickname,
                                                  OmegaGrid=nu,
                                                  Environment={'T':T_K,'p':p_torr*global_constants['atm_per_torr']},
                                                  Diluent=diluent_dict,
                                                  HITRAN_units=True,
                                                  IntensityThreshold=self.IntensityThreshold)
        elif profile.lower() == 'ht':
            _,sigma = absorptionCoefficient_HT(SourceTables=self.nickname,
                                                  OmegaGrid=nu,
                                                  Environment={'T':T_K,'p':p_torr*global_constants['atm_per_torr']},
                                                  Diluent=diluent_dict,
                                                  HITRAN_units=True,
                                                  IntensityThreshold=self.IntensityThreshold)
        if return_OD:
            return L_cm*sigma*mixing_ratio*air_density
        else:
            self.optical_depth = L_cm*sigma*mixing_ratio*air_density
 
class HAPI_Molecules(OrderedDict):
    '''
    a class built off the OrderedDict python class, each variable of which is 
    a HAPI_Molecule object with its nickname attribute as the key name
    '''
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def add_molecule(self,hapi_molecule):
        '''
        add a HAPI_Molecule object
        '''
        self.__setitem__(hapi_molecule.nickname,hapi_molecule)
    
    def fetch_tables(self,hapi_database_dir,
                    min_wavenumber=2242,max_wavenumber=2243.5,update_linelists=True):
        ''' 
        run HAPI_Molecule.fetach_table for all molecules
        hapi_database_dir:
            directory to save or that contains the hitran line lists for all molecules
            in this object
        min/max_wavenumber:
            boundaries of wavenumbers to fetch line lists, if update_linelist is True
        update_linelist:
            download/update line list if is True
        '''
        for (k,v) in self.items():
            self.logger.info('fetching hitran data for {}'.format(k))
            v.fetch_table(hapi_database_dir,
                    min_wavenumber=min_wavenumber,max_wavenumber=max_wavenumber,if_fetch=update_linelists)
            
    def get_optical_depths(self,nu,mixing_ratios,T_K,p_torr,L_cm,
                           profile='Voigt',return_ODs=False):
        ''' 
        run HAPI_Molecule.get_optical_depth for all molecules. see comments there 
        for details. major difference is that mixing_ratios should be a dictionary
        instead of a number. e.g., {'n2o':330e-9}
        '''
        if 'H2O_11' in mixing_ratios.keys():
            H2O_MR = mixing_ratios['H2O_11']
        elif 'H2O' in mixing_ratios.keys():
            H2O_MR = mixing_ratios['H2O']
        else:
            self.logger.warning('water vapor mixing ratio unknown, assuming 0')
            H2O_MR = 0.
        self.nu = nu
        ODs = np.array([v.get_optical_depth(
            nu,mixing_ratios[k],T_K,p_torr,L_cm,profile=profile,
            H2O_mixing_ratio=H2O_MR,return_OD=True) for (k,v) in self.items()])
        if return_ODs:
            return ODs
        else:
            self.optical_depths = ODs
    
    def plot_optical_depths(self,plot_names=None,ax=None,kwargs={}):
        ''' 
        plot selected molecules' optical depths
        plot_names:
            a list of molecule (nick)names to plot. plot all if is None
        ax:
            axes object to plot on. create fig/ax if is None
        '''
        if ax is None:
            self.logger.info('axes not supplied, creating one')
            fig,ax = plt.subplots(1,1,figsize=(10,5))
        else:
            fig = None
        figout = {}
        figout['fig'] = fig
        figout['ax'] = ax
        if plot_names is None:
            self.logger.info('plot all molecules')
            plot_names = list(self.keys())
            ODs = self.optical_depths.T
        else:
            plot_indices = [list(self.keys()).index(k) for k in plot_names]
            ODs = self.optical_depths[plot_indices,].T
        figout['ODs'] = ax.plot(self.nu,ODs,**kwargs)
        figout['leg'] = ax.legend(figout['ODs'],plot_names)
        return figout

def F_forward_model_TILDAS_142(channels,ref_channel,
                    T_K_cell,p_torr_cell,L_cm_cell,
                    N2O_41_cell,CO2_21_cell,CO2_22_cell,CO2_25_cell,H2O_11_cell,
                    hapi_molecules_cell,
                    T_K_ambient,p_torr_ambient,L_cm_ambient,
                    N2O_ambient,CO2_ambient,H2O_ambient,
                    hapi_molecules_ambient,
                    p0_wavenumber,p0_baseline,
                    p1_wavenumber=0,p1_baseline=0,
                    p2_wavenumber=0,p2_baseline=0,
                    p3_wavenumber=0,p3_baseline=0):
    '''
    forward model to model observed spectrum of the TILDAS_142 instrument
    inputs are largely hard-coded. create your own forward model for other configures
    '''
    poly_wavenumber = np.array([p3_wavenumber,p2_wavenumber,p1_wavenumber,p0_wavenumber],dtype=np.float64)
    nu = np.polyval(poly_wavenumber,channels-ref_channel)
    poly_baseline = np.array([p3_baseline,p2_baseline,p1_baseline,p0_baseline],dtype=np.float64)
    baseline = np.polyval(poly_baseline,channels-ref_channel)
    mixing_ratios_cell = {'N2O_41':np.float64(N2O_41_cell),
                          'CO2_21':np.float64(CO2_21_cell),
                          'CO2_22':np.float64(CO2_22_cell),
                          'CO2_25':np.float64(CO2_25_cell),
                          'H2O_11':np.float64(H2O_11_cell)}
    mixing_ratios_ambient = {'N2O':np.float64(N2O_ambient),
                             'CO2':np.float64(CO2_ambient),
                             'H2O':np.float64(H2O_ambient)}
    OD_cell = np.sum(hapi_molecules_cell.get_optical_depths(nu=nu,mixing_ratios=mixing_ratios_cell,
                   T_K=T_K_cell,p_torr=p_torr_cell,L_cm=L_cm_cell,return_ODs=True),axis=0)
    OD_ambient = np.sum(hapi_molecules_ambient.get_optical_depths(nu=nu,mixing_ratios=mixing_ratios_ambient,
                   T_K=T_K_ambient,p_torr=p_torr_ambient,L_cm=L_cm_ambient,return_ODs=True),axis=0)
    return baseline*np.exp(-OD_cell-OD_ambient)

class Fitting_Window(dict):
    '''
    class for a fitting window in TILDAS spectrum, based on python dictionary
    '''
    def __init__(self,window_name):
        '''
        window_name:
            give the fitting window a name. e.g., 'n2o_window' or 'full_window'
        '''
        self.logger = logging.getLogger(__name__)
        self.window_name = window_name
    
    def plot(self,ax=None,kwarg={}):
        '''
        quickly plot spectrum in this window, signal vs. channels
        '''
        if ax is None:
            self.logger.info('axes not supplied, creating one')
            fig,ax = plt.subplots(1,1,figsize=(10,5))
        else:
            fig = None
        figout = {}
        figout['fig'] = fig
        figout['ax'] = ax
        figout['spec'] = ax.plot(self['x'],self['y'],**kwarg)
        return figout
    
    def guess_wavcal(self,ref_channel,baseline_indices=None,line_knowledge_list=None):
        '''
        make a guess of the channels-wavenumber relationship by finding channel
        indices of major peaks
        ref_channel:
            the independent variable is channels-ref_channel
        baseline_indices:
            channel numbers (or indices, they are equivalent) that are largely absorption-free
            if is None, it is based on selected channels in the full_window (channels 0-400)
        line_knowledge_list:
            a dictionary of reasonable estimations of strong lines. see the following
            example after "if line_knowledge_list is None:"
        return:
            polynomial coefficients for channels-wavenumber relationship, the order
            is number of lines in line_knowledge_list-1
        '''
        if baseline_indices is None:
            baseline_indices = np.hstack((np.arange(0,35),np.arange(150,210),np.arange(270,330)))
        ybaseline = np.polyval(np.polyfit(self['x'][baseline_indices],self['y'][baseline_indices],3),self['x'])
        absorption = (ybaseline-self['y'])/ybaseline
        if line_knowledge_list is None:
            line_knowledge_list = [{'name':'n2o','index range':[50,150],'wavenumber':2242.453110,'absorption range':[0.1,0.25]},#n2o line
                                   {'name':'h2o','index range':[200,300],'wavenumber':2242.734230,'absorption range':[0.02,0.3]},#h2o line
                                   {'name':'co2','index range':[300,400],'wavenumber':2242.904508,'absorption range':[0.1,0.3]}] #co2 line
                                   
        line_wn_list = []
        line_idx_list = []
        for (i,lk) in enumerate(line_knowledge_list):
            spec =np.ma.masked_where((self['x']<lk['index range'][0]) | (self['x']>lk['index range'][1]),absorption)
            if np.nanmax(spec) > lk['absorption range'][1] or np.nanmax(spec) < lk['absorption range'][0]:
                self.logger.warning(lk['name']+' cannot be found')
            line_wn_list.append(lk['wavenumber'])
            line_idx_list.append(np.argmax(spec))
        wavcal_poly = np.polyfit(np.array(line_idx_list)-ref_channel,line_wn_list,len(line_wn_list)-1)
        return wavcal_poly
            
    def fit(self,forward_model_function,
            independent_vars_dict,
            param_names,param_prior,param_vary,
            max_nfev=100):
        '''
        The Fitting_Window class is where the core spectral fitting function is defined
        forward_model_function:
            name of forward model function, e.g., F_forward_model_TILDAS_142
        independent_vars_dict:
            keyword dictionary of independent variables in the forward model
        param_names,param_prior,param_vary:
            lists of the same length that specify property of the params object in lmfit
        max_nfev:
            max number of forward model evaluation
        '''
        if len(set([len(param_names),len(param_prior),len(param_vary)]))!=1:
            self.logger.error('parameter properties should have the same length')
            return
        p0_baseline_index = param_names.index('p0_baseline')
        # infer continuum level
        #param_prior[p0_baseline_index] = self['y'][self['x']==independent_vars_dict['ref_channel']][0]
        # this way ref_channel doesn't have to be in channels
        param_prior[p0_baseline_index] = np.nanmedian(self['y'])
        m1 = Model(func=forward_model_function,nan_policy='propagate',
                   independent_vars=independent_vars_dict.keys(),
                   param_names=param_names)
        [m1.set_param_hint(name=n,value=v,vary=ifv) for (n,v,ifv) in zip(param_names,param_prior,param_vary)];
        params = m1.make_params()
        # make sure the channels in independent vars are double
        independent_vars_dict['channels'] = np.float64(self['x'])
        independent_vars_dict['ref_channel'] = np.float64(independent_vars_dict['ref_channel'])
        self['fit_result'] = m1.fit(self['y'],params=params,**independent_vars_dict,max_nfev=max_nfev)
        
    def plot_fit(self):
        ''' 
        quick plot of fit result spectra
        '''
        fig,axs = plt.subplots(2,1,figsize=(10,5),sharex=True,constrained_layout=True)
        axs[0].plot(self['x'],self['y'],'ok')
        axs[0].plot(self['x'],self['fit_result'].init_fit,'-b')
        axs[0].plot(self['x'],self['fit_result'].best_fit,'-r')
        axs[1].plot(self['x'],self['fit_result'].residual,'-ok')

class TILDAS_Spectrum(dict):
    '''
    class of an aerodyne TILDAS sensor spectrum, based on python dictionary
    '''
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.nwindow = 0
    
    def trim_spectrum(self,window_name=None,spectrum_indices=np.arange(0,400),
                      zlo_indices=np.arange(470,479)):
        '''
        trim the spectrum to add one Fitting_Window object. expected to be called 
        within fit_spectrum below
        window_name:
            whatever you want to call the Fitting_Window object, e.g., full_window,
            n2o_window, co2_window. if is None, it will be window_1,2,...
        spectrum_indices:
            channels of the selected window
        zlo_indices:
            channels specifying the zero-level offset. use TILDAS_Spectrum.plot_spectrum 
            to plot and zoom
        '''
        zlo = np.nanmean(self['Spectrum'][zlo_indices])
        self.nwindow+=1
        if window_name is None:
            window_name = 'window_{}'.format(self.nwindow)
        self[window_name] = Fitting_Window(window_name)
        self[window_name].__setitem__('y',self['Spectrum'][spectrum_indices]-zlo)
        self[window_name].__setitem__('x',spectrum_indices)
    
    def fit_spectrum(self,window_name,
                     hapi_molecules_cell=None,
                     hapi_molecules_ambient=None,
                     database_dir_cell=None,
                     database_dir_ambient=None):
        '''
        expect lots of development in this function. use window_name to predefine fitting windows
        window_name:
            whatever you want to call the Fitting_Window object, e.g., full_window,
            n2o_window, co2_window.
        hapi_molecules_cell/ambient:
            the HAPI_Molecules objects corresponding to the multipass cell and
            ambient air. If None, generate here
        database_cell/ambient:
            paths to hitran files. won't be needed if hapi_molecules_cell/ambient
            are provided
        '''
        if window_name == 'full_window':
            if hapi_molecules_cell is None:
                self.logger.info('the HAPI_Molecules object for the cell is not provided. generating internally...')
                min_wavenumber=2242;max_wavenumber=2243.5
                mol_list_cell = ['N2O_41','CO2_21','CO2_22','CO2_25','H2O_11']
                moln_list_cell = [4,2,2,2,1]
                ison_list_cell = [1,1,2,5,1]
                minI_list_cell = [1e-21,1e-24,1e-24,1e-24,1e-26]
                p_torr_cell = self['PressureSample']
                T_K_cell = self['TemperatureSample']
                L_cm_cell = self['PathLengthSample']
                if database_dir_cell is None:
                    self.logger.error('you have to provide location of hitran files!')
                    return
                hms_cell = HAPI_Molecules()
                for (mol,moln,ison,minI) in zip(mol_list_cell,moln_list_cell,ison_list_cell,minI_list_cell):
                    hms_cell.add_molecule(HAPI_Molecule(nickname=mol,mol_id=moln,iso_id=ison,IntensityThreshold=minI))
                try:
                    hms_cell.fetch_tables(database_dir_cell,min_wavenumber,max_wavenumber,update_linelists=False)
                except Exception as e:
                    self.logger.warning(e)
                    self.logger.warning('no hitran files found. try downloading')
                    hms_cell.fetch_tables(database_dir_cell,min_wavenumber,max_wavenumber,update_linelists=True)
            if hapi_molecules_ambient is None:
                self.logger.info('the HAPI_Molecules object for the ambient air is not provided. generating internally...')
                min_wavenumber=2242;max_wavenumber=2243.5
                mol_list_ambient = ['N2O','CO2','H2O']
                moln_list_ambient = [4,2,1]
                ison_list_ambient = [1,[1,2,5],1]
                minI_list_ambient = [1e-19,1e-23,1e-25]
                p_torr_ambient = 760
                T_K_ambient = self['TemperatureSample']
                L_cm_ambient = 130
                if database_dir_ambient is None:
                    self.logger.error('you have to provide location of hitran files!')
                    return
                hms_ambient = HAPI_Molecules()
                for (mol,moln,ison,minI) in zip(mol_list_ambient,moln_list_ambient,ison_list_ambient,minI_list_ambient):
                    hms_ambient.add_molecule(HAPI_Molecule(nickname=mol,mol_id=moln,iso_id=ison,IntensityThreshold=minI))
                try:
                    hms_ambient.fetch_tables(database_dir_ambient,min_wavenumber,max_wavenumber,update_linelists=False)
                except Exception as e:
                    self.logger.warning(e)
                    self.logger.warning('no hitran files found. try downloading')
                    hms_ambient.fetch_tables(database_dir_ambient,min_wavenumber,max_wavenumber,update_linelists=True)
            
            spectrum_indices=np.arange(0,400)
            zlo_indices=np.arange(470,479)
            ref_channel = 200
            # # those two are reasonable first guesses
            # p0_wavenumber = 2242.63
            # p1_wavenumber = -1/570
            self.trim_spectrum(window_name,spectrum_indices,zlo_indices)
            # wavelength calibration
            baseline_indices = np.hstack((np.arange(0,35),np.arange(150,210),np.arange(270,330)))
            line_knowledge_list = [{'name':'n2o','index range':[50,150],'wavenumber':2242.453110,'absorption range':[0.1,0.25]},#n2o line
                                   {'name':'h2o','index range':[200,300],'wavenumber':2242.734230,'absorption range':[0.02,0.3]},#h2o line
                                   {'name':'co2','index range':[300,400],'wavenumber':2242.904508,'absorption range':[0.1,0.3]}] #co2 line
            wavcal_poly=self[window_name].guess_wavcal(ref_channel=ref_channel,
                                                       baseline_indices=baseline_indices,
                                                       line_knowledge_list=line_knowledge_list)
            p0_wavenumber = wavcal_poly[-1]
            p1_wavenumber = wavcal_poly[-2]
            p2_wavenumber = wavcal_poly[-3]
            param_names = ['T_K_cell','p_torr_cell','L_cm_cell',
                           'N2O_41_cell','CO2_21_cell','CO2_22_cell','CO2_25_cell','H2O_11_cell',
                           'T_K_ambient','p_torr_ambient','L_cm_ambient',
                           'N2O_ambient','CO2_ambient','H2O_ambient',
                           'p0_wavenumber','p0_baseline',
                           'p1_wavenumber','p1_baseline',
                           'p2_wavenumber','p2_baseline',
                           'p3_wavenumber','p3_baseline']
            param_prior = [T_K_cell,p_torr_cell,L_cm_cell,
                           330e-9,420e-6,420e-6,420e-6,0.01,
                           T_K_ambient,p_torr_ambient,L_cm_ambient,
                           330e-9,420e-6,0.01,
                           p0_wavenumber,1000,
                           p1_wavenumber,-1.8,
                           p2_wavenumber,0,
                           0,0]
            param_vary = np.array([0,1,0,
                                   1,1,1,1,1,
                                   0,0,0,
                                   0,0,0,
                                   1,1,
                                   1,1,
                                   1,1,
                                   1,1],dtype=np.bool)
            independent_vars_dict = {'channels':None,#will fill in Fitting_Window.fit
                                     'ref_channel':ref_channel,
                                     'hapi_molecules_cell':hms_cell,
                                     'hapi_molecules_ambient':hms_ambient}
            self[window_name].fit(F_forward_model_TILDAS_142,
                                  independent_vars_dict,
                                  param_names,param_prior,param_vary)            
    
    def plot_spectrum(self,ax=None):
        if ax is None:
            self.logger.info('axes not supplied, creating one')
            fig,ax = plt.subplots(1,1,figsize=(10,5))
        else:
            fig = None
        figout = {}
        figout['fig'] = fig
        figout['ax'] = ax
        figout['spec'] = ax.plot(self['Spectrum'])
        return figout

class TILDAS_Spectra(list):
    '''
    class of aerodyne TILDAS sensor spectra collection. it is a list of TILDAS_Spectrum object
    '''
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def merge(self,another_object):
        ''' 
        merge with another TILDAS_Spectra object
        '''
        if len(self) == 0:
            self.logger.info('the original object appears empty. returning a copy of new object')
            self.__dict__.update(another_object.__dict__)
            for s in another_object:
                self.append(s)
            return self
        if len(another_object) == 0:
            self.logger.info('the new object appears empty. nothing to add')
            return self
        if self.SpectrumLength != another_object.SpectrumLength:
            self.logger.error('inconsistent spectrum length!')
        self.logger.info('extending nspec from {} to {}'.format(self.nspec,self.nspec+another_object.nspec))
        self.nspec += another_object.nspec
        self.extend(another_object)
        return self
    
    def read_spb(self,fn):
        '''
        read and parse spb binary files. see page 56-57 of TDLwintel mannual
        fn:
            path to the spb file
        '''
        full = np.fromfile(fn,dtype=np.float64)
        idx = 0
        self.Revision = full[idx];idx+=1
        self.GlobalHeaderLength = int(full[idx]);idx+=1
        self.SpectralHeaderLength = int(full[idx]);idx+=1
        self.SpectrumLength = int(full[idx]);idx+=1
        self.nSpeciesUsed = int(full[idx]);idx+=1
        self.nLasersUsed = int(full[idx]);idx+=1
        # number of points for each laser
        self.nChansV = [full[i] for i in range(idx,idx+self.nLasersUsed)]
        idx+=self.nLasersUsed
        # tuning rate
        self.FreqWaves = [full[i] for i in range(idx,idx+self.SpectrumLength)]
        idx+=self.SpectrumLength
        # fitmarkers - what's that?
        self.FitMarks = [full[i] for i in range(idx,idx+self.SpectrumLength)]
        idx+=self.SpectrumLength
        self.PathLengthSample = full[idx];idx+=1
        self.PathLengthReference = full[idx];idx+=1
        # reference mixing ratios
        self.ReferenceMR = [full[i] for i in range(idx,idx+self.nSpeciesUsed)]
        idx+=self.nSpeciesUsed
        nspec = (len(full)-idx)/(6+2*self.nLasersUsed+self.nSpeciesUsed+self.SpectrumLength)
        if nspec != np.floor(nspec):
            self.logger.warning('last spectrum not complete?')
        nspec = int(np.floor(nspec))
        self.nspec = nspec
        n_bad_spec = 0
        for ispec in range(nspec):
            s = TILDAS_Spectrum()
            s['PathLengthSample'] = self.PathLengthSample
            # seems to be number of ms after 1904/1/1
            s['TimeStamp'] = full[idx];idx+=1
            s['DateTime'] = dt.datetime(1904,1,1)+dt.timedelta(seconds=s['TimeStamp']/1e3)
            s['Duration'] = full[idx];idx+=1
            s['PressureSample'] = full[idx];idx+=1
            s['TemperatureSample'] = full[idx];idx+=1
            s['PressureReference'] = full[idx];idx+=1
            s['TempReference'] = full[idx];idx+=1
            # peak positions
            s['FitPos'] = [full[i] for i in range(idx,idx+self.nSpeciesUsed)]
            idx+=self.nSpeciesUsed
            # laser linewidths
            s['LaserWidth'] = [full[i] for i in range(idx,idx+self.nLasersUsed)]
            idx+=self.nLasersUsed
            # scale factor for tuning rate
            s['TuningRateScaleFactor'] = [full[i] for i in range(idx,idx+self.nLasersUsed)]
            idx+=self.nLasersUsed
            s['Spectrum'] = full[idx:idx+self.SpectrumLength]
            idx+=self.SpectrumLength
            if s['TimeStamp'] == 0:
                self.logger.warning('seemingly bad spectrum at {}'.format(ispec))
                n_bad_spec+=1
                continue
            self.append(s)
        self.nspec -= n_bad_spec
        self.logger.info('{} spectra read from {}'.format(self.nspec,fn))
        return self
    
    def sample_gps(self,gps_filename):
        ''' 
        sample gps data to the TILDAS time stamps. 
        gps_filename:
            path to the gps ascii file saved by gps.py
        '''
        gps = pd.read_csv(gps_filename,header=None)
        gps_timestamp = gps[5].to_numpy()
        tildas_timestamp = np.array([dt.datetime.timestamp(s['DateTime']) for s in self])
        f_lat = interp1d(gps_timestamp,gps[2].to_numpy(),bounds_error=False)
        f_lon = interp1d(gps_timestamp,gps[3].to_numpy(),bounds_error=False)
        f_v = interp1d(gps_timestamp,gps[4].to_numpy(),bounds_error=False)
        for (i,s) in enumerate(self):
            s['Latitude'] = f_lat(tildas_timestamp[i])
            s['Longitude'] = f_lon(tildas_timestamp[i])
            s['Speed'] = f_v(tildas_timestamp[i])
    
    def plot_latlon(self,ax=None,zoom=9):
        ''' 
        plot coordinates
        '''
        if ax is None:
            self.logger.info('axes not supplied, creating one')
            fig,ax = plt.subplots(1,1,figsize=(10,5))
        else:
            fig = None
        figout = {}
        figout['fig'] = fig
        figout['ax'] = ax
        figout['spec'] = ax.plot(self.extract_array('Longitude'),self.extract_array('Latitude'),'o')
        try:
            import contextily as cx
            from pyproj import CRS
            if 'PROJ_LIB' not in os.environ:
                os.environ['PROJ_LIB'] = os.path.join(os.environ['CONDA_PREFIX'],'Library','share','proj')
                os.environ['GDAL_DATA'] = os.path.join(os.environ['CONDA_PREFIX'],'Library','share')
            cx.add_basemap(ax=ax,zoom=zoom,crs=CRS("EPSG:4326"))
        except Exception as e:
            self.logger.warning(e)
            self.logger.warning('sorry, basemap does not work')
        return figout
    
    def extract_array(self,extract_field,indices=None):
        ''' 
        extract one field from each TILDAS_Spectrum object to a numpy array
        extract_field:
            field name in TILDAS_Spectrum object. if 'spectral_mean' return mean
            of full_window
        indices:
            flexible to use only a subset of TILDAS_Spectrum objects
        '''
        if indices is None:
            indices = np.arange(len(self))
        if extract_field == 'spectral_mean':
            return np.array([np.nanmean(s['Spectrum'][0:400]-np.nanmean(s['Spectrum'][470:479])) for (i,s) in enumerate(self) if i in indices])
        else:
            return np.array([s[extract_field] for (i,s) in enumerate(self) if i in indices])
    
    def fit_dt_range(self,start_dt,end_dt,step=1,
                     window_name='full_window',
                     hapi_molecules_cell=None,
                     hapi_molecules_ambient=None,
                     database_dir_cell=None,
                     database_dir_ambient=None):
        ''' 
        selectively fit a subset of TILDAS_Spectrum objects by time. see
        TILDAS_Spectrum.fit_spectrum
        start/end_dt:
            start/end datetime between which spectra are fitted
        step:
            step size. 1 to fit every spectrum
        return:
            indices of TILDAS_Spectrum objects that are fitted
        '''
        ss_dt = self.extract_array('DateTime')
        idx = np.where((ss_dt>=start_dt)&(ss_dt<end_dt))[0]
        fitted_indices = np.arange(idx[0],idx[-1],step)
        # to be parallelized
        for i in fitted_indices:
            Time = time.time()
            self[i].fit_spectrum(window_name=window_name,
                                 hapi_molecules_cell=hapi_molecules_cell,
                                 hapi_molecules_ambient=hapi_molecules_ambient,
                                 database_dir_cell=database_dir_cell,
                                 database_dir_ambient=database_dir_ambient)
            self.logger.info('spectrum collected on {} fitted, took {:.3f} s'.format(self[i]['DateTime'].strftime('%Y%m%dT%H%M%S'),time.time()-Time))
        return fitted_indices
    
    def get_fitted_result(self,fitted_indices,param_name,window_name='full_window'):
        ''' 
        extract fit result of a param
        '''
        return np.array([s[window_name]['fit_result'].best_values[param_name] \
                         for (i,s) in enumerate(self) if i in fitted_indices])
    
    def plot_diagnostics(self,plot_fields=['spectral_mean','Speed','PressureSample','TemperatureSample']):
        '''
        quick plot of time series
        '''
        fig,axs = plt.subplots(len(plot_fields),1,sharex=True,constrained_layout=True)
        pdt = self.extract_array('DateTime')
        for (ax,f) in zip(axs,plot_fields):
            ax.plot(pdt,self.extract_array(f))
            ax.set_title(f);

