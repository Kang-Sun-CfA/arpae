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
                 ref_mixing_ratio=None,IntensityThreshold=0.,
                 profile='Voigt'):
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
        profile:
            line shape type. voigt and HT are supported for now
        '''
        self.logger = logging.getLogger(__name__)
        self.nickname = nickname
        self.mol_id = mol_id
        self.iso_id = iso_id
        self.profile = profile
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
                          air_density=None,profile=None,H2O_mixing_ratio=0.,
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
            line shape type. it becomes redundant after profile is included as
            an attribute at initiation. can still overwrite here
        H2O_mixing_ratio:
            mixing_ratio of water vapor
        return_OD:
            if True, return optical depth as an array. otherwise attached the 
            ODs as an attribute to the object
        '''
        if air_density is None:
            # note pressure should be converted to dry air partial pressure
            air_density = p_torr/(1+H2O_mixing_ratio)*global_constants['Pa_per_torr']/T_K/global_constants['kB']*1e-6
        if profile is None:
            profile = self.profile
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

def F_forward_model_frequency(channels,ref_channel,frequency,
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
    forward model to use "frequency" as base of wavenumber polynomial, instead of channels-ref_channel
    '''
    poly_wavenumber = np.array([p3_wavenumber,p2_wavenumber,p1_wavenumber,p0_wavenumber],dtype=np.float64)
    nu = np.polyval(poly_wavenumber,frequency)
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
        idx = np.isin(self['x'],baseline_indices)
        ybaseline = np.polyval(np.polyfit(self['x'][idx],self['y'][idx],3),self['x'])
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
            line_idx_list.append(self['x'][np.argmax(spec)])
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
        # [m1.set_param_hint(name=n,value=v,vary=ifv) for (n,v,ifv) in zip(param_names,param_prior,param_vary)];
        for (n,v,ifv) in zip(param_names,param_prior,param_vary):
            if n.lower()[0:3] == 'h2o':#make sure water vapor value does not go crazy
                m1.set_param_hint(name=n,value=v,vary=ifv,min=0,max=1)
            else:
                m1.set_param_hint(name=n,value=v,vary=ifv)
        params = m1.make_params()
        # make sure the channels in independent vars are double
        independent_vars_dict['channels'] = np.float64(self['x'])
        independent_vars_dict['ref_channel'] = np.float64(independent_vars_dict['ref_channel'])
        self['fit_result'] = m1.fit(self['y'],params=params,**independent_vars_dict,max_nfev=max_nfev)
        
    def plot_fit(self,xsource='nu'):
        ''' 
        quick plot of fit result spectra
        xsource:
            choose from 'x' or 'nu'
        '''
        fig,axs = plt.subplots(2,1,figsize=(10,5),sharex=True,constrained_layout=True)
        if xsource == 'x':
            xdata = self['x']
        elif xsource == 'nu':
            if 'nu' in self.keys():
                xdata = self['nu']
            else:
                p = np.array([self['fit_result'].best_values['p3_wavenumber'],
                              self['fit_result'].best_values['p2_wavenumber'],
                              self['fit_result'].best_values['p1_wavenumber'],
                              self['fit_result'].best_values['p0_wavenumber']])
                xdata = np.polyval(p,self['x']-self['ref_channel'])
        axs[0].plot(xdata,self['y'],'ok')
        axs[0].plot(xdata,self['fit_result'].init_fit,'-b')
        axs[0].plot(xdata,self['fit_result'].best_fit,'-r')
        axs[0].legend(['Data','Prior','Posterior'])
        axs[0].set_ylabel('Signal [mV]')
        axs[1].plot(xdata,self['fit_result'].residual,'-ok')
        axs[1].set_ylabel('Residual [mV]')
        rmse = np.sqrt(np.nanmean(np.power(self['fit_result'].residual,2)))
        rmse_r = rmse/self['fit_result'].best_values['p0_baseline']
        axs[1].legend(['residual rmse: {:.3f}, relative: {:.2%}'.format(rmse,rmse_r)])
        if xsource == 'x':
            axs[1].set_xlabel('Channels')
        else:
            axs[1].set_xlabel(r'Wavenumber [cm$^{-1}$]')
        figout = {'fig':fig,'axs':axs}
        return figout

class TILDAS_Spectrum(dict):
    '''
    class of an aerodyne TILDAS sensor spectrum, based on python dictionary
    '''
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.nwindow = 0
    
    def read_spe(self,fn):
        '''
        read spe file into a single spectrum
        fn:
            path to the spe file
        '''
        with open(fn) as f:
            Ls = f.readlines()
        nspec = np.int(Ls[3])
        spe_df = pd.read_csv(fn,header=3,nrows=nspec,sep='\t',names=['spec','fit','baseline','fitmarker','frequency'])
        self['Spectrum'] = np.array(spe_df['spec'])
        self['Frequency'] = np.array(spe_df['frequency'])
        for (i,l) in enumerate(Ls):
            if 'property - timestamp' in l:
                self['TimeStamp'] = np.float(Ls[i+1].strip('\n'))
                self['DateTime'] = dt.datetime(1904,1,1)+dt.timedelta(seconds=self['TimeStamp']/1e3)
            elif 'cell pressure' in l:
                self['PressureSample'] = np.float(Ls[i+1].strip('\n'))
            elif 'pathlength' in l:
                self['PathLengthSample'] = np.float(Ls[i+1].strip('\n'))
            elif 'cell temp' in l:
                self['TemperatureSample'] = np.float(Ls[i+1].strip('\n'))
        return self
                
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
        self[window_name].__setitem__('y',np.float64(self['Spectrum'][spectrum_indices]-zlo))
        self[window_name].__setitem__('x',np.float64(spectrum_indices))
        if 'Frequency' in self.keys():
            self[window_name].__setitem__('Frequency',np.float64(self['Frequency'][spectrum_indices]))
        else:
            self.logger.info('no Frequency in this spectrum, using presaved one')
            self[window_name].__setitem__('Frequency',static_data['Frequency'][spectrum_indices])
    
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
        p_torr_cell = self['PressureSample']
        T_K_cell = self['TemperatureSample']
        L_cm_cell = self['PathLengthSample']
        p_torr_ambient = 760
        T_K_ambient = self['TemperatureSample']
        L_cm_ambient = 130
        if window_name == 'full_window':
            if hapi_molecules_cell is None:
                self.logger.info('the HAPI_Molecules object for the cell is not provided. generating internally...')
                min_wavenumber=2242;max_wavenumber=2243.5
                mol_list_cell = ['N2O_41','CO2_21','CO2_22','CO2_25','H2O_11']
                moln_list_cell = [4,2,2,2,1]
                ison_list_cell = [1,1,2,5,1]
                minI_list_cell = [1e-21,1e-24,1e-24,1e-24,1e-26]
                if database_dir_cell is None:
                    self.logger.error('you have to provide location of hitran files!')
                    return
                hapi_molecules_cell = HAPI_Molecules()
                for (mol,moln,ison,minI) in zip(mol_list_cell,moln_list_cell,ison_list_cell,minI_list_cell):
                    hapi_molecules_cell.add_molecule(HAPI_Molecule(nickname=mol,mol_id=moln,iso_id=ison,IntensityThreshold=minI))
                try:
                    hapi_molecules_cell.fetch_tables(database_dir_cell,min_wavenumber,max_wavenumber,update_linelists=False)
                except Exception as e:
                    self.logger.warning(e)
                    self.logger.warning('no hitran files found. try downloading')
                    hapi_molecules_cell.fetch_tables(database_dir_cell,min_wavenumber,max_wavenumber,update_linelists=True)
            if hapi_molecules_ambient is None:
                self.logger.info('the HAPI_Molecules object for the ambient air is not provided. generating internally...')
                min_wavenumber=2242;max_wavenumber=2243.5
                mol_list_ambient = ['N2O','CO2','H2O']
                moln_list_ambient = [4,2,1]
                ison_list_ambient = [1,[1,2,5],1]
                minI_list_ambient = [1e-19,1e-23,1e-25]
                if database_dir_ambient is None:
                    self.logger.error('you have to provide location of hitran files!')
                    return
                hapi_molecules_ambient = HAPI_Molecules()
                for (mol,moln,ison,minI) in zip(mol_list_ambient,moln_list_ambient,ison_list_ambient,minI_list_ambient):
                    hapi_molecules_ambient.add_molecule(HAPI_Molecule(nickname=mol,mol_id=moln,iso_id=ison,IntensityThreshold=minI))
                try:
                    hapi_molecules_ambient.fetch_tables(database_dir_ambient,min_wavenumber,max_wavenumber,update_linelists=False)
                except Exception as e:
                    self.logger.warning(e)
                    self.logger.warning('no hitran files found. try downloading')
                    hapi_molecules_ambient.fetch_tables(database_dir_ambient,min_wavenumber,max_wavenumber,update_linelists=True)
            
            spectrum_indices=np.arange(10,400)
            zlo_indices=np.arange(470,479)
            ref_channel = 200
            
            self.trim_spectrum(window_name,spectrum_indices,zlo_indices)
            
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
                           2242.27,1000,
                           1,-1.8,
                           0,0,
                           0,0]
            param_vary = np.array([0,1,0,
                                   1,1,1,1,1,
                                   0,0,0,
                                   1,1,1,
                                   1,1,
                                   1,1,
                                   1,1,
                                   0,1],dtype=np.bool)
            independent_vars_dict = {'channels':None,#will fill in Fitting_Window.fit
                                     'ref_channel':ref_channel,
                                     'frequency':self[window_name]['Frequency'],
                                     'hapi_molecules_cell':hapi_molecules_cell,
                                     'hapi_molecules_ambient':hapi_molecules_ambient}
            self[window_name].fit(F_forward_model_frequency,
                                  independent_vars_dict,
                                  param_names,param_prior,param_vary)
            self[window_name]['ref_channel'] = ref_channel
            p = np.array([self[window_name]['fit_result'].best_values['p3_wavenumber'],
                          self[window_name]['fit_result'].best_values['p2_wavenumber'],
                          self[window_name]['fit_result'].best_values['p1_wavenumber'],
                          self[window_name]['fit_result'].best_values['p0_wavenumber']])
            self[window_name]['nu'] = np.polyval(p,self[window_name]['Frequency'])
        elif window_name == 'full_window_guess_nu':
            if hapi_molecules_cell is None:
                self.logger.info('the HAPI_Molecules object for the cell is not provided. generating internally...')
                min_wavenumber=2242;max_wavenumber=2243.5
                mol_list_cell = ['N2O_41','CO2_21','CO2_22','CO2_25','H2O_11']
                moln_list_cell = [4,2,2,2,1]
                ison_list_cell = [1,1,2,5,1]
                minI_list_cell = [1e-21,1e-24,1e-24,1e-24,1e-26]
                if database_dir_cell is None:
                    self.logger.error('you have to provide location of hitran files!')
                    return
                hapi_molecules_cell = HAPI_Molecules()
                for (mol,moln,ison,minI) in zip(mol_list_cell,moln_list_cell,ison_list_cell,minI_list_cell):
                    hapi_molecules_cell.add_molecule(HAPI_Molecule(nickname=mol,mol_id=moln,iso_id=ison,IntensityThreshold=minI))
                try:
                    hapi_molecules_cell.fetch_tables(database_dir_cell,min_wavenumber,max_wavenumber,update_linelists=False)
                except Exception as e:
                    self.logger.warning(e)
                    self.logger.warning('no hitran files found. try downloading')
                    hapi_molecules_cell.fetch_tables(database_dir_cell,min_wavenumber,max_wavenumber,update_linelists=True)
            if hapi_molecules_ambient is None:
                self.logger.info('the HAPI_Molecules object for the ambient air is not provided. generating internally...')
                min_wavenumber=2242;max_wavenumber=2243.5
                mol_list_ambient = ['N2O','CO2','H2O']
                moln_list_ambient = [4,2,1]
                ison_list_ambient = [1,[1,2,5],1]
                minI_list_ambient = [1e-19,1e-23,1e-25]
                if database_dir_ambient is None:
                    self.logger.error('you have to provide location of hitran files!')
                    return
                hapi_molecules_ambient = HAPI_Molecules()
                for (mol,moln,ison,minI) in zip(mol_list_ambient,moln_list_ambient,ison_list_ambient,minI_list_ambient):
                    hapi_molecules_ambient.add_molecule(HAPI_Molecule(nickname=mol,mol_id=moln,iso_id=ison,IntensityThreshold=minI))
                try:
                    hapi_molecules_ambient.fetch_tables(database_dir_ambient,min_wavenumber,max_wavenumber,update_linelists=False)
                except Exception as e:
                    self.logger.warning(e)
                    self.logger.warning('no hitran files found. try downloading')
                    hapi_molecules_ambient.fetch_tables(database_dir_ambient,min_wavenumber,max_wavenumber,update_linelists=True)
            
            spectrum_indices=np.arange(10,400)
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
                                   1,1,1,
                                   1,1,
                                   1,1,
                                   1,1,
                                   1,1],dtype=np.bool)
            independent_vars_dict = {'channels':None,#will fill in Fitting_Window.fit
                                     'ref_channel':ref_channel,
                                     'hapi_molecules_cell':hapi_molecules_cell,
                                     'hapi_molecules_ambient':hapi_molecules_ambient}
            self[window_name].fit(F_forward_model_TILDAS_142,
                                  independent_vars_dict,
                                  param_names,param_prior,param_vary)
            self[window_name]['ref_channel'] = ref_channel
            p = np.array([self[window_name]['fit_result'].best_values['p3_wavenumber'],
                          self[window_name]['fit_result'].best_values['p2_wavenumber'],
                          self[window_name]['fit_result'].best_values['p1_wavenumber'],
                          self[window_name]['fit_result'].best_values['p0_wavenumber']])
            self[window_name]['nu'] = np.polyval(p,self[window_name]['x']-self[window_name]['ref_channel'])
            
    
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
        if hasattr(self, 'main_df') and hasattr(another_object,'main_df'):
            self.main_df = pd.concat(self.main_df,another_object.main_df)
        return self
    
    def generate_main_df(self,str_file_list=[],stc_file_list=[],
                         TILDAS_Spectra_fields=['spectral_mean','PressureSample']):
        ''' 
        generate the main dataframe by incorporating str and stc files
        str_file_list:
            a list of str file names
        stc_file_list:
            a list of stc file names
        TILDAS_Spectra_fields:
            fields in self to be included in main_df
        yield a dataframe - main_df as one attribute of the object
        '''
        if len(str_file_list) == 0:
            self.logger.info('no str files')
            df_str = None
        else:
            df_list = []
            for f in str_file_list:
                with open(f, "r") as file:
                    first_line = file.readline()
                species = first_line.split(':')[-1].strip().split(',')
                species = ['time']+['{}_{}'.format(s,i) for (i,s) in enumerate(species)]
                df_list.append(pd.read_csv(f,header=None,
                                           skiprows=1,
                                           sep=' ',
                                           names=species,
                                           index_col=False,
                                           skipinitialspace=True))
            df_str = pd.concat(df_list)
        if len(stc_file_list) == 0:
            self.logger.info('no stc files')
            df_stc = None
        else:
            df_list = []
            for f in stc_file_list:
                df_list.append(pd.read_csv(f,header=1,skipinitialspace=True))
            df_stc = pd.concat(df_list)
            
        dict_spb = {'time':self.extract_array('TimeStamp')/1e3,
                    'DateTime':self.extract_array('DateTime')}
        [dict_spb.__setitem__(k, self.extract_array(k)) for k in TILDAS_Spectra_fields];
        df_spb = pd.DataFrame(dict_spb)
        
        # self.df_str = df_str
        # self.df_stc = df_stc
        if df_str is None and df_stc is None:
            self.main_df = df_spb.set_index('DateTime')
            return
        elif df_str is None:
            df_st = df_stc
        elif df_stc is None:
            df_st = df_str
        else:
            df_st = df_str.merge(df_stc,on='time')
        self.main_df = pd.merge_asof(df_spb, 
                                     df_st,on='time',
                                     tolerance=1e-2,
                                     direction='nearest').set_index('DateTime')
        if self.main_df.shape[0] != self.nspec:
            self.logger.warning('main_df has {} rows, whereas self has {} spectra'.format(self.main_df.shape[0],self.nspec))
    
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
            return np.array([np.nanmean(s['Spectrum'][10:400]-np.nanmean(s['Spectrum'][470:479])) for (i,s) in enumerate(self) if i in indices])
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
# frequency copied from 210729_163418_001_SIG.spe
static_data = {'Frequency':
              np.array([0.        , 0.00199959, 0.00399918, 0.00599877, 0.00799836,
                       0.00999795, 0.01199755, 0.01399713, 0.01599673, 0.01799631,
                       0.0199959 , 0.0219955 , 0.02399508, 0.02599481, 0.02799461,
                       0.02999437, 0.03199405, 0.03399362, 0.03599304, 0.0379923 ,
                       0.03999136, 0.04199018, 0.04398876, 0.04598703, 0.047985  ,
                       0.04998261, 0.05197985, 0.05397667, 0.05597302, 0.0579689 ,
                       0.05996423, 0.061959  , 0.06395318, 0.06594671, 0.06793959,
                       0.06993178, 0.07192327, 0.07391402, 0.07590401, 0.07789319,
                       0.07988156, 0.08186908, 0.08385571, 0.08584144, 0.08782623,
                       0.08981005, 0.0917929 , 0.09377477, 0.09575565, 0.09773552,
                       0.09971436, 0.10169219, 0.10366897, 0.10564471, 0.10761938,
                       0.109593  , 0.11156555, 0.113537  , 0.11550737, 0.11747667,
                       0.11944488, 0.121412  , 0.12337804, 0.125343  , 0.12730688,
                       0.12926968, 0.13123142, 0.13319208, 0.13515169, 0.13711025,
                       0.13906773, 0.14102414, 0.14297948, 0.14493373, 0.1468869 ,
                       0.14883896, 0.15078989, 0.15273969, 0.15468834, 0.15663579,
                       0.15858207, 0.16052713, 0.16247097, 0.16441356, 0.16635489,
                       0.16829494, 0.1702337 , 0.17217114, 0.17410725, 0.17604201,
                       0.17797537, 0.17990735, 0.1818379 , 0.18376702, 0.18569466,
                       0.18762083, 0.1895455 , 0.19146864, 0.19339024, 0.19531029,
                       0.19722879, 0.19914576, 0.20106121, 0.20297515, 0.20488758,
                       0.20679854, 0.20870801, 0.21061603, 0.21252261, 0.21442774,
                       0.21633147, 0.2182338 , 0.22013475, 0.22203433, 0.22393255,
                       0.22582943, 0.22772498, 0.2296192 , 0.23151213, 0.23340376,
                       0.23529412, 0.23718323, 0.2390711 , 0.24095777, 0.24284323,
                       0.24472753, 0.24661065, 0.24849262, 0.25037342, 0.2522531 ,
                       0.25413165, 0.25600908, 0.2578854 , 0.25976061, 0.2616347 ,
                       0.26350768, 0.26537953, 0.26725025, 0.26911984, 0.27098831,
                       0.27285562, 0.2747218 , 0.27658682, 0.27845068, 0.28031338,
                       0.28217491, 0.28403528, 0.28589448, 0.2877525 , 0.28960936,
                       0.29146503, 0.29331953, 0.29517287, 0.297025  , 0.29887593,
                       0.30072566, 0.30257419, 0.3044215 , 0.30626759, 0.30811248,
                       0.30995614, 0.31179859, 0.31363981, 0.31547983, 0.31731863,
                       0.3191562 , 0.32099256, 0.3228277 , 0.32466161, 0.32649429,
                       0.32832573, 0.33015595, 0.33198492, 0.33381265, 0.33563913,
                       0.33746434, 0.3392883 , 0.34111098, 0.34293238, 0.34475246,
                       0.34657125, 0.3483887 , 0.35020479, 0.35201954, 0.35383292,
                       0.35564492, 0.35745553, 0.35926474, 0.36107255, 0.36287894,
                       0.36468393, 0.36648748, 0.3682896 , 0.37009031, 0.37188958,
                       0.37368744, 0.37548385, 0.37727882, 0.37907237, 0.38086446,
                       0.38265511, 0.3844443 , 0.38623203, 0.38801829, 0.3898031 ,
                       0.39158646, 0.39336839, 0.39514889, 0.39692796, 0.39870565,
                       0.40048192, 0.40225682, 0.40403033, 0.40580244, 0.40757319,
                       0.40934256, 0.41111055, 0.41287717, 0.41464243, 0.41640632,
                       0.41816887, 0.41993009, 0.42168995, 0.42344849, 0.42520572,
                       0.42696162, 0.42871621, 0.43046949, 0.43222146, 0.43397213,
                       0.4357215 , 0.43746955, 0.4392163 , 0.44096175, 0.44270588,
                       0.44444869, 0.44619018, 0.44793035, 0.44966919, 0.4514067 ,
                       0.4531429 , 0.45487778, 0.45661135, 0.45834361, 0.46007458,
                       0.46180424, 0.46353261, 0.46525968, 0.46698545, 0.46870992,
                       0.47043309, 0.47215495, 0.47387551, 0.47559476, 0.4773127 ,
                       0.47902932, 0.48074461, 0.48245857, 0.48417118, 0.48588246,
                       0.48759237, 0.48930093, 0.49100813, 0.49271395, 0.49441839,
                       0.49612144, 0.49782311, 0.49952338, 0.50122225, 0.50291974,
                       0.50461583, 0.50631053, 0.50800384, 0.50969577, 0.51138629,
                       0.51307543, 0.51476318, 0.51644955, 0.51813451, 0.5198181 ,
                       0.5215003 , 0.52318111, 0.52486051, 0.52653852, 0.52821511,
                       0.52989029, 0.53156404, 0.53323637, 0.53490728, 0.53657677,
                       0.53824485, 0.53991151, 0.54157678, 0.54324064, 0.54490311,
                       0.5465642 , 0.54822389, 0.54988222, 0.55153919, 0.5531948 ,
                       0.55484907, 0.55650201, 0.55815363, 0.55980392, 0.5614529 ,
                       0.56310056, 0.56474692, 0.56639198, 0.56803573, 0.56967819,
                       0.57131936, 0.57295924, 0.57459782, 0.57623511, 0.57787111,
                       0.57950582, 0.58113924, 0.5827714 , 0.58440233, 0.58603206,
                       0.58766063, 0.58928804, 0.59091433, 0.59253954, 0.59416368,
                       0.5957868 , 0.59740891, 0.59903005, 0.60065023, 0.60226951,
                       0.60388791, 0.60550546, 0.60712219, 0.60873819, 0.61035349,
                       0.61196815, 0.6135822 , 0.6151957 , 0.61680871, 0.61842127,
                       0.62003344, 0.6216453 , 0.62325691, 0.62486833, 0.62647965,
                       0.62809092, 0.62970223, 0.63131366, 0.6329253 , 0.63453726,
                       0.63614961, 0.63776244, 0.63937586, 0.64098995, 0.64260478,
                       0.64422048, 0.64583712, 0.64745481, 0.64907362, 0.65069365,
                       0.65231502, 0.65393778, 0.65556204, 0.65718787, 0.65881536,
                       0.66044461, 0.66207567, 0.66370863, 0.66534359, 0.66698063,
                       0.6686198 , 0.67026122, 0.67190495, 0.67355105, 0.67519962,
                       0.67685074, 0.67850446, 0.68016085, 0.68181998, 0.68348189,
                       0.68514667, 0.68681436, 0.68848504, 0.69015875, 0.69183556,
                       0.69351548, 0.69519852, 0.69688468, 0.69857398, 0.70026641,
                       0.70196198, 0.70366064, 0.70536234, 0.70706703, 0.70877466,
                       0.71048514, 0.71219846, 0.71391452, 0.71563329, 0.71735467,
                       0.71907855, 0.72080481, 0.72253333, 0.724264  , 0.72599671,
                       0.7277313 , 0.7294676 , 0.73120544, 0.73294464, 0.73468503,
                       0.73642646, 0.73816873, 0.73991169, 0.74165519, 0.74339911,
                       0.74514328, 0.7468875 , 0.74863118, 0.75037486, 0.75211854,
                       0.75386222, 0.7556059 , 0.75734958, 0.75909326, 0.76083694,
                       0.76258062, 0.7643243 , 0.76606798, 0.76781166, 0.76955535,
                       0.77129903, 0.77304271, 0.77478639, 0.77653007, 0.77827375,
                       0.78001743, 0.78176112, 0.7835048 , 0.78524848, 0.78699216,
                       0.78873584, 0.79047952, 0.7922232 , 0.79396688, 0.79571057,
                       0.79745425, 0.79919793, 0.80094161, 0.80268529, 0.80442897,
                       0.80617265, 0.80791633, 0.80966001, 0.81140369, 0.81314737,
                       0.81489105, 0.81663473, 0.81837841, 0.8201221 , 0.82186578,
                       0.82360946, 0.82535314, 0.82709682, 0.8288405 , 0.83058418,
                       0.83232787, 0.83407155, 0.83581523, 0.83755891, 0.83930259,
                       0.84104627, 0.84278995, 0.84453364, 0.84627732, 0.848021  ,
                       0.84976468, 0.85150836, 0.85325204, 0.85499572, 0.8567394 ,
                       0.8567394 , 0.8577394 , 0.8587394 , 0.8597394 , 0.8607394 ,
                       0.8617394 , 0.8627394 , 0.8637394 , 0.8647394 , 0.8657394 ,
                       0.8667394 , 0.8677394 , 0.8687394 , 0.8697394 , 0.8707394 ,
                       0.8717394 , 0.8727394 , 0.8737394 , 0.8747394 , 0.8757394 ,
                       0.8767394 , 0.8777394 , 0.8787394 , 0.8797394 , 0.8807394 ,
                       0.8817394 , 0.8827394 , 0.8837394 , 0.8847394 , 0.8857394 ,
                       0.8867394 , 0.8877394 , 0.8887394 , 0.8897394 , 0.8907394 ,
                       0.8917394 , 0.8927394 , 0.8937394 , 0.8947394 , 0.8957394 ],dtype=np.float64)}