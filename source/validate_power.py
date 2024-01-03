#!/usr/bin/env python
# coding: utf-8
# @Author: Allison M. Campbell
# @Email: allison.m.campbell@pnnl.gov
# @Date: 2023-05-01 15:10:42
# @Last Modified by:   camp426
# @Last Modified time: 2023-05-01 15:10:42
#################################################


# In[13]:


import numpy as np
import read_timeseries_configs as rtc
import create_monthly_infrastructure as cmi
import evaluation_metrics_plots as emp
import create_generation_timeseries as cgt
import make_godeeep_timeseries as mgt
import os

# In[42]:
import importlib
importlib.reload(mgt)
importlib.reload(cmi)
importlib.reload(emp)
importlib.reload(rtc)
importlib.reload(cgt)

# In[1]:
# datadir = r'/Users/camp426/Documents/PNNL/Data/GODEEEP'
datadir = os.getcwd()

# %%
# Define the BAs in the analysis

gd_w, gd_s, wind_configs, solar_configs, ba_res_list = rtc.read_godeeep_data(datadir)
gd_timeseries = {
    'wind': gd_w,
    'solar': gd_s
}
# %%
slist_all = ba_res_list['solar']
wlist_all = ba_res_list['wind']

wlist = ['ERCO', 'ISNE', 'SWPP', 'CISO', 'BPAT', 'MISO', 'NYIS']
slist = ['ERCO', 'ISNE', 'SWPP', 'CISO']

BA_list = list(set(slist_all+wlist_all))
# In[46]:
# Create a dictionary of the installed capacities of each power plant, each resource, for each balancing authority, 2007 - 2020
# setting read_860m to True re-creates the metadata file on installed capacities
# setting read_860m to False reads the file {current_directory}/EIA860/all_years_860m.csv
read_860m = True
# setting write_860m to True writes to file the eia860 metadata here: {current_directory}/EIA860/all_years_860m.csv
# setting write_860m to False means we only hold the information in local memory
write_860m = True
# setting use_monthly to True relies on the monthly preliminary 860m reports
# setting use_monthly to False relies on the annual final reports
# RECOMMEND TO KEEP THIS SET TO FALSE, AS PRELIMINARY 860M REPORTS HAVE MANY ERRORS
use_monthly = False
if ('wind_configs' not in locals()) or ('solar_configs' not in locals()):
  print('reading in config files')
  wind_configs, solar_configs, _ = rtc.read_configs(datadir)
  print('done reading in config files')
validationdir = datadir+r'/validation_datasets'
eia860m_long = cmi.make_plants_eia860m(read_860m, write_860m, use_monthly, wind_configs, solar_configs, validationdir)

# %%
# add interconnection information
wecc = ['BPAT', 'CISO', 'IPCO', 'NWMT', 'PACE', 'PACW', 'PNM', 'PSCO', 'PSEI', 'WACM', 'WAUW', 'AZPS', 'LDWP', 'NEVP',
        'SRP', 'TEPC', 'BANC', 'GWA', 'EPE', 'PGE', 'IID', 'AVA', 'WWA', 'WALC', 'GRID', 'AVRN']
eic = ['AECI', 'ISNE', 'MISO', 'NYIS', 'PJM', 'SWPP', 'TVA', 'NBSO', 'CPLE', 'FPL', 'DUK', 'JEA', 'GVL', 'FPC', 'SEPA',
       'TEC', 'CPLW', 'SOCO', 'SC', 'FMPP', 'SPA', 'LGEE', 'SCEG', 'TAL']
erco = ['ERCO']
eia860m_long['interconnect'] = np.nan
eia860m_long.loc[eia860m_long['balancing authority code'].isin(wecc), 'interconnect'] = 'WECC'
eia860m_long.loc[eia860m_long['balancing authority code'].isin(eic), 'interconnect'] = 'EIC'
eia860m_long.loc[eia860m_long['balancing authority code'].isin(erco), 'interconnect'] = 'ERCOT'


# %%
# read in comparison datasets
eia923, eia930_w, eia930_s = rtc.read_eia_923_930(validationdir)
eia930 = {
    'solar': eia930_s,
    'wind': eia930_w
}
eia923 = {
    'solar': eia923[eia923['reported fuel type code'] == 'SUN'].copy(deep=True),
    'wind': eia923[eia923['reported fuel type code'] == 'WND'].copy(deep=True)
}

# %%
# create godeeep time series in a format ready for validation
# either create the dictionaries of generation based on the gd_w,gd_s cf time series and eia860m_long,
# or read the generation time series from file
# THIS STEP TAKES THE LONGEST TIME
make_plant_dicts = True
make_BA_dicts = True
save_BA_dicts = False  # alternative is to save only the BA hourly generation
all_power_hourly = mgt.make_formatted_timeseries(
    datadir, eia923, eia930, ba_res_list, gd_timeseries, make_plant_dicts, make_BA_dicts, save_BA_dicts, eia860m_long,
    solar_configs, wind_configs)

# %%
# validation

resource_bas = {
    'solar': slist,
    'wind': wlist
}
all_resource_bas = {
    'solar': slist_all,
    'wind': wlist_all
}

yr = list(range(2007, 2021))
mo = list(range(13))
da = list(range(32))
dates = [yr, mo, da]

validation_options = {
    'infrastructure_buildout': [
        'Y',
        emp.plot_grouped_bars,
        resource_bas,
        ['eia923', 'scada'],
        False,
        '',
        'by BA',
        False
    ],
    'interannual_variability': [  # this is the directory under plots
        'Y',                    # annual or monthly summary
        emp.plot_correlation,   # stat_function
        all_resource_bas,           # resource (wind/solar) and bas for that resource
        ['eia923'],             # comparison dataset
        True,                    # compute_cf
        True,               # mask bad data
        'by NERC region',   # spatial aggregation
        False               # seasonal disaggregation -- only used in diurnal plots
    ],
    'seasonal_cf_nerc': [
        'Seasonal',
        emp.plot_grouped_box,
        all_resource_bas,
        ['eia923'],
        '',
        '',
        'by NERC region',
        False
    ],
    'seasonal_cf_ba': [
        'Seasonal',
        emp.plot_grouped_box,
        {'wind': ['BPAT'],
         'solar': ['CISO']},
        ['eia923'],
        '',
        '',
        'by BA',
        False
    ],
    'diurnal_cf_seasonal': [
        'Diurnal',
        emp.plot_grouped_box,
        resource_bas,  # only use resource_bas for this
        ['scada'],
        '',
        '',
        'by BA',
        True
    ],
    'diurnal_cf': [
        'Diurnal',
        emp.plot_grouped_box,
        resource_bas,  # only use resource_bas for this
        ['scada'],
        '',
        '',
        'by BA',
        False
    ],
    'earth_mover_dist_cf': [
        dates,
        emp.plot_emd,
        resource_bas,  # only use resource_bas for this
        ['scada'],
        True,  # compute_cf
        '',
        '',
        False
    ],
    'earth_mover_dist_kwh': [
        dates,
        emp.plot_emd,
        resource_bas,  # only use resource_bas for this
        ['scada'],
        False,  # compute_cf
        '',
        '',
        False
    ],
    'daily_variability': [  # this is the directory under plots
        'D',                    # annual or monthly summary
        emp.plot_correlation,   # stat_function
        resource_bas,           # resource (wind/solar) and bas for that resource
        ['scada'],             # comparison dataset
        True,                    # compute_cf
        False,               # mask bad data
        'by BA',   # spatial aggregation
        False               # seasonal disaggregation -- only used in diurnal plots
    ]
}


# %%
this_analysis = ['seasonal_cf_nerc', 'seasonal_cf_ba']
# this_analysis = list(validation_options.keys())
validation = {x: validation_options[x] for x in this_analysis if x in validation_options}

for valplot, vals in validation.items():
  outdir = datadir+'/validation_results/'+valplot
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  freq, plot_func, resource_bas, comp, compute_cf, mask, agg, season_disagg = vals
  for res, these_bas in resource_bas.items():
    out = plot_func(all_power_hourly, eia860m_long, freq, res, these_bas, outdir,
                    compute_cf, comp, mask, agg, season_disagg, savef=True, showf=True)

# %%
