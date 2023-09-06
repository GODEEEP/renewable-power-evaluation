#!/usr/bin/env python
# coding: utf-8
# @Author: Allison M. Campbell
# @Email: allison.m.campbell@pnnl.gov
# @Date: 2023-05-01 15:10:42
# @Last Modified by:   camp426
# @Last Modified time: 2023-05-01 15:10:42
#################################################



# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import read_timeseries_configs as rtc
import create_monthly_infrastructure as cmi
import evaluation_metrics_plots as emp
import create_generation_timeseries as cgt
from scipy import stats
import os

# In[42]:
import importlib
importlib.reload(cmi)
importlib.reload(emp)
importlib.reload(rtc)
importlib.reload(cgt)
#%%

# In[1]:
datadir = r'/Users/camp426/Documents/PNNL/Data/GODEEEP'
# datadir = os.getcwd()

# In[27]:
gd_w,gd_s,wind_configs,solar_configs,ba_res_list = rtc.read_godeeep_data(datadir)
gd_timeseries ={
    'wind':gd_w,
    'solar':gd_s
}
#%%
slist_all = ba_res_list['solar']
wlist_all = ba_res_list['wind']

wlist = ['ERCO','ISNE','SWPP','CISO','BPAT','MISO','NYIS']
slist = ['ERCO','ISNE','SWPP','CISO']

BA_list = list(set(slist_all+wlist_all))
# In[46]:
# create a dictionary of the installed capacities of each power plant, each resource, for each balancing authority, 2007 - 2020
# set this to True if you want to re-create the metadata on installed capacities
# from the source data on EIA860 website
# setting this to False reads the file {top_directory}/EIA860/all_years_860m.csv
read_860m = True
# set this to True if you want to write to file the eia860 metadata
# setting this to False means we only hold the information in local memory
write_860m = True
# use the monthly preliminary 860m reports or the annual final reports
use_monthly = False
if ('wind_configs' not in locals()) or ('solar_configs' not in locals()):
    print('reading in config files')
    wind_configs,solar_configs,_ = rtc.read_configs(datadir)
    print('done reading in config files')
eia860m_long = cmi.make_plants_eia860m(read_860m,write_860m,use_monthly,wind_configs,solar_configs)
eia860m = {}
for thisBA in BA_list:
    eia860m[thisBA] = eia860m_long[(eia860m_long['balancing authority code']==thisBA)]

#%%
# change balancing authority code 'CA' to 'CISO'
eia860m_long.loc[eia860m_long['balancing authority code']=='CA','balancing authority code'] = 'CISO'
# change balancing authority code 'GRIS' to 'GRID'
eia860m_long.loc[eia860m_long['balancing authority code']=='GRIS','balancing authority code'] = 'GRID'
# change balancing authority code 'PJ' to 'PJM'
eia860m_long.loc[eia860m_long['balancing authority code']=='PJ','balancing authority code'] = 'PJM'
# change balancing authority code 'SW' to 'SWPP'
eia860m_long.loc[eia860m_long['balancing authority code']=='SW','balancing authority code'] = 'SWPP'
# useful code: eia860m_long[eia860m_long['plant id'].isin(cp_ids)][['plant id','balancing authority code','resource']].drop_duplicates().pivot(index='plant id',columns='balancing authority code',values='resource')
# change balancing authority code 'CP' to 'CPLE' for plant id 61527 for years [2017,2018]
eia860m_long.loc[(eia860m_long['balancing authority code']=='CP')&
                 (eia860m_long['plant id']==61527)&
                 (eia860m_long['year'].isin([2017,2018])),'balancing authority code'] = 'CPLE'
# change balancing authority code 'CP' to 'CPLE' for plant ids [61512,61520,61525,61528,61536]
eia860m_long.loc[(eia860m_long['balancing authority code']=='CP')&
                 (eia860m_long['plant id'].isin([61512,61520,61525,61528,61536])),'balancing authority code'] = 'CPLE'
# change balancing authority code 'IS' to 'ISNE' for plant id [61764,61765] for years [2018]
eia860m_long.loc[(eia860m_long['balancing authority code']=='IS')&
                 (eia860m_long['plant id'].isin([61764,61765]))&
                 (eia860m_long['year']==2018),'balancing authority code'] = 'ISNE'
# change balancing authority code 'IS' to 'ISNE' for plant ids [61518,61537,61538,61539,61541,61622,61629,61769,61770,61771,61774]
eia860m_long.loc[(eia860m_long['balancing authority code']=='IS')&
                 (eia860m_long['plant id'].isin([61518,61537,61538,61539,61541,61622,61629,61769,61770,61771,61774])),
                 'balancing authority code'] = 'ISNE'
# change balancing authority code 'DU' to 'DUK'
eia860m_long.loc[eia860m_long['balancing authority code']=='DU','balancing authority code'] = 'DUK'
# change balancing authority code 'ER' to 'ERCO'
eia860m_long.loc[eia860m_long['balancing authority code']=='ER','balancing authority code'] = 'ERCO'
# change balancing authority code 'MI' to 'MISO'
eia860m_long.loc[eia860m_long['balancing authority code']=='MI','balancing authority code'] = 'MISO'
# change balancing authority code 'NY' to 'NYIS'
eia860m_long.loc[eia860m_long['balancing authority code']=='NY','balancing authority code'] = 'NYIS'
# change balancing authority code 'CI' to 'CISO'
eia860m_long.loc[eia860m_long['balancing authority code']=='CI','balancing authority code'] = 'CISO'
# change balancing authority code 'PA' to 'PACW'
eia860m_long.loc[eia860m_long['balancing authority code']=='PA','balancing authority code'] = 'PACW'
# change balancing authority code 'PS' to 'PSCO'
eia860m_long.loc[eia860m_long['balancing authority code']=='PS','balancing authority code'] = 'PSCO'
#process:
# ci_ids = eia860m_long[eia860m_long['balancing authority code']=='CI']['plant id'].drop_duplicates().tolist()
# check to see if it's a one to one match for updating the ba
# eia860m_long[eia860m_long['plant id'].isin(ci_ids)][['plant id','balancing authority code','resource']].drop_duplicates().pivot(index='plant id',columns='balancing authority code',values='resource')
# if it is, then we can do the easy approach:
# change balancing authority code 'NY' to 'NYIS'
# eia860m_long.loc[eia860m_long['balancing authority code']=='NY','balancing authority code'] = 'NYIS'
def get_plantids(df,res,ba):
    return df[(df['balancing authority code']==ba)&(df['resource']==res)]['plant id'].drop_duplicates().tolist()

#%%
# add interconnection information
wecc = [ 'BPAT', 'CISO', 'IPCO', 'NWMT','PACE', 'PACW', 'PNM', 'PSCO', 'PSEI', 'WACM', 'WAUW', 'AZPS', 'LDWP', 'NEVP',
         'SRP', 'TEPC', 'BANC', 'GWA', 'EPE', 'PGE', 'IID', 'AVA', 'WWA', 'WALC', 'GRID', 'AVRN']
eic = ['AECI', 'ISNE', 'MISO', 'NYIS', 'PJM', 'SWPP', 'TVA', 'NBSO', 'CPLE', 'FPL', 'DUK', 'JEA', 'GVL', 'FPC', 'SEPA',
        'TEC', 'CPLW', 'SOCO', 'SC', 'FMPP', 'SPA', 'LGEE', 'SCEG', 'TAL']
erco = ['ERCO']
eia860m_long['interconnect'] = np.nan
eia860m_long.loc[eia860m_long['balancing authority code'].isin(wecc),'interconnect'] = 'WECC'
eia860m_long.loc[eia860m_long['balancing authority code'].isin(eic),'interconnect'] = 'EIC'
eia860m_long.loc[eia860m_long['balancing authority code'].isin(erco),'interconnect'] = 'ERCOT'


#%%

eia923,eia930_w,eia930_s = rtc.read_eia_923_930(datadir)
eia930 = {
    'solar':eia930_s,
    'wind':eia930_w
}
eia923 = {
    'solar':eia923[eia923['reported fuel type code']=='SUN'].copy(deep=True),
    'wind':eia923[eia923['reported fuel type code']=='WND'].copy(deep=True)
}

#%%
# MAKE PLANT-LEVEL GENERATION TIME SERIES
firstdate = '2007-01-01 01:00:00'
monthyears = cgt.get_monthyear(firstdate)
# either create the dictionaries of generation based on the gd_w,gd_s cf time series and eia860m_long,
# or read the generation time series from file
make_plant_dicts = True
# create dictionaries for each power plant
all_plant_gen = {}
for res in ['solar','wind']:
    if make_plant_dicts:
        all_plant_gen[res] = cgt.create_plantlevel_generation(gd_timeseries[res],eia860m_long,res,monthyears,solar_configs,wind_configs,datadir)
    else:
        all_plant_gen[res] = pd.read_csv(datadir+r'/historical_power/'+res+'_plant_generation.csv')

#%%
# use all_plant_gen dict to create the BA level generation using the plant_gen_id
make_BA_dicts = True
save_dicts = False # alternative is to save only the BA hourly generation
all_power_hourly = {}
if make_BA_dicts:
    for res,balist,sublist in zip(['solar','wind'],[slist_all,wlist_all],[slist,wlist]):#'wind',wlist,
        all_power_hourly[res] = cgt.create_balevel_generation(all_plant_gen[res],res,balist,gd_timeseries[res],eia930[res],eia923[res],eia860m,sublist,monthyears,datadir,save_dicts)
    print('completed creation of resource profile dict')
    if not save_dicts:
        for res in ['solar','wind']:
            bas = all_power_hourly[res].keys()
            df_tosave = pd.DataFrame(index=all_power_hourly['solar']['TEC'].index)
            for b in bas:
                godeeep_power = pd.DataFrame(all_power_hourly[res][b]['GODEEEP-'+res]).rename(columns={'GODEEEP-'+res:b})
                df_tosave = df_tosave.merge(godeeep_power,how='left',left_index=True,right_index=True)
            df_tosave.to_csv(datadir+r'/historical_power/'+res+'_BA_generation.csv')
else:
    all_power_hourly = {}
    for res,balist in zip(['solar','wind'],[slist_all,wlist_all]):
        BA_hr_diff = {}
        for thisBA in balist:
            BA_hr_diff[thisBA] = pd.read_csv(datadir+r'/historical_power/'+thisBA+'_'+res+'_generation.csv')
        all_power_hourly[res] = BA_hr_diff

# check for negative production
for r in ['wind','solar']:
    for ba in all_power_hourly[r].keys():
        if (all_power_hourly[r][ba]['GODEEEP-'+r]<0).any():
            print(r,ba)
#%%
# fix this above!
for r in wlist:
    all_power_hourly['wind'][r].columns = ['eia930', 'eia923', 'GODEEEP-wind', 'scada']
for r in slist:
    all_power_hourly['solar'][r].columns = ['eia930', 'eia923', 'GODEEEP-solar', 'scada']

#%%
# PLOT INFRASTRUCTURE ROLL OUT OVER TIME 
# WITH MONTHLY MAX PRODUCTION
emp.plot_infrastructure_over_time(wlist,slist,eia860m_long,all_power_hourly,datadir,plot_capnormdiff=False,savef=True)
# #%%
# # MAKE ANNUAL, SEASONAL PLOTS

# for res,balist in zip(['solar','wind'],[slist,wlist]):
#     comparison_datasets = ['eia923','scada']#,'eia930']
#             # [res+'-scada','eia923','eia930']
#             # select the comparison datasets for validation
#             # NOTE: the metrics are calculated based on the historical baseline 
#             # that overlaps between the two compared datasets 
#             # -- dates with NA are dropped for all datasets.
#             # EIA930 only goes back to July 2018, so the time baseline for the statistics
#             # calcs is over 7/2018 - 12/2020.
#     for freq in ['Y']:#['M','Y']:
#         # if freq == 'Y':
#         for bb in balist:
#             compute_cf = False
#             emp.plot_grouped_bars(all_power_hourly,eia860m_long,freq,res,bb,datadir,compute_cf,comparison_datasets,'','',savef=True)

#%%
resource_bas = {
    'solar': slist,
    'wind': wlist
}
all_resource_bas = {
    'solar': slist_all,
    'wind': wlist_all
}

yr = list(range(2007,2021))
mo = list(range(13))
da = list(range(32))
dates=[yr,mo,da]

validation_plots = {
    # 'infrastructure_buildout':[
    #     'Y',
    #     emp.plot_grouped_bars,
    #     resource_bas,
    #     ['eia923','scada'],
    #     False,
    #     '',
    #     'by BA',
    #     False 
    # ],
    # 'interannual_variability':[ # this is the directory under plots
    #     'Y',                    # annual or monthly summary
    #     emp.plot_correlation,   # stat_function
    #     all_resource_bas,           # resource (wind/solar) and bas for that resource
    #     ['eia923'],             # comparison dataset
    #     True,                    # compute_cf
    #     True,               # mask bad data
    #     'by NERC region',   # spatial aggregation
    #     False               # seasonal disaggregation -- only used in diurnal plots
    # ],
    # 'seasonal_cf_nerc':[
    #     'Seasonal',
    #     emp.plot_grouped_box,
    #     all_resource_bas,
    #     ['eia923'],
    #     '',
    #     '',
    #     'by NERC region',
    #     False
    # ],
    # 'seasonal_cf_ba':[
    #     'Seasonal',
    #     emp.plot_grouped_box,
    #     {'wind':['BPAT'],
    #      'solar':['CISO']},
    #     ['eia923'],
    #     '',
    #     '',
    #     'by BA',
    #     False
    # ],
    # 'diurnal_cf_seasonal':[
    #     'Diurnal',
    #     emp.plot_grouped_box,
    #     {'wind':['BPAT'],
    #      'solar':['CISO']},  # only use resource_bas for this
    #     ['scada'],
    #     '',
    #     '',
    #     'by BA',
    #     True
    # ],
    # 'diurnal_cf':[
    #     'Diurnal',
    #     emp.plot_grouped_box,
    #     {'wind':['BPAT'],
    #      'solar':['CISO']},  # only use resource_bas for this
    #     ['scada'],
    #     '', 
    #     '',
    #     'by BA',
    #     False
    # ],
    'earth_mover_dist_cf':[
        dates,
        emp.plot_emd,
        resource_bas,  # only use resource_bas for this
        ['scada'],
        True, # compute_cf
        '',
        '',
        False
    ]
    # 'earth_mover_dist_kwh':[
    #     dates,
    #     emp.plot_emd,
    #     resource_bas,  # only use resource_bas for this
    #     ['scada'],
    #     False, # compute_cf
    #     '',
    #     '',
    #     False
    # ]
    # 'daily_variability':[ # this is the directory under plots
    #     'D',                    # annual or monthly summary
    #     emp.plot_correlation,   # stat_function
    #     resource_bas,           # resource (wind/solar) and bas for that resource
    #     ['scada'],             # comparison dataset
    #     True,                    # compute_cf
    #     False,               # mask bad data
    #     'by BA',   # spatial aggregation
    #     False               # seasonal disaggregation -- only used in diurnal plots
    # ]
}


for valplot,vals in validation_plots.items():
    # print(valplot)
    outdir = datadir+r'/plots/'+valplot+'/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # print(vals)
    freq,plot_func,resource_bas,comp,compute_cf,mask,agg,season_disagg = vals
    # print(metrics,freq,stats,resource_bas)
    # stats_df_table = pd.DataFrame()
    for res, these_bas in resource_bas.items():
        # comp = ['eia923',f'{res}-scada']
        # print(valplot,freq,plot_func,resource_bas,comp,compute_cf)
        out = plot_func(all_power_hourly,eia860m_long,freq,res,these_bas,outdir,compute_cf,comp,mask,agg,season_disagg,savef=True,showf=True)
                        

#%%

#%%

#%%

# daily var to understand how to fluctuates throughout the day<br>
# better -- how does the bias vary by season?<br>
# nathalie's plot shows the median , and the inter-annual variability for that season

# In[ ]:


# 1. is there general bias? (esp in monthly)<br>
# 2. do we have the right level of intermittency at the daily and weekly level?<br>
# -- variation within the day, within the week<br>
# is it uniform throughout the hour? other?<br>
# show the distribution of intermittency<br>
# hourly variation for the season<br>
# 3. pct error on monthly, annual timescale -- relative bias on monthly timescale<br>
# --


