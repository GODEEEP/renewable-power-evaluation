#!/usr/bin/env python
# coding: utf-8
# @Author: Allison M. Campbell
# @Email: allison.m.campbell@pnnl.gov
# @Date: 2023-05-01 15:26:21
# @Last Modified by:   camp426
# @Last Modified time: 2023-05-01 15:26:21
#################################################

# %%
import pandas as pd
import numpy as np
# import os
import calendar
# %%


def get_BPA_operational_generation(resource,solar_configs,wind_configs,datadir):
    # compare BPA godeeep wind with BPA on their website
    # use the BPA published list of generators in/out of BA:
    # https://transmission.bpa.gov/business/operations/wind/Solar_and_Wind_Nameplate_LIST.pdf
    raw = get_eia860y(2020,['OA','OP','SB','OS','RE'],solar_configs,wind_configs,datadir)
    ba_list = pd.read_excel(datadir+r'/BPA/Solar_and_Wind_Nameplate_LIST.xlsx',skiprows=10,sheet_name=resource)
    ba_list['Operating Date'] = pd.to_datetime(ba_list['Operating Date'])
    ba_list = ba_list[['Max Production EIA','Plant ID','Gen ID','Plant Name EIA','Operating Date']]
    ba_list = ba_list.rename(columns={'Plant ID':'plant id','Plant Name EIA':'plant name','Gen ID':'generator id','Max Production EIA':'nameplate capacity (mw)'})
    first_year = max(2007,ba_list.sort_values(by='Operating Date')['Operating Date'].dt.year[0])
    first_month = min(1,ba_list.sort_values(by='Operating Date')['Operating Date'].dt.month[0])
    moremonthyears = []
    for y in range(first_year,2021,1):
        for m in range(1,13,1):
            moremonthyears.append((y,m))
    bacols = ['plant id','plant name','generator id','nameplate capacity (mw)','month','year']
    ba_plants = ba_list[ba_list['Operating Date']<str(first_year)].reset_index(drop=True)
    ba_plants['month'] = first_month
    ba_plants['year'] = first_year
    ba_plants = ba_plants[bacols]
    lastm = moremonthyears[0][1]
    lasty = moremonthyears[0][0]
    for y,m in moremonthyears[1:]:
        thismonthplants = ba_plants.loc[(ba_plants['month'] == lastm)&(ba_plants['year'] == lasty)].copy()
        thismonthplants.loc[:,'month'] = m
        thismonthplants.loc[:,'year'] = y
        newplants = ba_list[(ba_list['Operating Date'].dt.month == m)&(ba_list['Operating Date'].dt.year == y)].copy()
        if len(newplants)>0:
            newplants.loc[:,'month'] = m
            newplants.loc[:,'year'] = y
            newplants = newplants[bacols]
            if any(newplants['nameplate capacity (mw)']<0):
                # remove plants that are moving to another BA from thismonthplants
                dropplants = newplants[newplants['nameplate capacity (mw)']<0]
                dropplantids = thismonthplants['plant id'].isin(dropplants['plant id'].tolist())
                dropplantgenids = thismonthplants['generator id'].isin(dropplants['generator id'].tolist())
                thismonthplants = thismonthplants[~(dropplantids&dropplantgenids)]
                dropplantids = newplants['plant id'].isin(dropplants['plant id'].tolist())
                dropplantgenids = newplants['generator id'].isin(dropplants['generator id'].tolist())
                newplants = newplants[~(dropplantids&dropplantgenids)]
            # then add the actual new generators
            phases = pd.concat([thismonthplants,newplants])
            if any(phases[['plant id','generator id']].duplicated()):
                aggregatedplants = pd.DataFrame(phases[phases[['plant id','generator id']].duplicated(keep=False)].groupby(['plant id','plant name','generator id'])['nameplate capacity (mw)'].sum()).reset_index()
                aggregatedplants.loc[:,'month'] = m
                aggregatedplants.loc[:,'year'] = y
                dropplants = phases[phases[['plant id','generator id']].duplicated(keep=False)]
                dropplantids = phases['plant id'].isin(dropplants['plant id'].tolist())
                dropplantgenids = phases['generator id'].isin(dropplants['generator id'].tolist())
                thismonthplants = pd.concat([phases[~(dropplantids&dropplantgenids)],aggregatedplants]).reset_index(drop=True)
                dropplantids = newplants['plant id'].isin(dropplants['plant id'].tolist())
                dropplantgenids = newplants['generator id'].isin(dropplants['generator id'].tolist())
                newplants = None
            thismonthplants = pd.concat([thismonthplants,newplants])
        ba_plants = pd.concat([ba_plants,thismonthplants])
        lastm = m
        lasty = y
    ba_plants['generator id'] = ba_plants['generator id'].astype(str)
    ba_plants = ba_plants.merge(raw[['plant id','generator id','plant_code_unique','scale_nameplate']],how='left',left_on=['plant id','generator id'],right_on=['plant id','generator id'])
    ba_plants['plant_gen_id'] = ba_plants['plant id'].astype(str)+'_'+ba_plants['generator id']
    return ba_plants
    # ba_plants.to_csv(datadir+r'/BPA/monthlyBPAcapacity.csv',index=False)

def make_eia860m_from_annual(monthyears_before,raw):
    raw = raw.rename(columns={'Balancing Authority Code':'balancing authority code'})
    bacols = ['plant id', 'generator id', 'plant name', 'nameplate capacity (mw)',
       'balancing authority code', 'resource', 'month', 'year',
       'plant_code_unique', 'scale_nameplate']
    first_year = monthyears_before[0][0]
    first_month = monthyears_before[0][1]
    # everything before first year
    ba_plants = raw[(raw['year']<=first_year)].reset_index(drop=True)
    ba_plants['month'] = first_month
    ba_plants['year'] = first_year
    ba_plants = ba_plants[bacols]
    lastm = monthyears_before[0][1]
    lasty = monthyears_before[0][0]
    for my in monthyears_before[1:]:
        m = my[1]
        y = my[0]
        thismonthplants = ba_plants.loc[(ba_plants['month'] == lastm)&(ba_plants['year'] == lasty)].copy()
        thismonthplants.loc[:,'month'] = m
        thismonthplants.loc[:,'year'] = y
        newplants = raw[(raw['month'] == m)&(raw['year'] == y)].copy()
        if len(newplants)>0:
            newplants.loc[:,'month'] = m
            newplants.loc[:,'year'] = y
            newplants = newplants[bacols]
            thismonthplants = pd.concat([thismonthplants,newplants])
        ba_plants = pd.concat([ba_plants,thismonthplants])
        lastm = m
        lasty = y
    return ba_plants

# reads in the 860m files through July 2016
def make_eia860m_from_monthly(solar_configs,wind_configs,datadir):
    eia860m = pd.DataFrame()
    monthyears = []
    for y in range(2020,2015,-1):
        if y == 2016:
            m_end = 6
        else:
            m_end = 0
        for m in range(12,m_end,-1):
            monthyears.append((y,m))
    for y,m in monthyears:
        print(y,m)
        month = calendar.month_name[m].lower()
        if (y >= 2021) or (y >= 2020 and m>=11):
            nrows = 2
        else:
            nrows = 1
        eia_df = pd.read_excel(datadir+r'/EIA860/eia860'+str(y)+'/'+month+'_generator'+str(y)+'.xlsx',skiprows=nrows,sheet_name='Operating',verbose=True)
        eia_df.columns = [i.replace('\n','').lower() for i in eia_df.columns]
        if eia_df.columns.isin(['energy source code','balancing authority code','status']).sum() == 3:
            # if we have all these columns, then we can use the eia860m
            df = eia_df[(eia_df['energy source code'].isin(['WND','SUN']))&
                (eia_df['status']=='(OP) Operating')&(~eia_df['plant state'].isin(['AK','HI']))][['plant id', 'generator id', 'plant name', 'nameplate capacity (mw)','balancing authority code', 'energy source code']]
            if df[df.isna().any(axis=1)].shape[0] > 0:
                fix_df = eia_df[(eia_df['energy source code'].isin(['WND','SUN']))&
                    (eia_df['status']=='(OP) Operating')&(~eia_df['plant state'].isin(['AK','HI']))][['plant id', 'generator id', 'plant name', 'nameplate capacity (mw)','balancing authority code', 'energy source code','status','plant state']]
                fix_df = fix_df[fix_df.isna().any(axis=1)]
                # fixing the balancing authority code with information from this year's plant sheet
                # print(fix_df.columns[fix_df.isna().any()].tolist())
                fix_df['balancing authority code'] = fix_eia860m_ba_from_annual_plant(fix_df,y)
                fix_df.loc[:,'plant id'] = fix_df.loc[:,'plant id'].astype(int)
                fix_df = fix_df.rename(columns={'energy source code':'resource'})
                fix_df['resource'] = np.where(fix_df['resource']=='WND','wind','solar')
                keep_df = df[~df.isna().any(axis=1)].reset_index(drop=True)
                keep_df.loc[:,'plant id'] = keep_df.loc[:,'plant id'].astype(int)
                keep_df = keep_df.rename(columns={'energy source code':'resource'})
                keep_df['resource'] = np.where(keep_df['resource']=='WND','wind','solar')
                df = pd.concat([keep_df,fix_df[keep_df.columns]])
            else:
                df['plant id'] = df['plant id'].astype(int)
                df = df.rename(columns={'energy source code':'resource'})
                df['resource'] = np.where(df['resource']=='WND','wind','solar')
        df['month'] = m
        df['year'] = y
        eia860m = pd.concat([eia860m,df])
    # fix the ba with information from the rest of the history
    fix_plants = eia860m[eia860m.isna().any(axis=1)]['plant id'].drop_duplicates().tolist()
    for fp in fix_plants:
        correct_ba = eia860m[(eia860m['plant id']==fp)&(~eia860m['balancing authority code'].isna())]['balancing authority code'].drop_duplicates().tolist()
        if len(correct_ba)>0:
            eia860m.loc[(eia860m['plant id']==fp)&(eia860m['balancing authority code'].isna()),['balancing authority code']] = correct_ba[0]
        else:
            eia860m.loc[(eia860m['plant id']==fp)&(eia860m['balancing authority code'].isna()),['balancing authority code']] = np.nan
    eia860m = eia860m[~eia860m.isna().any(axis=1)]
    eia860m = eia860m.groupby('year').apply(get_pcu_pgi_scalenamep_from_configs,solar_configs,wind_configs,datadir)
    eia860m = eia860m.reset_index(drop=True)
    # fix the plant_code_unique and scale_nameplate for plants/gens that do not have pcu, pgi, scalen from configs
    fix_pcu_sn = eia860m[eia860m.isna().any(axis=1)][['plant id','generator id']].drop_duplicates().reset_index(drop=True)
    for fp,fg in zip(fix_pcu_sn['plant id'],fix_pcu_sn['generator id']):
        correct_pcu = eia860m[(eia860m['plant id']==fp)&(eia860m['generator id']==fg)&(~eia860m['plant_code_unique'].isna())]['plant_code_unique'].drop_duplicates().tolist()
        if len(correct_pcu)>0:
            eia860m.loc[(eia860m['plant id']==fp)&(eia860m['generator id']==fg)&(eia860m['plant_code_unique'].isna()),['plant_code_unique']] = correct_pcu[0]
            correct_sn = eia860m[(eia860m['plant id']==fp)&(eia860m['generator id']==fg)&(~eia860m['scale_nameplate'].isna())]['scale_nameplate'].drop_duplicates().tolist()
            eia860m.loc[(eia860m['plant id']==fp)&(eia860m['generator id']==fg)&(eia860m['scale_nameplate'].isna()),['scale_nameplate']] = correct_sn
        else:
            drop_idx = eia860m.loc[(eia860m['plant id']==fp)&(eia860m['generator id']==fg)].index
            eia860m = eia860m.drop(index = drop_idx)
    eia860m = eia860m.reset_index(drop=True)
    eia860m.loc[eia860m['plant_gen_id'].isna(),'plant_gen_id'] = eia860m.loc[eia860m['plant_gen_id'].isna(),'plant id'].astype(int).astype(str)+'_'+eia860m.loc[eia860m['plant_gen_id'].isna(),'generator id']
    eia860m = eia860m.drop_duplicates()
    eia860m.to_csv(datadir+r'/EIA860/all_860m.csv',index=False)
    return eia860m

def fix_eia860m_ba_from_annual_plant(eia_df,y,datadir):
    # need the annual 860 plant sheet to get the BA
    eia860y_p = pd.read_excel(datadir+r'/EIA860/eia860'+str(y)+'/2___Plant_Y'+str(y)+'.xlsx',skiprows=1,sheet_name='Plant')
    eia860y_p = eia860y_p.rename(columns={'Plant Code':'plant id','Balancing Authority Code':'balancing authority code'})
    eia860y_p = eia860y_p[~eia860y_p['State'].isin(['AK','HI'])]
    outdf = eia860y_p[['plant id','balancing authority code']].merge(eia_df[['plant id','generator id']],how='right',left_on='plant id',right_on='plant id')
    return outdf['balancing authority code'].values


def get_eia860y(yr,status,solar_configs,wind_configs,datadir):
    eia860_wind = pd.read_excel(datadir+r'/EIA860/eia860'+str(yr)+'/3_2_Wind_Y'+str(yr)+'.xlsx',skiprows=1,sheet_name='Operable')
    eia860_solar = pd.read_excel(datadir+r'/EIA860/eia860'+str(yr)+'/3_3_Solar_Y'+str(yr)+'.xlsx',skiprows=1,sheet_name='Operable')
    eia860_solar = eia860_solar[['Plant Code','Plant Name','Generator ID','Status','State','Operating Month',
           'Operating Year','Nameplate Capacity (MW)']]
    eia860_solar['resource'] = 'solar'
    eia860_wind = eia860_wind[['Plant Code','Plant Name','Generator ID','Status','State','Operating Month',
           'Operating Year','Nameplate Capacity (MW)']]
    eia860_wind['resource'] = 'wind'
    eia860_resource = pd.concat([eia860_wind,eia860_solar])
    eia860_plant = pd.read_excel(datadir+r'/EIA860/eia860'+str(yr)+'/2___Plant_Y'+str(yr)+'.xlsx',skiprows=1,sheet_name='Plant')
    eia860_y = eia860_resource[(eia860_resource['Status'].isin(status))&(~eia860_resource['State'].isin(['AK','HI']))].merge(
        eia860_plant[['Plant Code','Balancing Authority Code']],
        how='left',left_on='Plant Code',right_on='Plant Code')
    # scale namplate will be used to divide the nameplate capacity by half
    # for generators whose num turbines exceeds 300
    # default is 1 -- this value will only be changed to 0.5 for
    # the handful of generators in split_id and split_gen
    eia860_y['scale_nameplate'] = 1.
    eia860_y = eia860_y.merge(eia860_y.groupby(['Plant Code','resource'])['Generator ID'].count()>1,how='left',left_on=['Plant Code','resource'],right_on=['Plant Code','resource'])
    eia860_y = eia860_y.rename(columns={'Generator ID_y':'multiple_generators'})
    split_id =  ['10597','50532','50535','50821']
    split_gen = ['WGN1','WGNS','WGNS','GEN1']
    eia860_y.loc[(eia860_y['Plant Code'].isin([int(i) for i in split_id]))&(eia860_y['Generator ID_x'].isin(split_gen)),['multiple_generators']] = True
    eia860_y.loc[(eia860_y['Plant Code'].isin([int(i) for i in split_id]))&(eia860_y['Generator ID_x'].isin(split_gen)),['scale_nameplate']] = 0.5
    for i,g in zip(split_id,split_gen):
        eia860_y = pd.concat([eia860_y,(eia860_y[(eia860_y['Plant Code']==int(i))&(eia860_y['Generator ID_x']==g)])])
    eia860_y = eia860_y.reset_index(drop=True)
    eia860_y = eia860_y.rename(columns={'Plant Code':'plant id','Plant Name':'plant name','Generator ID_x':'generator id','Nameplate Capacity (MW)':'nameplate capacity (mw)','Operating Month':'month','Operating Year':'year'})
    eia860_y_w = eia860_y[eia860_y['resource']=='wind'].copy()
    eia860_y_w = eia860_y_w.merge(wind_configs[['plant_code','generator_id','plant_code_unique','plant_gen_id']],how='left',left_on=['plant id','generator id'],right_on=['plant_code','generator_id']).drop(columns=['plant_code','generator_id'])
    eia860_y_s = eia860_y[eia860_y['resource']=='solar'].copy()
    eia860_y_s = eia860_y_s.merge(solar_configs[['plant_code','generator_id','plant_code_unique','plant_gen_id']],how='left',left_on=['plant id','generator id'],right_on=['plant_code','generator_id']).drop(columns=['plant_code','generator_id'])
    eia860_y = pd.concat([eia860_y_s,eia860_y_w])
    # give each generator a generator key using logic of the reV generation code
    # eia860_y['plant_code_unique'] = np.nan
    # eia860_y.loc[eia860_y['multiple_generators']==True,'plant_code_unique']= \
    #     eia860_y.loc[eia860_y['multiple_generators']==True,'Plant Code'].astype(str)+\
    #     '_'+eia860_y.loc[eia860_y['multiple_generators']==True].groupby('Plant Code').cumcount().astype(str)
    # eia860_y.loc[eia860_y['plant_code_unique'].str.endswith('_0')==True,'plant_code_unique'] = eia860_y['Plant Code']
    # eia860_y.loc[eia860_y['multiple_generators']==False,'plant_code_unique'] = eia860_y['Plant Code']
    # eia860_y['plant_code_unique'] = eia860_y['plant_code_unique'].astype(str)
    return eia860_y

def get_pcu_pgi_scalenamep_from_configs(df,solar_configs,wind_configs,datadir):
    thisyear = df.year.drop_duplicates().values[0]
    status = ['OP','SB','OA','OS','RE']
    print(thisyear)
    merged_df = df.merge(get_eia860y(thisyear,status,solar_configs,wind_configs,datadir)[['plant id','generator id','resource','plant_code_unique','plant_gen_id','scale_nameplate']],
                         how='left',
                         left_on=['plant id','generator id','resource'],
                         right_on=['plant id','generator id','resource'])
    return merged_df

def make_plants_eia860m(read_eia860,write_eia860,use_monthly,wind_configs,solar_configs,datadir):
    # pull in the monthly eia860 df
    if read_eia860 == True:
        if use_monthly == True:
            eia860monthly = make_eia860m_from_monthly(solar_configs,wind_configs,datadir)
            eia860monthly['plant id'] = eia860monthly['plant id'].astype(int)
            if len(eia860monthly[eia860monthly['plant_gen_id'].isna()])>0:
                na_pc = eia860monthly[eia860monthly['plant_gen_id'].isna()][['plant id','resource']].drop_duplicates()
                for pc,res in zip(na_pc['plant id'],na_pc['resource']):
                    if len(eia860monthly[(eia860monthly['plant id']==pc)&(eia860monthly['plant_code_unique'].isna())])>0:
                        eia860monthly.loc[(eia860monthly['plant id']==pc)&(eia860monthly['resource']==res),['plant_code_unique']] = str(pc)
                    if res == 'solar':
                        eia860monthly.loc[(eia860monthly['plant_code_unique']==str(pc))&(eia860monthly['resource']==res),['plant_gen_id']] = \
                            solar_configs.loc[solar_configs['plant_code_unique']==str(pc),['plant_gen_id']].values

                    else:
                        eia860monthly.loc[(eia860monthly['plant_code_unique']==str(pc))&(eia860monthly['resource']==res),['plant_gen_id']] = \
                            wind_configs.loc[wind_configs['plant_code_unique']==str(pc),['plant_gen_id']].values
            monthyears = []
            for y in range(2016,2006,-1):
                if y == 2016:
                    m_beg = 6
                else:
                    m_beg = 12
                for m in range(m_beg,0,-1):
                    monthyears.append((y,m))
            # use the eia860_2016 annual file to gather monthly information
            # for the remainder of months and years -- back through jan 2007
            # use the eia860_annual to import information about plant_code_unique and scale_nameplate
            eia860m_before = make_eia860m_from_annual(sorted(monthyears),get_eia860y(2016,['OP','OA'],solar_configs,wind_configs,datadir))
            eia860m_before_s = eia860m_before[eia860m_before['resource']=='solar'].merge(solar_configs[['plant_code_unique','plant_gen_id']],how='left',left_on='plant_code_unique',right_on='plant_code_unique')
            if len(eia860m_before_s[eia860m_before_s['plant_code_unique'].isna()])>0:
                df_nona = eia860m_before_s[~eia860m_before_s['plant_code_unique'].isna()].reset_index(drop=True)
                df_na = eia860m_before_s[eia860m_before_s['plant_code_unique'].isna()].reset_index(drop=True)
                df_na['plant_code_unique'] = df_na['plant id'].astype(str)
                df_na['plant_gen_id'] = df_na['plant id'].astype(str)+'_'+df_na['generator id']
                eia860m_before_s = pd.concat([df_nona,df_na]).sort_values(by=['year','month','plant_code_unique']).reset_index(drop=True)
            eia860m_before_w = eia860m_before[eia860m_before['resource']=='wind'].merge(wind_configs[['plant_code_unique','plant_gen_id']],how='left',left_on='plant_code_unique',right_on='plant_code_unique')
            if len(eia860m_before_w[eia860m_before_w['plant_code_unique'].isna()])>0:
                df_nona = eia860m_before_w[~eia860m_before_w['plant_code_unique'].isna()].reset_index(drop=True)
                df_na = eia860m_before_w[eia860m_before_w['plant_code_unique'].isna()].reset_index(drop=True)
                df_na['plant_code_unique'] = df_na['plant id'].astype(str)
                df_na['plant_gen_id'] = df_na['plant id'].astype(str)+'_'+df_na['generator id']
                eia860m_before_w = pd.concat([df_nona,df_na]).sort_values(by=['year','month','plant_code_unique']).reset_index(drop=True)
            eia860m_before = pd.concat([eia860m_before_s,eia860m_before_w]).reset_index(drop=True)
            eia860m = pd.concat([eia860monthly,eia860m_before]).sort_values(by=['year','month','plant_code_unique']).reset_index(drop=True)
            eia860m['plant id'] = eia860m['plant id'].astype(int)
            pgi_list = eia860m[eia860m.isna().any(axis=1)]['plant_gen_id'].tolist()
            fix_ba_df = eia860m[(eia860m['plant_gen_id'].isin(pgi_list))&(~eia860m['balancing authority code'].isna())][['plant_gen_id','balancing authority code']].drop_duplicates()
            fix_ba_eia860m = eia860m[(eia860m['plant_gen_id'].isin(pgi_list))&(eia860m['balancing authority code'].isna())]
            fix_ba_eia860m = fix_ba_eia860m.drop(columns=['balancing authority code'])
            keep_eia860m = eia860m[~((eia860m['plant_gen_id'].isin(pgi_list))&(eia860m['balancing authority code'].isna()))]
            fix_ba_eia860m = fix_ba_eia860m.merge(fix_ba_df,how='left',left_on='plant_gen_id',right_on='plant_gen_id')
            eia860m = pd.concat([keep_eia860m,fix_ba_eia860m]).sort_values(by=['year','month'])
            # if plant_gen_id is nan, 
            # we need to find other months that contain the plant_gen_id information
            # we find this in the solar/wind_config files.
            # if the plant code/id is not in the solar/wind_config files,
            # that means we have not modeled the plant/generator at all 
            # -- we will remove it from the eia860m file -- so far haven't found any
            if len(eia860m[eia860m['plant_gen_id'].isna()])>0:
                na_pc = eia860m[eia860m['plant_gen_id'].isna()][['plant id','resource']].drop_duplicates()
                for pc,res in zip(na_pc['plant id'],na_pc['resource']):
                    if len(eia860m[(eia860m['plant id']==pc)&(eia860m['plant_code_unique'].isna())])>0:
                        eia860m.loc[(eia860m['plant id']==pc)&(eia860m['resource']==res),['plant_code_unique']] = str(pc)
                    if (res == 'solar'): 
                        if len(solar_configs[solar_configs['plant_code_unique']==str(pc)])>0:
                            eia860m.loc[(eia860m['plant_code_unique']==str(pc))&(eia860m['resource']==res),['plant_gen_id']] = \
                                solar_configs.loc[solar_configs['plant_code_unique']==str(pc),['plant_gen_id']].values
                        else:
                            print('solar plant not in config file: ',pc)
                    elif (res == 'wind'):
                        if len(wind_configs[wind_configs['plant_code_unique']==str(pc)])>0:
                            eia860m.loc[(eia860m['plant_code_unique']==str(pc))&(eia860m['resource']==res),['plant_gen_id']] = \
                                wind_configs.loc[wind_configs['plant_code_unique']==str(pc),['plant_gen_id']].values
                        else:
                            print('wind plant not in config file: ',pc)
        else:
            monthyears = []
            for y in range(2020,2006,-1):
                m_beg = 12
                for m in range(m_beg,0,-1):
                    monthyears.append((y,m))
            eia860m = make_eia860m_from_annual(sorted(monthyears),get_eia860y(2020,['OP','OA'],solar_configs,wind_configs,datadir))
            eia860m_w = eia860m[eia860m['resource']=='wind'].merge(wind_configs[['plant_code_unique','plant_gen_id']],how='left',left_on='plant_code_unique',right_on='plant_code_unique')
            if len(eia860m_w[eia860m_w['plant_code_unique'].isna()])>0:
                df_nona = eia860m_w[~eia860m_w['plant_code_unique'].isna()].reset_index(drop=True)
                df_na = eia860m_w[eia860m_w['plant_code_unique'].isna()].reset_index(drop=True)
                df_na['plant_code_unique'] = df_na['plant id'].astype(str)
                df_na['plant_gen_id'] = df_na['plant id'].astype(str)+'_'+df_na['generator id']
                eia860m_w = pd.concat([df_nona,df_na]).sort_values(by=['year','month','plant_code_unique']).reset_index(drop=True)
            eia860m_s = eia860m[eia860m['resource']=='solar'].merge(solar_configs[['plant_code_unique','plant_gen_id']],how='left',left_on='plant_code_unique',right_on='plant_code_unique')
            if len(eia860m_s[eia860m_s['plant_code_unique'].isna()])>0:
                df_nona = eia860m_s[~eia860m_s['plant_code_unique'].isna()].reset_index(drop=True)
                df_na = eia860m_s[eia860m_s['plant_code_unique'].isna()].reset_index(drop=True)
                df_na['plant_code_unique'] = df_na['plant id'].astype(str)
                df_na['plant_gen_id'] = df_na['plant id'].astype(str)+'_'+df_na['generator id']
                eia860m_s = pd.concat([df_nona,df_na]).sort_values(by=['year','month','plant_code_unique']).reset_index(drop=True)
            eia860m = pd.concat([eia860m_s,eia860m_w]).reset_index(drop=True)

        # identify power plants 'plantid_genid' which have moved out of BPA
        fix_bpa_inventory = {'BPAT':['58324_SIE23'],
                            'PACW':['56666_1','56360_1'],
                            'PGE' :['56485_1','56485_2','56485_3','58571_1'],
                            'AVRN':['55871_Ph 1','56359_PH2','57320_1','56361_1','56468_1','56468_1','56789_1','57096_1','57319_1','57333_1']}
        # if these plantid_genid are in eia860m[BPA] after they are no longer in bpa_monthly,
        # remove them from eia860m[BPA] and place them in eia860m[pacw/pge/avrn]
        # find the month each plant_gen_id is not longer in bpa_monthly
        bpa_monthly = get_BPA_operational_generation('wind',solar_configs,wind_configs,datadir)
        for ba in fix_bpa_inventory:
            for p in fix_bpa_inventory[ba]:
                mo,ye = bpa_monthly[bpa_monthly['plant_gen_id']==p].sort_values(by=['year','month'],ascending=False)[['month','year']].values[0]
                movedate = pd.to_datetime(str(ye)+'/'+str(mo))
                mo,ye = bpa_monthly[bpa_monthly['plant_gen_id']==p].sort_values(by=['year','month'],ascending=True)[['month','year']].values[0]
                firstdate = pd.to_datetime(str(ye)+'/'+str(mo))
                monthyears = []
                for y in range(max(2007,firstdate.year),2021,1):
                    if y == max(2007,firstdate.year):
                        m_beg = firstdate.month
                    else:
                        m_beg = 1
                    for m in range(m_beg,13,1):
                        monthyears.append((m,y))
                for m,y in monthyears:
                    thisdate = pd.to_datetime(str(y)+'/'+str(m))
                    if thisdate <= movedate:
                        eia860m.loc[(eia860m['resource']=='wind')&
                                    (eia860m['plant_gen_id']==p)&
                                    (eia860m['month']==m)&
                                    (eia860m['year']==y),['balancing authority code']] = 'BPAT'
                    else:
                        eia860m.loc[(eia860m['resource']=='wind')&
                                    (eia860m['plant_gen_id']==p)&
                                    (eia860m['month']==m)&
                                    (eia860m['year']==y),['balancing authority code']] = ba
        # final step -- remove all BPAT from eia860m, and replace with our bpa_monthly
        eia860m = eia860m[~((eia860m['balancing authority code']=='BPAT')&(eia860m['resource']=='wind'))]
        monthyears = []
        for y in range(2007,2021):
            for m in range(1,13):
                monthyears.append((y,m))
        for y,m in monthyears:
            bpa_pgi_this_my = bpa_monthly[(bpa_monthly['year']==y)&(bpa_monthly['month']==m)]['plant_gen_id'].tolist()
            idx_to_drop = eia860m[(eia860m['year']==y)&(eia860m['month']==m)&(eia860m['plant_gen_id'].isin(bpa_pgi_this_my))].index
            eia860m = eia860m.drop(idx_to_drop,axis=0)
        bpa_monthly['balancing authority code'] = 'BPAT'
        bpa_monthly['resource'] = 'wind'
        eia860m = pd.concat([eia860m,bpa_monthly]).sort_values(by=['year','month','plant_code_unique']).reset_index(drop=True)
        # eia860m = eia860m.drop_duplicates()
        # fix two plants in BPA -- plant_gen_id: 56468_2, 56790_1
        # these plants do not show up in AVRN until 2019, 
        # but they left BPA for AVRN in July 2018
        # this is an artifact of the eia860 database and needs 
        # to be manually corrected.
        fix_BPAT_AVRN = eia860m[(eia860m['plant_gen_id'].isin(['56468_2','56790_1']))&(eia860m['month'].isin([7,8,9,10,11,12]))&(eia860m['year']==2019)].copy()
        eia860m_keep = eia860m[~((eia860m['plant_gen_id'].isin(['56468_2','56790_1']))&(eia860m['month'].isin([7,8,9,10,11,12]))&(eia860m['year']==2019))].copy()
        fix_BPAT_AVRN['year'] = 2018
        eia860m = pd.concat([eia860m_keep,fix_BPAT_AVRN]).sort_values(by=['resource','balancing authority code','year','month','plant_gen_id']).reset_index(drop=True)
        # each plant_gen_id should exist once per month
        monthyears = []
        for y in range(2007,2021):
            for m in range(1,13):
                monthyears.append((y,m))
        eia860m_fix_dups = pd.DataFrame()
        for y,m in monthyears:
            for res in ['wind','solar']:
                eia860m_fix_dups_this_month = eia860m[(eia860m['month']==m)&(eia860m['year']==y)&(eia860m['resource']==res)]
                gt1 = pd.DataFrame(eia860m[(eia860m['month']==m)&(eia860m['year']==y)&(eia860m['resource']==res)&(eia860m['scale_nameplate']==1)]['plant_gen_id'].value_counts())
                gt1_id = gt1[gt1['count']>1].index.tolist()
                if len(gt1_id)>0:
                    gt1_df = eia860m[(eia860m['month']==m)&(eia860m['year']==y)&(eia860m['resource']==res)&(eia860m['plant_gen_id'].isin(gt1_id))]
                    eia860m_nodups = eia860m[(eia860m['month']==m)&(eia860m['year']==y)&(eia860m['resource']==res)].drop(gt1_df.index,axis=0)
                    # print(eia860m[(eia860m['month']==m)&(eia860m['year']==y)&(eia860m['resource']==res)&(eia860m['plant_gen_id'].isin(gt1_id))])
                    gt1_df.loc[:,['fix_plant_gen_id']]=gt1_df[['plant id','generator id']].astype(str).agg('_'.join,axis=1)
                    fixed_dups = gt1_df[gt1_df['plant_gen_id']==gt1_df['fix_plant_gen_id']]
                    fixed_dups = fixed_dups.drop(columns=['fix_plant_gen_id']).drop_duplicates()
                    eia860m_fix_dups_this_month= pd.concat([eia860m_nodups,fixed_dups])
                eia860m_fix_dups = pd.concat([eia860m_fix_dups,eia860m_fix_dups_this_month])
        eia860m = eia860m_fix_dups.copy()

        pgi_noba = eia860m[eia860m.isna().any(axis=1)][['plant id','generator id']].drop_duplicates()
        for i,row in pgi_noba.iterrows():
            pi,gi = row['plant id'],row['generator id']
            bas = eia860m[(eia860m['plant id']==pi)&(eia860m['generator id']==gi)]['balancing authority code'].dropna().drop_duplicates().values[0]
            # subset the eia860m by the pi, gi, and if bac is nan
            eia860m.loc[(eia860m['plant id']==pi)&(eia860m['generator id']==gi)&(eia860m['balancing authority code'].isna()),['balancing authority code']] = bas
            # fixeia860m = eia860m[(eia860m['plant id']==pi)&(eia860m['generator id']==gi)&(eia860m['balancing authority code'].isna())]
            eia860m = eia860m.drop_duplicates()
            for y,m in monthyears:
                gt1 = pd.DataFrame(eia860m.loc[(eia860m['plant id']==pi)&(eia860m['generator id']==gi)&(eia860m['year']==y)&(eia860m['month']==m)]['plant_gen_id'].value_counts())
                gt1_id = gt1[gt1['count']>1].index.tolist()
                if len(gt1_id)>0:
                    print(pi,gi,y,m)

        # for plants that have not been split, there should be only 1 plant_code_unique matched to the plant_gen_id
        nunique_eia860m = pd.DataFrame(eia860m[eia860m['scale_nameplate']==1.0].groupby(['plant_gen_id','year','month'])['plant_code_unique'].nunique())
        pgi_dups = nunique_eia860m[nunique_eia860m['plant_code_unique']>1].index.get_level_values('plant_gen_id').drop_duplicates().tolist()
        if len(pgi_dups) > 0:
            print('there are duplicate plant_gen_ids -- check ',pgi_dups)

        # spot fixing remaining plants. process to do this:
        # ci_ids = eia860m_long[eia860m_long['balancing authority code']=='CI']['plant id'].drop_duplicates().tolist()
        # check to see if it's a one to one match for updating the ba
        # eia860m_long[eia860m_long['plant id'].isin(ci_ids)][['plant id','balancing authority code','resource']].drop_duplicates().pivot(index='plant id',columns='balancing authority code',values='resource')
        # if it is, then we can do the easy approach:
        # change balancing authority code 'NY' to 'NYIS'
        # eia860m_long.loc[eia860m_long['balancing authority code']=='NY','balancing authority code'] = 'NYIS'
        # change balancing authority code 'CA' to 'CISO'
        eia860m.loc[eia860m['balancing authority code']=='CA','balancing authority code'] = 'CISO'
        # change balancing authority code 'GRIS' to 'GRID'
        eia860m.loc[eia860m['balancing authority code']=='GRIS','balancing authority code'] = 'GRID'
        # change balancing authority code 'PJ' to 'PJM'
        eia860m.loc[eia860m['balancing authority code']=='PJ','balancing authority code'] = 'PJM'
        # change balancing authority code 'SW' to 'SWPP'
        eia860m.loc[eia860m['balancing authority code']=='SW','balancing authority code'] = 'SWPP'
        # change balancing authority code 'CP' to 'CPLE' for plant id 61527 for years [2017,2018]
        eia860m.loc[(eia860m['balancing authority code']=='CP')&
                        (eia860m['plant id']==61527)&
                        (eia860m['year'].isin([2017,2018])),'balancing authority code'] = 'CPLE'
        # change balancing authority code 'CP' to 'CPLE' for plant ids [61512,61520,61525,61528,61536]
        eia860m.loc[(eia860m['balancing authority code']=='CP')&
                        (eia860m['plant id'].isin([61512,61520,61525,61528,61536])),'balancing authority code'] = 'CPLE'
        # change balancing authority code 'IS' to 'ISNE' for plant id [61764,61765] for years [2018]
        eia860m.loc[(eia860m['balancing authority code']=='IS')&
                        (eia860m['plant id'].isin([61764,61765]))&
                        (eia860m['year']==2018),'balancing authority code'] = 'ISNE'
        # change balancing authority code 'IS' to 'ISNE' for plant ids [61518,61537,61538,61539,61541,61622,61629,61769,61770,61771,61774]
        eia860m.loc[(eia860m['balancing authority code']=='IS')&
                        (eia860m['plant id'].isin([61518,61537,61538,61539,61541,61622,61629,61769,61770,61771,61774])),
                        'balancing authority code'] = 'ISNE'
        # change balancing authority code 'DU' to 'DUK'
        eia860m.loc[eia860m['balancing authority code']=='DU','balancing authority code'] = 'DUK'
        # change balancing authority code 'ER' to 'ERCO'
        eia860m.loc[eia860m['balancing authority code']=='ER','balancing authority code'] = 'ERCO'
        # change balancing authority code 'MI' to 'MISO'
        eia860m.loc[eia860m['balancing authority code']=='MI','balancing authority code'] = 'MISO'
        # change balancing authority code 'NY' to 'NYIS'
        eia860m.loc[eia860m['balancing authority code']=='NY','balancing authority code'] = 'NYIS'
        # change balancing authority code 'CI' to 'CISO'
        eia860m.loc[eia860m['balancing authority code']=='CI','balancing authority code'] = 'CISO'
        # change balancing authority code 'PA' to 'PACW'
        eia860m.loc[eia860m['balancing authority code']=='PA','balancing authority code'] = 'PACW'
        # change balancing authority code 'PS' to 'PSCO'
        eia860m.loc[eia860m['balancing authority code']=='PS','balancing authority code'] = 'PSCO'

    else:
        eia860m = pd.read_csv(datadir+r'/EIA860/all_years_860m.csv')

    if write_eia860== True:
        eia860m.to_csv(datadir+r'/EIA860/all_years_860m.csv',index=False)
    return eia860m

# %%
