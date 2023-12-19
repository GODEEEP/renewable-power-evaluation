import pandas as pd
import numpy as np
from datetime import timedelta
import glob
import re
import create_monthly_infrastructure as cmi
from datetime import datetime


def read_configs(datadir):
    # read in reV/SAM config files
    wind_configs = pd.read_csv(datadir+r'/tgw-gen-historical/wind/eia_wind_configs.csv')
    wind_configs['plant_gen_id'] = wind_configs['plant_code'].astype(str) + '_' + wind_configs['generator_id']
    solar_configs = pd.read_csv(datadir+r'/tgw-gen-historical/solar/eia_solar_configs.csv')
    sc_keep = solar_configs[~solar_configs['generator_id'].str.contains(',')]
    sc_fix = solar_configs[solar_configs['generator_id'].str.contains(',')]
    sc_fix = sc_fix.assign(**{'generator_id':sc_fix['generator_id'].str.split(',')}).explode('generator_id')
    solar_configs = pd.concat([sc_keep,sc_fix]).reset_index(drop=True)
    solar_configs['plant_gen_id'] = solar_configs['plant_code'].astype(str) + '_' + solar_configs['generator_id']
    pc = sc_fix['plant_code'].drop_duplicates().tolist()
    # gc = sc_fix['generator_id'].drop_duplicates().tolist()
    return wind_configs,solar_configs,pc

def read_godeeep_data(datadir):
    # extract the time series from the reV runs which used the GODEEEP-Wind dataset
    gd_w = pd.DataFrame()
    for yr in range(2006,2021):
        w = pd.read_csv(datadir+r'/tgw-gen-historical/wind/historical/wind_gen_cf_'+str(yr)+'.csv')
        gd_w = pd.concat([gd_w,w])
    gd_w['datetime'] = pd.to_datetime(gd_w['datetime'])
    gd_w = gd_w.set_index('datetime')
    # check for errors in the capacity factor
    gt1list = []
    for cc in gd_w:
        if gd_w[cc].max()> 0.885:
            gd_w[cc] = 0.885*(gd_w[cc]/gd_w[cc].max())
            print('wind: ',cc,gd_w[cc].max())
            gt1list.append(cc)
        elif gd_w[cc].max() == 0:
            print(cc,' cf = 0')
    gd_s = pd.DataFrame()
    for yr in range(2006,2021):
        s = pd.read_csv(datadir+r'/tgw-gen-historical/solar/historical_bc/solar_gen_cf_'+str(yr)+'_bc.csv')
        gd_s = pd.concat([gd_s,s])
    gd_s['datetime'] = pd.to_datetime(gd_s['datetime'])
    gd_s = gd_s.set_index('datetime')
    # check for errors in the capacity facgtor
    for cc in gd_s:
        if gd_s[cc].max()> 1:
            # gd_w[cc] = 0.885*(gd_w[cc]/gd_w[cc].max())
            print('solar: ',cc,gd_s[cc].max())
    # align the gd_s dataset with the solar config files
    # using the framework of solar generator in the solar config files
    wind_configs,solar_configs,pc = read_configs(datadir)
    eia860_y = cmi.get_eia860y(2020,['OP'],solar_configs,wind_configs,datadir+r'/validation_datasets')
    pi_pcu = eia860_y.loc[(eia860_y['resource']=='solar')][['plant id','plant_code_unique']]      
    pi_pcu['plant id'] = pi_pcu['plant id'].astype('str')
    fix_gd_s = pi_pcu[pi_pcu['plant id'].isin([str(i) for i in pc])
                      ].merge(gd_s[[str(i) for i in pc]].T,
                              how='left',
                              left_on='plant id',
                              right_index=True).drop(columns='plant id').set_index('plant_code_unique').T
    gd_s = gd_s.drop(columns=[str(i) for i in pc]).merge(fix_gd_s,how='left',left_index=True,right_index=True)
    gd_s.index = pd.to_datetime(gd_s.index)
    # check to see if all power plants have been modeled
    # and to build the ba resource list for each resource -- for all bas
    BA_list = eia860_y['Balancing Authority Code'].drop_duplicates().tolist()
    BA_list = [i for i in BA_list if isinstance(i,str)]
    ba_res_list = {}
    for res,gd_cf in zip(['solar','wind'],[gd_s,gd_w]):
        ba_res_list[res] = eia860_y[(eia860_y['resource']==res)]['Balancing Authority Code'].drop_duplicates().tolist()
        eia860_2020_list = eia860_y[(eia860_y['resource']==res)]['plant_code_unique'].drop_duplicates().tolist()
        missing_plants = [i for i in eia860_2020_list if i not in gd_cf.columns]
        if (len(missing_plants) > 0) and (np.nan not in missing_plants):
            print('There are missing power plants in the '+res+'config file')
            droplist = eia860_y[(eia860_y['resource']==res)&(eia860_y['plant_code_unique'].isin([i[0] for i in missing_plants]))]['Balancing Authority Code'].drop_duplicates().tolist()
            print('removing these BAs from current study: ',droplist)
            BA_list = [i for i in BA_list if i not in droplist]
        else:
            print('all '+res+' power plants have been modeled')
    return gd_w,gd_s,wind_configs,solar_configs,ba_res_list

def read_eia_923_930(datadir):   
    # column names for the eia923 files
    cols = ['plant id','plant name', 'reported fuel type code','netgen january',
    'netgen february',
    'netgen march',
    'netgen april',
    'netgen may',
    'netgen june',
    'netgen july',
    'netgen august',
    'netgen september',
    'netgen october',
    'netgen november',
    'netgen december','year']

    # read in the EIA923 monthly data
    eia923 = pd.DataFrame()
    for yr in range(2007,2021):
        if yr == 2007:
            skipr= 7
            filename = 'f906920_2007.xls'
        elif yr == 2008:
            skipr = 7
            filename = 'eia923December2008.xls'
        elif yr == 2009:
            skipr = 7
            filename = 'EIA923 SCHEDULES 2_3_4_5 M Final 2009 REVISED 05252011.XLS'
        elif yr == 2010:
            skipr = 7
            filename = 'EIA923 SCHEDULES 2_3_4_5 Final 2010.xls'
        elif yr == 2011:
            skipr = 5
            filename = 'EIA923_Schedules_2_3_4_5_2011_Final_Revision.xlsx'
        elif yr == 2013:
            skipr = 5
            filename = 'EIA923_Schedules_2_3_4_5_2013_Final_Revision.xlsx'
        else:
            skipr = 5
            filename = 'EIA923_Schedules_2_3_4_5_M_12_'+str(yr)+'_Final_Revision.xlsx'
        eia923_df = pd.read_excel(datadir+r'/EIA923/f923_'+str(yr)+'/'+filename,skiprows=skipr,sheet_name='Page 1 Generation and Fuel Data')
        eia923_df.columns = [i.replace('\n',' ') for i in eia923_df.columns]
        eia923_df = eia923_df.replace('.',0)
        eia923_df.columns = [i.lower() for i in eia923_df.columns]
        if 'netgen_jan' in eia923_df.columns:
            eia923_df = eia923_df.rename(columns={'netgen_jan':'netgen january','netgen_feb':'netgen february','netgen_mar':'netgen march',
                                                'netgen_apr':'netgen april','netgen_may':'netgen may','netgen_jun':'netgen june','netgen_jul':'netgen july',
                                                'netgen_aug':'netgen august','netgen_sep':'netgen september','netgen_oct':'netgen october',
                                                'netgen_nov':'netgen november','netgen_dec':'netgen december'})
        eia923_df = eia923_df[eia923_df['reported fuel type code'].isin(['WND','SUN'])]
        eia923_df = eia923_df[cols]
        eia923 = pd.concat([eia923,eia923_df])

    # read in the EIA930 hourly wind data
    eia930_w = pd.read_csv(datadir+r'/EIA930/actual_wind.csv')
    eia930_w['time_stamp'] = pd.to_datetime(eia930_w['time_stamp'])
    eia930_w = eia930_w.set_index('time_stamp')
    # read in the EIA930 hourly solar data
    eia930_s = pd.read_csv(datadir+r'/EIA930/actual_solar.csv')
    eia930_s['time_stamp'] = pd.to_datetime(eia930_s['time_stamp'])
    eia930_s = eia930_s.set_index('time_stamp')
    return eia923,eia930_w,eia930_s

# functions to extract the ba self-reported generation

def get_reported_ba_scada(ba,resource,datadir):
    print('reading in scada data for ',ba)
    validationdir = datadir+r'/validation_datasets'
    if ba == 'CISO':
        scada_df = get_caiso(resource,validationdir)
    elif ba == 'BPAT':
        scada_df = get_bpa(validationdir)
    elif ba == 'MISO':
        scada_df = get_miso(validationdir)
    elif ba == 'ISNE':
        scada_df = get_isone(resource,validationdir)
    elif ba == 'NYIS':
        scada_df = get_nyiso(validationdir)
    elif ba == 'ERCO':
        scada_df = get_ercot(resource,validationdir)
    elif ba == 'SWPP':
        scada_df = get_swpp(resource,validationdir)
    return scada_df

def get_bpa(datadir):
    bafiles = sorted(glob.glob(datadir+r'/BPA/*.xls'))
    ba_scada = pd.DataFrame()
    for f in bafiles:
        y = int(re.findall(r"(\d+).xls", f)[0])
        if y == 7:
            skip = 18
        elif y in [8,9,10]:
            skip = 20
        elif y in [2011,2012,2013,2014,2015,2016]:
            skip = 22
        elif y == 2017:
            skip = 25
        else:
            skip = 23
        if y in [7,8,9,10]:
            ba_df = pd.read_excel(f,skiprows=skip)
            cols = ['DateTime',
                   'TOTAL WIND GENERATION  IN BPA CONTROL AREA (MW; SCADA 79687)']
            janjun = ba_df[~ba_df['DateTime'].isnull()][cols]
            juldec = ba_df[~ba_df['DateTime.1'].isnull()][[i+'.1' for i in cols]]
            juldec.columns = cols
            ba_yr = pd.concat([janjun,juldec])
            ba_yr.columns = ['datetime','BPA-Wind']
            # ba_yr['datetime'] = pd.to_datetime(ba_yr['datetime'])
        else:
            janjun = pd.read_excel(f,skiprows=skip,sheet_name='January-June')
            juldec = pd.read_excel(f,skiprows=skip,sheet_name='July-December')
            cols = ['Date/Time',
                    'TOTAL WIND GENERATION  IN BPA CONTROL AREA (MW; SCADA 79687)']
            ba_yr = pd.concat([janjun[cols],juldec[cols]])
            ba_yr.columns = ['datetime','BPA-Wind']
            ba_yr['datetime'] = pd.to_datetime(ba_yr['datetime'],format='%m/%d/%y %H:%M')
        ba_scada = pd.concat([ba_scada,ba_yr])
    fivemin_idx = pd.date_range(start='2007-01-01 08:00:00',end='2021-01-01 07:55:00',freq='5T',tz='UTC').values.astype('datetime64[s]')
    ba_scada['datetime'] = fivemin_idx
    ba_scada = ba_scada.set_index('datetime')
    ba_scada = ba_scada.resample('H').mean(numeric_only=True)
    ba_scada.index = ba_scada.index.tz_localize('UTC')
    return ba_scada

def get_miso(datadir):
    fi_list = ['20081231_hwd_HIST.csv',
               '20091231_hwd_HIST.csv',
               '20101231_hwd_HIST.csv',
               '20111231_hwd_HIST.csv',
               '20121231_hwd_HIST.csv',
               '20131231_hwd_HIST.csv',
               '20141231_hwd_HIST.csv',
               '20151231_hwd_hist.csv',
               '20161231_hwd_hist.csv',
               '20171231_hwd_HIST.csv',
               '20181231_hwd_HIST.csv',
               '20191231_hwd_HIST.csv',
               '20201231_hwd_HIST.csv']
    ba_scada = pd.DataFrame()
    for fi in fi_list:
        mi = pd.read_csv(datadir+r'/MISO/'+fi,skiprows=7)#,parse_dates=[['Market Day\t','Hour Ending']])
        mi = mi[:-1]
        ba_scada = pd.concat([ba_scada,mi])
    # ba_scada = ba_scada[:-1]
    # ba_scada['Hour Ending'] = ba_scada['Hour Ending'].astype(int)
    # ba_scada['Hour Ending'] = ba_scada['Hour Ending']-1
    # ba_scada['datetime'] = pd.to_datetime(ba_scada['Market Day\t']+' '+ba_scada['Hour Ending'].astype(str)+':'+np.zeros(len(ba_scada)).astype(str))
    est_hour_idx = pd.date_range(start='2008-01-01 05:00:00',end='2021-01-01 04:55:00',freq='H',tz='UTC').values.astype('datetime64[s]')
    ba_scada['datetime'] = est_hour_idx
    ba_scada['Wind'] = ba_scada['MWh']
    ba_scada['Wind'] = ba_scada['Wind'].astype(float)
    ba_scada = ba_scada.set_index('datetime')
    ba_scada.index = ba_scada.index.tz_localize('UTC')
    return pd.DataFrame(ba_scada['Wind'])

def parse_date(date_string):
    format_strings = ['%Y-%m-%dT%H:%M:%S.%fZ',  # format with microseconds
                      '%Y-%m-%dT%H:%M:%SZ',     # format without microseconds
                     ]
    for format_string in format_strings:
        try:
            return datetime.strptime(date_string, format_string)
        except ValueError:
            pass
    return None

def interp_bw_two(df):
    df = df.set_index('datetime')
    df = df.resample('1H').interpolate(method='linear')
    df = df.reset_index()
    return df

def get_nyiso(datadir):
    folders = sorted(glob.glob(datadir+r'/NYISO/*csv'))
    ba_scada = pd.DataFrame()
    for fo in folders:
        files = sorted(glob.glob(fo+r'/*.csv'))
        for fi in files:
            ny = pd.read_csv(fi)#,parse_dates=[['Market Day\t','Hour Ending']])
            ny = ny[ny['Fuel Category']=='Wind'].reset_index(drop=True)
            # I think the units should be MW, not MWh
            # bc it's the interval power -- MW delivered in 5 min
            # so the MWh over the hour would be the average
            # over 12 intervals
            if 'Gen MWh' in ny.columns:
                ny['Wind'] = ny['Gen MWh']
            elif 'Gen MW' in ny.columns:
                ny['Wind'] = ny['Gen MW']
            ny['Time Stamp'] = pd.to_datetime(ny['Time Stamp'])
            ny['datetime'] = ny['Time Stamp']
            ny['Date'] = ny['datetime'].dt.date
            ny['Hour'] = ny['datetime'].dt.hour
            ny['Interval'] = (ny.datetime.dt.minute/5+1).astype('int')
            ny = ny[['datetime','Date','Hour','Time Zone','Interval','Wind']]
            ba_scada = pd.concat([ba_scada,ny])
    ba_scada = ba_scada.reset_index(drop=True)
    # revert EDT to EST
    ba_scada.loc[ba_scada['Time Zone'] == 'EDT',['datetime']] = ba_scada.loc[ba_scada['Time Zone'] == 'EDT',['datetime']] - timedelta(hours=1)
    # take hourly average
    ba_scada = ba_scada.set_index('datetime')
    ba_scada = ba_scada.resample('H').mean(numeric_only=True)
    # add 5 hours to convert to utc
    ba_scada.index = ba_scada.index + timedelta(hours=5)
    ba_scada.index = ba_scada.index.tz_localize('UTC')
    return pd.DataFrame(ba_scada['Wind'])

def get_ercot(resource,datadir):
    files = sorted(glob.glob(datadir+r'/ERCOT/FuelMixReport_PreviousYears/*.xls*'))
    pattern = r'^[a-zA-Z]{3}\d{4}'
    ba_scada = pd.DataFrame()
    for fi in files:
        xl = pd.ExcelFile(fi)
        snlist = xl.sheet_names
        snlist = [i for i in snlist if re.match(pattern, i)]
        if len(snlist) == 0:
            pattern = r'^[a-zA-Z]{3}\d{2}'
            snlist = xl.sheet_names
            snlist = [i for i in snlist if re.match(pattern, i)]
        if len(snlist) == 0:
            pattern = r'^[A-Z][a-z]{2}$'
            snlist = xl.sheet_names
            snlist = [i for i in snlist if re.match(pattern, i)]
        for i in snlist:
            df = xl.parse(i)
            # datefuel = ['Date - Fuel','DateFuel','Date-Fuel']
            df_col = [i for i in df.columns.tolist() if i == 'Date - Fuel' or i == 'DateFuel' or i == 'Date-Fuel']
            if len(df_col) > 0:
                df_col = [i for i in df.columns.tolist() if i == 'Date - Fuel' or i == 'DateFuel' or i == 'Date-Fuel'][0]
                delim = re.findall(r'\d+([\s_ -]+)\w*',df[df_col].values[0])[0]
                df[['Date','Fuel']] = df[df_col].str.split(delim,expand=True)
                df = df.drop(columns={df_col})
            df_datefuel = df[['Date','Fuel']]
            df = df.drop(['Date','Fuel'],axis=1)
            totaldrop = [i for i in df.columns if i == 'Total' or i == 'Daily MWH' or i == 'Settlement Type']
            df = df.drop(totaldrop,axis=1)
            if df.columns.shape[0] == 96:
                col_list = ['HE'+str(i)+'_int'+str(j) for i in range(1,25) for j in range(1,5)]
                df.columns = col_list
            else:
                col_list = ['HE'+str(i)+'_int'+str(j) for i in range(1,26) for j in range(1,5)]
                df.columns = col_list
            df= df_datefuel.merge(df,how='left',left_index=True,right_index=True)
            df = pd.melt(df,id_vars=['Date','Fuel'],value_vars=col_list)
            df[['HE','int']] = df['variable'].str.split('_',expand=True)
            df['HE'] = df['HE'].str[2:].astype(int)
            df['int'] = df['int'].str[3:].astype(int)
            df = df.dropna()
            df.loc[(df['Fuel']=='Sun'),['Fuel']] = 'Solar'
            df.loc[(df['Fuel']=='Wnd'),['Fuel']] = 'Wind'
            df = df[df['Fuel'].isin(['Wind','Solar'])]
            ba_scada = pd.concat([ba_scada,df])
    ba_scada['Date'] = pd.to_datetime(ba_scada['Date'],format='%m/%d/%y')
    ba_scada = ba_scada.sort_values(by=['Date','HE','int'])
    ba_scada = ba_scada[ba_scada['Fuel']==resource.capitalize()]
    if resource == 'wind':
        idx = pd.date_range(start='2007-01-01 06:15:00',periods=490944,freq='15T',tz='UTC').values.astype('datetime64[s]')
    elif resource == 'solar':
        idx = pd.date_range(start='2011-07-01 05:15:00',periods=333316,freq='15T',tz='UTC').values.astype('datetime64[s]')
    ba_scada = ba_scada.sort_values(by=['Date','HE','int']).set_index(idx)
    ba_scada['value'] = ba_scada['value'].astype(float)
    ba_scada = pd.DataFrame(ba_scada.rename(columns={'value':resource.capitalize()})[resource.capitalize()]).resample('H').sum()
    ba_scada.index = ba_scada.index.tz_localize('UTC')
    return pd.DataFrame(ba_scada[resource.capitalize()])

def get_caiso(resource,datadir,curt=False):
    fi_list = ['ProductionAndCurtailmentsData-May1_2014-May31_2017.xlsx','ProductionAndCurtailmentsData-Jun1_2017-Dec31_2017.xlsx',
              'ProductionAndCurtailmentsData_2018.xlsx','ProductionAndCurtailmentsData_2019.xlsx','ProductionAndCurtailmentsData_2020.xlsx']
    ba_scada = pd.DataFrame()
    ba_curt = pd.DataFrame()
    for fi in fi_list:
        ba_scada = pd.concat([ba_scada,pd.read_excel(datadir+r'/CAISO/'+fi,sheet_name='Production')])
        ba_curt = pd.concat([ba_curt,pd.read_excel(datadir+r'/CAISO/'+fi,sheet_name='Curtailments')])
    ba_scada['Date'] = pd.to_datetime(ba_scada['Date'].dt.date)
    if curt == True:
        # ba_curt = ba_curt.fillna(0)
        ba_scada = ba_scada.merge(ba_curt,how='left',left_on=['Date','Hour','Interval'],right_on=['Date','Hour','Interval'])
        ba_scada = ba_scada.fillna(0)
    # ba_scada['datetime'] = ba_scada['Date']
    fivemin_idx = pd.date_range(start='2014-05-01 07:00:00',periods=701868,freq='5T',tz='UTC').values.astype('datetime64[s]')
    ba_scada['datetime'] = fivemin_idx
    ba_scada = ba_scada.set_index('datetime')
    ba_scada = ba_scada.resample('H').mean(numeric_only=True)
    ba_scada.index = ba_scada.index.tz_localize('UTC')
    if curt == True:
        return pd.DataFrame(ba_scada[resource.capitalize()]-ba_scada[resource.capitalize()+' Curtailment'])
    else:
        return pd.DataFrame(ba_scada[resource.capitalize()])

def get_isone(resource,datadir):
    if resource == 'wind':
        fi_list = ['hourly_wind_gen_2011_2014.xlsx',
            'hourly_wind_gen_2015.xlsx',
            'hourly_wind_gen_2016.xlsx',
            'hourly_wind_gen_2017.xlsx',
            'hourly_wind_gen_2018.xlsx',
            'hourly_wind_gen_2019.xlsx',
            'hourly_wind_gen_2020.xlsx']
        ba_scada = pd.DataFrame()
        for fi in fi_list:
            isne = pd.read_excel(datadir+r'/ISONE/'+fi,sheet_name='HourlyData')
            if fi == 'hourly_wind_gen_2011_2014.xlsx':
                isne.columns = ['year','Date','Hour','Wind']
            else:
                isne.columns = ['year','Date','Hour','Wind','freq']
            if fi == 'hourly_wind_gen_2017.xlsx':
                isne = isne[:-1]
            isne.loc[(isne['Hour'] == '02X'),['Hour']] = '02'
            isne['Hour'] = isne['Hour'].astype('int')
            isne['Hour'] = isne['Hour'] -1
            isne['datetime'] = pd.to_datetime(isne['Date'].astype(str)+' '+isne['Hour'].astype(str)+':'+np.zeros(len(isne)).astype(str))
            isne['diff'] = isne['datetime'] - isne['datetime'].shift(1)
            if fi == 'hourly_wind_gen_2011_2014.xlsx':
                # need to do some interpolation for values which have been removed
                # they were removed bc the number of plants producing in that interval
                # is lower than allowed for public knowledge
                springfwd = [datetime(2011,3,13,2,0,0),
                             datetime(2012,3,11,2,0,0),
                             datetime(2013,3,10,2,0,0),
                             datetime(2014,3,9,2,0,0)]
                isne.loc[isne['datetime'].isin(springfwd),['diff']] = pd.Timedelta('1H')
                mask = isne['diff']>pd.Timedelta('1H')
                flagged_mask = mask.shift(-1) | mask
                isne_interp = isne[flagged_mask]
                isne_nointerp = isne[~isne.isin(isne_interp).all(axis=1)]
                isne_interp = isne_interp.reset_index(drop=True)
                mask = isne_interp['diff'] == pd.Timedelta('1H')
                isne_interp['group'] = mask.cumsum()
                isne_interp = isne_interp.groupby('group').apply(interp_bw_two).reset_index(drop=True)
                isne = pd.concat([isne_nointerp,isne_interp])
                isne = isne.sort_values(by='datetime')
                isne = isne[['year','Date','Hour','Wind','datetime']]
            ba_scada = pd.concat([ba_scada,isne])

    elif resource == 'solar':
        fi_list = ['hourly_solar_gen_'+str(i)+'.xlsx' for i in range(2011,2021)]
        ba_scada = pd.DataFrame()
        for fi in fi_list:
            isne = pd.read_excel(datadir+r'/ISONE/'+fi,sheet_name='HourlyData')
            isne.columns = ['year','Date','Hour','Solar','freq']
            isne.loc[(isne['Hour'] == '02X'),['Hour']] = '02'
            isne['Hour'] = isne['Hour'].astype('int')
            isne['Hour'] = isne['Hour'] -1
            isne['datetime'] = pd.to_datetime(isne['Date'].astype(str)+' '+isne['Hour'].astype(str)+':'+np.zeros(len(isne)).astype(str))
            isne['Solar'] = isne['Solar'].fillna(0)
            ba_scada = pd.concat([ba_scada,isne])
    est_hour_idx = pd.date_range(start='2011-01-01 05:00:00',end='2021-01-01 04:55:00',freq='H',tz='UTC').values.astype('datetime64[s]')
    ba_scada['datetime'] = est_hour_idx
    ba_scada = ba_scada.set_index('datetime')
    ba_scada.index = ba_scada.index.tz_localize('UTC')
    return pd.DataFrame(ba_scada[resource.capitalize()])

def get_swpp(resource,datadir):
    fi_list = ['GenMix_'+str(i)+'.csv' for i in range(2011,2021)]
    y_pattern = re.compile(r'\d{4}')
    ba_scada = pd.DataFrame()
    for fi in fi_list:
        swpp = pd.read_csv(datadir+r'/SWPP/'+fi)
        y = int(re.findall(y_pattern,fi)[0])
        if y < 2016:
            swpp = swpp[['GMTTime','Solar','Wind']]
            swpp = swpp.rename(columns={'GMTTime':'datetime'})
        elif y < 2018:
            swpp = swpp[['GMT MKT Interval',' Solar',' Wind']]
            swpp = swpp.rename(columns={'GMT MKT Interval':'datetime',' Solar':'Solar',' Wind':'Wind'})
        else:
            swpp = swpp[['GMT MKT Interval',' Solar Self',' Wind Self']]
            swpp = swpp.rename(columns={'GMT MKT Interval':'datetime',' Solar Self':'Solar',' Wind Self':'Wind'})
        if swpp.iloc[-1].isna().all():
            swpp = swpp.iloc[:-1]
        ba_scada = pd.concat([ba_scada,swpp])
    ba_scada = ba_scada.reset_index(drop=True)
    ba_scada['datetime'] = ba_scada['datetime'].apply(lambda x: parse_date(x))
    ba_scada = ba_scada.set_index('datetime')
    ba_scada = ba_scada.resample('H').mean(numeric_only=True)
    ba_scada.index = ba_scada.index.tz_localize('UTC')
    return pd.DataFrame(ba_scada[resource.capitalize()])
