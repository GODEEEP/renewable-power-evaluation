import pandas as pd
import calendar
import read_timeseries_configs as rtc

def get_monthyear(firstdate):
    monthyear = []
    for y in range(2020,pd.to_datetime(firstdate).year-1,-1):
        if y == pd.to_datetime(firstdate).year:
            m_end = pd.to_datetime(firstdate).month-1
        else:
            m_end = 0
        for m in range(12,m_end,-1):
            monthyear.append((y,m))
    return monthyear



# this function is used in one place below to convert the eia923 monthly series
# to an hourly average
def resample_hourly_8760(df):
    leapyr = [2008,2012,2016,2020]
    thisyear = [*set(df.index.year)]
    df = pd.DataFrame({'eia923':df['eia923']/(df.index.days_in_month*24)})
    df.loc[(df.index[-1]+pd.DateOffset(months=1)),:] = 0#
    if thisyear[0] in leapyr:
        if thisyear[0] == 2020:
            remove_hrs = -24
        else:
            remove_hrs = -25 # consider changing this later to detect the last year
    else:
        remove_hrs = -1
    return df.resample('H').ffill()[:remove_hrs]

def cf_to_gen(df,gd,res,solar_configs,wind_configs):
    # this assumes that only wind plants have been split/scaled, and there is
    # wind -- a many:1 match between the plant_code_unique and plant_gen_id 
    # solar -- a 1:many match between plant_code_unique and plant_gen_id
    # accounts for plant_code_uniqe needing to be grouped/summed for the wind
    # plants that were split -- these are summed to a single plant_gen_id

    df_out = pd.DataFrame()
    for pcu,np in zip(df['plant_code_unique'],df['nameplate capacity (mw)']):
        # print(gd[pcu])
        if pcu not in gd.columns:
            if (res == 'solar')& (len(solar_configs[solar_configs['plant_code_unique']==pcu])!=0):
                    print('plant_code_unique '+pcu+' not in columns -- this solar pcu was modeled')
            elif (res == 'wind')& (len(wind_configs[wind_configs['plant_code_unique']==pcu])!=0):
                    print('plant_code_unique '+pcu+' not in columns -- this wind pcu was modeled')
            df_out = pd.DataFrame(index=gd.index,columns=df['plant_gen_id'])
            # print('here')
            # df_out[pcu] = np.nan#np.empty(len(gd))*np.nan
            # print('here')
        else:
            if (res == 'solar') & (len(gd[pcu].shape)>1):
                # print('duplicated solar columns')
                gd_temp = gd[pcu].loc[:,~gd[pcu].columns.duplicated()].copy()
                df_out[pcu] = gd_temp*np
            else:
                df_out[pcu] = gd[pcu]*np
        # print(df_out[pcu].head())
    return df_out.sum(axis=1)

def create_plantlevel_generation(gd_df,eia860m_long,res,monthyears,solar_configs,wind_configs,datadir):
    gd_df = gd_df[gd_df.index.year!=2006]
    eia860m_r = eia860m_long[eia860m_long['resource']==res]
    gd_r = pd.DataFrame(columns=eia860m_r['plant_gen_id'].drop_duplicates().tolist(),index=gd_df.index)
    for y,m in monthyears:
        print(y,m)
        pm = eia860m_r[(eia860m_r['month']==m)&(eia860m_r['year']==y)]
        pm.loc[:,['nameplate capacity (mw)']] = pm['nameplate capacity (mw)']*pm['scale_nameplate']
        gd_df_m = gd_df[(gd_df.index.year==y)&(gd_df.index.month==m)]
        gd_r_cols = list(pm.groupby('plant_gen_id').groups.keys())
        gd_r.loc[(gd_r.index.year==y)&(gd_r.index.month==m),gd_r_cols]=pm.groupby('plant_gen_id').apply(cf_to_gen,pd.DataFrame(gd_df_m),res,solar_configs,wind_configs).T
    # check for production greater than capacity or equal to zero
    for pgi in gd_r.columns:
        max_pgi = gd_r[pgi].max()
        this_np = eia860m_r[eia860m_r['plant_gen_id']==pgi]['nameplate capacity (mw)'].max()
        thiscf = gd_r[pgi].max()/this_np
        if thiscf > 1:
            print(pgi,' gt 1')
        elif (thiscf < 0.5) & (thiscf > 0):
            print(pgi,thiscf,' lt 0.5')
        elif thiscf == 0:
            print(pgi,' = 0')
    gd_r.to_csv(datadir+r'/historical_power/'+res+'_plant_generation.csv')
    return gd_r


def create_balevel_generation(all_plant_gen,res,balist,gd_df,eia930_df,eia923_df,eia860m,subset_list,monthyears,datadir,save_dicts):
    BA_hr_diff = {}
    for thisBA in balist:
        BA_hr_diff[thisBA] = pd.DataFrame()
        if thisBA in subset_list:
            BA_hr_diff[thisBA][res+'-scada'] = rtc.get_reported_ba_scada(thisBA,res.lower(),datadir)
            final_cols = ['eia930','eia923','GODEEEP-'+res,res+'-scada']
        else:
            final_cols = ['eia930','eia923','GODEEEP-'+res]
        plants_monthly = eia860m[thisBA][eia860m[thisBA]['resource']==res].copy(deep=True)
        # this contains a placeholder for all plants in this BA that ever existed between
        # 2007 and 2020
        BA = pd.DataFrame(index=gd_df.index,columns=plants_monthly['plant_gen_id'].drop_duplicates().tolist())
        # multiply plant level capacities through the capacity factors
        for y,m in monthyears:
            pm = plants_monthly[['balancing authority code','nameplate capacity (mw)','resource','month','year','plant_gen_id']].drop_duplicates()
            pm = pm[(pm['month']==m)&(pm['year']==y)]
            cap =pm[['plant_gen_id','nameplate capacity (mw)']]
            cap = cap.set_index(cap['plant_gen_id'].astype(str))
            pgi_thismonth = pm['plant_gen_id'].drop_duplicates().tolist()
            # pull in the production from the plant_gen_ids for that month
            BA.loc[(BA.index.month==m)&(BA.index.year==y),pgi_thismonth] = all_plant_gen.loc[
                (all_plant_gen.index.month==m)&(all_plant_gen.index.year==y),pgi_thismonth]
        if thisBA == 'GRIS':
            fixedBA = 'GRID'
            BA_hr = pd.DataFrame({'GODEEEP-'+res:BA.sum(axis=1)}).merge(eia930_df.loc[eia930_df['eia_ba_name']==fixedBA]['value'],how='left',left_index=True,right_index=True)
        else:
            BA_hr = pd.DataFrame({'GODEEEP-'+res:BA.sum(axis=1)}).merge(eia930_df.loc[eia930_df['eia_ba_name']==thisBA]['value'],how='left',left_index=True,right_index=True)
        BA_hr=BA_hr.rename(columns={'value':'eia930'})
        # first_mo,first_ye = plants_monthly.sort_values(by=['year','month'])[['month','year']].iloc[0]
        # month_idx = pd.date_range(start=str(first_ye)+'-'+str(first_mo)+'-01 01:00:00',end='2020-12-01 01:00:00',freq='MS',tz='UTC').values.astype('datetime64[s]')
        BA_hr_diff[thisBA].index.name = 'datetime'
        BA_hr_diff[thisBA] = BA_hr_diff[thisBA].merge(BA_hr,how='outer',left_index=True,right_index=True)#pd.concat([BA_hr_diff[thisBA],BA_hr],axis=1)
        BA_hr_diff[thisBA].index = pd.to_datetime(BA_hr_diff[thisBA].index).values.astype('datetime64[s]')
        # pd.date_range(start = BA_hr_diff[thisBA].index[0],periods=len(BA_hr_diff[thisBA]),freq='1H',tz='UTC').values.astype('datetime64[s]')
        BA_m = BA_hr_diff[thisBA].resample('MS').sum()
        eia923_ba = eia923_df[eia923_df['plant id'].isin(plants_monthly['plant id'].drop_duplicates().tolist())]
        for yr in [*set([i[0] for i in monthyears])]:
            months = BA_m.loc[BA_m.index.year==yr].index.month
            for m in months:
                month_col = ['netgen '+calendar.month_name[m].lower()]
                pm = plants_monthly[['balancing authority code','resource','month','year','plant id']].drop_duplicates()
                pm = pm[(pm['month']==m)&(pm['year']==yr)]
                pi_thismonth = pm['plant id'].drop_duplicates().tolist()
                BA_m.loc[(BA_m.index.year==int(yr))&(BA_m.index.month==m),'eia923'] = \
                    eia923_ba.loc[(eia923_ba['year']==yr)&(eia923_ba['plant id'].isin(pi_thismonth)),month_col].sum().values
        # BA_m = BA_m.iloc[:-1]
        # remove last day for the leap year
        thisBA_eia923 = BA_m.groupby(BA_m.index.year).apply(resample_hourly_8760)[1:]
        thisBA_eia923.index.names = ['year','datetime']
        thisBA_eia923 = thisBA_eia923.droplevel('year')
        BA_hr_diff[thisBA]['eia923'] = thisBA_eia923
        BA_hr_diff[thisBA] = BA_hr_diff[thisBA][final_cols]
        BA_hr_diff[thisBA] = BA_hr_diff[thisBA].astype('float')
        if save_dicts:
            BA_hr_diff[thisBA].to_csv(datadir+r'/historical_power/'+thisBA+'_'+res+'_generation.csv')
    return BA_hr_diff