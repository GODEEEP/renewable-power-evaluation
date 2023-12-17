#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar
import math
from scipy import stats
import seaborn as sns
import os

#%%
# plot the infrastructure over time
def plot_infrastructure_over_time(wlist,slist,eia860m_long,all_power_hourly,outdir,plot_capnormdiff=False,savef=False):
    tabledir = outdir+r'/tables'
    if not os.path.exists(tabledir):
        os.makedirs(tabledir)
    if savef:
        plotdir = outdir+r'/plots'
        if not os.path.exists(plotdir):
            os.makedirs(plotdir)
    for this_list,res in zip([wlist,slist],['wind','solar']):
        for this_ba in this_list:
            ba_hourly_power = all_power_hourly[res][this_ba][['GODEEEP-'+res,'scada']]
            ba_hourly_power = ba_hourly_power.groupby([ba_hourly_power.index.year,ba_hourly_power.index.month]).max().reset_index()
            ba_hourly_power = ba_hourly_power.rename(columns={'level_0':'year','level_1':'month'})
            ba_hourly_power['year-mo'] = ba_hourly_power[['year','month']].astype(str).agg('-'.join,axis=1)

            tmp = eia860m_long[(eia860m_long['balancing authority code']==this_ba)&(eia860m_long['resource']==res)].groupby(['year','month'])['nameplate capacity (mw)'].sum().reset_index()
            tmp['year-mo'] = tmp[['year','month']].astype(str).agg('-'.join,axis=1)

            ba_list = eia860m_long[(eia860m_long['balancing authority code']==this_ba)&
                                    (eia860m_long['resource']==res)]
            ba_list = ba_list.groupby(['year','month'])['nameplate capacity (mw)'].sum().reset_index()
            ba_list['year-mo'] = ba_list[['year','month']].astype(str).agg('-'.join,axis=1)
            ba_hourly_power = ba_hourly_power[ba_hourly_power['year-mo'].isin(ba_list['year-mo'].tolist())].reset_index(drop=True)
            ba_hourly_power['capacity'] = tmp['nameplate capacity (mw)']
            ba_hourly_power['capacity percent diff'] = (ba_hourly_power['scada']-ba_hourly_power['GODEEEP-'+res])/ba_hourly_power['capacity']
            ba_hourly_power['percent diff'] = (ba_hourly_power['scada']-ba_hourly_power['GODEEEP-'+res])/ba_hourly_power['scada']

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(1, 1, 1)
            majorx_ticks = pd.RangeIndex(0,len(tmp),12)
            minorx_ticks = pd.RangeIndex(0,len(tmp),6)
            yscale = len(str((ba_hourly_power['scada'].max()*1.2).astype(int)))
            majory_ticks = pd.RangeIndex(0,(ba_hourly_power['scada'].max()*1.2).astype(int),250*10**(yscale-4))
            plt.plot(tmp['year-mo'],tmp['nameplate capacity (mw)'],label='godeeep-capacity',color='tab:blue',linestyle='--')
            plt.plot(ba_list['year-mo'],ba_list['nameplate capacity (mw)'],label=this_ba+'-capacity',color='k',linestyle='--')
            plt.plot(ba_hourly_power['year-mo'],ba_hourly_power['GODEEEP-'+res],label='godeeep-power',color='tab:blue',linestyle='-')
            plt.plot(ba_hourly_power['year-mo'],ba_hourly_power['scada'],label=this_ba+'-power',color='k',linestyle='-')
            ax.set_xticks(majorx_ticks)
            ax.set_xticks(minorx_ticks, minor=True)
            ax.set_yticks(majory_ticks)
            # plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
            ax.grid(which='minor', alpha=0.2)
            ax.grid(which='major', alpha=0.8)
            plt.title(this_ba+' '+res.capitalize() +' Installed Capacity and Max Monthly Production',fontsize=18)
            plt.legend()
            plt.xticks(rotation = 45,fontsize=14) # Rotates X-Axis Ticks by 45-degrees
            plt.yticks(fontsize = 14) # Rotates X-Axis Ticks by 45-degrees
            plt.ylabel('MW',fontsize=16)
            plt.tight_layout()
            if savef:
                plt.savefig(plotdir+r'/'+this_ba+'_'+res+'_infrastructure_over_time.png')
            plt.show()
            plt.close(fig)

            if plot_capnormdiff:
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                majorx_ticks = pd.RangeIndex(0,len(tmp),12)
                minorx_ticks = pd.RangeIndex(0,len(tmp),6)
                plt.plot(ba_hourly_power['year-mo'],ba_hourly_power['capacity percent diff'],color='tab:blue',linestyle='-',alpha=0.9)
                ax.set_xticks(majorx_ticks)
                ax.set_xticks(minorx_ticks, minor=True)
                plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
                ax.grid(which='minor', alpha=0.2)
                ax.grid(which='major', alpha=0.8)
                plt.title(this_ba+' '+res.capitalize()+' Capacity Normalized Monthly Max Production Difference')
                if savef:
                    plt.savefig(plotdir+r'/'+this_ba+'_'+res+'_cap_norm_diff.png')
                plt.show()
                plt.close(fig)
    return

#%%
# yearly sum, bar plot for each -- scada and eia923 against our simulation
# each comparison dataset will be a key in a dict
# within the dict, the index is the bar label
# the column headers are the components of the bar

def setup_timeseries(df,compute_cf,capacity,cols):
    df = df[(df.index.year>2006)&(df.index.year<2021)]
    if compute_cf:
        # sig figs only for cf space, not for total energy
        df = df.merge(capacity,how='left',left_on=[df.index.year,df.index.month],right_on=['year','month']).set_index(df.index)
        df = df[cols+['nameplate capacity (mw)']]
        # df[cols] = df[cols].div(df['nameplate capacity (mw)'],axis=0)
    # dd = dd.dropna(axis=0)
    df = df[cols]
    return df
                # dd,freq,res,comp,datadir,thesecols,BA,'balevel_'+sname,savef
def plot_whiskers(dd,freq,res,comp,plotdir,thesecols,region,agg,savef,showf):
    dd = dd[dd['nameplate capacity (mw)']>0]
    # print(dd.columns)
    # print(comp)
    # print(freq)
    if freq == 'Seasonal':
        dd = dd.groupby([dd.index.year,dd.index.month]).sum()
        dd.index.set_names(['Year','Month'],inplace=True)
        xaxis = [calendar.month_name[i] for i in sorted(dd.index.get_level_values('Month').drop_duplicates())]
        rotate = 45
        grouping = 'Month'
        grp_unit = ''
    elif freq == 'Diurnal':
        dd = dd.dropna(axis=0)
        xaxis = [i+1 for i in sorted(dd.index.hour.drop_duplicates())]
        dd.index = pd.MultiIndex.from_tuples(list(zip(*[dd.index.date,dd.index.hour])),names=['Date','Hour'])
        rotate = 0
        grouping = 'Hour'
        grp_unit = ' [UTC]'
    # print(grouping)
    dd[thesecols] = dd[thesecols].div(dd['nameplate capacity (mw)'],axis=0)
    dd = pd.melt(dd.reset_index(),id_vars=[grouping],value_vars=['GODEEEP-'+res,comp[0]],var_name='dataset',value_name='Capacity Factor')
        # dd.groupby(['dataset','Hour']).mean().unstack('dataset') --> this is how we get the hourly/monthly values
    dd = dd.sort_values(by=grouping)
    if grouping == 'Month':
        dd['Month'] = [calendar.month_name[i] for i in dd['Month']]
    bias = calc_diff(dd.groupby(['dataset',grouping]).mean().unstack('dataset').droplevel(0,axis=1),'mean',comp,res)
    bias_label = []
    for col in dd['dataset'].drop_duplicates().tolist():
        if col == 'GODEEEP-'+res:
            bias_label.append(col)
        else:
            bias_label.append(col+', bias = '+str(round(bias[comp[0]],2)))#+biasunit
    fig,_ = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
    ax = sns.boxplot(x=grouping,
                     y='Capacity Factor',
                     data=dd,
                     hue='dataset')
                    #  order=['January','February','March','May','June','July','August','September','October','November','December'])
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                alpha=0.5)
    ax.set(
        axisbelow=True)
    width = 0.15
    x = np.arange(len(xaxis))
    ax.set_ylabel('Capacity Factor',fontsize=16)
    ax.set_xlabel(grouping+grp_unit,fontsize=16)
    # print(region)
    ax.set_title(region+' '+'Capacity Factor'+' '+res.capitalize(),fontsize=18)
    ax.set_xticks(x + width, xaxis)
    ax.set_xticklabels(xaxis,rotation=rotate, fontsize=14)
    ax.set_ylim([0,1.1])
    ax.tick_params(axis='y',labelsize=14)
    ax.legend(loc='upper left',fontsize=14)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                alpha=0.5)
    plt.tight_layout()
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=bias_label)
    if savef:
        plt.savefig(plotdir+r'/'+region+'_'+res+'_'+agg+'.png')
    if showf:
        plt.show()
    plt.close(fig)
    return


winter = [12,1,2]
spring = [3,4,5]
summer = [6,7,8]
fall= [9,10,11]
seasons = [winter,spring,summer,fall]
season_names = ['DJF','MAM','JJA','SON']
                    # all_power_hourly,eia860m_long,freq,res,these_bas,outdir,compute_cf,comp,mask,agg,savef=False
def plot_grouped_box(all_power_hourly,eia860m_long,freq,res,balist,outdir,compute_cf,comp,mask,aggregation,season_disagg,savef=False,showf=False):
    tabledir = outdir+r'/tables'
    if not os.path.exists(tabledir):
        os.makedirs(tabledir)
    if savef:
        plotdir = outdir+r'/plots'
        if not os.path.exists(plotdir):
            os.makedirs(plotdir)
    if aggregation == 'by BA':
        for BA in balist:
            dd = all_power_hourly[res][BA].copy(deep=True)[['GODEEEP-'+res]+comp]
            thesecols = dd.columns.tolist()
            capacity = eia860m_long[(eia860m_long['balancing authority code']==BA)&(eia860m_long['resource']==res)].groupby(['year','month'])['nameplate capacity (mw)'].sum().reset_index()
            capacity['year-mo'] = capacity[['year','month']].astype(str).agg('-'.join,axis=1)
            dd = dd[(dd.index.year>2006)&(dd.index.year<2021)]
            dd = dd.merge(capacity,how='left',left_on=[dd.index.year,dd.index.month],right_on=['year','month']).set_index(dd.index)
            dd = dd[thesecols+['nameplate capacity (mw)']]
            if season_disagg == True:
                for season,sname in zip(seasons,season_names):
                    dd_plot = dd[dd.index.month.isin(season)]
                    plot_whiskers(dd_plot,freq,res,comp,plotdir,thesecols,BA+' '+sname,'balevel_'+sname,savef,showf)
            else:
                plot_whiskers(dd,freq,res,comp,plotdir,thesecols,BA,'balevel',savef,showf)
    elif aggregation == 'by NERC region':
        for bas_in_nerc,nerc_name in zip([wecc,eic,erco],nerc_zones):
            plot_bas = [i for i in bas_in_nerc if i in balist]
            nerc_wide = pd.DataFrame(columns=['GODEEEP-'+res]+comp+['nameplate capacity (mw)'],
                                    index=pd.date_range(start='2007-01-01',end='2021-01-01',inclusive='left',freq='1H',tz='UTC').values.astype('datetime64[s]'))
            nerc_eachba = np.zeros(list(nerc_wide.shape)+[len(plot_bas)]) 
            for BA,ba_idx in zip(plot_bas,range(len(plot_bas))):
                dd = all_power_hourly[res][BA].copy(deep=True)[['GODEEEP-'+res]+comp]
                thesecols = dd.columns.tolist()
                capacity = eia860m_long[(eia860m_long['balancing authority code']==BA)&(eia860m_long['resource']==res)].groupby(['year','month'])['nameplate capacity (mw)'].sum().reset_index()
                capacity['year-mo'] = capacity[['year','month']].astype(str).agg('-'.join,axis=1)
                dd = dd.merge(capacity,how='left',left_on=[dd.index.year,dd.index.month],right_on=['year','month']).set_index(dd.index)
                dd = dd[thesecols+['nameplate capacity (mw)']]
                dd = dd[(dd.index.year>2006)&(dd.index.year<2021)].fillna(0)
                # 3. place the 'dd' in nerc_eachba
                nerc_eachba[:len(dd),:,ba_idx] = dd
            nerc_wide.iloc[:,:] = nerc_eachba.sum(axis=2)
            dd = plot_whiskers(nerc_wide,freq,res,comp,plotdir,thesecols,nerc_name,'nerclevel',savef,showf)
    return


                    # all_power_hourly,eia860m_long,freq,res,these_bas,outdir,compute_cf,comp,mask,agg,savef=True
def plot_grouped_bars(all_power_hourly,eia860m_long,freq,res,balist,outdir,compute_cf,comparison_datasets,mask,agg,s_agg,savef=False,showf=False):
    tabledir = outdir+r'/tables'
    if not os.path.exists(tabledir):
        os.makedirs(tabledir)
    if savef:
        plotdir = outdir+r'/plots'
        if not os.path.exists(plotdir):
            os.makedirs(plotdir)
    if freq=='Y':
        idx = list(range(2007,2021))+['mean']
    stats_df = pd.DataFrame(columns=pd.MultiIndex.from_product([balist,comparison_datasets]),index=idx,dtype=pd.Int64Dtype())
    for BA in balist:
        dd = all_power_hourly[res][BA].copy(deep=True)[['GODEEEP-'+res]+comparison_datasets]
        thesecols = dd.columns.tolist()
        dd = dd[(dd.index.year>2006)&(dd.index.year<2021)]
        if compute_cf:
            # sig figs only for cf space, not for total energy
            capacity = eia860m_long[(eia860m_long['balancing authority code']==BA)&(eia860m_long['resource']==res)].groupby(['year','month'])['nameplate capacity (mw)'].sum().reset_index()
            capacity['year-mo'] = capacity[['year','month']].astype(str).agg('-'.join,axis=1)
            dd = dd.merge(capacity,how='left',left_on=[dd.index.year,dd.index.month],right_on=['year','month']).set_index(dd.index)
            dd = dd[thesecols+['nameplate capacity (mw)']]
            dd[thesecols] = dd[thesecols].div(dd['nameplate capacity (mw)'],axis=0)
            dd = dd[thesecols]
        if compute_cf:
            energyunit = ' Capacity Factor'
            cf_fig = 'cf'
        else:
            energyunit = ' Energy [GWh x1e6]'
            cf_fig = 'energy'
        if freq == 'Y':
            if compute_cf:
                summarize = 'mean'
                biasunit = ''
                ylab = 'Annual Average'
                plotdd = dd.resample(freq).mean()
            else:
                summarize = 'pct'
                biasunit = ' %'
                ylab = 'Annual'
                plotdd = dd.resample(freq).sum()/1e6
            bias = (dd.resample(freq).apply(calc_diff,summarize,comparison_datasets,res)*100)
            bias.index = bias.index.year
            bias = bias.replace([np.inf,-np.inf],np.nan)
            bias.loc['mean'] = bias.mean().astype(int)
            bias = bias.round(0).apply(lambda x: [str(i).rstrip('0').rstrip('.') for i in x])
            add_yrs = len([i for i in list(range(2007,2021)) if i not in plotdd.index.year])
            yrs_idx = pd.date_range(start='12/31/2007',periods=add_yrs,freq='Y',unit='s')#.values#.astype('datetime64[s]')
            add_rows = pd.DataFrame(columns=plotdd.columns.tolist(),index=yrs_idx)
            plotdd = pd.concat([add_rows,plotdd])
            plotdd.index = pd.to_datetime(plotdd.index,utc=True)
            xaxis = plotdd.index.year
            yfig = 10
        elif freq == 'M':
            ylab = 'Monthly Average'
            bias = dd.groupby(dd.index.month).apply(calc_diff,'mean',thesecols,res).mean().round(2)
            if compute_cf:
                biasunit = ''
            else:
                biasunit = ' GWh'
            dd = dd.groupby(dd.index.month).mean()
            # bias = calc_pctdiff(dd,thesecols,res).round(2)
            xaxis = [calendar.month_name[i] for i in dd.index]
            yfig = 10
        stats_df[BA] = bias
        bias_dict = {}
        for col in plotdd.columns:
            if col == 'GODEEEP-'+res:
                bias_dict[col] = col
            else:
                bias_dict[col] = col+', bias = '+str(bias.loc['mean',col])+biasunit
        ddict = plotdd.to_dict('list')
        x = np.arange(len(xaxis))
        width = 0.15
        multiplier = 0
        fig, ax = plt.subplots(figsize=(yfig, 6))
        for dataset,cf in ddict.items():
            offset = width*multiplier
            rects = ax.bar(x + offset,cf,width,label=bias_dict[dataset])
            multiplier+=1
        ax.set_ylabel(ylab+energyunit,fontsize=16)
        ax.set_title(BA+' '+ylab+' '+res.capitalize(),fontsize=18)
        ax.set_xticks(x + width, xaxis)
        ax.set_xticklabels(xaxis,rotation=45, fontsize=14)
        if compute_cf:
            ax.set_ylim([0,0.85])
        ax.tick_params(axis='y',labelsize=14)
        ax.legend(loc='upper left',fontsize=14)
        ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                    alpha=0.5)
        plt.tight_layout()
        if savef:
            plt.savefig(plotdir+r'/'+BA+'_'+freq+'_'+res+'_'+cf_fig+'.png')
        if showf:
            plt.show()
        plt.close(fig)
    stats_df.to_csv(tabledir+r'/infrastructure_'+summarize+'_'+freq+'_'+res+'_'+cf_fig+'.csv')
    return

def standard_scale(x):
    return (x-np.mean(x))/np.std(x)

wecc = [ 'BPAT', 'CISO', 'IPCO', 'NWMT','PACE', 'PACW', 'PNM', 'PSCO', 'PSEI', 'WACM', 'WAUW', 'AZPS', 'LDWP', 'NEVP',
         'SRP', 'TEPC', 'BANC', 'GWA', 'EPE', 'PGE', 'IID', 'AVA', 'WWA', 'WALC', 'GRID', 'AVRN']
eic = ['AECI', 'ISNE', 'MISO', 'NYIS', 'PJM', 'SWPP', 'TVA', 'NBSO', 'CPLE', 'FPL', 'DUK', 'JEA', 'GVL', 'FPC', 'SEPA',
        'TEC', 'CPLW', 'SOCO', 'SC', 'FMPP', 'SPA', 'LGEE', 'SCEG', 'TAL']
erco = ['ERCO']
nerc_zones = ['wecc','eic','erco']

def plot_correlation(all_power_hourly,eia860m_long,freq,res,these_bas,outdir,compute_cf,comparison_datasets,mask_bad_data,agg,s_agg,savef=False,showf=False):
    tabledir = outdir+r'/tables'
    if not os.path.exists(tabledir):
        os.makedirs(tabledir)
    if savef:
        plotdir = outdir+r'/plots'
        if not os.path.exists(plotdir):
            os.makedirs(plotdir)
    if agg == 'by BA':
        mbd = ''
        a_n = 'byba'
        stats_df = pd.DataFrame(columns=['r2'],index=these_bas)
        for BA in these_bas:
            fig, ax = plt.subplots(figsize=(6, 6))
            # print(BA, res, comparison_datasets)
            dd = all_power_hourly[res][BA].copy(deep=True)[['GODEEEP-'+res]+comparison_datasets]
            thesecols = dd.columns.tolist()
            capacity = eia860m_long[(eia860m_long['balancing authority code']==BA)&(eia860m_long['resource']==res)].groupby(['year','month'])['nameplate capacity (mw)'].sum().reset_index()
            capacity['year-mo'] = capacity[['year','month']].astype(str).agg('-'.join,axis=1)
            dd = dd.merge(capacity,how='left',left_on=[dd.index.year,dd.index.month],right_on=['year','month']).set_index(dd.index)
            dd = dd[thesecols+['nameplate capacity (mw)']]
            dd = dd[(dd.index.year>2006)&(dd.index.year<2021)].fillna(0)
            # hourly cf
            dd_cf = dd[thesecols].div(dd['nameplate capacity (mw)'],axis=0)
            dd_cf = dd_cf.dropna(axis=0)
            plt.scatter(dd_cf['GODEEEP-'+res],dd_cf[comparison_datasets[0]],alpha=0.6,c='grey')#,label=BA)
            plt.xlabel('GODEEEP-'+res.capitalize())
            plt.ylabel(comparison_datasets[0].upper())
            plt.title(BA+' '+res.capitalize()+' Hourly Variability')
            plt.show()
            # average r2 across all days of hourly cf
            # print(dd_cf.head())
            daily_r = []
            for g in list(dd_cf.groupby([dd_cf.index.year,dd_cf.index.month,dd_cf.index.day])):
                if ~(g[1].sum() == 0).any():
                    daily_r.append(calc_r(g[1],'',res,comparison_datasets[0])[0])
            # return(dd_cf)
            print('Daily Correlation Coef in '+BA+': ',np.average(np.array(daily_r)))
            stats_df.loc[BA] = np.average(np.array(daily_r))
            plt.hist(np.array(daily_r),
                    density=True,bins=60,alpha=.9)
            plt.title(BA+' '+res.capitalize()+' Daily $r$')
            plt.show()
    elif agg == 'by NERC region':
        a_n = 'byregion'
        stats_df = pd.DataFrame(columns=pd.MultiIndex.from_product([nerc_zones,['GODEEEP-'+res]+comparison_datasets]),index=list(range(2007,2021)))
        # subset these_bas by interconnect
        for bas_in_nerc,nerc_name in zip([wecc,eic,erco],nerc_zones):
            fig, ax = plt.subplots(figsize=(6, 6))
            plot_bas = [i for i in bas_in_nerc if i in these_bas]
            nerc_wide = pd.DataFrame(columns=['GODEEEP-'+res]+comparison_datasets+['nameplate capacity (mw)'],
                                    index=pd.date_range(start='2007-01-01',end='2021-01-01',inclusive='left',freq='1H',tz='UTC').values.astype('datetime64[s]'))
            nerc_eachba = np.zeros(list(nerc_wide.shape)+[len(plot_bas)]) 
            for BA,ba_idx in zip(plot_bas,range(len(plot_bas))):
                dd = all_power_hourly[res][BA].copy(deep=True)[['GODEEEP-'+res]+comparison_datasets]
                thesecols = dd.columns.tolist()
                capacity = eia860m_long[(eia860m_long['balancing authority code']==BA)&(eia860m_long['resource']==res)].groupby(['year','month'])['nameplate capacity (mw)'].sum().reset_index()
                capacity['year-mo'] = capacity[['year','month']].astype(str).agg('-'.join,axis=1)
                dd = dd.merge(capacity,how='left',left_on=[dd.index.year,dd.index.month],right_on=['year','month']).set_index(dd.index)
                dd = dd[thesecols+['nameplate capacity (mw)']]
                dd = dd[(dd.index.year>2006)&(dd.index.year<2021)].fillna(0)
                annual_cf = dd.resample(freq).sum().fillna(0)
                annual_cf[thesecols] = annual_cf[thesecols].div(annual_cf['nameplate capacity (mw)'],axis=0)
                annual_cf = annual_cf.dropna(axis=0)
                if mask_bad_data == True:
                    mbd = '_masked'
                    # identify which year(s) in this ba have a cf < 0.05
                    # 1. find the years for this BA -- create a new frame from dd/don't overwrite dd
                    low_cf_yr_idx_annual = annual_cf.loc[(annual_cf<0.05).any(axis=1)].index.year
                    annual_cf = annual_cf.loc[(annual_cf[thesecols]>=0.05).any(axis=1)]
                    monthly_cf = dd.groupby([dd.index.year,dd.index.month]).sum().fillna(0)
                    monthly_cf[thesecols] = monthly_cf[thesecols].div(monthly_cf['nameplate capacity (mw)'],axis=0)
                    monthly_cf = monthly_cf.fillna(0)
                    low_cf_mo_idx,low_cf_yr_idx = monthly_cf.loc[(monthly_cf==0).any(axis=1)].index.get_level_values(1),monthly_cf.loc[(monthly_cf==0).any(axis=1)].index.get_level_values(0)
                    # 2. mask these years as 0 production in 'dd'
                    dd.loc[((dd.index.year.isin(low_cf_yr_idx))&(dd.index.month.isin(low_cf_mo_idx)))|
                        (dd.index.year.isin(low_cf_yr_idx_annual)),thesecols] = 0
                else:
                    mbd = ''
                # 3. place the 'dd' in nerc_eachba
                nerc_eachba[:len(dd),:,ba_idx] = dd
                if (res == 'solar') & (nerc_name=='wecc'):
                    corr_coef,slope,intercept = calc_r(annual_cf[thesecols].astype(float),'',res,comparison_datasets[0])
                plt.scatter(annual_cf['GODEEEP-'+res],annual_cf[comparison_datasets[0]],alpha=0.6,c='grey')#,label=BA)
            nerc_wide.iloc[:,:] = nerc_eachba.sum(axis=2)
            nerc_wide = nerc_wide.resample(freq).sum()
            nerc_wide = nerc_wide[nerc_wide['nameplate capacity (mw)']>0]
            nerc_wide[thesecols] = nerc_wide[thesecols].div(nerc_wide['nameplate capacity (mw)'],axis=0)
            nerc_wide = nerc_wide.loc[(nerc_wide[thesecols]>=0.05).any(axis=1)]
            nerc_wide.index = nerc_wide.index.year
            stats_df[nerc_name] = nerc_wide[thesecols].round(2)
            plt.scatter(nerc_wide['GODEEEP-'+res],nerc_wide[comparison_datasets[0]])
            corr_coef,slope,intercept = calc_r(nerc_wide[thesecols].astype(float),'',res,comparison_datasets[0])
            xax = np.linspace(0, 1.0, num=100)
            plt.plot(xax, intercept + slope*xax, 'r', label='$r$ = '+str(corr_coef.round(2)))
            plt.legend()
            ax.set_xlabel('GODEEEP-'+res.capitalize()+' Average Annual Capacity Factor',fontsize=12)
            ax.set_xlim([0,0.6])
            ax.set_ylabel(comparison_datasets[0].upper()+' Average Annual Capacity Factor',fontsize=12)
            ax.set_ylim([0,0.6])
            ax.set_title(nerc_name.upper()+' '+res.capitalize()+' Inter-Annual Variability',fontsize=18)
            ax.grid(which='minor', alpha=0.2)
            ax.grid(which='major', alpha=0.8)    
            plt.tight_layout()
            if savef:
                plt.savefig(plotdir+r'/'+nerc_name+'_'+res+'_'+'correlation'+mbd+'.png')
            if showf:
                plt.show()
            plt.close(fig) 
    stats_df.to_csv(tabledir+r'/cf_'+res+mbd+'_'+a_n+'.csv')
    return


#%%
def calc_pdf(data, smooth=None):
    hist = np.histogram(data, bins=100)
    hist_dist = stats.rv_histogram(hist)
    X = np.linspace(0, max(data), 100)
    Y = hist_dist.pdf(X)
    return X, Y


# In[ ]:
# good explanation of why to use EMD:
# https://jeremykun.com/2018/03/05/earthmover-distance/
# https://blog.demofox.org/2022/07/27/calculating-the-similarity-of-histograms-or-pdfs-using-the-p-wasserstein-distance/
# we want a similarity metric where if we separate these histograms with more zeros, it tells us that they are less similar, and if we remove zeros or make them overlap, it tells us that they are more similar.


def calc_emd(data1, data2):
    pdf1, pdf2, emd = [], [], []
    timeslice = np.intersect1d(data1.columns,data2.columns)
    for ts in timeslice:
        _data1, _data2 = data1[ts],data2[ts]
        X, _pdf1 = calc_pdf(_data1)
        X, _pdf2 = calc_pdf(_data2)
        if _pdf1.size==0: continue
        _emd  = stats.wasserstein_distance(_pdf1, _pdf2)
        pdf1.append(_pdf1)
        pdf2.append(_pdf2)
        emd.append(_emd)
    pdf1 = np.stack(pdf1, axis=1)
    pdf2 = np.stack(pdf2, axis=1)
    emd  = np.array(emd)
    return pdf1, pdf2, emd

#%%
# https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
def reject_outliers(data, m=3.5):
    return data[abs(data - np.mean(data)) <= m * np.std(data)]

# In[ ]:
def get_emd_scada(all_power_hourly,eia860m_long,balist,res,dates,comp,tabledir,normalize=False,robust=True):
    yr,mo,da = dates[0],dates[1],dates[2]
    emd_scada = []
    emd_scada_robust = []
    for bb in balist:
        dd = all_power_hourly[res][bb].copy(deep=True)
        df_emd = dd.loc[(dd.index.year.isin(yr))&(dd.index.month.isin(mo))&(dd.index.day.isin(da)),['GODEEEP-'+res,'scada']]
        if df_emd.isnull().values.any():
            df_emd = df_emd.dropna()
        idx =  pd.MultiIndex.from_arrays([df_emd.index.time, df_emd.index.floor('D')])
        pt1 = df_emd.set_index(idx)['GODEEEP-'+res].unstack()
        pt2 = df_emd.set_index(idx)[comp].unstack()
        pt1 = pt1.dropna(axis='columns')
        pt2 = pt2.dropna(axis='columns')
        pt1.to_csv(tabledir+r'/emd_gd_'+res+'_'+bb+'.csv')
        pt2.to_csv(tabledir+r'/emd_sc_'+res+'_'+bb+'.csv')
        emd = calc_emd(pt1,pt2)[2]
        if normalize:
            capacity = eia860m_long[(eia860m_long['balancing authority code']==bb)&(eia860m_long['resource']==res)].groupby(['year','month'])['nameplate capacity (mw)'].sum().reset_index()
            capacity['year-mo'] = capacity[['year','month']].astype(str).agg('-'.join,axis=1)
            capacity['days'] = capacity.apply(lambda x: calendar.monthrange(x['year'],x['month'])[1],axis=1)
            capacity = capacity.reindex(capacity.index.repeat(capacity.days)).reset_index(drop=True)
            startday = len(capacity) - len(emd)
            emd = emd/capacity.loc[startday:,'nameplate capacity (mw)']*100
        else:
            emd = emd*1e3
        emd_scada.append(emd)
        emd_scada_robust.append(reject_outliers(emd))
    if robust:
        return emd_scada_robust
    else:
        return emd_scada


def plot_emd(all_power_hourly,eia860m_long,dates,res,these_bas,outdir,compute_cf,comparison_datasets,mask,agg,s_agg,savef=False,showf=False):
    tabledir = outdir+r'/tables'
    if not os.path.exists(tabledir):
        os.makedirs(tabledir)
    if savef:
        plotdir = outdir+r'/plots'
        if not os.path.exists(plotdir):
            os.makedirs(plotdir)
    emd = dict(zip(these_bas,get_emd_scada(all_power_hourly,eia860m_long,these_bas,res,dates,comparison_datasets[0],tabledir,normalize=compute_cf,robust=True)))
    # return get_emd_scada(all_power_hourly,eia860m_long,these_bas,res,dates,comparison_datasets[0],normalize=compute_cf,robust=True)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
    labels,data = [*zip(*emd.items())]
    for l,d in zip(labels,data):
        pd.DataFrame(d).to_csv(tabledir+r'/'+l+'_'+res+'_raw_daily_emd.csv')
    ax1.boxplot(data,notch=False, sym='+', vert=True, whis=1.5)
    plt.yscale('log')
    plt.xticks(range(1, len(labels) + 1), labels)
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                    alpha=0.5)
    ax1.set(axisbelow=True)
    ax1.set_xticklabels(emd.keys(),
                        rotation=30, fontsize=16)
    if compute_cf:
        unit = ' [% of Capacity]'
        filen = 'pct'
        ax1.set_ylim([1e-10,50])
    else:
        unit = ' [kWh]'
        filen = 'kwh'
    ax1.set_ylabel('Earth Mover Distance'+unit,fontsize=16)
    ax1.set_title(res.capitalize()+' Intra-Day Hourly Variability',fontsize=18)
    ax1.tick_params(axis='y',labelsize=14)
    plt.tight_layout()
    if savef:
        plt.savefig(plotdir+r'/emd_'+res+'_'+filen+'.png')
    if showf:
        plt.show()
    plt.close()
    emd_means = pd.DataFrame(index=these_bas,columns=[res.capitalize()])
    emd_medians = pd.DataFrame(index=these_bas,columns=[res.capitalize()])
    for e in emd.keys():
        emd_means.loc[e,res.capitalize()] = np.mean(emd[e]).round(3)
        emd_medians.loc[e,res.capitalize()] = np.median(emd[e]).round(3)
    emd_means.to_csv(tabledir+r'/emd_means_robust_'+filen+'_'+res+'.csv')
    emd_medians.to_csv(tabledir+r'/emd_medians_robust_'+filen+'_'+res+'.csv')
    return

#%%
# CALCULATE STATISTICS FOR PAPER -- need to fix -- these are calculating something different
# than the annual and seasonal bias
def calc_rmse(thisgroup,_,comp,res):
    return math.sqrt(np.square(np.subtract(thisgroup['GODEEEP-'+res],thisgroup[comp[0]])).mean())

def calc_r(g,_,res,comp):
    result = stats.linregress(g['GODEEEP-'+res],g[comp])
    return result.rvalue,result.slope,result.intercept


def calc_diff(thisgroup,summarize,comp,res):
    # average pct diff across the frequency interval
    if summarize == 'pct':
        diff = pd.Series(index=comp)
        for cc in comp:
            subset_group = thisgroup[['GODEEEP-'+res,cc]].dropna(axis=0)
            if (len(subset_group)>0) and (subset_group[cc].sum()!=0):
                diff[cc] = (subset_group['GODEEEP-'+res].sum() - subset_group[cc].sum())/subset_group[cc].sum()
            else:
                diff[cc] = np.nan
        # print(diff)
        return diff
    else:
        gd = pd.concat([thisgroup['GODEEEP-'+res]]*len(comp),axis=1)
        return (gd.values - thisgroup[comp]).apply(summarize)
