import pandas as pd
import create_generation_timeseries as cgt
import os


def make_formatted_timeseries(
        datadir, eia923, eia930, ba_res_list, gd_timeseries, make_plant_dicts, make_BA_dicts, save_BA_dicts,
        eia860m_long, solar_configs, wind_configs):
  slist_all = ba_res_list['solar']
  wlist_all = ba_res_list['wind']

  wlist = ['ERCO', 'ISNE', 'SWPP', 'CISO', 'BPAT', 'MISO', 'NYIS']
  slist = ['ERCO', 'ISNE', 'SWPP', 'CISO']
  BA_list = list(set(slist_all+wlist_all))
  eia860m = {}
  for thisBA in BA_list:
    eia860m[thisBA] = eia860m_long[(eia860m_long['balancing authority code'] == thisBA)]

  # MAKE PLANT-LEVEL GENERATION TIME SERIES
  firstdate = '2007-01-01 01:00:00'
  monthyears = cgt.get_monthyear(firstdate)

  # create dictionaries for each power plant
  all_plant_gen = {}
  for res in ['solar', 'wind']:
    if make_plant_dicts:
      all_plant_gen[res] = cgt.create_plantlevel_generation(
          gd_timeseries[res], eia860m_long, res, monthyears, solar_configs, wind_configs, datadir)
    else:
      all_plant_gen[res] = pd.read_csv(
          datadir+r'/historical_power/'+res+'_plant_generation.csv', index_col='datetime', parse_dates=True)

  # use all_plant_gen dict to create the BA level generation using the plant_gen_id
  all_power_hourly = {}
  if make_BA_dicts:
    if 'all_plant_gen' in locals():
      print('Reading scada data')
      for res, balist, sublist in zip(['solar', 'wind'], [slist_all, wlist_all], [slist, wlist]):  # 'wind',wlist,
        print(res)
        all_power_hourly[res] = cgt.create_balevel_generation(
            all_plant_gen[res],
            res, balist, gd_timeseries[res],
            eia930[res],
            eia923[res],
            eia860m, sublist, monthyears, datadir, save_BA_dicts)
      print('completed creation of resource profile dict')
    else:
      print('need all_plant_gen dict to create BA dict \n')
      print('set make_plant_dicts to True')
    if save_BA_dicts:
      for res in ['solar', 'wind']:
        bas = all_power_hourly[res].keys()
        # index_tosave = pd.DataFrame(index=all_power_hourly['solar']['TEC'].index)
        for b in bas:
          # godeeep_power = pd.DataFrame(all_power_hourly[res][b]['GODEEEP-'+res]).rename(columns={'GODEEEP-'+res: b})
          # df_tosave = index_tosave.merge(godeeep_power, how='left', left_index=True, right_index=True)
          # df_tosave.to_csv(datadir+r'/historical_power/'+b+'_'+res+'_generation.csv', index_label='datetime')
          fn = datadir+r'/historical_power/'+b+'_'+res+'_generation.csv'
          all_power_hourly[res][b].to_csv(fn, index_label='datetime')
  else:
    all_power_hourly = {}
    for res, balist in zip(['solar', 'wind'], [slist_all, wlist_all]):
      BA_hr_diff = {}
      for thisBA in balist:
        fn = datadir+r'/historical_power/'+thisBA+'_'+res+'_generation.csv'
        if os.path.exists(fn):
          BA_hr_diff[thisBA] = pd.read_csv(fn, index_col='datetime', parse_dates=True)
      all_power_hourly[res] = BA_hr_diff

  # check for negative production
  for r in ['wind', 'solar']:
    for ba in all_power_hourly[r].keys():
      if (all_power_hourly[r][ba]['GODEEEP-'+r] < 0).any():
        print(r, ba)

  for r in wlist:
    all_power_hourly['wind'][r].columns = ['eia930', 'eia923', 'GODEEEP-wind', 'scada']
  for r in slist:
    all_power_hourly['solar'][r].columns = ['eia930', 'eia923', 'GODEEEP-solar', 'scada']

  return all_power_hourly
