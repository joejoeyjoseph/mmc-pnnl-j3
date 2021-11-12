import os
os.environ['PROJ_LIB'] = '/home/jlee/.conda/envs/mmc_sgp/share/proj'

import glob
import xarray as xr
import wrf
from netCDF4 import Dataset
import numpy as np
import pandas as pd

file_dir = '/projects/wfip2les/cdraxl/2020100300/'
# file_dir = '/home/jlee/wfip/test_case/'
out_dir = '/home/jlee/wfip/'

file_list = glob.glob(file_dir+'custom_wrfout_d03*')
file_list.sort()

base = Dataset((file_dir+'wrfout_d03_2020-10-03_12_00_00'), 'r')

fino_ij = wrf.ll_to_xy(base, 55.006928, 13.154189)
baltic_ij = wrf.ll_to_xy(base, 54.9733, 13.1778)

hgt = wrf.g_geoht.get_height(base, msl=False)

def get_target_ws(hgt_list, target_hgt, ws):
    
    near_z = min(enumerate(hgt_list), key=lambda x: abs(x[1]-target_hgt))
    
    alpha = np.log(ws[near_z[0]+1]/ws[near_z[0]])/np.log(hgt_list[near_z[0]+1]/near_z[1])

    target_ws = ws[near_z[0]]*(target_hgt/near_z[1])**alpha
    
    return np.round(target_ws, 4)

fino_df = pd.DataFrame(columns=['time', 'wind-speed_62m', 'wind-speed_72m', 'wind-speed_82m', 'wind-speed_92m'])
baltic_df = pd.DataFrame(columns=['time', 'wind-speed_78-25m'])

for file in file_list:
    
    print(file)
    
    fino_hgt = hgt[:, fino_ij[1], fino_ij[0]]

    ds = xr.open_dataset(file)

    one_time = ds['Times'].values[0].decode('utf-8').replace('_', ' ')
    
    #####
    
    u_fino = ds['U'][:, :, fino_ij[1], int(fino_ij[0].values):int(fino_ij[0].values)+2].squeeze().mean(axis=1)
    v_fino = ds['V'][:, :, int(fino_ij[1].values):int(fino_ij[1].values+2), fino_ij[0]].squeeze().mean(axis=1)
    ws_fino = np.sqrt(u_fino**2 + v_fino**2)

    ws62 = get_target_ws(fino_hgt, 62, ws_fino)
    ws72 = get_target_ws(fino_hgt, 72, ws_fino)
    ws82 = get_target_ws(fino_hgt, 82, ws_fino)
    ws92 = get_target_ws(fino_hgt, 92, ws_fino)

    dum_fino_df = pd.DataFrame(data={'time': [one_time], 'wind-speed_62m': [ws62.values], 
                                     'wind-speed_72m': [ws72.values], 'wind-speed_82m': [ws82.values], 
                                     'wind-speed_92m': [ws92.values]})

    fino_df = fino_df.append(dum_fino_df)
    
    #####
    
    baltic_hgt = hgt[:, baltic_ij[1], baltic_ij[0]]
    
    u_baltic = ds['U'][:, :, baltic_ij[1], 
                       int(baltic_ij[0].values):int(baltic_ij[0].values)+2].squeeze().mean(axis=1)
    v_baltic = ds['V'][:, :, int(baltic_ij[1].values):int(baltic_ij[1].values+2), 
                       baltic_ij[0]].squeeze().mean(axis=1)
    ws_baltic = np.sqrt(u_baltic**2 + v_baltic**2)
    
    ws7825 = get_target_ws(baltic_hgt, 78.25, ws_baltic)
    
    dum_baltic_df = pd.DataFrame(data={'time': [one_time], 'wind-speed_78-25m': [ws7825.values]})

    baltic_df = baltic_df.append(dum_baltic_df)
    
fino_df.set_index('time', inplace=True)
fino_df.to_csv(out_dir+'nrel_fino.csv')

baltic_df.set_index('time', inplace=True)
baltic_df.to_csv(out_dir+'nrel_baltic.csv')
