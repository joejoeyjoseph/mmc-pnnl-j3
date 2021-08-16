import xarray as xr
import wrf
from netCDF4 import Dataset
import os
import glob

year_list = [2018, 2019]

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

print('starting')

for year in year_list: 

    year = str(year)

    file_dir = '/gpfs/wolf/atm118/proj-shared/jbk/wrfstat/'+year
    file_list = glob.glob(file_dir+'/*/*/*.nc')
    print(len(file_list))

    out_dir = '/ccsopen/home/leec813/trim_lasso_wrfstat/'

    for nc_file in file_list:

        nc_file_dir = os.path.dirname(nc_file)

        print(nc_file_dir) 
        print(nc_file)

        wip_file = out_dir+'test.nc'
        out_file = out_dir+nc_file[-34:-27]+nc_file_dir[-1]+nc_file[-27:]

        # drop data below x meters
        height_thres = 500

        csp_head = 'CSP_'
        csp_var = ['Z', 'U', 'V', 'P', 'THV', 'TH', 'QV', 'QC', 'U2', 'V2', 'UW', 'VW', 'THV2', 'QV2',
                   'THDT_LS', 'QVDT_LS', 'UDT_LS', 'VDT_LS', 'W', 'W2']
        csp_list = [csp_head+i for i in csp_var]

        cst_head = 'CST_'
        cst_var = ['UST', 'FSNS', 'FLNS', 'SH', 'LH', 'PS', 'PRECT', 'CLDTOT', 'LWP']
        cst_list = [cst_head+i for i in cst_var]

        all_var_list = csp_list+cst_list+['XLAT', 'XLONG', 'XTIME', 'Times']

        ds = xr.open_dataset(nc_file)

        # create dummy XLAT XLONG
        ds = ds.assign(XLAT=lambda ds: ds.CSS_IWP)
        ds = ds.assign(XLONG=lambda ds: ds.CSS_IWP)

        if os.path.isfile(wip_file):
            os.remove(wip_file)

        ds[all_var_list].to_netcdf(wip_file)

        ds_wip = ds[all_var_list]

        wrft = Dataset(wip_file, 'r')

        csp_w = wrf.getvar(wrft, 'CSP_W', timeidx=wrf.ALL_TIMES)
        csp_w2 = wrf.getvar(wrft, 'CSP_W2', timeidx=wrf.ALL_TIMES)

        csp_w_arr = wrf.destagger(csp_w, 1)
        csp_w2_arr = wrf.destagger(csp_w2, 1)

        ds_wip['CSP_W'] = (('Time', 'bottom_top'), csp_w_arr)
        ds_wip['CSP_W2'] = (('Time', 'bottom_top'), csp_w2_arr)

        ds_out = ds_wip.sel(bottom_top=slice(0, ds_wip['CSP_Z'].where(ds_wip['CSP_Z'][1, :] < 500, drop=True).shape[1]))

        ds_out = ds_out.drop(['XLAT', 'XLONG'])

        if os.path.isfile(out_file):
            os.remove(out_file)

        ds_out.to_netcdf(out_file)

        os.remove(wip_file)

print('finish')
