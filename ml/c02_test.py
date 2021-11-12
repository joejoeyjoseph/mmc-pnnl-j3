import xarray as xr 
import wrf
from netCDF4 import Dataset
import os
import glob

year_list = [2018]

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

print('starting')

for year in year_list:

    year = str(year)

file_dir = '/gpfs/wolf/atm118/proj-shared/jbk/wrfstat/'+year
file_list = glob.glob(file_dir+'/*/*/*.nc')
# nc_file = file_dir+'wrfstat_d01_2018-05-14_12:00:00.nc'

print(file_list[0])
print(os.path.dirname(file_list[0]))
print(os.path.dirname(file_list[0])[-1])

out_dir = '/ccsopen/home/leec813/trim_lasso_wrfstat/'
wip_file = out_dir+'test.nc'
out_file = out_dir+'out.nc'


