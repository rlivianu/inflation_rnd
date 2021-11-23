from src import *

path = 'C:\\Users\\pmxph7\\OneDrive - The University of Nottingham\\PhD\\inflation\\us_data'
fs = [path + '\\' + str(i) + 'Y.csv' for i in range(1, 11)]
swap_fs = path + '\\' + 'SWIL.csv'
ois_fs = path + '\\' + 'OIS.csv'

data = InflationData(fs, swap_fs, ois_fs)
data.compute_implied_vol()
data.fit_splines()
data.timplied_vols[0].head()
