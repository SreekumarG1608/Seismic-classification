import librosa as lib
import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt

file = pd.read_csv("20HZ.txt")
file = pd.DataFrame(file)
time_series = file[['TIMESERIES XA_BHW01__HH_D']].copy()
time_series.rename(columns = {'TIMESERIES XA_BHW01__HH_D':'Amp'}, inplace = True)
N=len(file.index)
time_series['Time'] = np.linspace(0,(N-1)/100,N)

#convert 100 sps to 20 sps
index_20Hz = [*filter(lambda x: x%5==0,range(0,len(time_series.index)))]
time_series_20hz = time_series.iloc[index_20Hz].copy()


#Assigning ids, event-111 noise-000
np_time = time_series['Time']
np_label = []
for val in np_time:
    if(val>17 and val<36):
        np_label.append('111')
    else:
        np_label.append('000')
time_series['label'] = np_label
#select_range = time_series.loc[time_series['Time']<133]

#tsfresh feature extraction
from tsfresh import extract_features
if __name__ == '__main__':
    extracted_features_timeseries = extract_features(timeseries_container=time_series,column_id='label', column_sort='Time', n_jobs=8)
    extracted_features_timeseries = extracted_features_timeseries.rename_axis('Label Ids')
    print(extracted_features_timeseries)
    extracted = extracted_features_timeseries