#fix the issues with labelling the rows (make is applicable for all the data)


import librosa as lib
import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt

#reads csv file and labels the amplitude column   
file = pd.read_csv("ASCII_3.txt")
file = pd.DataFrame(file)
time_series = file.loc[[0]].copy()
time_series.columns = ['Amp'] 


#adds column of time(in seconds)
N=len(file.index)
time_series['Time'] = np.linspace(0,(N-1)/100,N)

#plots the data
plt.plot(time_series['Time'],time_series['Amp'])
plt.title('Amplitude Vs Time')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()


