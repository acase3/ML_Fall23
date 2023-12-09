
from obspy.signal.trigger import plot_trigger


from obspy.signal.trigger import classic_sta_lta


import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import obspy
import h5py
from obspy import UTCDateTime
import numpy as np
from obspy.clients.fdsn.client import Client
import matplotlib.pyplot as plt

file_name = "merged.hdf5"
csv_file = "merged.csv"

def make_plot(tr, title='', ylab=''):
    '''
    input: trace

    '''

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(tr.times("matplotlib"), tr.data, "k-")
    ax.xaxis_date()
    fig.autofmt_xdate()
    #plt.ylabel('counts')
    #plt.title('Raw Data')
    plt.show()

def make_stream(dataset):
    '''
    input: hdf5 dataset
    output: obspy stream

    '''
    data = np.array(dataset)

    tr_Z = obspy.Trace(data=data[:, 2])
    tr_Z.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_Z.stats.delta = 0.01
    tr_Z.stats.channel = dataset.attrs['receiver_type'] + 'Z'
    tr_Z.stats.station = dataset.attrs['receiver_code']
    tr_Z.stats.network = dataset.attrs['network_code']


    stream = obspy.Stream([tr_Z])

    return stream


def makeTrace(dataset):
    data = np.array(dataset)

    tr_Z = obspy.Trace(data=data[:])
    tr_Z.stats.delta = 0.01
    tr_Z.stats.sampling_rate = 100

    return tr_Z




#5.12 second waveforms of p wave arrivals - want 512 counts
#trace starttime + (pwavesample/100) = p wave arrival time
#if the model looks at waveforms every 5 seconds, we cant garentee that the p wave will arraive at the beginning of those 5 sec
#random 2 numbers that add to 5 --- -pwave arrival time is start of wave form + pwave arrival time is end = 5sec
######
#features: displacement waveforms, maximum amp/average amp, 10 dominant frequencies in the waveform
######
np_data_array = np.empty((0, 512))
#####
# reading the csv file into a dataframe:
df_full = pd.read_csv(csv_file)
print(f'total events in csv file: {len(df_full)}')
# filterering the dataframe
df = df_full[(df_full.trace_category == 'earthquake_local') & (df_full.source_distance_km <= 500) & (df_full.source_magnitude > 2)]
df_noise = df_full[(df_full.trace_category == 'noise')]
print(f'total events selected: {len(df)}')
print(f'total events selected: {len(df_noise)}')
# making a list of trace names for the selected data
ev_list = df['trace_name'].to_list()[18440:22298]
noise_list = df_noise['trace_name'].to_list()[11540:15398]
#print(len(noise_list))
# retrieving selected waveforms from the hdf5 file:
dtfl = h5py.File(file_name, 'r')

y_pred = np.empty(len(ev_list))
y_real = np.empty(len(ev_list))
for c, evi in enumerate(ev_list):
    dataset = dtfl.get('data/' + str(evi))
    # waveforms, 3 channels: first row: E channel, second row: N channel, third row: Z channel
    data = np.array(dataset)
    #print(np.shape(data))
    import random
    p_arrival = dataset.attrs['p_arrival_sample']
    x = random.randint(50, 450)
    while p_arrival - x < 1:  # if the random x is too early, pick another
        x = random.randint(50, 450)
    y = int(500 - x)
    start = int(dataset.attrs['p_arrival_sample'] - x)
    end = int(dataset.attrs['p_arrival_sample'] + y)
    wave = data[start:end, 2]
    print(len(wave))
    #print(np.shape(wave))
    st_D = makeTrace(wave)
    cft = classic_sta_lta(st_D.data, int(1 * 100), int(2 * 100))
    print('go!')
    #print(cft)
    #plot_trigger(st_D, cft, 1.5, 0.5)
    threshold = 1.5 #you have ot set this manually...it means that you have to tune the threshold to each station :(
    # 1 == EQ, 0 == noise
    if max(cft) >= threshold:
        y_pred[c] = 1
        y_real[c] = 1
    else:
        y_pred[c] = 0
        y_real[c] = 1
    w = c / 100
    if w.is_integer() == True:
        print('another 100 done!')
        print(c)


df_y_pred = pd.DataFrame(y_pred)
df_y_pred.to_csv('EQ_y_predf.csv')
df_y_real = pd.DataFrame(y_real)
df_y_real.to_csv('EQ_y_realf.csv')

y_predn = np.empty(len(noise_list))
y_realn = np.empty(len(noise_list))

for c, evi in enumerate(noise_list):
    dataset = dtfl.get('data/' + str(evi))
    # waveforms, 3 channels: first row: E channel, second row: N channel, third row: Z channel
    data = np.array(dataset)
    #print(np.shape(data))
    import random
    x = random.randint(0, 2000)
    y = int(x + 500)
    start = int(x)
    end = int(y)
    wave = data[start:end, 2]
    #print(np.shape(wave))
    st_D = makeTrace(wave)
    cft = classic_sta_lta(st_D.data, int(1 * 100), int(2 * 100))
    #print(cft)
    #plot_trigger(st_D, cft, 1.5, 0.5)
    threshold = 1.5 #you have ot set this manually...it means that you have to tune the threshold to each station :(
    # 1 == EQ, 0 == noise
    if max(cft) >= threshold:
        y_predn[c] = 1
        y_realn[c] = 0
    else:
        y_predn[c] = 0
        y_realn[c] = 0
    w = c / 100
    if w.is_integer() == True:
        print('another 100 done!')
        print(c)

df_y_predn = pd.DataFrame(y_predn)
df_y_predn.to_csv('EQ_y_prednf.csv')
df_y_realn = pd.DataFrame(y_realn)
df_y_realn.to_csv('EQ_y_realnf.csv')
#comparing how they do
