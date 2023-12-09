
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
ev_list = df['trace_name'].to_list()[:27000]
noise_list = df_noise['trace_name'].to_list()[20000:41000]
#print(len(noise_list))
# retrieving selected waveforms from the hdf5 file:
dtfl = h5py.File(file_name, 'r')


for c, evi in enumerate(noise_list):
    dataset = dtfl.get('data/' + str(evi))
    # waveforms, 3 channels: first row: E channel, second row: N channel, third row: Z channel
    data = np.array(dataset)
    st_D = make_stream(dataset)

    client = Client("IRIS")
    try:
        inventory = client.get_stations(network=dataset.attrs['network_code'],
                                        station=dataset.attrs['receiver_code'],
                                        starttime=UTCDateTime(dataset.attrs['trace_start_time']),
                                        endtime=UTCDateTime(dataset.attrs['trace_start_time']) + 60,
                                        loc="*",
                                        channel="*",
                                        level="response")
        st_D = st_D.remove_response(inventory=inventory, output="DISP", plot=False)
    except:
        continue
    else:
        import random

        x = random.randint(0, 2000)
        # x = 250
        # print(x)
        y = int(x+500)
        # print(x+y)
        st_D = st_D.remove_response(inventory=inventory, output="DISP", plot=False)
        start = int(x)
        end = int(y)
        # print(len(st_D[2].data[start:end]))
        wave = st_D[0].data[start:end]

        maxvalD = max(abs(wave))
        avevalD = sum(wave) / len(wave)  # this is a feature!
        amp_val = maxvalD / avevalD
        minvalD = abs(min(wave))
        normvalD = max(maxvalD, minvalD)
        normD = wave

        from scipy.fft import fft, fftfreq

        # Number of samples in normalized_tone
        N = 500

        yf = fft(normD)
        xf = fftfreq(N, 1 / 100)
        #plt.plot(xf, np.abs(yf))
        #plt.show()
        yf_pos = yf[0:250]
        xf_pos = xf[0:250]
        index = np.argsort(yf_pos)[-10:]
        #print(index)
        temp_freq = np.empty(10)
        for h in np.arange(0, len(index), 1):
            temp_freq[h] = xf_pos[index[h]]
        #print(temp_freq)
        temp_array = np.transpose(normD)
        #print(np.shape(temp_array))
        #print(np.shape(temp_freq))
        temp_array = np.append(temp_array, temp_freq)
        temp_array = np.append(temp_array, amp_val)
        label = 'Noise'
        temp_array = np.append(temp_array, label)
        temp_array = np.transpose(temp_array)
        #np_data_array = np.vstack((np_data_array, temp_array))
        w = c / 100
        if w.is_integer() == True:
            print('another 100 done!')
            print(c)
        # print(temp_array.shape)
        #fig = plt.figure()
        np_data_array = np.vstack((np_data_array, temp_array))
        #ax = fig.add_subplot(313)
        #plt.plot(normD, 'k')
        #plt.rcParams["figure.figsize"] = (8, 5)
        #legend_properties = {'weight': 'bold'}
        #plt.tight_layout()
        #ymin, ymax = ax.get_ylim()
        # print(dataset.attrs['p_arrival_sample'])
        # pl = plt.vlines(dataset.attrs['p_arrival_sample'], ymin, ymax, color='b', linewidth=2, label='P-arrival')
        # sl = plt.vlines(dataset.attrs['s_arrival_sample'], ymin, ymax, color='r', linewidth=2, label='S-arrival')
        # cl = plt.vlines(dataset.attrs['coda_end_sample'], ymin, ymax, color='aqua', linewidth=2, label='Coda End')
        # plt.legend(handles=[pl, sl, cl], loc='upper right', borderaxespad=0., prop=legend_properties)
        #plt.ylabel('Amplitude counts', fontsize=12)
        #ax.set_xticklabels([])
        #plt.show()
df_waveforms_n = pd.DataFrame(np_data_array)
df_waveforms_n.to_csv('waveforms_noisenonnorm.csv')
for c, evi in enumerate(ev_list):
    dataset = dtfl.get('data/'+str(evi))
    # waveforms, 3 channels: first row: E channel, second row: N channel, third row: Z channel
    data = np.array(dataset)
    st_D = make_stream(dataset)

    client = Client("IRIS")
    try:
        inventory = client.get_stations(network=dataset.attrs['network_code'],
                                        station=dataset.attrs['receiver_code'],
                                        starttime=UTCDateTime(dataset.attrs['trace_start_time']),
                                        endtime=UTCDateTime(dataset.attrs['trace_start_time']) + 60,
                                        loc="*",
                                        channel="*",
                                        level="response")
        st_D = st_D.remove_response(inventory=inventory, output="DISP", plot=False)

    except:
        print('no station response info...skipping')
        continue
    else:
        import random
        p_arrival = dataset.attrs['p_arrival_sample']
        x = random.randint(1, 450)
        while p_arrival-x < 1: #if the random x is too early, pick another
            x = random.randint(1, 450)
        y = int(500 - x)
        #print(x+y)
        st_D = st_D.remove_response(inventory=inventory, output="DISP", plot=False)
        start = int(dataset.attrs['p_arrival_sample'] - x)
        end = int(dataset.attrs['p_arrival_sample'] + y)
        wave = st_D[0].data[start:end]
        #print(start)
        #print(end)
        #print(len(wave))


        maxvalD = max(abs(wave))
        avevalD = sum(wave)/len(wave) #this is a feature!
        amp_val = maxvalD / avevalD
        minvalD = abs(min(wave))
        normvalD = max(maxvalD, minvalD)
        normD = wave

        from scipy.fft import fft, fftfreq

        # Number of samples in normalized_tone
        N = 500

        yf = fft(normD)
        xf = fftfreq(N, 1 / 100)
        #plt.plot(xf, np.abs(yf))
        #plt.show()
        yf_pos = yf[0:250]
        xf_pos = xf[0:250]
        index = np.argsort(yf_pos)[-10:]
        #print(index)
        temp_freq = np.empty(10)
        for h in np.arange(0,len(index), 1):
            temp_freq[h] = xf_pos[index[h]]
        #print(temp_freq)
        temp_array = np.transpose(normD)
        #print(np.shape(temp_array))
        #print(np.shape(temp_freq))
        temp_array = np.append(temp_array, temp_freq)
        temp_array = np.append(temp_array, amp_val)
        label = 'EQ'
        temp_array = np.append(temp_array, label)
        temp_array = np.transpose(temp_array)

    #print(temp_array.shape)
        #fig = plt.figure()
        np_data_array = np.vstack((np_data_array, temp_array))
        w = c/100
        if w.is_integer() == True:
            print('another 100 done!')
            print(c)
        #ax = fig.add_subplot(313)
        #plt.plot(normD, 'k')
        #plt.rcParams["figure.figsize"] = (8, 5)
        #legend_properties = {'weight': 'bold'}
        #plt.tight_layout()
        #ymin, ymax = ax.get_ylim()
        #print(dataset.attrs['p_arrival_sample'])
        #pl = plt.vlines(dataset.attrs['p_arrival_sample'], ymin, ymax, color='b', linewidth=2, label='P-arrival')
        #sl = plt.vlines(dataset.attrs['s_arrival_sample'], ymin, ymax, color='r', linewidth=2, label='S-arrival')
        #cl = plt.vlines(dataset.attrs['coda_end_sample'], ymin, ymax, color='aqua', linewidth=2, label='Coda End')
        #plt.legend(handles=[pl, sl, cl], loc='upper right', borderaxespad=0., prop=legend_properties)
        #plt.ylabel('Amplitude counts', fontsize=12)
        #ax.set_xticklabels([])
        #plt.show()
        #make_plot(st_A[2], title='Acceleration', ylab='meters/second**2')
print('done...moving on to noise')


print(np_data_array.shape)

#x = np.arange(0,6000)
#plt.plot(x, np_data_array[6,:6000])
#plt.show()
df_waveforms = pd.DataFrame(np_data_array)
df_waveforms.to_csv('waveforms_unnorm.csv')




