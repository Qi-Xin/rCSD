import numpy as np
import scipy
import matplotlib. pyplot as plt
from scipy import signal
import copy
from matplotlib import rcParams, rcParamsDefault


def use_pdf_plot(**kwargs):
    SMALL_SIZE = 5
    MEDIUM_SIZE = 6
    BIG_SIZE = 7
    # rcParams['font.size'] = SMALL_SIZE
    rcParams['lines.linewidth'] = 1
    rcParams['axes.linewidth'] = 0.8  # Adjust as needed.
    rcParams['axes.labelsize'] = MEDIUM_SIZE
    rcParams['axes.titlesize'] = BIG_SIZE
    rcParams['figure.titlesize'] = BIG_SIZE
    rcParams['legend.fontsize'] = MEDIUM_SIZE
    rcParams['xtick.labelsize'] = MEDIUM_SIZE
    rcParams['ytick.labelsize'] = MEDIUM_SIZE
    rcParams['xtick.major.size'] = 2  # length of x-axis major ticks
    rcParams['ytick.major.size'] = 2  # length of y-axis major ticks
    rcParams['xtick.major.pad'] = 1  # Adjust as needed
    rcParams['ytick.major.pad'] = 1  # Adjust as needed

    for key, value in kwargs.items():
        rcParams[key] = value
        
    rcParams['xtick.major.width'] = rcParams['axes.linewidth']
    rcParams['ytick.major.width'] = rcParams['axes.linewidth']

def use_default_plot():
    rcParams.update(rcParamsDefault)


##### Frequency domain analysis 

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butterworth_bandpass_filter(x, lowcut, highcut, fs, order=4):
    """Butter bandpass filter.

    Args:
        x: Input signal.
        fs: Sampling frequency.
        order: The order of the Butterworth filter.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, x)
    return y

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def low_pass_filter(raw_signal, fs, cutoff=100.):
    b, a = butter_lowpass(cutoff, fs, order=3)
    signal_filt = signal.filtfilt(b, a, raw_signal, axis=1)
    return signal_filt

def get_phase(raw_signal, fs, lowcut=8., highcut=12., npadding=0):
    """Get instantaneous phase of a time series / multiple time series

    Args:
        raw_signal (np array): (nx,nt)
        fs (float): sampling rate
        lowcut (float, optional): lower threshold. Defaults to 8..
        hightcut (float, optional): higher threshold. Defaults to 12..

    Returns:
        np array: instantaneous phase at each time point
    """
    only_one_row = False
    if raw_signal.ndim == 1:
        only_one_row = True
        raw_signal = raw_signal[None,:]
    b, a = butter_bandpass(lowcut, highcut, fs=fs, order=3)
    signal_filt = signal.filtfilt(b, a, raw_signal, axis=1)
    
    analytic_signal = signal.hilbert(signal_filt,axis=1)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.angle(analytic_signal)
    # instantaneous_phase_conti = np.unwrap(instantaneous_phase)
    # instantaneous_frequency = (np.diff(instantaneous_phase_conti) /
    #                         (2.0*np.pi) * fs)
    power = amplitude_envelope.mean(axis=1)
    instantaneous_power = amplitude_envelope
    instantaneous_phase = instantaneous_phase[:,npadding:-npadding]
    instantaneous_power = instantaneous_power[:,npadding:-npadding]
    if only_one_row:
        instantaneous_phase = instantaneous_phase.squeeze()
        instantaneous_power = instantaneous_power.squeeze()
        power = power.squeeze().item()
    return instantaneous_phase, instantaneous_power

def get_power_phase(data, npadding, lowcut, highcut):
    data = copy.copy(data)
    nx, nt, ntrial = data.shape
    phase = np.zeros((nx, nt-2*npadding, ntrial))
    power = np.zeros((nx, nt-2*npadding, ntrial))

    for itrial in range(ntrial):
        raw_signal = data[:,:,itrial]
        instantaneous_phase, instantaneous_power = get_phase(raw_signal, 500, lowcut=lowcut, highcut=highcut, 
                                                                   npadding=npadding)
        phase[:,:,itrial] = instantaneous_phase
        power[:,:,itrial] = instantaneous_power
    return phase, power

def get_power_spectrum(
        x,
        fs,
        output_figure_path=None,
        show_figure=False):
    """Gets the power spectrum."""
    num_per_segment = 2 ** 12
    f, Pxx_den = signal.welch(x, fs, nperseg=1024)

    plt.figure()
    # plt.semilogy(f, Pxx_den)
    plt.plot(f, Pxx_den)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')

    if output_figure_path:
        plt.savefig(output_figure_path)
        print('Save figure to: ', output_figure_path)
    if show_figure:
        plt.show()
    plt.close()

    return f, Pxx_den

def get_spectrogram(
    x,
    fs,
    time_offset=0,
    output_figure_path=None,
    show_figure=True):
    """Get the spectrum along time.

    Args:
        x: Input signal.
        fs: Sampling frequency.
    """
    # `nfft` > `nperseg` means apply zero padding to make the spectrum look
    # smoother, but it does not provide extra informaiton. `noverlap` is the
    # overlap between adjacent sliding windows, the larger, the more overlap.
    # num_per_segment = 2 ** 8
    num_per_segment = 250
    f, t, Sxx = signal.spectrogram(
        x, fs,
        nperseg=num_per_segment,
        noverlap=num_per_segment // 50 * 49,
        nfft=num_per_segment * 8)
    t = np.array(t) + time_offset
    # Used to debug the positions of the sliding window.
    # print(np.array(t))

    plt.figure(figsize=(10, 8))
    # plt.pcolormesh(t, f, np.log(Sxx))  # The log scale plot.
    # plt.pcolormesh(t, f, Sxx, vmax=np.max(Sxx) / 10)
    plt.pcolormesh(t, f, Sxx, vmax=200)
    plt.ylim(0, 100)
    plt.colorbar()
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    if output_figure_path:
        plt.savefig(output_figure_path)
        print('Save figure to: ', output_figure_path)
    if show_figure:
        plt.show()
    plt.close()


##### LFP/CSD utils

def check_and_get_size(lfp):
    if lfp.ndim == 3:
        nx, nt, ntrial = lfp.shape
    elif lfp.ndim == 2:
        nx, nt = lfp.shape
        ntrial = 1
        lfp = lfp[:,:,None]     # Convert single trial LFP data to three dimensional
    else:
        raise AssertionError("LFP data must be either two dimensional (single trial) or three dimensional (multiple trials)!")
    return nx, nt, ntrial, lfp

def moving_average_single_row(x, pooling_size=1, moving_size=1):
    assert np.mod(moving_size,2) == 1, "Moving average kernel width must be an odd number!"
    assert np.mod(len(x),pooling_size) == 0, "Pooling parameter must be exactly divisible towards length!"
    """ Doing pooling """
    x_temp = x.reshape(-1,pooling_size)
    x_temp = x_temp.mean(axis=1)
    """ Doing moving average """
    weight_correct = np.convolve(np.ones(len(x_temp)), np.ones(moving_size), 'same') / moving_size
    x_smooth = np.convolve(x_temp, np.ones(moving_size), 'same') / moving_size  /weight_correct  # 
    return x_smooth

def moving_average(lfp, pooling_size=1, moving_size=1):
    nx, nt, ntrial, lfp = check_and_get_size(lfp)
    lfp_smooth = np.zeros((nx, nt, ntrial))
    for itrial in range(ntrial):
        for i in range(nx):
            temp = moving_average_single_row(lfp[i, :, itrial], pooling_size, moving_size)
            lfp_smooth[None, :, None] = temp[None, :, None]
    return lfp_smooth
    
def pooling(data, merge):
    new_data = np.zeros((data.shape[0], int(data.shape[1]/merge), data.shape[2]))
    for i in range(merge):
        new_data += data[:, i::merge, :]
    new_data = new_data/merge
    return new_data

def normalize(x):
    return x/np.max(np.abs(x), axis=(0, 1))

def normalize_var(x):
    return x/np.std(x)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# Add correct label for each figure
def add_label_csd(the_title,the_yticks):
    yticks_num = 5
    #plt.colorbar()
    plt.yticks( np.linspace(0,the_yticks.shape[0],yticks_num),
               np.linspace(the_yticks.min(),the_yticks.max(),yticks_num).astype(int) )
    plt.xlabel('Time (frame)')
    plt.ylabel('Depth (micron)')
    plt.gca().set_title(the_title)
    plt.gca().xaxis.tick_bottom()

def plot_im (arr , v1 , v2):
    plt.xticks([]) 
    plt.yticks([])
    p = plt. imshow (arr , vmin =-np. nanmax (np. abs ( arr )), vmax =np. nanmax (np. abs ( arr )),
                     cmap='bwr', extent =[ np. min (v1), np. max (v1), np. max (v2), np. min (v2)])
    return p 

def plot_im_new (arr,v1,v2,vmin,vmax,yticks):
    if yticks == 0:
        plt.yticks([])
    p = plt. imshow (arr , vmin = vmin, vmax = vmax,
                     cmap='bwr',extent =[ np. min (v1), np. max (v1), np. max (v2), np. min (v2)])
    return p 

def sort_grid (x):
    xsrt = x[x[:, 1]. argsort ()]
    xsrt = xsrt [ xsrt [:, 0]. argsort ( kind ='mergesort ')]
    return xsrt

def expand_grid (x1 ,x2):
    """
    Creates ( len (x1)* len (x2), 2) array of points from two vectors .
    : param x1: vector 1, (nx1 , 1)
    : param x2: vector 2, (nx2 , 1)
    : return : ( nx1 *nx2 , 2) points
    """
    lc = [(a, b) for a in x1 for b in x2]
    return np. squeeze (np. array (lc))

def reduce_grid (x):
    """
    Undoes expand_grid to take (nx , 2) array to two vectors containing
    unique values of each col .
    : param x: (nx , 2) points
    : return : x1 , x2 each vectors
    """
    x1 = np. unique (x [: ,0])
    x2 = np. unique (x [: ,1])
    return x1 , x2

def mykron (A, B):
    """
    Efficient Kronecker product .
    """
    a1 , a2 = A. shape
    b1 , b2 = B. shape
    C = np. reshape (A[:, np. newaxis , :, np. newaxis ] * B[np. newaxis , :, np. newaxis , :], (a1*b1 , a2*b2))
    return C

def comp_eig_D (Ks , Kt , sig2n ):
    """
    Computes eigvecs and diagonal D for inversion of kron (Ks , Kt) + sig2n*I
    : param Ks: spatial covariance
    : param Kt: temporal covariance
    : param sig2n : noise variance
    : return : eigvec (Ks), eigvec (Kt), Dvec
    """
    nx = Ks. shape [0]
    nt = Kt. shape [0]
    evals_t , evec_t = scipy . linalg . eigh (Kt)
    evals_s , evec_s = scipy . linalg . eigh (Ks)
    Dvec = np. repeat ( evals_s , nt) * np. tile ( evals_t , nx) + sig2n *np. ones(nx*nt)
    return evec_s , evec_t , Dvec



