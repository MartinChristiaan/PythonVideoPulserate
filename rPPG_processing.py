import numpy as np
import scipy.io as sio
from scipy import signal
import matplotlib.pyplot as plt

# Params
fs_video = 20;

fftlength = 300
noverlap  = fftlength-20

# Load Data
data_path = "C:\\Users\\marti\\Google Drive\\VHM_Personal\\Framework_Matlab\\Data"
data = sio.loadmat(data_path + "\\stationary.mat")
rPPG = np.array( data['rPPG'])

R = 0
G = 1
B = 2

bpf_div= 60 * fs_video / 2
b_BPF40220,a_BPF40220 = signal.butter(10, ([40/bpf_div, 220/bpf_div]),  'bandpass') 

k = int((rPPG.shape[1]-fftlength)/(fftlength-noverlap))
T = np.arange(fftlength/2, rPPG.shape[1]-fftlength/2,fftlength-noverlap).astype(int)


STFT_X = np.zeros((k, int(fftlength/2 + 1)),dtype = np.complex128)
fft_roi = range(int(fftlength/2+1)) # We only care about this part of the fft because it is symmetric anyway
for ii in range(k):
    stride = np.arange(-fftlength/2,fftlength/2).astype(int) + T[ii]
    col_c = np.zeros((3,stride.shape[0]))
    #print(stride)
    ## Chrominance method
    skin_vec = [1,0.66667,0.5]
    for col in [R,G,B]:
        col_stride = rPPG[col,stride]
        y_ACDC = signal.detrend(col_stride/np.mean(col_stride))
        col_c[col] = y_ACDC * skin_vec[col]

    X_chrom = col_c[R]-col_c[G]
    Y_chrom = col_c[R] + col_c[G] - 2* col_c[B]


    Xf = signal.filtfilt(b_BPF40220,a_BPF40220,X_chrom) # Applies band pass filter
    Yf = signal.filtfilt(b_BPF40220,a_BPF40220,Y_chrom)
    Nx = np.std(Xf)
    Ny = np.std(Yf)
    alpha_CHROM = Nx/Ny
    x_stride_method = Xf- alpha_CHROM*Yf
    STFT_X[ii]  = np.fft.fft(x_stride_method,fftlength)[fft_roi]

f = np.linspace(0,fs_video/2,fftlength/2 + 1);
normalized_amplitude = np.abs(STFT_X)/np.max(np.abs(STFT_X))
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')

X,Y = np.meshgrid(np.arange(k),f*60)

surf = ax.plot_surface(X,Y, np.transpose( normalized_amplitude))

#plt.figure()
#plt.pl
#plt.surf(range(k), f*60, normalized_amplitude ,'EdgeColor','interp')
#plt.ylabel('frequency (BPM)')
#plt.xlabel('time (stride index)')
plt.show()
#title(legend_method{Method_Choice})
#axis xy
#axis tight
#view(0,90);
#ylim([30 220])