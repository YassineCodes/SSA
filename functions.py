# 
#
# function.py
# ------------------------------------------------------------------------------
import numpy as np
from numpy import dot
import pandas as pd
from scipy.linalg import svd
import math
from scipy import signal
import statistics as st
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
#from fastdtw import fastdtw
#from dtw import dtw
import random

#the main pre-processing function
def wavelet_denoise(data, wavelet, noise_sigma):
    '''Filter accelerometer data using wavelet denoising

    Modification of F. Blanco-Silva's code at: https://goo.gl/gOQwy5
    '''
    
    import scipy
    import pywt

    wavelet = pywt.Wavelet(wavelet)
    levels  = min(1, (np.floor(numpy.log2(data.shape[0]))).astype(int))

    # Francisco's code used wavedec2 for image data
    wavelet_coeffs = pywt.wavedec(data, wavelet, level=levels)
    threshold = noise_sigma*np.sqrt(2*numpy.log2(data.size))

    new_wavelet_coeffs = map(lambda x: pywt.threshold(x, threshold, mode='soft'),
                             wavelet_coeffs)

    return pywt.waverec(list(new_wavelet_coeffs), wavelet)
class SSA(object):
    
    __supported_types = (pd.Series, np.ndarray, list)
    
    def __init__(self, tseries, L, save_mem=True):
        """
        Decomposes the given time series with a singular-spectrum analysis. Assumes the values of the time series are
        recorded at equal intervals.
        
        Parameters
        ----------
        tseries : The original time series, in the form of a Pandas Series, NumPy array or list. 
        L : The window length. Must be an integer 2 <= L <= N/2, where N is the length of the time series.
        save_mem : Conserve memory by not retaining the elementary matrices. Recommended for long time series with
            thousands of values. Defaults to True.
        
        Note: Even if an NumPy array or list is used for the initial time series, all time series returned will be
        in the form of a Pandas Series or DataFrame object.
        """
        
        # Tedious type-checking for the initial time series
        if not isinstance(tseries, self.__supported_types):
            raise TypeError("Unsupported time series object. Try Pandas Series, NumPy array or list.")
        
        # Checks to save us from ourselves
        self.N = len(tseries)
        if not 2 <= L <= self.N/2:
            raise ValueError("The window length must be in the interval [2, N/2].")
        
        self.L = L
        self.orig_TS = pd.Series(tseries)
        self.K = self.N - self.L + 1
        
        # Embed the time series in a trajectory matrix
        self.X = np.array([self.orig_TS.values[i:L+i] for i in range(0, self.K)]).T
        
        # Decompose the trajectory matrix
        self.U, self.Sigma, VT = np.linalg.svd(self.X)
        self.d = np.linalg.matrix_rank(self.X)
        
        self.TS_comps = np.zeros((self.N, self.d))
        
        if not save_mem:
            # Construct and save all the elementary matrices
            self.X_elem = np.array([ self.Sigma[i]*np.outer(VT[i,:]) for i in range(self.d) ])
            #self.X_elem = np.array([ self.Sigma[i]*np.outer(self.U[:,i], VT[i,:]) for i in range(self.d) ])
            # Diagonally average the elementary matrices, store them as columns in array.           
            for i in range(self.d):
                X_rev = self.X_elem[i, ::-1]
                self.TS_comps[:,i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0]+1, X_rev.shape[1])]
            
            self.V = VT.T
        else:
            # Reconstruct the elementary matrices without storing them
            for i in range(self.d):
                X_elem = self.Sigma[i]*np.outer(self.U[:,i], VT[i,:])
                X_rev = X_elem[::-1]
                self.TS_comps[:,i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0]+1, X_rev.shape[1])]
            
            self.X_elem = "Re-run with save_mem=False to retain the elementary matrices."
            
            # The V array may also be very large under these circumstances, so we won't keep it.
            self.V = "Re-run with save_mem=False to retain the V matrix."
        
        # Calculate the w-correlation matrix.
        self.calc_wcorr()
            
    def components_to_df(self, n=0):
        """
        Returns all the time series components in a single Pandas DataFrame object.
        """
        if n > 0:
            n = min(n, self.d)
        else:
            n = self.d
        
        # Create list of columns - call them F0, F1, F2, ...
        cols = ["F{}".format(i) for i in range(n)]
        return pd.DataFrame(self.TS_comps[:, :n], columns=cols, index=self.orig_TS.index)
            
    
    def reconstruct(self, indices):
        """
        Reconstructs the time series from its elementary components, using the given indices. Returns a Pandas Series
        object with the reconstructed time series.
        
        Parameters
        ----------
        indices: An integer, list of integers or slice(n,m) object, representing the elementary components to sum.
        """
        if isinstance(indices, int): indices = [indices]
        
        ts_vals = self.TS_comps[:,indices].sum(axis=1)
        return pd.Series(ts_vals, index=self.orig_TS.index)
    
    def calc_wcorr(self):
        """
        Calculates the w-correlation matrix for the time series.
        """
             
        # Calculate the weights
        w = np.array(list(np.arange(self.L)+1) + [self.L]*(self.K-self.L-1) + list(np.arange(self.L)+1)[::-1])
        
        def w_inner(F_i, F_j):
            return w.dot(F_i*F_j)
        
        # Calculated weighted norms, ||F_i||_w, then invert.
        F_wnorms = np.array([w_inner(self.TS_comps[:,i], self.TS_comps[:,i]) for i in range(self.d)])
        F_wnorms = F_wnorms**-0.5
        
        # Calculate Wcorr.
        self.Wcorr = np.identity(self.d)
        self.Power = []
        for i in range(self.d):
            for j in range(i+1,self.d):
                self.Wcorr[i,j] = abs(w_inner(self.TS_comps[:,i], self.TS_comps[:,j]) * F_wnorms[i] * F_wnorms[j])
                self.Wcorr[j,i] = self.Wcorr[i,j]
                if self.Wcorr[i,j] >= 0.8 and self.Wcorr[i,j] <= self.Wcorr[i,j-1]:
                    self.Power.append(i)
                    self.Power.append(j)
                    
        return self.Power   
    def plot_wcorr(self, min=None, max=None):
        """
        Plots the w-correlation matrix for the decomposed time series.
        """
        if min is None:
            min = 0
        if max is None:
            max = self.d
        
        if self.Wcorr is None:
            self.calc_wcorr()
        
        ax = plt.imshow(self.Wcorr)
        plt.xlabel(r"$\tilde{F}_i$")
        plt.ylabel(r"$\tilde{F}_j$")
        plt.colorbar(ax.colorbar, fraction=0.045)
        ax.colorbar.set_label("$W_{i,j}$")
        plt.clim(0,1)
        
        # For plotting purposes:
        if max == self.d:
            max_rnge = self.d-1
        else:
            max_rnge = max
        
        plt.xlim(min-0.5, max_rnge+0.5)
        plt.ylim(max_rnge+0.5, min-0.5)
        return self.Wcorr
def specific(move):
    window = 17 # samples
    accel_ssa = SSA(move, window)  
    plt.close()
    Wcorr = accel_ssa.plot_wcorr(max=17)
    totalSum = sum([Wcorr[i][j] for i in range(1,len(Wcorr)) for j in range(i)])
    # print(np.mean(totalSum))
    threshold = 0.7
    # list of components
    p = []
    # ----------
    tmp = []
    for s in range(3, window):
            if Wcorr[s, s-1] >= threshold:
                if s!=3:
                    token = False
                    if len(tmp) > 1:
                        highCorr = False
                        all = 0
                        for t in range(len(tmp)):
                            if Wcorr[s, tmp[t]] >=0.3:
                                all = all + 1
                            if len(tmp) == all:
                                token = True
                            if Wcorr[s, tmp[t]] >=0.9:
                                highCorr = True
                                p.append(s)
                        if highCorr == False:
                            p = []
                    else:
                        token = True
                        if Wcorr[s, s-1] >= threshold or Wcorr[s, s-2]>=threshold:
                            p.append(s)
                    if token != True:
                        
                        tmp = []
                        tmp.append(s)
                else:
                    tmp.append(s)
                    tmp.append(s-1)
            elif s!=3:
                if len(tmp) > 1: 
                    p.append(tmp[-1])
                    tmp = []
                else:
                    tmp = []
                    tmp.append(s)
            else:
                if Wcorr[s-1, s-2] >= 0.9:
                    p.append(s-1)
                tmp.append(s)
    
    
    fig, axes = plt.subplots(ncols=1, nrows= 2, figsize=(10,10))
    for i, ax in enumerate(axes.flatten()):
        if i == 0:
            move[1:500].plot(ax=ax)
            ax.set_title('Raw {}'.format("Accelerometer"),fontweight="bold", size=16)
        else:            
            accel_ssa.reconstruct(list(set(p)))[1:500].plot(ax=ax) 
            ax.set_title('Reconstructed Best SSA {}'.format("Accelerometer"),fontweight="bold", size=16)
            plt.show()
            print("Selected components: ", p)
            returned_components = {'type': 'data'}
            returned_components.update({'best' : accel_ssa.reconstruct(list(set(p)))})
            for idx in range(1, window):
                component = []
                component.append(idx)
                returned_components.update({idx : accel_ssa.reconstruct(list(set(component)))})
                plt.plot(accel_ssa.reconstruct(list(set(component)))[1:500])
                plt.show()
    return(returned_components)#

def get_y_rotation(x,y,z):
    radians = math.atan2(x, dist(y,z))
    return -math.degrees(radians)
def get_x_rotation(x,y,z):
    radians = math.atan2(y, dist(x,z))
    return math.degrees(radians)
def dist(a, b):
    return math.sqrt((a * a) + (b * b))
def complementary_filter(accx, accy, accz, gyrox, gyroy, gyroz):
    time_diff = 0.01
    K = 0.98
    K1 = 1 - K
        
    gyro_total_x = np.cumsum(gyrox)
    gyro_total_y = np.cumsum(gyroy)
    gyro_total_z = np.cumsum(gyroz)
    rotation_x = []
    rotation_y = []
    cutoffHZ = 4
            
    
    sampleHz = 33  # the sample frequency should be cross validated
    nyqHZ = sampleHz / 2
    d, a = signal.butter(2, cutoffHZ / nyqHZ, "low")  # lowFilter - generic filter model is 9
    f, b = signal.butter(2, cutoffHZ / nyqHZ, "high")  # highFilter - generic filter model is 9
    for j in range(0,len(accx)):
        rotation_x.append(get_x_rotation(accx[j], accy[j], accz[j]))
        rotation_y.append(get_y_rotation(accx[j], accy[j], accz[j]))
       
    last_x_ = gyro_total_x
    last_y_ = gyro_total_y
    last_x__ = gyro_total_x
    last_y__ = gyro_total_y
    
    for i in range(0,(len(accx)-1)):
        j = i + 1
        last_x_[j] = K *(last_x_[i] + gyro_total_x[j])     #complementary filter
        last_y_[j] = K *(last_y_[i] + gyro_total_y[j])    #complementary filter
    for i in range(0,(len(accx)-1)):
        j = i + 1
        last_x__[j] =  (K1 * rotation_x[j])    #complementary filter
        last_y__[j] =  (K1 * rotation_y[j])    #complementary filter
    
    last_x = signal.filtfilt(f, b, last_x_) + signal.filtfilt(d, a, last_x__)
    last_y = signal.filtfilt(f, b, last_y_) + signal.filtfilt(d, a, last_y__)
    return pd.DataFrame({'AccX': last_x , 'AccY':last_y})
      
def transform (signal_data):
    Acc = signal_data[["AccX","AccY","AccZ"]]
    Gyro = signal_data[["GyroX","GyroY","GyroZ"]]
    Compass = signal_data[["CompassX","CompassY","CompassZ"]]

    #Normalize Gyroscope signal
    Gyro = Gyro.reset_index(drop=True)
    Gyro.loc[:,'GyroX'] = Gyro.loc[:,'GyroX']/np.std(Gyro.loc[:,'GyroX'])
    Gyro.loc[:,'GyroY'] = Gyro.loc[:,'GyroY']/np.std(Gyro.loc[:,'GyroY'])
    Gyro.loc[:,'GyroZ'] = Gyro.loc[:,'GyroZ']/np.std(Gyro.loc[:,'GyroZ'])

    #Normalize Compass signal
    Compass = Compass.reset_index(drop=True)
    Compass.loc[:,'CompassX'] = Compass.loc[:,'CompassX']/np.std(Compass.loc[:,'CompassX'])
    Compass.loc[:,'CompassY'] = Compass.loc[:,'CompassY']/np.std(Compass.loc[:,'CompassY'])
    Compass.loc[:,'CompassZ'] = Compass.loc[:,'CompassZ']/np.std(Compass.loc[:,'CompassZ'])

    Acc.columns = ["X","Y","Z"]
    Gyro.columns = ["X","Y","Z"]
    Compass.columns = ["X","Y","Z"]

    cct = pd.concat([np.transpose(Acc), np.transpose(Gyro), np.transpose(Compass)], axis=1)

    cct_length = len(cct.columns)
    cct = np.asmatrix(cct)

    u, s, vh = svd(cct)

    # create m x n Sigma matrix
    Sigma = np.zeros((cct.shape[0], cct.shape[1]))
    # populate Sigma with n x n diagonal matrix

    Sigma[:cct.shape[0], :cct.shape[0]] = np.diag(s)
    # reconstruct matrix

    cct = Sigma.dot(vh)

    cct = pd.DataFrame(cct)
    cct = cct.T


    cct.columns = ["TRSFX","TRSFY","TRSFZ"]
    signal_data = pd.concat([signal_data, cct.iloc[range(0,((int(cct_length/3))))].reset_index(drop=True)], axis=1)
    cct.columns = ["Gyro_TRSFX","Gyro_TRSFY","Gyro_TRSFZ"]
    signal_data = pd.concat([signal_data, cct.iloc[range((int(cct_length/3)), (int((cct_length/3)*2)))].reset_index(drop=True)], axis=1)
    cct.columns = ["Compass_TRSFX","Compass_TRSFY","Compass_TRSFZ"]
    signal_data = pd.concat([signal_data, cct.iloc[range((int((cct_length/3)*2)),(int(cct_length)))].reset_index(drop=True)], axis=1)
    return signal_data