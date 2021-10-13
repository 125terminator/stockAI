import numpy as np
import pandas as pd
from scipy.stats import zscore

class Inputs:
    def __init__(self, df):
        self.df = df
        self.moving_averages = []
        self.rsi_list = []
        self.dema_list = []
        self.tema_list = []
        self.stoch_list = []
        self.vin_list = []
        self.vip_list = []
        self.cci_list = []
        self.inputs = []
        self.prepare_inputs()

    def prepare_inputs(self):
        self.compute_moving_averages()
        self.compute_rsi()
        self.compute_dema()
        self.compute_tema()
        self.compute_stoch()
        self.compute_vortex_indicator_pos()
        self.compute_vortex_indicator_neg()
        self.compute_cci()
        self.inputs = np.concatenate((self.moving_averages, \
            self.rsi_list, self.dema_list, self.tema_list, \
            self.stoch_list, self.vip_list, self.vin_list, self.cci_list), axis=0)

    # Exponential moving average
    def ema(self, series, n):
        return series.ewm(span=n, min_periods=n).mean()

    # Double exponential average
    def DEMA(self, n):
        EMA = self.ema(self.df.Close, n)
        return 2*EMA - self.ema(EMA,n)

    # Triple exponential average
    def TEMA(self, n):
        EMA = self.ema(self.df.Close, n)
        EEMA = self.ema(EMA, n)
        return 3*EMA - 3*EEMA + self.ema(EEMA, n)

    def rsi(self, n=14):
        diff = self.df.Close.diff(1)
        which_dn = diff < 0
        up, dn = diff, diff*0
        up[which_dn], dn[which_dn] = 0, -up[which_dn]
        emaup = self.ema(up, n)
        emadn = self.ema(dn, n)
        rsi = 100 * emaup / (emaup + emadn)
        return np.array(rsi)

    def compute_tema(self):
        periods = [30, 60, 180, 375, 375*5, 375*10]
        for period in periods:
            m = self.TEMA(period)
            self.tema_list.append(m)

        self.tema_list =  zscore(self.tema_list, axis=1, nan_policy='omit')

    def compute_dema(self):
        periods = [30, 60, 180, 375, 375*5, 375*10]
        for period in periods:
            m = self.DEMA(period)
            self.dema_list.append(m)

        self.dema_list =  zscore(self.dema_list, axis=1, nan_policy='omit')
        
    def compute_rsi(self):
        periods = [30, 60, 180, 375, 375*5, 375*10]
        for period in periods:
            m = self.rsi(period)
            self.rsi_list.append(m)
        
        self.rsi_list = zscore(self.rsi_list, axis=1, nan_policy='omit')
    
    def compute_moving_averages(self):
        periods = [1, 30, 60, 180, 375, 375*5, 375*10]
        moving_averages = []
        for period in periods:
            m = self.df.Close.rolling(period).mean()
            moving_averages.append(m)
        
        self.moving_averages = np.array(moving_averages)
        for i in range(self.moving_averages.shape[1]):
            self.moving_averages[:, i] = self.moving_averages[0, i] - self.moving_averages[:, i]

        self.moving_averages = np.delete(self.moving_averages, 0, 0)
        self.moving_averages = zscore(self.moving_averages, axis=1, nan_policy='omit')