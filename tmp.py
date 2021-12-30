import numpy as np
import pandas as pd
# from scipy.stats import zscore
import math

def flatten(t):
    return [item for sublist in t for item in sublist]
    
periods = [2, 4, 8, 10, 14, 20, 30]
def zscore(a, axis=1, nan_policy='omit'):
    return np.array(a)
class Inputs:
    def __init__(self, df, df2):
        '''
        inputs: 
          df2 -> has data per hour
          df -> has data per day

          self.df -> first contains data per hour then will be split and made 7hr single data
        '''
        self.df = df2
        self.moving_averages = []
        self.rsi_list = []
        self.dema_list = []
        self.tema_list = []
        self.stoch_list = []
        self.vin_list = []
        self.vip_list = []
        self.cci_list = []
        self.avg_true_list = []
        self.macd_signal_cross = []
        self.obv_cross_list = []
        self.inputs = []
        self.desc_inputs = []
        self.prepare_inputs()

        self.inputs = np.transpose(self.inputs)
        # split and made 7hr single data
        self.inputs = np.split(self.inputs, [i for i in range(7, self.inputs.shape[0], 7)], axis=0)
        self.inputs = np.array([i.flatten() for i in self.inputs])
        self.first_non_nan = 0
        for i in range(self.inputs.shape[0]):
            for j in range(self.inputs.shape[1]):
                if math.isnan(self.inputs[i, j]):
                    self.first_non_nan = max(self.first_non_nan, i)

        self.test_days = 30
        self.start_day = self.first_non_nan + 1
        self.end_day = self.start_day + 1664
        self.test_start_day = self.end_day
        self.test_end_day = self.end_day+self.test_days
        
        self.x_train = self.inputs[self.start_day:self.end_day, :]
        self.x_test = self.inputs[self.test_start_day:self.test_end_day, :]

        y = np.array((df.Close > df.Open).shift(-1).fillna(False), dtype=np.int8)

        self.y_train = y[self.start_day:self.end_day]
        self.y_test = y[self.test_start_day:self.test_end_day]

        print('start day: ', self.start_day)
        print('end day: ', self.end_day)
        print('start test day: ', self.test_start_day)
        print('end test day: ', self.test_end_day)
        
        # print('inputs: ', self.inputs)
        # print('input description: ', self.desc_inputs)

    def prepare_inputs(self):
        # technical parameters with open values
        # moving_averages, rsi, dema, tema, macd
        self.moving_averages = np.concatenate(
            (
                self.compute_moving_averages(),\
                self.compute_moving_averages(open=True)
            ), axis=0
        )
        self.rsi_list = np.concatenate(
            (
                self.compute_rsi(),\
                self.compute_rsi(open=True)
            ), axis=0
        )
        self.dema_list = np.concatenate(
            (
                self.compute_dema(),\
                self.compute_dema(open=True)
            ), axis=0
        )
        self.tema_list = np.concatenate(
            (
                self.compute_tema(),\
                self.compute_tema(open=True)
            ), axis=0
        )
        self.macd_signal_cross = np.concatenate(
            (
                self.compute_macd(),\
                self.compute_macd(open=True)
            ), axis=0
        )
        self.compute_stoch()
        self.compute_vortex_indicator_pos()
        self.compute_vortex_indicator_neg()
        self.compute_cci()
        self.compute_obv()
        self.compute_average_true_range()

        data = (
                self.moving_averages, \
                self.rsi_list, \
                self.dema_list, \
                self.tema_list,\
                self.stoch_list, \
                self.vip_list, \
                self.vin_list, \
                self.cci_list, \
                self.avg_true_list, \
                self.macd_signal_cross, \
                self.obv_cross_list
        )
        
        self.desc_inputs = [
                ["moving_averages"], \
                ["rsi"], \
                ["dema"], \
                ["tema"], \
                ["stoch"], \
                ["vortex_indicator_pos"], \
                ["vortex_indicator_neg"], \
                ["cci"], \
                ["average_true_range"], \
                ["macd_and_signal_cross"], \
                ["obv"]
        ]
        for i in range(len(data)):
            self.desc_inputs[i] *= data[i].shape[0]
        self.desc_inputs = np.array(flatten(self.desc_inputs))

        self.inputs = np.concatenate(data, axis=0)

    # Exponential moving average
    def ema(self, series, n):
        return series.ewm(span=n, min_periods=n, ignore_na=True).mean()

    def macd(self, n, open=False):
        if open:
            exp1 = self.ema(self.df.Open, 12*n).shift(-1).fillna(0)
            exp2 = self.ema(self.df.Open, 26*n).shift(-1).fillna(0)
            macd = exp1-exp2
            exp3 = self.ema(macd, 9*n)
        else:
            exp1 = self.ema(self.df.Close, 12*n)
            exp2 = self.ema(self.df.Close, 26*n)
            macd = exp1-exp2
            exp3 = self.ema(macd, 9*n)
        return np.array(2*(macd>exp3)-1)

    # Double exponential average
    def DEMA(self, n, open=False):
        if open:
            EMA = self.ema(self.df.Open, n).shift(-1).fillna(0)
        else:
            EMA = self.ema(self.df.Close, n)
        return 2*EMA - self.ema(EMA,n)

    # Triple exponential average
    def TEMA(self, n, open=False):
        if open:
            EMA = self.ema(self.df.Open, n).shift(-1).fillna(0)
        else:
            EMA = self.ema(self.df.Close, n)
        EEMA = self.ema(EMA, n)
        return 3*EMA - 3*EEMA + self.ema(EEMA, n)

    def stoch(self, n=14):
        smin = self.df.Low.rolling(n).min()
        smax = self.df.High.rolling(n).max()
        stoch_k = 100 * (self.df.Close - smin) / (smax - smin + 1)
        return np.array(stoch_k)

    def stoch_signal(self, n=14, d_n=3):
        stoch_k = pd.Series(self.stoch(n), name='stoch_k')
        stoch_d = stoch_k.rolling(d_n).mean()
        return np.array(stoch_d)

    def rsi(self, n=14, open=False):
        if open:
            diff = self.df.Open.diff(1).shift(-1).fillna(0)
        else:
            diff = self.df.Close.diff(1)
        which_dn = diff < 0
        up, dn = diff, diff*0
        up[which_dn], dn[which_dn] = 0, -up[which_dn]
        emaup = self.ema(up, n)
        emadn = self.ema(dn, n)
        rsi = 100 * emaup / (emaup + emadn + 1)
        return np.array(rsi)

    def average_true_range(self, n=14):
        cs = self.df.Close.shift(1)
        tr = self.df.High.combine(cs, max) - self.df.Low.combine(cs, min)

        atr = np.zeros(len(self.df.Close))
        atr[0] = tr[1::].mean()
        for i in range(1, len(atr)):
            atr[i] = (atr[i-1] * (n-1) + tr.iloc[i]) / float(n)

        return np.array(atr)

    def vortex_indicator_pos(self, n=14):
        tr = self.df.High.combine(self.df.Close.shift(1), max) - self.df.Low.combine(self.df.Close.shift(1), min)
        trn = tr.rolling(n).sum()

        vmp = np.abs(self.df.High - self.df.Low.shift(1))
        vmm = np.abs(self.df.Low - self.df.High.shift(1))

        vip = vmp.rolling(n).sum() / ( trn + 1 )
        return np.array(vip)

    def vortex_indicator_neg(self, n=14):
        tr = self.df.High.combine(self.df.Close.shift(1), max) - self.df.Low.combine(self.df.Close.shift(1), min)
        trn = tr.rolling(n).sum()

        vmp = np.abs(self.df.High - self.df.Low.shift(1))
        vmm = np.abs(self.df.Low - self.df.High.shift(1))

        vin = vmm.rolling(n).sum() / trn
        return np.array(vin)

    def cci(self, n=20, c=0.015):
        pp = (self.df.High + self.df.Low + self.df.Close) / 3.0
        cci = (pp - pp.rolling(n).mean()) / (c * pp.rolling(n).std()+1)
        return np.array(cci)

    def compute_obv(self):
        obv = (np.sign(self.df.Close.diff()) * self.df.Volume).fillna(0).cumsum()
        self.obv_cross_list.append(np.array(2*(obv>obv.shift(1))-1))
        self.obv_cross_list = np.array(self.obv_cross_list)

    def compute_average_true_range(self):
        global periods
        for period in periods:
            m = self.average_true_range(period)
            self.avg_true_list.append(m)

        self.avg_true_list = zscore(self.avg_true_list, axis=1, nan_policy='omit')

    def compute_vortex_indicator_pos(self):
        global periods
        for period in periods:
            m = self.vortex_indicator_pos(period)
            self.vip_list.append(m)

        self.vip_list = zscore(self.vip_list, axis=1, nan_policy='omit')

    def compute_vortex_indicator_neg(self):
        global periods
        for period in periods:
            m = self.vortex_indicator_neg(period)
            self.vin_list.append(m)

        self.vin_list = zscore(self.vin_list, axis=1, nan_policy='omit')

    def compute_cci(self):
        global periods
        for period in periods:
            m = self.cci(period)
            self.cci_list.append(m)

        self.cci_list = zscore(self.cci_list, axis=1, nan_policy='omit')

    def compute_stoch(self):
        global periods
        for period in periods:
            m = self.stoch(period)
            self.stoch_list.append(m)

        self.stoch_list.append(self.stoch_signal())
        self.stoch_list = zscore(self.stoch_list, axis=1, nan_policy='omit')

    def compute_tema(self, open=False):
        global periods
        tema_list = []
        for period in periods:
            m = self.TEMA(period, open)
            tema_list.append(m)

        return zscore(tema_list, axis=1, nan_policy='omit')

    def compute_dema(self, open=False):
        global periods
        dema_list = []
        for period in periods:
            m = self.DEMA(period, open)
            dema_list.append(m)

        return zscore(dema_list, axis=1, nan_policy='omit')
        
    def compute_rsi(self, open=False):
        global periods
        rsi_list = []
        for period in periods:
            m = self.rsi(period, open)
            rsi_list.append(m)
        
        return np.array(rsi_list)/100
    
    def compute_macd(self, open=False):
        periods = [1]
        macd_signal_cross = []
        for period in periods:
            macd_signal_cross.append(self.macd(period, open))

        return np.array(macd_signal_cross)
    
    def compute_moving_averages(self, open=False):
        periods = [1, 2, 4, 8, 10, 14, 20, 30]
        moving_averages = []
        for period in periods:
            if open:
                m   = self.df.Open.rolling(period).mean().shift(-1).fillna(0)
            else:
                m   = self.df.Close.rolling(period).mean()
            moving_averages.append(m)
        
        moving_averages = np.array(moving_averages)
        for i in range(moving_averages.shape[1]):
            moving_averages[:, i] = moving_averages[0, i] - moving_averages[:, i]

        moving_averages = np.delete(moving_averages, 0, 0)
        return zscore(moving_averages, axis=1, nan_policy='omit')


if __name__ == "__main__":
    inp = Inputs(df, df2)