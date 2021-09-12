from scipy.stats import zscore


class Robo:
    def __init__(self, ann, df):
        self.ann = ann
        self.df = df
        self.bought = False
        self.moving_averages = np.array()
    
    def fitness(self):
        self.prepare_inputs()
        self.ann.forward_propagation(self.moving_averages)

    def prepare_inputs(self):
        self.get_moving_averages()

    def get_moving_averages(self):
        periods = [1, 3, 5, 7, 11, 15, 19, 23, 27, 35, 41, 50, 61]
        moving_averages = []
        for period in periods:
            m = df.Close.rolling(period).mean()
            moving_averages.append(m[periods[-1]-1:])
        self.moving_averages = zscore(moving_averages)