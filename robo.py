from scipy.stats import zscore
from broker_calc import broker_calc


class Robo:
    def __init__(self, ann, df, money):
        self.ann = ann
        self.df = df
        self.money = money
        self.init_money = money
        # self.bought -> [has stocks, number of stocks, price of each stock]
        self.bought = [False, 0, 0]
        self.moving_averages = []
    
    def fitness(self):
        self.prepare_inputs()
        output = self.ann.forward_propagation(self.moving_averages)
        # 0 -> buy if money
        # 1 -> sell if has stocks
        # 2 -> hold
        for i in range(60, output.shape[1]):
            ind = output[:, i].argmax()

            if ind == 0:
                bought_price = self.df.Close[i]
                if self.money >= bought_price:
                    stock_qty = self.money // bought_price
                    self.bought = [True, stock_qty, bought_price]
                    self.money = self.money - stock_qty * bought_price

            elif ind == 1:
                if self.bought[0] == True:
                    sell_price = self.df.Close[i]
                    buy_price = self.bought[2]
                    stock_qty = self.bought[1]

                    holdings = stock_qty*sell_price
                    holdings -= broker_calc(buy_price, sell_price, stock_qty)
                    self.money += holdings
                    self.bought = [False, 0, 0]

            elif ind == 2:
                pass

        return self.money - self.init_money

    def prepare_inputs(self):
        self.get_moving_averages()

    def get_moving_averages(self):
        periods = [1, 3, 5, 7, 11, 15, 19, 23, 27, 35, 41, 50, 61]
        moving_averages = []
        for period in periods:
            m = self.df.Close.rolling(period).mean()
            moving_averages.append(m)
        self.moving_averages = zscore(moving_averages)