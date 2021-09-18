from datetime import datetime
from dateutil.parser import parse
from random import randrange

from scipy.stats import zscore
import numpy as np

from broker_calc import get_net_profit


class Robo:
    def __init__(self, ann, df, money):
        self.ann = ann
        self.df = df
        self.money = money
        self.init_money = money
        # self.bought -> [has stocks, number of stocks, price of each stock]
        self.bought = [False, 0, 0]
        self.moving_averages = []
        self.time_ind = []
        self.inputs = []
        self.profit = 0

    def sell_stocks(self):
        if self.bought[0] == True:
            sell_price = self.df.Close[i]
            buy_price = self.bought[2]
            stock_qty = self.bought[1]

            net_profit = get_net_profit(buy_price, sell_price, stock_qty)

            # if we are getting profit after selling stocks then
            # the amount is not credited imediately to our margin
            # profit is credited after one day
            # if loss then our margin is decreased
            if net_profit > 0:
                self.profit += net_profit
                net_profit
            holdings = buy_price*stock_qty + net_profit
            self.money += holdings
            self.bought = [False, 0, 0]

    def fitness(self):
        self.prepare_inputs()
        output = self.ann.forward_propagation(self.inputs)
        # output index meaning
        # 0 -> buy if money
        # 1 -> sell if has stocks
        # 2 -> hold

        for i in range(output.shape[1]):
            ind = output[:, i].argmax()

            # Do not buy stocks after 15:10
            # only sell the stock
            if self.df.Date[i] > 356:
                if ind == 1:
                    self.sell_stocks()
                
                # sell harshly and immediately at 15:18
                if self.df.Date[i] >= 364:
                    self.sell_stocks()

            else:

                if ind == 0:
                    bought_price = self.df.Close[i]
                    if self.bought[0] == False and self.money >= bought_price:
                        stock_qty = self.money // bought_price
                        self.bought = [True, stock_qty, bought_price]
                        self.money = self.money - stock_qty * bought_price

                elif ind == 1:
                    self.sell_stocks()
                    

        return self.money + self.profit

    def prepare_inputs(self):
        self.get_moving_averages()

    def get_moving_averages(self):
        periods = [1, 3, 5, 7, 11, 15, 19, 23, 27, 35, 41, 50, 61, 128]
        moving_averages = []
        for period in periods:
            m = self.df.Close.rolling(period).mean()
            moving_averages.append(m)
        self.moving_averages = zscore(moving_averages)


class RoboTest:
    def __init__(self, df, money):
            self.df = df
            self.money = money
            self.init_money = money
            # self.bought -> [has stocks, number of stocks, price of each stock]
            self.bought = [False, 0, 0]
            self.moving_averages = []
            self.time_ind = []
            self.inputs = []
        
    def fitness(self):
        start_date_index = 128
        end_index = 10000
        while self.df.Date[start_date_index] != 1:
            start_date_index += 1

        print(start_date_index, self.df.Time[start_date_index])

        delivery_trade = False

        for i in range(start_date_index, self.df.shape[0]):
            ind = randrange(3)

            if self.df.Date[i]==1 and self.bought[0]:
                delivery_trade = True
            
            if i >= end_index and self.bought[0]==False:
                break

            if ind == 0:
                bought_price = self.df.Close[i]
                if self.bought[0] == False and self.money >= bought_price:
                    stock_qty = self.money // bought_price
                    self.bought = [True, stock_qty, bought_price]
                    self.money = self.money - stock_qty * bought_price
                    print(df.Time[i], "Bought at = %f" %(stock_qty*bought_price), end = '\t')

            elif ind == 1:
                if self.bought[0] == True:
                    sell_price = self.df.Close[i]
                    buy_price = self.bought[2]
                    stock_qty = self.bought[1]
                    
                    if delivery_trade:
                        net_profit = get_net_profit(buy_price, sell_price, stock_qty, True)
                        delivery_trade = False
                    else:
                        net_profit = get_net_profit(buy_price, sell_price, stock_qty)

                    holdings = buy_price*stock_qty + net_profit
                    print("Sold at = %f\t Profit = %f" %(holdings, net_profit), buy_price, sell_price, stock_qty)
                    self.money += holdings
                    self.bought = [False, 0, 0]

        return self.money