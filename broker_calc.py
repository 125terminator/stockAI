import math

def jsround(value):
    x = math.floor(value)
    if (value - x) < .50:
        return x
    else:
        return math.ceil(value)

def to_fixed(a, fix):
    return round(a, fix)

def get_net_profit(buy_price, sell_price, qty, delivery=False):
    brokerage_buy = to_fixed(min((buy_price * qty * 0.0003), 20), 2)
    brokerage_sell = to_fixed(min((sell_price * qty * 0.0003), 20), 2)
    brokerage = to_fixed(brokerage_buy + brokerage_sell, 2)

    turnover = to_fixed((buy_price+sell_price)*qty, 2)

    if delivery:
        stt_total = jsround(to_fixed(turnover * 0.001, 2))
        brokerage = 0
        stamp_charges = to_fixed(buy_price*qty*0.00015, 2)
    else:
        stt_total = jsround(to_fixed((sell_price * qty) * 0.00025, 2))
        stamp_charges = to_fixed((buy_price*qty)*0.00003, 2)

    exc_trans_charge = to_fixed(0.0000345*turnover, 2)
    cc = 0

    stax = to_fixed(0.18 * (brokerage + exc_trans_charge), 2)

    sebi_charges = to_fixed(turnover*0.000001, 2)

    total_tax = to_fixed(brokerage + stt_total + exc_trans_charge + cc + stax + sebi_charges + stamp_charges, 2)

    breakeven = 0
    if qty > 0:
        breakeven = to_fixed(total_tax / qty, 2)

    # print('Profit if no charges = {}'.format())
    # print(' Turnover = {}\n Brokerage = {}\n STT total = {}\n Exc txn = {}\n GST = {}\n Sebi chg = {}\n Stamp duty = {}\n Total tax = {}'.format(turnover, brokerage, stt_total, exc_trans_charge, stax, sebi_charges, stamp_charges, total_tax))

    net_profit = to_fixed(((sell_price - buy_price) * qty) - total_tax, 2)

    return net_profit

if __name__ == "__main__":
    buy_price = float(input())
    sell_price = float(input())
    qty = int(input())
    print(broker_calc(buy_price, sell_price, qty))