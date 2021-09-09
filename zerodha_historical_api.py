from requests import Request, Session
from datetime import datetime, timedelta
import json
from time import sleep

import pandas as pd
import numpy as np

def candles_to_csv(array):
    array = np.array(array)
    df = pd.DataFrame(array)
    df.rename(columns={0: 'date', 1: 'open', 2: 'open_high', 3:'open_low', 4: 'close', 5: 'volume'}, inplace=True)
    df.to_csv('reliance.csv')

def get_candles(url):
    req = Request('GET', url, headers=headers)
    resp = s.send(req.prepare())
    candles = []
    if resp.status_code == 200:
        data = resp.content.decode('utf-8')
        data = json.loads(data)
        if data["status"] == "success":
            for candle in data["data"]["candles"]:
                # time, open_price, open_high, open_low, close_price, volume, _ = candle
                candles.append(candle)

    resp.close()
    return candles

if __name__ == "__main__":
    s = Session()
    RELIANCE_ID = 738561
    # date format %Y-%m-%d
    frm = datetime.strptime( "2015-03-01", "%Y-%m-%d" )
    to = datetime.strptime( "2015-03-01", "%Y-%m-%d" )
    today = datetime.tody()
    url = "https://kite.zerodha.com/oms/instruments/historical/{stock_id}/minute?user_id=GH1868&oi=1&from={from_}&to={to_}"
    headers = {
        "accept": "*/*",
        "accept-language": "en-IN,en-GB;q=0.9,en-US;q=0.8,en;q=0.7,hi;q=0.6",
        "authorization": "enctoken UpyNg+0QwdYUT/CVom42kKqBTp7Jp4XzJcblM2Z1koY336UL5IaVHlrdfNSMG8W9DnzxtdjK0DRX713UI9OgNMKz4dhdVzPuHXJgynnLrh61jUERfwVYuA==",
        "sec-ch-ua": "\"Chromium\";v=\"92\", \" Not A;Brand\";v=\"99\", \"Google Chrome\";v=\"92\"",
        "sec-ch-ua-mobile": "?0",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "referrer": "https://kite.zerodha.com/static/build/chart.html?v=2.9.3",
        "referrerPolicy": "strict-origin-when-cross-origin",
        "method": "GET",
        "mode": "cors",
        "credentials": "include"
    }
    candles = []
    for i in range( 100 ):
        # print( frm, to, to-frm )
        frm = to + timedelta( days=1 )
        to = frm + timedelta( days=30 )
        if frm > today:
            break
        tmp_url = url.format( stock_id=RELIANCE_ID, from_=frm.strftime( "%Y-%m-%d" ), to_=to.strftime( "%Y-%m-%d" ) )
        candles.extend( get_candles( tmp_url ) )
        sleep(1)
    candles_to_csv(candles)

    s.close()