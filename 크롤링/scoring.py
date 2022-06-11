import pandas as pd
import FinanceDataReader as fdr
import yfinance as yf
import datetime as dt
from dateutil.relativedelta import relativedelta
import matplotlib as plt
import requests
from bs4 import BeautifulSoup
import re
import time
import numpy as np

import crawling
from crawling import consensus_crawling_DB, consensus_crawling_DB_a_month_ago

# 상장법인 코드 추출
stocks = fdr.StockListing('KRX')
stockcode = stocks[['Symbol', 'Market', 'Name']]
data = consensus_crawling_DB
data_ago = consensus_crawling_DB_a_month_ago

# 종목코드 호출 함수
def code(stock):
    a = stockcode[stockcode['Name'] == stock]
    a_code = a['Symbol'].item()
    a_market = a['Market'].item()
    if a_market == 'KOSPI':
        a_market = '.KS'
    elif a_market == 'KOSDAQ':
        a_market = '.KQ'
    return a_code + a_market

# 어제 수익률 보기
def yesterday_rtn(name):
    a = code(name)
    today = dt.datetime.strptime('2022-05-17','%Y-%m-%d')
    weekago = today - relativedelta(weeks=1)
 
    df = yf.download(a, start=str(weekago), end=str(today), progess=False)
    df['daily_rtn'] = df['Adj Close'].pct_change()
    last_rtn = df['daily_rtn'][-1:].item()
    return last_rtn

# 한달동안 일별 수익률 추이 보기
def monthly_rtn_graph(name):
    a = code(name)
    today = dt.datetime.strptime('2022-05-17','%Y-%m-%d')
    monthago = today - relativedelta(months=1)

    df = yf.download(a, start=str(monthago), end=str(today), progess=False)
    df['daily_rtn'] = df['Adj Close'].pct_change()

    return df['daily_rtn'].plot()

stocks['Symbol'] = stocks['Symbol'].astype(str)


# 종목명 입력하면 애널리스트 보고서 제목 호출해주는 함수
def consensus_report_titles(name, data):
    new_list = [x[3] for x in data if x[1] == name]

    return new_list

# 종목명 입력하면 기준일로부터 한달간 적정주가 호출해주는 함수
def consensus_report_price(name, data):
    new_list = [x[4] for x in data if x[1] == name]

    return new_list

# 종목명 입력하면 매수/매도 의견 호출해주는 함수
def consensus_report_view(name, data):
    new_list = [x[5] for x in data if x[1] == name]

    return new_list

# 종목명 입력하면 기준일로부터 두달 전 적정주가 호출해주는 함수
def consensus_report_price_a_month_ago(name, data_ago):
    new_list = [x[4] for x in data_ago if x[1] == name]

    return new_list

# 종목명 입력하면 기준일로부터 두달 전 매수/매도 의견 호출해주는 함수
def consensus_report_view_a_month_ago(name, data_ago):
    new_list = [x[5] for x in data_ago if x[1] == name]

    return new_list


def achievement_ratio(name, data):

    ticker = code(name)[:6]
    start_date = '2022-05-17'
    current_price = fdr.DataReader(ticker, start_date, start_date)['Close'].item()

    price = []
    for i in range(len(consensus_report_price(name,data))):
        price.append(int(consensus_report_price(name,data)[i].replace(',','')))

    achievement_ratio = current_price / np.mean(price)

    return achievement_ratio


def target_change(name, data, data_ago):

    month_ago_price = consensus_report_price_a_month_ago(name, data_ago)
    price_now = consensus_report_price(name, data)
    
    for i in range(len(month_ago_price)):
        month_ago_price[i] = int(month_ago_price[i].replace(',',''))
    
    for i in range(len(price_now)):
        price_now[i] = int(price_now[i].replace(',',''))
    
    return np.mean(price_now) / np.mean(month_ago_price)


def news_num_change(name):
    
    a = keyword_daily_news_trend(port['종목명'].to_list(), "2022-04-30", "2022-05-31")
    b = a.loc['2022-05-17'][name]
    c = a.loc['2022-05-18'][name]

    return (c-b) / b

# 네이버증권 종목토론방 크롤링 함수(기준일 기준)
def reply_vol(name):
    
    base_url_2 = "https://finance.naver.com/item/board.naver?code=%s&page={}"%(code(name)[:6])

    b=[]
    c=[]
    i=1
    k=0
    while i:
        url = base_url_2.format(i)
        headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 6.0; WOW64; rv:24.0) Gecko/20100101 Firefox/24.0' }

        resp = requests.get(url, headers=headers)
        soup = BeautifulSoup(resp.text, "html.parser")
        reply_date = soup.find_all('span', {'class':'tah p10 gray03'})
        reply_date_list = list(map(str, reply_date))

        for j in range(len(reply_date_list)):
            if '05.17' in reply_date_list[j] :
                b.append(reply_date[j].string)

            if '05.16' in reply_date_list[j] :
                c.append(reply_date[j].string)

            if '05.15' in reply_date_list[j]:
                k=1

        if k == 1 :
            break

        i += 1

    b=[i for i in b if '2022' in i]
    c = [i for i in c if '2022' in i]

    # print(b) 
    # print(c)

    return (len(b) - len(c)) / len(c)



    