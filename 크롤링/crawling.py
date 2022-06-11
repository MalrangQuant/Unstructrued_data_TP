import datetime as dt
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

stocks = fdr.StockListing('KRX')
stockcode = stocks[['Symbol', 'Market', 'Name']]

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

# 노이즈 제거 함수
def remove_noise_and_split_title(title):
    in_code = ''
    in_name = ''

    for code, name in stocks[['Symbol','Name']].values:
        if code in title and name in title:
            in_code = code
            in_name = name
    
    # 한글, 영어, 숫자 외 노이즈 제거
    clean_title = re.sub('[^A-Za-z0-9가-힣]',' ', title)

    # 기업명 코드 수정
    clean_title = clean_title.replace(in_code,' ')
    clean_title = clean_title.replace(in_name,' ')
    while ' ' * 2 in clean_title: 
        clean_title = clean_title.replace(' ' * 2, ' ')
    
    if in_name == '':
        return [None]
    else: 
        return [in_name, in_code, clean_title]

# 한경 컨센서스 크롤링 함수(기준일로부터 한달)
def consensus_crawling_DB():
    last = False
    page = 1
    today = '2022-05-17'
    start_date = (dt.datetime.strptime('2022-05-17','%Y-%m-%d') - relativedelta(months=1)).strftime("%Y-%m-%d")
    data = []
    while last == False:
        base_url = 'http://hkconsensus.hankyung.com/apps.analysis/analysis.list?&sdate={}&edate={}&report_type=CO&order_type=&now_page={}'.format(start_date, today, page)
        html = requests.get(base_url, headers={'User-Agent':'Gils'}).content
        soup = BeautifulSoup(html,'lxml')
        table = soup.find("div",{'class':'table_style01'}).find('table')
        for tr in table.find_all("tr")[1:]:
            record = []
            for i , td in enumerate(tr.find_all("td")[:6]):
                if i == 1:
                    record += remove_noise_and_split_title(td.text)
                elif i == 3:
                    record.append(td.text.replace(" ","").replace("\r","").replace("\n",""))
                else:
                    record.append(td.text)

            if None not in record:
                data.append(record)

        print('Loading page number {}...'.format(page))
    
        page_place = soup.find("div",{"class":"paging"})

        try:
            page_a_list = page_place.find_all('a')
            page_text = page_a_list[-2].string

        except AttributeError or NameError:
            pass
        
        if page_text == None:
            page = page + 1

        elif int(page_text)>page:
            page = page + 1

        else:
            last = True

        time.sleep(1)

    return data

# 한경 컨센서스 크롤링 함수(한달전 기준)
def consensus_crawling_DB_a_month_ago():
    last = False
    page = 1
    today = (dt.datetime.strptime('2022-05-17','%Y-%m-%d') - relativedelta(months=1)).strftime("%Y-%m-%d")
    start_date = (dt.datetime.strptime('2022-05-17','%Y-%m-%d') - relativedelta(months=2)).strftime("%Y-%m-%d")
    data = []
    while last == False:
        base_url = 'http://hkconsensus.hankyung.com/apps.analysis/analysis.list?&sdate={}&edate={}&report_type=CO&order_type=&now_page={}'.format(start_date, today, page)
        html = requests.get(base_url, headers={'User-Agent':'Gils'}).content
        soup = BeautifulSoup(html,'lxml')
        table = soup.find("div",{'class':'table_style01'}).find('table')
        for tr in table.find_all("tr")[1:]:
            record = []
            for i , td in enumerate(tr.find_all("td")[:6]):
                if i == 1:
                    record += remove_noise_and_split_title(td.text)
                elif i == 3:
                    record.append(td.text.replace(" ","").replace("\r","").replace("\n",""))
                else:
                    record.append(td.text)

            if None not in record:
                data.append(record)
        print('Loading page number {}...'.format(page))
    
        ## 페이지 마지막이면 자동으로 크롤링 종료
        page_place = soup.find("div",{"class":"paging"})

        try:
            page_a_list = page_place.find_all('a')
            page_text = page_a_list[-2].string

        except AttributeError or NameError:
            pass
        
        if page_text == None:
            page = page + 1

        elif int(page_text)>page:
            page = page + 1

        else:
            last = True

        time.sleep(1)

    return data

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
