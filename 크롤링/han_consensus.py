from pydoc import pager
from bleach import clean
import requests
from bs4 import BeautifulSoup
import datetime as dt
import pandas as pd
import FinanceDataReader as fdr
from dateutil.relativedelta import relativedelta
import re
import time

stocks = fdr.StockListing('KRX')
stocks['Symbol'] = stocks['Symbol'].astype(str)
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

def consensus_crawling_DB():
    last = False
    page = 1
    today = dt.datetime.today().strftime("%Y-%m-%d")
    start_date = (dt.datetime.today() - relativedelta(months=1)).strftime("%Y-%m-%d")
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

def consensus_crawling_DB_a_month_ago():
    last = False
    page = 1
    today = (dt.datetime.today() - relativedelta(months=1)).strftime("%Y-%m-%d")
    start_date = (dt.datetime.today() - relativedelta(months=2)).strftime("%Y-%m-%d")
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


data = consensus_crawling_DB()
data_a_month_ago = consensus_crawling_DB_a_month_ago()

def consensus_report_titles(name):
    new_list = [x[3] for x in data if x[1] == name]
    return new_list

def consensus_report_price(name):
    new_list = [x[4] for x in data if x[1] == name]
    return new_list

def consensus_report_view(name):
    new_list = [x[5] for x in data if x[1] == name]
    return new_list

def consensus_report_price_a_month_ago(name):
    new_list = [x[4] for x in data if x[1] == name]
    return new_list

def consensus_report_view_a_month_ago(name):
    new_list = [x[5] for x in data if x[1] == name]
    return new_list

