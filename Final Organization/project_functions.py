import requests
import numpy as np
from config import *
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import os
import json
import pandas as pd
import FinanceDataReader as fdr
import yfinance as yf
import FinanceDataReader as fdr

from dateutil.relativedelta import relativedelta
from docopt import docopt #해당 모듈을 어떻게 쓸지 자동으로 알려 줌
from warnings import filterwarnings

import nltk

from collections import defaultdict
from sklearn.pipeline import Pipeline

from es_corpus_reader import EsCorpusReader
from es_corpus_reader_keyword import EsCorpusReader_key
from es_corpus_reader_title import EsCorpusReader_title
from korean_text_normalizer import KoreanTextNormalizer
from gensim_vectorizer import GensimTfidVectorizer
from konlpy.tag import Okt
from PIL import Image
from wordcloud import ImageColorGenerator
from wordcloud import WordCloud
from collections import Counter
from math import pi
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from konlpy.tag import Hannanum
from config import * # config안의 정보 가져오기
from transformers import pipeline
import re

'''만든거'''
import crawling
import scoring


filterwarnings('ignore')

rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

# font_path = "C:/Windows/Fonts/malgun.ttf"
# font = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font)
# plt.rcParams['axes.unicode_minus'] = False



def df_plotting(df, title='Insert Title'):

    column_idx = df.columns.to_list()
    time_idx = df.index.to_list()

    plt.figure(figsize=(20,12), facecolor='w', dpi=300)
    
    my_palette = plt.cm.get_cmap("Set2", len(df.columns))

    for idx, i in enumerate(column_idx):
        color = my_palette(idx)
        plt.plot(time_idx, df[i], color=color, label=i, linewidth=3)

    plt.title(title, size=40)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=15, rotation=30)
    plt.yticks(fontsize=15)
    plt.show()

def daily_news_df(date_from, date_to):
    
    date_from = dt.datetime.strptime(date_from, '%Y-%m-%d').isoformat()[0:10]
    date_to = dt.datetime.strptime(date_to, '%Y-%m-%d').isoformat()[0:10]

    #date_interval = list(pd.date_range(dt.datetime.strptime(date_from, '%Y-%m-%d'), dt.datetime.strptime(date_to, '%Y-%m-%d'), freq='d'))  

    url = f"{ELASTIC_SEARCH_URL}/news/_search"

    query_day = {
        "size": 0,
            "query":{
                "range":{
                "created_at": {
                    "gte": date_from,
                    "lte": date_to
                    }
                }
            },
            "aggs": {
                "group_by_date":{
                "date_histogram": {
                    "field": "created_at",
                    "interval": "day"
                }
            }
        }
    }

    headers = {
        'Content-Type' : 'application/json'
    }

    resp = requests.get(
        url,
        data=json.dumps(query_day),
        headers=headers,
        auth= ELASTIC_SEARCH_AUTH
    )

    data = resp.json()

    daily = data['aggregations']['group_by_date']['buckets']
    df_daily = pd.DataFrame(data = daily)[['key_as_string', 'doc_count']]
    df_daily = df_daily.rename(columns={"key_as_string" : "date"})
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily = df_daily.set_index('date')

    return df_daily

def top_publisher(num_of_publisher, date_from, date_to):
    
    date_from = dt.datetime.strptime(date_from, '%Y-%m-%d').isoformat()[0:10]
    date_to = dt.datetime.strptime(date_to, '%Y-%m-%d').isoformat()[0:10]

    url = f"{ELASTIC_SEARCH_URL}/news/_search"

    query_publisher = {
    "size": 0,
        "query":{
            "range":{
                "created_at": {
                    "gte": date_from,
                    "lte": date_to
                    }
                }
            },
    "aggs": {
        "group_by_publisher":{
            "terms": {
                "field": "publisher.keyword",
                "size": num_of_publisher
                }
            }
        }
    }

    headers = {
        'Content-Type' : 'application/json'
    }

    resp = requests.get(
        url,
        data=json.dumps(query_publisher),
        headers=headers,
        auth= ELASTIC_SEARCH_AUTH
    )

    data = resp.json()

    dict = data['aggregations']['group_by_publisher']['buckets']

    df = pd.DataFrame(data=dict)

    # publisher_list = []
    # for i in range(len(dict)):
    #     publisher_list.append(dict[i]['key'])

    return df['key'].to_list()

def publisher_daily_news_trend (publisher_list, date_from, date_to):
    
    time_idx = daily_news_df(date_from, date_to).index.to_list()

    date_from = dt.datetime.strptime(date_from, '%Y-%m-%d').isoformat()[0:10]
    date_to = dt.datetime.strptime(date_to, '%Y-%m-%d').isoformat()[0:10]

    df = pd.DataFrame(index=time_idx)

    url = f"{ELASTIC_SEARCH_URL}/news/_search"

    headers = {
    'Content-Type' : 'application/json'
    }


    for keyword in publisher_list:
        query = {
            "size": 0,
            "query":{
                "range":{
                    "created_at": {
                        "gte": date_from,
                        "lte": date_to
                        }
                    }
                },
                "aggs": {
                    "group_by_date":{
                        "date_histogram": {
                            "field": "created_at",
                            "interval": "day"
                        }
                    }
                },
                "query":{
                    "match":{
                    "publisher": keyword
                    }
                }
        }
        
        resp = requests.get(
            url,
            data=json.dumps(query),
            headers=headers,
            auth= ELASTIC_SEARCH_AUTH
        )

        data = resp.json()
        data = data['aggregations']['group_by_date']['buckets']
        df_tmp = pd.DataFrame(data = data)[['key_as_string', 'doc_count']]
        df_tmp = df_tmp.rename(columns={"key_as_string" : "date"})
        df_tmp['date'] = pd.to_datetime(df_tmp['date'])
        df_tmp = df_tmp.set_index('date')

        df[keyword] = df_tmp['doc_count']


    return df
    
def keyword_daily_news_trend (keyword_list, date_from, date_to):

    time_idx = daily_news_df(date_from, date_to).index.to_list()

    date_from = dt.datetime.strptime(date_from, '%Y-%m-%d').isoformat()[0:10]
    date_to = dt.datetime.strptime(date_to, '%Y-%m-%d').isoformat()[0:10]

    df = pd.DataFrame(index=time_idx)

    url = f"{ELASTIC_SEARCH_URL}/news/_search"

    headers = {
    'Content-Type' : 'application/json'
    }


    for keyword in keyword_list:
        query = {
                "size": 0,
                    "query":{
                        "range":{
                        "created_at": {
                            "gte": date_from,
                            "lte": date_to
                            }
                        }
                },
                "aggs": {
                    "group_by_date":{
                        "date_histogram": {
                            "field": "created_at",
                            "interval": "day"
                        }
                    }
                },
                "query":{
                    "match":{
                    "body": keyword
                    }
                }
            }

        resp = requests.get(
            url,
            data=json.dumps(query),
            headers=headers,
            auth= ELASTIC_SEARCH_AUTH
        )

        data = resp.json()
        data = data['aggregations']['group_by_date']['buckets']
        df_tmp = pd.DataFrame(data = data)[['key_as_string', 'doc_count']]
        df_tmp = df_tmp.rename(columns={"key_as_string" : "date"})
        df_tmp['date'] = pd.to_datetime(df_tmp['date'])
        df_tmp = df_tmp.set_index('date')

        df[keyword] = df_tmp['doc_count']

    
    df = df.fillna(0)

    return df

def compare_relative_doc_count (df):

    total = daily_news_df("2022-05-01", "2022-05-31")

    column_list = df.columns.to_list()

    tmp = pd.DataFrame(index=df.index)

    for i in column_list:
        tmp[f'{i}_ratio'] = df[i] / total['doc_count']

    return tmp
    
def fetch_news_docs(date_from, date_to, page):
    
    date_from = date_from.isoformat()[0:10]
    date_to = date_to.isoformat()[0:10]

    query = {
        "query": {
            "range":{
                "created_at": {
                    "gte": date_from,
                    "lte": date_to
                }
            }
        },
        "size": 10,
        "from": page * 10
    }

    headers = {
        'Content-Type': 'application/json'
    }

    resp = requests.get(
        f'{ELASTIC_SEARCH_URL}/news/_search',
        headers = headers,
        data = json.dumps(query),
        auth = ELASTIC_SEARCH_AUTH
    )

    assert resp.status_code == 200

    data = json.loads(resp.text)
    hits = data['hits']['hits']

    return hits

def get_titles(hits):

    return hits['_source']['title']

def download_total_title (date_from, date_to):

    date_from = dt.datetime.strptime(date_from, '%Y-%m-%d').isoformat()[0:10]
    date_to = dt.datetime.strptime(date_to, '%Y-%m-%d').isoformat()[0:10]

    date_interval = list(pd.date_range(date_from, date_to, freq='d'))

    # print("Downloading data from OpenSearch server")

    total_title_list = []

    for i in range(len(date_interval)):

        str_date = date_interval[i].strftime('%Y-%m-%d')

        # print(f"{str_date} Start!")

        daily_title_list = []

        start_date = date_interval[i]
        end_date = date_interval[i] + relativedelta(days=1)

        for page in range(1000):

#            print('.', end='', flush=True)

            hits = fetch_news_docs(start_date, end_date, page)

            if len(hits) == 0:
                break

            for doc in hits:
            
                daily_title_list.append(get_titles(doc))

        total_title_list.append(daily_title_list)

        # print("***" + str_date + "End!" + "***")

    # print("Daily Data end!")

    return total_title_list

def total_title_show (total_title_list, date_from, date_to):

    date_from = dt.datetime.strptime(date_from, '%Y-%m-%d').isoformat()[0:10]
    date_to = dt.datetime.strptime(date_to, '%Y-%m-%d').isoformat()[0:10]

    date_interval = list(pd.date_range(date_from, date_to, freq='d'))

    for i in range(len(date_interval)):

        str_date = date_interval[i].strftime('%Y-%m-%d')

        print(str_date)
        print(total_title_list[i] + '\n')

    #return date_interval

def txt_title_save (total_title_list, date_from, date_to):

    date_from = dt.datetime.strptime(date_from, '%Y-%m-%d').isoformat()[0:10]
    date_to = dt.datetime.strptime(date_to, '%Y-%m-%d').isoformat()[0:10]

    date_interval = list(pd.date_range(date_from, date_to, freq='d'))

    with open('title_list_allday.txt','w',encoding='UTF-8') as f:

        for i in range(len(total_title_list)):
        
            str_date = date_interval[i].strftime('%Y-%m-%d')

            f.write('\n' + '[' + str_date + ']' + '\n\n')
        
            for titles in total_title_list[i]:

                f.write(titles + '\n')

def fetch_news_docs(date_from, date_to, page):
    
    date_from = date_from.isoformat()[0:10]
    date_to = date_to.isoformat()[0:10]

    query = {
        "query": {
            "range":{
                "created_at": {
                    "gte": date_from,
                    "lte": date_to
                }
            }
        },
        "size": 10,
        "from": page * 10
    }

    headers = {
        'Content-Type': 'application/json'
    }

    resp = requests.get(
        f'{ELASTIC_SEARCH_URL}/news/_search',
        headers = headers,
        data = json.dumps(query),
        auth = ELASTIC_SEARCH_AUTH
    )

    assert resp.status_code == 200

    data = json.loads(resp.text)
    hits = data['hits']['hits']

    return hits

def get_titles(hits):

    return hits['_source']['title']

def download_total_title (date_from, date_to):

    date_from = dt.datetime.strptime(date_from, '%Y-%m-%d').isoformat()[0:10]
    date_to = dt.datetime.strptime(date_to, '%Y-%m-%d').isoformat()[0:10]

    date_interval = list(pd.date_range(date_from, date_to, freq='d'))

    # print("Downloading data from OpenSearch server")

    total_title_list = []

    for i in range(len(date_interval)):

        str_date = date_interval[i].strftime('%Y-%m-%d')

        # print(f"{str_date} Start!")

        daily_title_list = []

        start_date = date_interval[i]
        end_date = date_interval[i] + relativedelta(days=1)

        for page in range(1000):

#            print('.', end='', flush=True)

            hits = fetch_news_docs(start_date, end_date, page)

            if len(hits) == 0:
                break

            for doc in hits:
            
                daily_title_list.append(get_titles(doc))

        total_title_list.append(daily_title_list)

        # print("***" + str_date + "End!" + "***")

    # print("Daily Data end!")

    return total_title_list

def total_title_show (total_title_list, date_from, date_to):

    date_from = dt.datetime.strptime(date_from, '%Y-%m-%d').isoformat()[0:10]
    date_to = dt.datetime.strptime(date_to, '%Y-%m-%d').isoformat()[0:10]

    date_interval = list(pd.date_range(date_from, date_to, freq='d'))

    for i in range(len(date_interval)):

        str_date = date_interval[i].strftime('%Y-%m-%d')

        print(str_date)
        print(total_title_list[i] + '\n')

    #return date_interval

def txt_title_save (total_title_list, date_from, date_to):

    date_from = dt.datetime.strptime(date_from, '%Y-%m-%d').isoformat()[0:10]
    date_to = dt.datetime.strptime(date_to, '%Y-%m-%d').isoformat()[0:10]

    date_interval = list(pd.date_range(date_from, date_to, freq='d'))

    with open('title_list_allday.txt','w',encoding='UTF-8') as f:

        for i in range(len(total_title_list)):
        
            str_date = date_interval[i].strftime('%Y-%m-%d')

            f.write('\n' + '[' + str_date + ']' + '\n\n')
        
            for titles in total_title_list[i]:

                f.write(titles + '\n')

def fetch_news_docs_keyword(date_from, date_to, page, keyword):
    
    date_from = date_from.isoformat()[0:10]
    date_to = date_to.isoformat()[0:10]

    query = {
        "query": {
            "bool": {
                "must":[
                    {
                        "match":{
                            "body": keyword
                        }
                    },
                    {
                    "range":{
                        "created_at": {
                            "gte": date_from,
                            "lt": date_to
                            }
                        }
                    }
                ]
            },
        },
        "size": 10,
        "from": page * 10
    }

    headers = {
        'Content-Type': 'application/json'
    }

    resp = requests.get(
        f'{ELASTIC_SEARCH_URL}/news/_search',
        headers = headers,
        data = json.dumps(query),
        auth = ELASTIC_SEARCH_AUTH
    )

    assert resp.status_code == 200

    data = json.loads(resp.text)
    hits = data['hits']['hits']

    return hits

def get_bodies(hits):

    return hits['_source']['body']


def download_total_title_keyword (date_from, date_to, keyword):

    date_from = dt.datetime.strptime(date_from, '%Y-%m-%d').isoformat()[0:10]
    date_to = dt.datetime.strptime(date_to, '%Y-%m-%d').isoformat()[0:10]

    date_interval = list(pd.date_range(date_from, date_to, freq='d'))

    # print("Downloading data from OpenSearch server")

    total_title_list = []

    for i in range(len(date_interval)):

        str_date = date_interval[i].strftime('%Y-%m-%d')

        # print(f"{str_date} Start!")

        daily_title_list = []

        start_date = date_interval[i]
        end_date = date_interval[i] + relativedelta(days=1)

        for page in range(1000):

#            print('.', end='', flush=True)

            hits = fetch_news_docs_keyword(start_date, end_date, page, keyword)

            if len(hits) == 0:
                break

            for doc in hits:
            
                daily_title_list.append(get_titles(doc))

        total_title_list.append(daily_title_list)

        # print("***" + str_date + "End!" + "***")

    # print("Daily Data end!")

    return total_title_list

def download_total_bodies_keyword (date_from, date_to, keyword):

    date_from = dt.datetime.strptime(date_from, '%Y-%m-%d').isoformat()[0:10]
    date_to = dt.datetime.strptime(date_to, '%Y-%m-%d').isoformat()[0:10]

    date_interval = list(pd.date_range(date_from, date_to, freq='d'))

    # print("Downloading data from OpenSearch server")

    total_title_list = []

    for i in range(len(date_interval)):

        str_date = date_interval[i].strftime('%Y-%m-%d')

        # print(f"{str_date} Start!")

        daily_title_list = []

        start_date = date_interval[i]
        end_date = date_interval[i] + relativedelta(days=1)

        for page in range(1000):

#            print('.', end='', flush=True)

            hits = fetch_news_docs_keyword(start_date, end_date, page, keyword)

            if len(hits) == 0:
                break

            for doc in hits:
            
                daily_title_list.append(get_bodies(doc))

        total_title_list.append(daily_title_list)

        # print("***" + str_date + "End!" + "***")

    # print("Daily Data end!")

    return total_title_list
    
def classification_top_n(count_dict, n=3):
    
    for group in range(len(count_dict)):
        count_dict[group].sort()

    top_n = dict(sorted(count_dict.items(), reverse=True, key=lambda x: len(x[1]))[:n])

    i = 1

    for key in top_n.keys():
    
        print(f"*** TOP {i} Section ***")
        
        for idx, content in enumerate(top_n[key]):
            
            if idx < 5:
                print(f"{content}")
            else:
                pass
        i += 1

def classification_top_1(count_dict, n=1):
    
    for group in range(len(count_dict)):
        count_dict[group].sort()

    top_n = dict(sorted(count_dict.items(), reverse=True, key=lambda x: len(x[1]))[:n])

    i = 1

    for key in top_n.keys():
        
        for idx, content in enumerate(top_n[key]):
            
            if idx < 5:
                print(f"{content}")
            else:
                pass
        i += 1

def daily_k_means_top3(date_from, date_to):

    date_from = (dt.datetime.strptime(date_from, '%Y-%m-%d').astimezone(dt.timezone.utc) + dt.timedelta(hours=9)).isoformat()[0:10]
    date_to = (dt.datetime.strptime(date_to, '%Y-%m-%d').astimezone(dt.timezone.utc) + dt.timedelta(hours=9)).isoformat()[0:10]

    date_interval = list(pd.date_range(date_from, date_to, freq='d'))

    total_doc_count = daily_news_df(date_from, date_to)

    for time in date_interval:

        str_time = time.strftime('%Y-%m-%d')

        reader = EsCorpusReader(date_from=time, date_to=time+dt.timedelta(days=1))

        doc_num = total_doc_count.loc[time]['doc_count']
 
        corpus = list(reader.titles(n=100))


        model = Pipeline([
            ('normalizer', KoreanTextNormalizer()),
            ('vectorizer', GensimTfidVectorizer())
        ])

        vectors = model.fit_transform(corpus)

        num_means = 20
        distance = nltk.cluster.cosine_distance

        kmeans = nltk.cluster.KMeansClusterer(
            num_means=num_means,
            distance=distance,
            avoid_empty_clusters=True
        )

        kmeans.cluster(vectors)

        classified = defaultdict(list)

        for doc, vec in zip(corpus, vectors):
            group = kmeans.classify(vec)
            mean = kmeans.means()[group]
            dist = distance(vec, mean)

            entry = (dist, doc)

            classified[group].append(entry)

        print(f"-------- {str_time}'s TOP 3 ISSUES --------")
        classification_top_n(classified, n=3)
        print("\n")
        
        # for group in range(len(classified)):
        #     print(f'*** Group {group} ***')

        #     classified[group].sort() #작은거에서 큰걸로 그룹안에 점수 정렬함 (양극단값이 아래로 내려가고 중앙값이 가운데로 올라옴)

        #     print("Topic Size: {}".format(len(classified[group])))

        #     for idx, x in enumerate(classified[group]):
        #         print(f'{idx}: {x}')

        #         if idx > 5:
        #             break

        #     print()

    # return classified

def daily_k_means_top_keyword(date_from, date_to, keyword):

    date_from = (dt.datetime.strptime(date_from, '%Y-%m-%d').astimezone(dt.timezone.utc) + dt.timedelta(hours=9)).isoformat()[0:10]
    date_to = (dt.datetime.strptime(date_to, '%Y-%m-%d').astimezone(dt.timezone.utc) + dt.timedelta(hours=9)).isoformat()[0:10]

    date_interval = list(pd.date_range(date_from, date_to, freq='d'))

    total_doc_count = keyword_daily_news_trend([keyword], date_from, date_to)

    for time in date_interval:

        str_time = time.strftime('%Y-%m-%d')

        reader = EsCorpusReader_key(date_from=time, date_to = time+dt.timedelta(days=1), keyword=keyword)

        doc_num = total_doc_count.loc[time][total_doc_count.columns[0]]
 
        corpus = list(reader.titles(n=doc_num))

        model = Pipeline([
            ('normalizer', KoreanTextNormalizer()),
            ('vectorizer', GensimTfidVectorizer())
        ])

        vectors = model.fit_transform(corpus)

        num_means = 8
        distance = nltk.cluster.cosine_distance

        kmeans = nltk.cluster.KMeansClusterer(
            num_means=num_means,
            distance=distance,
            avoid_empty_clusters=True
        )

        kmeans.cluster(vectors)

        classified = defaultdict(list)

        for doc, vec in zip(corpus, vectors):
            group = kmeans.classify(vec)
            mean = kmeans.means()[group]
            dist = distance(vec, mean)

            entry = (dist, doc)

            classified[group].append(entry)

        print(f'''-------- {str_time}'s TOP ISSUE FOR "{keyword}" --------''')
        classification_top_1(classified, n=1)
        print("\n")
#함수들 모음

# 날짜: 자동으로 설정
# 티커: 함수이용

# 해당하는 종목코드 추출

def code_list_create (port_list):
    code_list = []
    for i in port_list:
        code_list.append(code(i))

    return code_list
    

def code(name):

    a = stockcode[stockcode['Name'] == name]
    a_code = a['Symbol'].item()
    a_market = a['Market'].item()
    
    if a_market == 'KOSPI':
        a_market = '.KS'
    
    elif a_market == 'KOSDAQ':
        a_market = '.KQ'
    
    return a_code + a_market

def port_weight_creator (port, date_from, date_to):
    
    port_list = port['종목명'].to_list()

    port = port.set_index('종목명')

    port_dict = {}

    for k in port_list:
        port_dict[code(k)] = k

    code_list = code_list_create(port_list)

    date_from = pd.to_datetime(date_from) - dt.timedelta(days=5)
    date_to = pd.to_datetime(date_to) + dt.timedelta(days=1)
    str_date_from = date_from.strftime("%Y-%m-%d")
    str_date_to = date_to

    df = yf.download(code_list, start=str_date_from, end=date_to, progress=False)['Adj Close']
    df2 = df.pct_change()

    for i in list(port_dict.keys()):
        df.rename(columns={i: port_dict[i]}, inplace=True)

    for i in list(port_dict.keys()):
        df2.rename(columns={i: port_dict[i]}, inplace=True)


    port['weight'] = df.iloc[-1,:].T
    port['weight'] = (port['수량'] * port['weight']) / (port['수량'] * port['weight']).sum()
    port['price'] = df.iloc[-1,:].T
    port['yet_ret'] = df2.iloc[-1,:].T

    return port

# 어제 수익률 보기
def yesterday_rtn(list, date_from, date_to):

    date_from = pd.to_datetime(date_from) - dt.timedelta(days=5)

    df = yf.download(list, start=date_from, end=date_to, progress=False)

    df['daily_rtn'] = df['Adj Close'].pct_change()
    last_rtn = df['daily_rtn'][-1].item()
    
    return last_rtn

# 한달동안 일별 수익률 추이 보기
def monthly_rtn_graph(name):
    
    b = code(name)
    
    today = dt.datetime.today()
    monthago = dt.datetime.today() - relativedelta(months=1)

    df = yf.download(b, start=str(monthago), end=str(today), progress=False)
    df['daily_rtn'] = df['Adj Close'].pct_change()

    return df['daily_rtn'].plot()



def port_individual_chart (date_from, date_to, port):

    date_from = pd.to_datetime(date_from)
    date_to = pd.to_datetime(date_to)

    if date_from == date_to:
        date_to = date_to + dt.timedelta(days=1)
    else:
        date_to = date_to

    str_date_from = date_from.strftime("%Y-%m-%d")
    str_date_to = date_to.strftime("%Y-%m-%d")

    port_stock_list = port.index.to_list()
    idx_num = len(port_stock_list)

    if (idx_num % 2) == 0:
        sub_idx_1 = int(idx_num / 2)
        sub_idx_2 = 2

    else:
        sub_idx_1 = int((idx_num + 1) / 2)
        sub_idx_2 = 2

    fig = plt.figure(figsize=(15, 27), dpi=300)
    fig.set_facecolor('w')

    my_palette = plt.cm.get_cmap("Set2", len(port_stock_list))

    for idx, i in enumerate(port_stock_list):
        
        color = my_palette(idx)

        tmp =yf.download(code(i), str_date_from, str_date_to, interval='2m', progress=False)[['Close']][:-1]

        ax = plt.subplot(sub_idx_1, sub_idx_2, idx+1)
        ax.plot(tmp.index, tmp['Close'], color=color, label=i)
        ax.legend()
        plt.xticks(rotation=30)
        plt.title(f"{i} 주가 그래프 ({str_date_from})", fontsize=20)

    plt.tight_layout()
    plt.show()

def classification_top_1_return_title (count_dict, n):
    
    for group in range(len(count_dict)):
        count_dict[group].sort()

    top_n = dict(sorted(count_dict.items(), reverse=True, key=lambda x: len(x[1]))[:n])

    i = 1

    tmp_list = []

    for key in top_n.keys():
        
        for idx, content in enumerate(top_n[key]):
            
            if idx < 1:
                tmp_list.append(content)
            else:
                pass
        i += 1

    return tmp_list

def download_total_url_from_title (date_from, date_to, title):

    date_from = dt.datetime.strptime(date_from, '%Y-%m-%d').isoformat()[0:10]
    date_to = dt.datetime.strptime(date_to, '%Y-%m-%d').isoformat()[0:10]

    date_interval = list(pd.date_range(date_from, date_to, freq='d'))

    # print("Downloading data from OpenSearch server")

    total_title_list = []

    for i in range(len(date_interval)):

        str_date = date_interval[i].strftime('%Y-%m-%d')

        # print(f"{str_date} Start!")

        daily_title_list = []

        start_date = date_interval[i]
        end_date = date_interval[i] + relativedelta(days=1)

        for page in range(1000):

#            print('.', end='', flush=True)

            hits = fetch_news_docs_title(start_date, end_date, page, title)

            if len(hits) == 0:
                break

            for doc in hits:
            
                daily_title_list.append(get_bodies(doc))

        total_title_list.append(daily_title_list)

        # print("***" + str_date + "End!" + "***")

    # print("Daily Data end!")

    return total_title_list


def daily_k_means_top_keyword_with_url(date_from, date_to, keyword):

    date_from = (dt.datetime.strptime(date_from, '%Y-%m-%d').astimezone(dt.timezone.utc) + dt.timedelta(hours=9)).isoformat()[0:10]
    date_to = (dt.datetime.strptime(date_to, '%Y-%m-%d').astimezone(dt.timezone.utc) + dt.timedelta(hours=9)).isoformat()[0:10]

    date_interval = list(pd.date_range(date_from, date_to, freq='d'))

    total_doc_count = keyword_daily_news_trend([keyword], date_from, date_to)

    for time in date_interval:

        str_time = time.strftime('%Y-%m-%d')

        reader = EsCorpusReader_key(date_from=time, date_to = time+dt.timedelta(days=1), keyword=keyword)

        doc_num = total_doc_count.loc[time][total_doc_count.columns[0]]
 
        corpus = list(reader.titles(n=doc_num))

        model = Pipeline([
            ('normalizer', KoreanTextNormalizer()),
            ('vectorizer', GensimTfidVectorizer())
        ])

        vectors = model.fit_transform(corpus)

        num_means = 8
        distance = nltk.cluster.cosine_distance

        kmeans = nltk.cluster.KMeansClusterer(
            num_means=num_means,
            distance=distance,
            avoid_empty_clusters=True
        )

        kmeans.cluster(vectors)

        classified = defaultdict(list)

        for doc, vec in zip(corpus, vectors):
            group = kmeans.classify(vec)
            mean = kmeans.means()[group]
            dist = distance(vec, mean)

            entry = (dist, doc)

            classified[group].append(entry)

        print(f'''-------- {str_time}'s 3 TOP ISSUES FOR "{keyword}" --------''')
        returned_title = classification_top_1_return_title(classified, n=3)

        count = 1
        for i in returned_title:
            reader = EsCorpusReader_title(date_from=time, date_to = time+dt.timedelta(days=1), title=i[1])
            corpus = list(reader.urls(n=1))[0]

            print(f"{count} ISSUE : {i[1]}")
            print(f"URL = {corpus}")
            count += 1

        print("\n")

def my_port_issue_with_url(date_from, date_to, port_list):

    for i in port_list:
        daily_k_means_top_keyword_with_url(date_from, date_to, i)

def sentiment_hugging_face_docs(date_from, date_to, port_list):

    date_from = (dt.datetime.strptime(date_from, '%Y-%m-%d').astimezone(dt.timezone.utc) + dt.timedelta(hours=9)).isoformat()[0:10]
    date_to = (dt.datetime.strptime(date_to, '%Y-%m-%d').astimezone(dt.timezone.utc) + dt.timedelta(hours=9)).isoformat()[0:10]

    doc_list = []
    for i in port_list:
        doc_list.append(download_total_title_keyword(date_from, date_to, i)[0])

    # return doc_list
    sentiment_score = hugging_list(doc_list)

    return sentiment_score
    

def hugging_list (doc_list):
    '''list가 있어요, 그 안에 리스트가 또 있어요 하하 [ [삼성전자 글들...] ], [셀트리온 글들...], [ㅇㄹㅇㄹㅇㄹ]  ] '''
    sentiment_score = list()

    for i in range(len(doc_list)):

        result = classifier(doc_list[i])

        neutral_score = 1
        positive_score = 1
        negative_score = 1


        for k in range(len(result)):
            #print(f"제목 : {i[k]}")
            #print(f"이 뉴스는 {round(result[k]['score'] * 100, 2)}% 확률로 {result[k]['label']} 입니다.\n")

            if result[k]['label'] == 'neutral':
                neutral_score *= 0
            elif result[k]['label'] == 'positive':
                positive_score += 1 * result[k]['score']
            else:
                negative_score += (-1) * result[k]['score']

        if len(result) == 0:
            total_score = np.nan
            sentiment_score.append(total_score) 
        else:
            total_score = (neutral_score + positive_score + negative_score) / len(result) 
            sentiment_score.append(total_score) 

    # print(f"neutral : {neutral_score/len(result)}")
    # print(f"positive : {positive_score/len(result)}")
    # print(f"negative : {negative_score/len(result)}")

        # print(f"Sentiment Score of {port_list[i]} : {total_score}")

    return sentiment_score

def port_create_stn_news_score (date_from, date_to, port):

    port_list = port.index.to_list()

    score = sentiment_hugging_face_docs(date_from, date_to, port_list)

    port['snt_news_title_score'] = score

    return port

## 뉴스개수 증감 factor
def news_num_change(name): 
    
    a = keyword_daily_news_trend(port_stock_list, "2022-04-30", "2022-05-31")
    b = a.loc['2022-05-17'][name]
    c = a.loc['2022-05-18'][name]

    return (c-b) / b

## 뉴스개수 증감 factor
def news_num_change(name): 
    
    a = keyword_daily_news_trend(port_stock_list, "2022-04-30", "2022-05-31")
    b = a.loc['2022-05-17'][name]
    c = a.loc['2022-05-18'][name]

    return (c-b) / b

def port_create_snt_consensus_score (port):
    ## 한경컨센서스 센티멘트 factor 점수 df에 추가하기
    doc_list_consensus = []
    for stockname in port.index.to_list():
        doc_list_consensus.append(scoring.consensus_report_titles(stockname, data)) # 포트폴리오 종목별 한경컨센서스 제목 불러 오기

    sentiment_score_consensus = hugging_list(doc_list_consensus) # 한경컨센서스 제목 센티멘트 분석

    port['snt_consensus_title_score'] = sentiment_score_consensus # df에 한경컨센서스 제목 센티멘트 점수 추가

    return port

def port_create_achievement_ratio_score (port):
    achievement_ratio_port = []
    for stockname in port.index.to_list():
        achievement_ratio_port.append(scoring.achievement_ratio(stockname, data))

    port['achievement_ratio'] = achievement_ratio_port

    return port

def port_create_target_ratio_score (port):
    target_change_port = []
    for stockname in port.index.to_list():
        target_change_port.append(scoring.target_change(stockname,data,data_ago))

    port['target_change'] = target_change_port

    return port

def port_create_reply_vol_score (port):
    reply_vol_port = []
    for stockname in port.index.to_list():
        reply_vol_port.append(scoring.reply_vol(stockname))

    port['reply_vol'] = reply_vol_port    

    return port

def port_create_news_num_change_score (port):
    news_num_change_port = []
    for stockname in port.index.to_list():
        news_num_change_port.append(news_num_change(stockname))

    port['news_num_change'] = news_num_change_port

    return port

def rader_graph (port):

    labels = port.columns[4:]
    num_labels = len(labels)

    tmp = port[labels]
    tmp = tmp.fillna(0)

    tmp.rename(columns={"snt_news_title_score": "뉴스들 갬성", "snt_consensus_title_score":"애널들 갬성", "achievement_ratio": "적정가 대비 현재주가", "target_change":"전문가들 말바꿈 정도", "reply_vol":"토론방 시끌벅적도", "news_num_change":"기자들 시끌벅적도"}, inplace=True)
    labels = tmp.columns.to_list()


    angles = [x/float(num_labels)*(2*pi) for x in range(num_labels)]
    angles += angles[:1]

    my_palette = plt.cm.get_cmap("Set2", len(tmp.index))

    x_angles = [0 for x in range(300)]
    y_angles = [x/float(300)*(2*pi) for x in range(300)]

    fig = plt.figure(figsize=(15, 27), dpi=300)
    fig.set_facecolor('w')

    idx_num = len(tmp.index)

    if (idx_num % 2) == 0:
        sub_idx_1 = int(idx_num / 2)
        sub_idx_2 = 2

    else:
        sub_idx_1 = int((idx_num + 1) / 2)
        sub_idx_2 = 2

    for i, row in tmp.iterrows():

        idx = tmp.index.to_list().index(i)
        sub_idx_3 = idx + 1

        color = my_palette(idx)


        data = tmp.iloc[idx].to_list()    
        data += data[:1]

        ax = plt.subplot(sub_idx_1, sub_idx_2, sub_idx_3, polar=True)
        ax.set_theta_offset(pi/2)
        ax.set_theta_direction(-1)

        plt.xticks(angles[:-1], labels, fontsize=13)
        ax.tick_params(axis='x', width='major', pad=15)

        ax.set_rlabel_position(0)
        plt.yticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5], ['-1.5', '-1', '-0.5', '0', '0.5', '1', '1.5'], fontsize=9)
        plt.ylim(-1.5,1.5)

        ax.plot(y_angles, x_angles, color='grey', linewidth=1.7)
        ax.plot(angles, data, color=color, linewidth=2, linestyle='solid')
        ax.fill(angles, data, color=color, alpha=0.4)

        plt.title(i, size=30, color=color, x=-0.15, y=1.1, ha='center', fontweight='bold')

    plt.tight_layout(pad=5)
    plt.show()

def total_port_df (port):

    port_weight = port['weight']
    port_cal = port[port.columns[4:]]

    port_idx = ['나의 포트폴리오']

    total_port = pd.DataFrame(index=port_idx, columns=['stockname', 'weight', 'price', 'return'])

    port = port.fillna(0)

    for i in port.columns[4:]:
        
        total_port[i] = [port_weight.T @ port[i]]

    return total_port

def return_port_weather_score (port):

    port = port.dropna(axis=1)

    return round((port.T.sum() / len(port.columns)).values[0], 3)
    
def preprocess(date_from, date_to, keyword_list):
    
    buffer = []

    for keyword in keyword_list:
        key = download_total_title_keyword (date_from, date_to, keyword)
        key = sum(key,[])
        

        key=' '.join(s for s in key)
        buffer.append(key)
    
    
    
    return buffer 

def plot_wordcloud(bodies,weight,score=None):

    '''글 '''
    '''
    각 종목들의 기사 body들과 포트폴리오 가중치를 인수로 받아 
    가중치만큼 반영하여 빈도수에 가중하여 워드클라우드로 출력
    '''


    weight = np.array(weight)*1000
    
    np.rint(weight)

    buffer=[]
    
    for i,v in enumerate(bodies):
        okt = Okt()
        noun_list = list(okt.nouns(v))
        pos_list = list(okt.pos(v))
        noun_list=[x for x in noun_list if len(x)>1]

        noun_list*= int(weight[i])

        buffer.append(noun_list)
        
    buffer = sum(buffer,[]) # 리스트의 리스트를 하나의 리스트로 전환
        

    dancing_mask = np.array(Image.open('./wordcloud/psy2.jpg').convert("RGBA"))
    sunny_mask = np.array(Image.open('./wordcloud/sunny.jpg').convert("RGBA"))
    sunny_cloudy_mask = np.array(Image.open('./wordcloud/sunny_clouds.jpg').convert("RGBA"))
    cloudy_mask = np.array(Image.open('./wordcloud/cloud2.png').convert("RGBA"))
    rainy_mask = np.array(Image.open('./wordcloud/raindrop.png').convert("RGBA"))

    if score is None:
        mask = None

    else:
        if score >= 1:
            mask = dancing_mask
            
        elif score >= 0.5:
            mask = sunny_mask
            
        elif score >= 0:
            mask = sunny_cloudy_mask
            
        elif score >= -0.5:
            mask = cloudy_mask
            
        elif score < -0.5:
            mask = rainy_mask        

    if mask is not None:
    
        wordcloud = WordCloud(font_path='./wordcloud/HANYheadM.ttf', mask=mask, width=800, height=800,background_color='white', color_func=ImageColorGenerator(mask))
    
    else:
        wordcloud = WordCloud(font_path='./wordcloud/HANYheadM.ttf', mask=mask, width=800, height=800,background_color='white')


    count = Counter(buffer)
    wordcloud = wordcloud.generate_from_frequencies(count)
    array = wordcloud.to_array()

    fig = plt.figure(figsize=(20,20), dpi=300)
    plt.imshow(array, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    return


def wordcloud_creator_with_keyword (date_from, date_to, keyword_list, weight, score=None):
    
    '''weight는 포트폴리오 웨이트는 리스트로 들어가야댐, Score는 날씨 정보'''

    buffer = preprocess(date_from, date_to, keyword_list)

    plot_wordcloud(buffer, weight, score)

def port_result (date_from, date_to, port):
    
    try:
        port_individual_chart(date_from, date_to, port)
    except:
        pass

    my_port_issue_with_url(date_from, date_to, port.index.to_list())

    port = port_create_stn_news_score(date_from, date_to, port)
    port = port_create_snt_consensus_score(port)
    port = port_create_achievement_ratio_score(port)
    port = port_create_target_ratio_score(port)
    port = port_create_reply_vol_score(port)
    port = port_create_news_num_change_score(port)
    
    rader_graph(port)
    rader_graph(total_port_df(port))
    wordcloud_creator_with_keyword(date_from, date_to, port.index.to_list(), port.weight.to_list(), return_port_weather_score(total_port_df(port)))

