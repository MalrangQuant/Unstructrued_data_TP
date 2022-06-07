import requests
import numpy as np
from config import *
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

def daily_news():
    url = f"{ELASTIC_SEARCH_URL}/news/_search"

    query_day = """
    {
    "size": 0,
    "aggs": {
        "group_by_date":{
        "date_histogram": {
            "field": "created_at",
            "interval": "day"
        }
        }
    }
    }
    """

    headers = {
        'Content-Type' : 'application/json'
    }

    resp = requests.get(
        url,
        data=query_day,
        headers=headers,
        auth= ELASTIC_SEARCH_AUTH
    )

    data = resp.json()

    daily = data['aggregations']['group_by_date']['buckets']

    df_daily = pd.DataFrame(data = daily)

    df_daily['date'] = pd.to_datetime(df_daily['key_as_string'])

    df_daily = df_daily.set_index('date')

    plt.figure(figsize=(12,8), dpi=400)
    plt.plot(df_daily['doc_count'], color='b', label = 'Daily News')
    plt.legend()
    plt.xticks(rotation = 30)
    plt.xlabel('Date')
    plt.ylabel('Number of News')
    plt.show()
    return df_daily
    

def top_publisher():
    url = f"{ELASTIC_SEARCH_URL}/news/_search"

    query_publisher = """
    {
    "size": 0,
    "aggs": {
        "group_by_publisher":{
        "terms": {
            "field": "publisher.keyword"
        }
        }
    }
    }
    """

    headers = {
        'Content-Type' : 'application/json'
    }

    resp = requests.get(
        url,
        data=query_publisher,
        headers=headers,
        auth= ELASTIC_SEARCH_AUTH
    )

    data = resp.json()

    dict = data['aggregations']['group_by_publisher']['buckets'][1:6]

    publisher_list = []
    for i in range(len(dict)):
        publisher_list.append(dict[i]['key'])

    return publisher_list

def news_trends_by_publisher(publisher_list):
    url = f"{ELASTIC_SEARCH_URL}/news/_search"
    
    df_publisher = pd.DataFrame()

    for pub in publisher_list:
        query = """
            {
                "size": 0,
                "aggs": {
                    "group_by_publisher":{
                        "terms":{
                            "field":"publisher.keyword",
                            "include": "%s"
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
                }
            }
        """ %str(pub)
        headers = {
        'Content-Type' : 'application/json'
        }

        resp = requests.get(
            url,
            data=query.encode('utf-8'),
            headers=headers,
            auth= ELASTIC_SEARCH_AUTH
        )

        data = resp.json()

        publisher_daily = data['aggregations']['group_by_publisher']['buckets'][0]['group_by_date']['buckets']
        df_temp = pd.DataFrame(data = publisher_daily)
        df_temp['date'] = pd.to_datetime(df_temp['key_as_string'])
        df_temp = df_temp.set_index('date')
        df_temp = df_temp['2022-05-01':'2022-05-31']
        df_publisher[pub] = df_temp['doc_count']

    plt.figure(figsize=(15,9))
    plt.plot(df_publisher.iloc[:,[0]], label = 'News1')
    plt.plot(df_publisher.iloc[:,[1]], label = 'Newsis')
    plt.plot(df_publisher.iloc[:,[2]], label = 'E-daily')
    plt.plot(df_publisher.iloc[:,[3]], label = 'Financial news')
    plt.plot(df_publisher.iloc[:,[4]], label = 'Money Today')
    plt.legend(loc = 'upper left')
    plt.xlabel('Date')
    plt.ylabel('Number of News')
    plt.xticks(rotation = 30)
    plt.show()
    
    return df_publisher


def news_trends_by_keyword(keyword_list, df_daily):
    url = f"{ELASTIC_SEARCH_URL}/news/_search"
    
    df_keyword = pd.DataFrame()
    df_ratio = pd.DataFrame()

    for keyword in keyword_list:
        query = """
            {
                "size": 0,
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
                    "body": "%s"
                    }
                }
            }
        """ %str(keyword)

        headers = {
        'Content-Type' : 'application/json'
        }

        resp = requests.get(
            url,
            data=query.encode('utf-8'),
            headers=headers,
            auth= ELASTIC_SEARCH_AUTH
        )

        data = resp.json()
        keyword_daily = data['aggregations']['group_by_date']['buckets']
        df_temp = pd.DataFrame(data = keyword_daily)
        df_temp['date'] = pd.to_datetime(df_temp['key_as_string'])
        df_temp = df_temp.set_index('date')
        df_temp = df_temp['2022-05-01':'2022-05-31']
        df_keyword[keyword] = df_temp['doc_count']

        df_ratio[keyword] = (df_keyword[keyword] / df_daily['doc_count']) * 100
    df_ratio.head()
    plt.figure(figsize=(15,9))
    plt.plot(df_keyword.iloc[:,[0]], label = 'War')
    plt.plot(df_keyword.iloc[:,[1]], label = 'Inflation')
    plt.plot(df_keyword.iloc[:,[2]], label = 'Crypto Currency')
    plt.legend(loc = 'upper left')
    plt.xlabel('Date')
    plt.ylabel('Number of News')
    plt.xticks(rotation = 30)

    plt.figure(figsize=(15,9))
    plt.plot(df_ratio.iloc[:,[0]], label = 'War')
    plt.plot(df_ratio.iloc[:,[1]], label = 'Inflation')
    plt.plot(df_ratio.iloc[:,[2]], label = 'Crypto Currency')
    plt.legend(loc = 'upper left')
    plt.xlabel('Date')
    plt.ylabel('Ratio [%]')
    plt.xticks(rotation = 30)
    plt.show()
    return df_keyword


def collect_titles():
    url = f"{ELASTIC_SEARCH_URL}/news/_search"
    title = []
    for i in np.arange(1,32,1):
        if i < 10:
            query_body = """
            {
            "from": 0, 
            "size" : 7000,
            "query":{
                "match": {
                "created_at": "2022-05-0%s"
                }
            },
            "sort": [
                {
                "created_at": {
                    "order": "asc"
                }
                }
            ]
            }
            """%str(i)
        elif i > 9:
            query_body = """
            {
            "from": 0, 
            "size" : 8000,
            "query":{
                "match": {
                "created_at": "2022-05-%s"
                }
            },
            "sort": [
                {
                "created_at": {
                    "order": "asc"
                }
                }
            ]
            }
            """%str(i)
        headers = {
            'Content-Type' : 'application/json'
            }

        resp = requests.get(
            url,
            data=query_body,
            headers=headers,
            auth= ELASTIC_SEARCH_AUTH
        )
        if resp.status_code != 200:
            continue
        data = resp.json()
        print('Crawling date {}...'.format(i))
        temp = []
        for j in range(len(data['hits']['hits'])):
            temp.append(data['hits']['hits'][j]['_source']['title'])
        title.append(temp)
    return title