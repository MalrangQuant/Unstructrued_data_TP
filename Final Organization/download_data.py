"""Download Data

Usage:
    download_data.py
    download_data.py [--from <date>] [--to <date>]
    download_data.py (-h | --help)
    download_data.py (-v | --version)

Options:
    -h --help         Show this screen.
    -v --version      Show version.
    --from <date>     First date from which data will be downloaded
    --to <date>       Last date to which data will be downloaded

"""

import os
import pdb
import datetime as dt
import requests
import json
import pandas as pd

from config import *
from dateutil.relativedelta import relativedelta
from docopt import docopt #해당 모듈을 어떻게 쓸지 자동으로 알려 줌

def parse_arguments():
    args = docopt(__doc__, version="0.1")
    
    if args['--to']:
        date_to = dt.datetime.fromisoformat(args['--to'])
    else:
        date_to = dt.datetime.now()

    if args['--from']:
        date_from = dt.datetime.fromisoformat(args['--from'])
    else:
        date_from = date_to - relativedelta(days=1)

    return date_from, date_to


def fetch_news_docs(date_from, date_to, page):
    
    query = {
        "query": {
            "range":{
                "created_at": {
                    "gte": date_from.isoformat(),
                    "lt": date_to.isoformat()
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
    

def fetch_news_docs_with_keyword (date_from, date_to, page, keyword):

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
                            "gte": date_from.isoformat(),
                            "lt": date_to.isoformat()
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

def fetch_news_docs_with_title (date_from, date_to, page, title):

    query = {
        "query": {
            "bool": {
                "must":[
                    {
                        "match":{
                            "title": title
                        }
                    },
                    {
                    "range":{
                        "created_at": {
                            "gte": date_from.isoformat(),
                            "lt": date_to.isoformat()
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


def download_data (date_from, date_to):
    print("Downloading data from OpenSearch server")

    if not os.path.exists('data'):
        os.makedirs('data')

    for page in range(1000):
        print('.', end='', flush=True)

        hits = fetch_news_docs(date_from, date_to, page)

        if len(hits) == 0:
            break

        for doc in hits:

            filename = f"data/news-{doc['_id']}.txt"

            with open(filename, 'w', encoding='utf-8') as f:
                text = json.dumps(doc['_source'], ensure_ascii=False)
                f.write(text)

def download_data (date_from, date_to):

    date_interval = list(pd.date_range(date_from, date_to, freq='d'))

    print("Downloading data from OpenSearch server")

    df = pd.DataFrame()

    for i in range(len(date_interval)-1):

        tmp = []

        start_date = date_interval[i]
        end_date = date_interval[i+1]

        for page in range(1000):

            print('.', end='', flush=True)

            hits = fetch_news_docs(start_date, end_date, page)

            if len(hits) == 0:
                break

            for doc in hits:
                
                tmp.append(doc)
        
        df[f'{start_date}'] = [tmp]

    print("Daily Data end!")

    return df


if __name__ == "__main__":
    date_from, date_to = parse_arguments()

    download_data(date_from, date_to)
