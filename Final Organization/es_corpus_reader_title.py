import datetime as dt
import pdb

from dateutil.relativedelta import relativedelta
from download_data import fetch_news_docs_with_title
class EsCorpusReader_title:

    def __init__(self, date_from=None, date_to=None, title=None):
        if date_to:
            self.date_to = date_to
        else:
            self.date_to = dt.datetime.now()

        if date_from:
            self.date_from = date_from
        else:
            self.date_from = self.date_to - relativedelta(days=1)

        
        self.title = title

        self._buffer = []
        self._next_page = 0


    def clear(self):
        self._buffer = []
        self._next_page = 0

    def fetch_next_page(self):
        docs = fetch_news_docs_with_title(self.date_from, self.date_to, self._next_page, self.title)

        self._buffer += docs
        self._next_page += 1

    def docs(self, n=-1):

        self.clear()

        while True:

            if n == 0:
                return

            if len(self._buffer) == 0:
                self.fetch_next_page()

            if len(self._buffer) == 0:
                return 
            
            doc = self._buffer.pop(0) #pop은 메모리에서 사라지면서 꺼내줌

            n -= 1

            yield doc

    def urls(self, n=-1):
        for doc in self.docs(n):
            yield doc['_source']['naver_url']

    def titles(self, n=-1):
        for doc in self.docs(n):
            yield doc['_source']['title']

if __name__ == '__main__':
    reader = EsCorpusReader_key(date_from=dt.datetime(2022, 5, 1), date_to=dt.datetime(2022, 5, 31), keyword="옴바니밧메훔")

    for idx, title in enumerate(reader.titles(n=10)):
        print(f'News #{idx}: ', title)