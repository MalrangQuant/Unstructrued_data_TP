import pdb
import re
import datetime as dt

from sklearn.base import BaseEstimator, TransformerMixin
from konlpy.tag import Hannanum

from es_corpus_reader import EsCorpusReader

'''한글을 토큰으로 바꿔줘 봅시당'''


class KoreanTextNormalizer(BaseEstimator, TransformerMixin): ##클래스 다중상속 (기본은 BaseEstimator로 가고, TransformerMixing을 끼워쓰게땅)
    
    def __init__(self):
        self.hannanum = Hannanum()

    def fit(self, X, y=None):
        return self
    
    #애는 할게 없으니까 (x, y를 받는데 쓸일 없으니깡)

    def transform(self, docs):

        def generator():
            for doc in docs:
                doc = re.sub(r'[^\wㄱ-ㅎ가-힣&%美中]', ' ', doc)
                tokens = self.hannanum.morphs(doc)
                yield tokens

        return list(generator())


if __name__ == "__main__":

    reader = EsCorpusReader(date_from=dt.datetime(2022,4,1))

    corpus = list(reader.titles(n=10))

    normalizer = KoreanTextNormalizer()
    normalized = normalizer.fit_transform(corpus)  #transform mix안에서 호출하는 방식임..

    for idx, x in enumerate(normalized):
        print(f'#{idx}: {x}')