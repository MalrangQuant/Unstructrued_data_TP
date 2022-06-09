import enum
import pdb
import gensim
import datetime as dt
from markupsafe import EscapeFormatter

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.matutils import sparse2full
from es_corpus_reader import EsCorpusReader
from korean_text_normalizer import KoreanTextNormalizer

class GensimTfidVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.lexicon = None
        self.tfidf = None

    
    def fit(self, docs, y=None):
        self.lexicon = gensim.corpora.Dictionary(docs) #이떄 토큰화된 docs가 들어간다
        self.tfidf = gensim.models.TfidfModel(dictionary=self.lexicon, normalize=True)

        return self

    def transform(self, docs):

        def generator():
            for doc in docs:
                vec = self.lexicon.doc2bow(doc)
                vec = self.tfidf[vec] #tfidf feature extraction (이걸 이용하면)

                yield sparse2full(vec, len(self.lexicon)) #이걸 가져와서 만들어낸당

        return list(generator())

    

if __name__ == "__main__":

    reader = EsCorpusReader(date_from=dt.datetime(2022, 4, 1))
    corpus = list(reader.titles(n=10))

    for idx, x in enumerate(corpus):
        print(f"News #{idx}: {x}")

    model = Pipeline([
        ('normalizer', KoreanTextNormalizer()),
        ('vectorlizer', GensimTfidVectorizer())
    ])

    vectors = model.fit_transform(corpus)

    for idx, vec in enumerate(vectors):
        print(f'Vector #{idx}: {vec}')
