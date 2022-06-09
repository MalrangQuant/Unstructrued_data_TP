import nltk
import datetime as dt
import pdb

from collections import defaultdict
from sklearn.pipeline import Pipeline

from es_corpus_reader import EsCorpusReader
from korean_text_normalizer import KoreanTextNormalizer
from gensim_vectorizer import GensimTfidVectorizer

if __name__ == '__main__':

    reader = EsCorpusReader(date_from=dt.datetime(2022,4,12,0,0,0), date_to=dt.datetime(2022,4,12,23,59,59))
    corpus = list(reader.titles(n=5196))


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

    for group in range(len(classified)):
        print(f'*** Group {group} ***')

        classified[group].sort()

        print("Topic Size: {}".format(len(classified[group])))

        for idx, x in enumerate(classified[group]):
            print(f'{idx}: {x}')

            if idx > 5:
                break

        print()



''' 인플레이션이나 이런거는 EsCorpusReader에서 날짜나 키워드 추가해서 바꾸면 된다. '''