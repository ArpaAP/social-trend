import os
import numpy as np
import itertools
from collections import Counter
from tqdm import tqdm
import pandas as pd

from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import sqlite3

### VARIABLES ###

# collection 폴더에서 불러올 DB입니다. 
# DB 파일 이름이 data-221107-111306.db 이면 해당 변수는 '221107-111306'이 됩니다.
dataname = '221105-202616'

# 아래는 M1 아키텍처 기반 MacOS 전용 설정입니다. M1에서는 JVM 경로를 수동으로 지정해야 했습니다.
JVM_PATH = '/Library/Java/JavaVirtualMachines/zulu-15.jdk/Contents/Home/bin/java'
okt = Okt(jvmpath=JVM_PATH)

###

# okt = Okt()

def max_sum_sim(doc_embedding, candidate_embeddings, words, top_n, nr_candidates):
    # 문서와 각 키워드들 간의 유사도
    distances = cosine_similarity(doc_embedding, candidate_embeddings)

    # 각 키워드들 간의 유사도
    distances_candidates = cosine_similarity(candidate_embeddings, 
                                            candidate_embeddings)

    # 코사인 유사도에 기반하여 키워드들 중 상위 top_n개의 단어를 pick.
    words_idx = list(distances.argsort()[0][-nr_candidates:])
    words_vals = [candidates[index] for index in words_idx]
    distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

    # 각 키워드들 중에서 가장 덜 유사한 키워드들간의 조합을 계산
    min_sim = np.inf
    candidate = None
    for combination in itertools.combinations(range(len(words_idx)), top_n):
        sim = sum([distances_candidates[i][j] for i in combination for j in combination if i != j])
        if sim < min_sim:
            candidate = combination
            min_sim = sim

    return [words_vals[idx] for idx in candidate]

def mmr(doc_embedding, candidate_embeddings, words, top_n, diversity):

    # 문서와 각 키워드들 간의 유사도가 적혀있는 리스트
    word_doc_similarity = cosine_similarity(candidate_embeddings, doc_embedding)

    # 각 키워드들 간의 유사도
    word_similarity = cosine_similarity(candidate_embeddings)

    # 문서와 가장 높은 유사도를 가진 키워드의 인덱스를 추출.
    # 만약, 2번 문서가 가장 유사도가 높았다면
    # keywords_idx = [2]
    keywords_idx = [np.argmax(word_doc_similarity)]

    # 가장 높은 유사도를 가진 키워드의 인덱스를 제외한 문서의 인덱스들
    # 만약, 2번 문서가 가장 유사도가 높았다면
    # ==> candidates_idx = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10 ... 중략 ...]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    # 최고의 키워드는 이미 추출했으므로 top_n-1번만큼 아래를 반복.
    # ex) top_n = 5라면, 아래의 loop는 4번 반복됨.
    for _ in range(top_n - 1):
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # MMR을 계산
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # keywords & candidates를 업데이트
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]

print('loading database')

conn = sqlite3.connect(f'./collection/data-{dataname}.db')
conn.row_factory = sqlite3.Row
cur = conn.cursor()

print('filtering empty content')
docs = [x['content'] for x in cur.execute('SELECT content FROM News') if x['content']]

print('importing model')
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

n_gram_range = (0, 3)

mss_ls = []
mmr_ls = []

count = 0
success_count = 0
ignore_count = 0

for doc in tqdm(docs, desc='extracting keywords'):
    count += 1
    tokenized_doc = okt.pos(doc)
    tokenized_nouns = ' '.join([word[0] for word in tokenized_doc if word[1] == 'Noun'])

    # print('품사 태깅 10개만 출력 :',tokenized_doc[:10])
    # print('명사 추출 :',tokenized_nouns)

    try:
        count_vector = CountVectorizer(ngram_range=n_gram_range).fit([tokenized_nouns])
    except ValueError:
        ignore_count += 1
        continue

    candidates = count_vector.get_feature_names_out()

    # print('trigram 개수 :',len(candidates))
    # print('trigram 다섯개만 출력 :',candidates[:5])

    doc_embedding = model.encode([doc])
    candidate_embeddings = model.encode(candidates)

    # top_n = 5
    # distances = cosine_similarity(doc_embedding, candidate_embeddings)
    # keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]
    # print(keywords)

    try:
        result_mss = max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n=15, nr_candidates=20)
        result_mmr = mmr(doc_embedding, candidate_embeddings, candidates, top_n=15, diversity=0.4)
    except:
        ignore_count += 1
        continue

    success_count += 1

    mss_ls.extend(result_mss)
    mmr_ls.extend(result_mmr)

mss_c = Counter(mss_ls)
mmr_c = Counter(mmr_ls)

mss_df = pd.DataFrame(mss_c.most_common())
mmr_df = pd.DataFrame(mmr_c.most_common())

print(mss_c.most_common(20), mmr_c.most_common(20), sep='\n\n')

mss_df.to_csv(f'./extract/mss_{dataname}_{"-".join(map(str, n_gram_range))}.csv', index=False, encoding='utf-8-sig')
mmr_df.to_csv(f'./extract/mmr_{dataname}_{"-".join(map(str, n_gram_range))}.csv', index=False, encoding='utf-8-sig')