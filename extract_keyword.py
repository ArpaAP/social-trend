import numpy as np
import itertools

from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

doc = """
윤석열 대통령이 1일 경기도 부천 한 장례식장에 마련된 이태원 참사 희생자 빈소를 찾아 조문한 뒤 유가족을 위로하고 있다. 대통령실 제공

윤석열 대통령이 이태원 참사 사망자들의 빈소 두 곳을 직접 찾아 조문하고 유가족을 위로했다.

윤 대통령은 1일 저녁 경기도 부천의 한 장례식장에서 이태원 사고로 딸을 잃은 아버지를 위로했다고 대통령실 이재명 부대변인이 이날 서면 브리핑에서 전했다.

윤 대통령은 고인의 부친 손을 붙잡고 “뭐라고 위로의 말씀을 드려야 할지 모르겠다”며 머리를 숙였다. 고인의 남동생에게는 “아버지를 잘 보살펴 드리라”고 당부의 말을 건넸다.

윤 대통령은 이어 서울의 한 장례식장을 찾아 사고로 부인과 딸을 잃은 유가족을 만나 애도했다.

윤석열 대통령이 1일 서울 용산구 이태원역 1번 출구 앞 이태원 참사 추모 공간을 방문, 헌화하고 있다. 대통령실 제공, 연합뉴스

이날 조문은 갑작스러운 사고로 가족을 잃은 유가족에게 위로의 마음을 전하고 싶다는 윤 대통령의 뜻에 따라 이뤄졌다고 이 부대변인은 설명했다.

앞서 윤 대통령은 지난달 31일 서울광장에 마련된 정부 합동분향소를 찾은 데 이어 이날 사고 현장 인근의 합동분향소를 한 차례 더 방문하기도 했다.
"""

JVM_PATH = '/Library/Java/JavaVirtualMachines/zulu-15.jdk/Contents/Home/bin/java'

okt = Okt(jvmpath=JVM_PATH)

tokenized_doc = okt.pos(doc)
tokenized_nouns = ' '.join([word[0] for word in tokenized_doc if word[1] == 'Noun'])

# print('품사 태깅 10개만 출력 :',tokenized_doc[:10])
# print('명사 추출 :',tokenized_nouns)

n_gram_range = (0, 1)

count = CountVectorizer(ngram_range=n_gram_range).fit([tokenized_nouns])
candidates = count.get_feature_names_out()

# print('trigram 개수 :',len(candidates))
# print('trigram 다섯개만 출력 :',candidates[:5])

model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
doc_embedding = model.encode([doc])
candidate_embeddings = model.encode(candidates)

# top_n = 5
# distances = cosine_similarity(doc_embedding, candidate_embeddings)
# keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]
# print(keywords)

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

print(max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n=15, nr_candidates=20))
print(mmr(doc_embedding, candidate_embeddings, candidates, top_n=15, diversity=0.4))