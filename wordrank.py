from krwordrank.hangle import normalize
from krwordrank.word import KRWordRank

texts = [
"""
이재명 더불어민주당 대표가 2015년 성남시장 재직 당시 해외 출장 중 김문기 성남도시개발공사 개발1처장과 함께 찍은 사진. 이기인 국민의힘 경기도의원 제공

대장동 개발사업 특혜·로비 비리로 재판받고 있는 유동규 전 성남도시개발공사 기획본부장이 2015년 당시 성남시장이던 이재명 더불어민주당 대표, 고(故) 김문기 전 성남도개공 개발사업1처장과 함께 골프를 쳤던 호주의 골프장 이름과 위치 등을 검찰에 구체적으로 진술한 것으로 알려졌다.

유동규 전 본부장은 최근 검찰에서 2015년 1월 6~16일 당시 성남시장이던 이 대표, 김문기씨 등과 함께 호주, 뉴질랜드로 출장 갔던 상황을 자세히 밝혔다고 2일 조선일보가 보도했다.

보도에 따르면 이 대표의 선거법 위반 혐의 공소장에는 ‘2015년 1월 12일 이 대표와 유 전 본부장, 김 전 처장이 호주에서 함께 골프를 쳤다’고 명시돼 있다. 이와 관련해 유 전 본부장은 당시 골프를 쳤던 장소를 밝히며 “이 대표와 나, 김문기씨가 함께 카트를 탔다”고도 진술한 것으로 전해졌다.

유 전 본부장은 지난달 24일 한국일보 인터뷰에서도 “(이 대표가) 김문기를 몰라? (나랑) 셋이 호주에서 같이 골프 치고 카트까지 타고 다녔으면서”라고 말한 바 있다.

유 전 본부장의 이 같은 진술은 이 대표에게 불리하게 작용할 전망이다. 이 대표의 선거법 위반 사건 재판은 오는 22일 2차 준비기일이 열릴 예정인데, 검찰은 향후 이 대표 재판에서 유 전 본부장의 진술을 증거로 제출할 것으로 알려졌다. 검찰은 유 전 본부장을 증인으로 신청할지 여부도 검토 중이다.

유동규 전 성남도시개발공사 본부장이 지난달 28일 오전 서울 서초구 서초동 서울중앙지법에서 열리는 대장동 개발 사업 로비·특혜 의혹 관련 공판에 출석하고 있다. 뉴시스

한편 유 전 본부장은 지난해 대선 과정에서 이 대표가 ‘김문기를 모른다’고 한 벌언을 접하고 ‘꼬리 자르기’를 당했다고 생각해 주변에 섭섭함을 토로한 것으로 전해졌다. 그전까지만 해도 ‘의리’를 지키겠다며 입을 다물고 있던 유 전 본부장이 이 일을 계기로 심경에 변화가 생겼다는 해석도 나온다.

유 전 본부장은 이 대표의 최측근 김용(구속) 민주연구원 부원장의 ‘불법 대선 자금’ 8억여원 수수 혐의에 대해 검찰에 진술하다가도 “이 대표가 김씨를 몰랐을 수가 없다”는 취지로 말한 것으로 전해졌다.

이 대표는 지난 대선 때 본인이 성남시장이던 때에는 김 전 처장을 몰랐다고 수차례 말했고, 검찰은 이 대표의 이런 말이 거짓이라며 지난 9월 이 대표를 공직선거법상 ‘허위 사실 공표’ 혐의로 기소했다. 검찰은 공소장에서 ‘이 대표가 대장동 사업에 대해 김 전 처장으로부터 대면 보고를 수시로 받았다’고 적시했다. 대장동 사업 핵심 실무자였던 김 전 처장은 지난해 12월 검찰 수사를 받던 도중 극단적 선택을 했다."""
]
texts = [normalize(text, english=True, number=True) for text in texts]

wordrank_extractor = KRWordRank(
    min_count = 1, # 단어의 최소 출현 빈도수 (그래프 생성 시)
    max_length = 10, # 단어의 최대 길이
    verbose = True
    )

beta = 0.85    # PageRank의 decaying factor beta
max_iter = 10

keywords, rank, graph = wordrank_extractor.extract(texts, beta, max_iter)

for word, r in sorted(keywords.items(), key=lambda x:x[1], reverse=True)[:30]:
    print('%8s:\t%.4f' % (word, r))