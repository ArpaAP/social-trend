# social-trend
호산고 로봇공학반 2학년 &lt;응용 프로그래밍 개발> 과목 프로젝트 - 크롤링 및 API를 통한 사회 트렌드 분석


## 프로젝트 구조
```bash
.
├── collection/         # 뉴스 기사 수집 데이터 저장 경로
│   └── data-*.db       # sqlite3 db 파일
├── datas/
│   └── media.json      # 네이버뉴스 언론사 데이터
├── extract/            # 키워드 추출 결과 저장 경로
│   ├── mss_*.csv       # Max Sum Similarity 알고리즘을 통해 얻은 결과물
│   └── mmr_*.csv       # Maximal Marginal Relevance 알고리즘을 통해 얻은 결과물
├── collect_news.py     # 뉴스 기사 수집 코드
├── extract_keyword.py  # 키워드 추출 코드
├── requirements.txt    # 라이브러리 의존성 명세
└── wordrank.py         # KRWordRank 활용한 키워드 추출 코드. 시도만 해보고 사용하지 않는 더미 코드
```


## 참고 자료
**딥러닝을 이용한 자연어 처리 입문(유원준/안상준)** - [한국어 키버트(Korean KeyBERT)를 이용한 키워드 추출](https://wikidocs.net/159468)