# ChatBot

## 2018.07.24 ~ 2018.07.31 (1주차)
+ 데이터 수집
+ 데이터 전처리 계획

## 학습 데이터 전처리 과정 계획
+ 텍스트의 토큰화
+ 빈도수가 낮은 단어 제거
+ 시작 토큰과 끝 토큰 붙이기
+ 단어를 index로 mapping (벡터화)

## 해결해야할 문제
+ 단어를 벡터화 할 때, 띄워쓰기 단위로 할지 명사 단위로 할지 정해야한다.
+ 인코더 디코더에 입력값을 줄 때, 띄워쓰기 단위로 줄것인지 형태소분석을 통해 명사, 조사 등 품사단위로 줄것인지 정해야한다.
+ 추가적인 데이터 수집을 위해 씨네스트 자막자료실(http://cineaste.co.kr/bbs/board.php?bo_table=psd_caption&sca=%ED%95%9C%EA%B8%80&page=3&page=1) 에서 웹 크롤링을 통해 데이터를 수집하려고 시도했으나, 헤더,쿠기 문제로 잘 안되고 있다.

---

## 2018.07.31 ~ 2018.08.07 (2주차)



### 데이터 수집처
+ http://gom.gomtv.com/main/index.html?ch=subtitles&pt=l&menu=subtitles (곰플레이어 한글자막)
+ https://ithub.korean.go.kr/user/total/database/corpusManager.do 국립국어원 언어정보나눔터 (대화 데이터)

### 참고 논문/사이트
+ https://github.com/YBIGTA/DeepNLP-Study/wiki (논문 스터디)
+ https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf (Sequence to Sequence Learning with NN)
+ https://arxiv.org/pdf/1706.03762 (Attention Is All You Need)
+ http://www.cs.toronto.edu/~lcharlin/papers/vhred_aaai17.pdf (A Hierarchical Recurrent Encoder-Decoder for Generative Context-Aware Query Suggestion)
+ http://www.cs.toronto.edu/~lcharlin/papers/vhred_aaai17.pdf (A Hierarchical Latent Variable Encoder-Decoder  Model for Generating Dialogues)
