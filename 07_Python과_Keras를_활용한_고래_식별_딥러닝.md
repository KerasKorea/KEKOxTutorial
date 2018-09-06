## Python과 Keras를 활용한 고래 식별 딥러닝

[Humpback Whale Identification Challenge on Kaggle (원문 바로가기)](https://medium.com/@evily.yang/humpback-whale-identification-challenge-on-kaggle-7c550a342df3)  


> 이 글은 저자가 Kaggle의 고래 식별 대회에 참여한 경험을 바탕으로 하고 있습니다. Keras를 활용하여 딥러닝으로 이미지 식별 문제를 해결한 과정을 설명하고, 과정에서 겪었던 어려움들을 서술하고 있습니다.  

* 딥러닝
* CNN
* AutoEncoder

[캐글 대회 링크 바로가기](https://www.kaggle.com/c/whale-categorization-playground/leaderboard)

### 요약

#### 도구 (Tools)
- 파이썬
- 케라스
- 텐서플로우
- GPU 2개

#### 사용한 메소드 (Methods)
- 오토인코더 (Auto Encoder)
- 합성곱 신경망 (CNN) : VGG19, ResNet50
- 트랜스퍼 러닝 (Transfer Learning)
- 로스(Loss) 함수에 다른 클래스 가중치 사용하기
- 데이터 수 늘이기 (Data Augmentation)

#### 해결된 문제
- 224x224 보다 큰 이미지 처리
- 케라스로 여러 개 GPU 사용하기

#### 해결 안 된 문제
- 데이터 불균형 문제
- 낮은 정확도 (점수 : 0.32653, 벤치마크 : 0.36075 보다 낮음)


### 개요


### 데이터 불러오기

### 데이터 처리하기

### 오토인코더 (Auto Encoder)

### 이미지 분류

### 결론

### 참고문서
* [참고 사이트 1]()
* [참고 사이트 2]()

