## Neural Style Transfer : tf.keras와 eager execution를 이용한 딥러닝 미술 작품 만들기(Neural Style Transfer: Creating Art with Deep Learning using tf.keras and eager execution)
[원문 링크](https://medium.com/tensorflow/neural-style-transfer-creating-art-with-deep-learning-using-tf-keras-and-eager-execution-7d541ac31398)
> 

* keras
* eager execution
* transfer

### 주요 개념 설명

#### 갖고 있어야할 특정 개념들 : 

#### Style fransfer를 수행하는 일반적인 과정 : 

### 구현

#### 컨텐츠와 스타일 표현 정의

**왜 중간 계층을?**

#### 모델

#### 손실 함수(컨텐츠와 스타일 격차)를 정의 및 생성

**콘텐츠 손실관련 :**

**스타일 손실관련 :**

#### 기울기 하강(Gradient Descent) 실행

#### 손실과 기울기 계산하기

#### Style fransfer 절차를 실행, 적용

### 참고문서
* [참고 사이트 1]()
* [참고 사이트 2]()