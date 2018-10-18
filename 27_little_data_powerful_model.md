# 작은 데이터셋으로 강력한 이미지 분류 모델 설계하기

원문: [Building powerful image classification models using very little data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html), Francois Chollet

번역: 허남규, [itsnamgyu@gmail.com](mailto:itsnamgyu@gmail.com)

이 글은 적은 양의 데이터를 가지고 강력한 이미지 분류 모델을 구축하는 방법을 소개합니다. 수백에서 수천 장 정도의 작은 데이터셋을 가지고도 강력한 성능을 낼 수 있는 모델을 만들어볼 것입니다.

저희는 세 가지 방법을 다룰 것입니다.

- 작은 네트워크를 처음부터 학습 (앞으로 사용할 방법의 평가 기준)
- 기존 네트워크의 *bottleneck feature* 사용
- 기존 네트워크의 상단부 레이어 *fine-tuning*

저희가 다룰 케라스 기능은 다음과 같습니다.

- `ImageDataGenerator`: 실시간 이미지 augmentation
- `flow`: `ImageDataGenerator` 디버깅
- `fit_generator`: `ImageDataGenerator`를 이용한 케라스 모델 학습
- `Sequential`: 케라스 내 모델 설계 인터페이스
- `keras.applications`: 케라스에서 기존 이미지 분류 네트워크 불러오기
- 레이어 동결 (freezing)을 이용한 fine-tuning
