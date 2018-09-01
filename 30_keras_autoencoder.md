# Building Autoencoders in Keras

https://blog.keras.io/building-autoencoders-in-keras.html

> 이 문서에서는 autoencoder에 대한 일반적인 질문에 답하며 다음의 모델에 해당하는 코드를 다룹니다.

- 완전 연결 레이어 (fully connected layer) 에 기반한 간단한 autoencoder
- a sparse autoencoder
- a deep fully-connected autoencoder 
- a deep convolutional autoencoder
- an image denoising model
- a sequence-to-sequence autoencoder
- a variational autoencoder

Note: 모든 예제 코드는 2017년 3월 14일에 Keras 2.0 API에 업데이트 되었습니다. 예제 코드를 실행하기 위해서는 Keras 버전 2.0 이상이 필요합니다. 

----

## autoencoder(오토인코더)는 무엇일까요?

<img src="https://blog.keras.io/img/ae/autoencoder_schema.jpg">

**"Autoencoding"**은 데이터 압축 알고리즘으로 … 

또한 autoencoder 라는 용어가 사용되는 대부분의 경우, 압축과 압축해제 함수는 신경망으로 구현됩니다. 

1) autoencoder는 date-specific 합니다. 오토인코더는 이제껏 훈련된 데이터와 비슷한 데이터로만 압축될 수 있습니다. 예를 들어 말하자면, 오토인코더는 MPEG-2 Audio Layer III (MP3) 압축 알고리즘과는 다릅니다. MP3 알고리즘은 일반적으로 소리에 관한 압축이지만 특정한 종류의 소리에 관한 것은 아닙니다. 얼굴 사진에 대해 학습된 오토인코더는 나무의 사진을 압축하는 데에는 좋은 성능을 내지 못하는데 그 이유는 오토인코더가 배우는 특징은 얼굴 특유의 것이기 때문입니다. 

2) autoencoder는 손실이 있습니다. 즉, 압축 해제된 결과물은 원본 보다 안좋아집니다 (MP3나 JPEG 압축처럼..!). 이는 손실없는 산술 압축과는 다릅니다. 

3)  autoencoder는 예제 데이터로부터 자동적으로 학습하는데 이는 유용한 성질입니다: 데이터로부터 자동적으로 학습한다는 의미는 특정 종류의 입력값에 대해 잘 작동하는 특별한 형태의 알고리즘을 쉽게 훈련시킬 수 있다는 말입니다. 이는 새로운 공학적 방법 필요 없이 단지 데이터를 적절히 훈련시키면 됩니다. 

오토인코더를 만들기 위해서는 세 가지가 필요합니다

- 인코딩 함수 (encoding function)
- 디코딩 함수 (decoding function )
- 원본에 대해 압축된 표현과 압축 해제된 표현(representation) 간 정보 손실량 간의 거리 함수 (즉 손실 함수)

 The encoder and decoder will be chosen to be parametric functions (typically neural networks), and to be differentiable with respect to the distance function, so the parameters of the encoding/decoding functions can be optimize to minimize the reconstruction loss, using Stochasic Gradient Descent. 

 간단! 또한 이 실습 예제의 오토인코더를 시작하기 위해 이러한 단어들을 이해할 필요가 없습니다. 

## Are they good at data compression? 



