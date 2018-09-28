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

**"Autoencoding"** 은 데이터 압축 알고리즘으로 압축 함수와 압축해제 함수는 다음과 같은 세가지 특징을 갖습니다: 1) date-specific, 2) 손실, 3) learned automatically from examples. 

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



## 오토인코더는 데이터 압축에 좋을까요?

 일반적으로는 그렇지 않습니다. 사진 압축에서 JPEG와 같은 기본 알고리즘보다 나은 작업을 수행하는 오토인코더를 개발하는 것은 꽤 어렵습니다. 일반적으로 JPEG와 같은 성능을 달성할 수 있는 유일한 방법은 사진을 매우 특정한 유형의 사진으로 제한하는 것입니다. 오토인코더가 data-specific 하다는 점 때문에 오토인코더는 실제 데이터 압축 문제에 적용하기에 비실용적입니다. 따라서 오토인코더는 훈련된 것과 비슷한 데이터에서만 사용될 수 있고 오토인코더를 일반적인 데이터에 대해 사용하기 위해서는 많은 훈련 데이터가 필요합니다. 하지만 미래에는 바뀔 수도 있습니다, 누가 알겠어요?



## 오토인코더는 어디에 좋을까요? 

 오토인코더는 실제 응용에서는 거의 사용되지 않습니다. 2012년 오토인코더는 deep convolutional neural network [1] , 그러나 이는 ?

 

오늘날 오토인코더의 두 가지 흥미로운 실제 응용분야는 data denosing 과 데이터 시각화를 위한 차원 축소입니다. 적절한 dimensionality와 sparsity contraints를 사용하면, 오토인코더는 PCA나 다른 기법들보다 더 흥미로운 data projection을 배울 수 있습니다. 



특히 2차원 시각화에 대하여, t-SNE는 거의 최고의 알고리즘입니다. 하지만 이는 상대적으로 낮은 차원의 데이터를 요구합니다. 따라서 높은 차원의 데이터에서 유사(similarity) 관계를 시각화하는 좋은 전략은 먼저 오토인코더를 사용하여 데이터를 낮은 차원으로 압축합니다. 그리고나서 압축된 데이터를 t-SNE를 사용하여 2차원 평면으로 매핑합니다. 이미 케라스의 휼륭한 parametric implementation이 Kyle McDonald에 의해 개발되어있고 [github](https://github.com/kylemcdonald/Parametric-t-SNE/blob/master/Parametric%20t-SNE%20(Keras).ipynb) 에서 볼 수 있습니다. 그 밖에도, [scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) 에도 간단하고 실용적이게 구현되어 있습니다. 



## 그렇다면 오토 인코더는 무엇이 중요할까요?

Their main claim to fame comes from being featured in many introductory machine learning classes available online. As a result, a lot of newcomers to the field absolutely love autoencoders and can't get enough of them. This is the reason why this tutorial exists! (??)



오토인코더가 수많은 연구와 집중을 끌어들이는 이유는 오토인코더가 unsupervised learning(비지도학습)의 문제를 풀어낼 잠재적인 수단으로 오랜동안 생각되어왔기 때문입니다. 즉, 오토인코더는 



## 매우 간단한 오토인코더를 만들어봅시당 

간단한 것에서 시작합시다. 

```python
from keras.layers import Input, Dense
from keras.models import Model

# 
```







[1]: dfdfd



