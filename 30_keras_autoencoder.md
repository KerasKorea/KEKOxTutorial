# Building Autoencoders in Keras

https://blog.keras.io/building-autoencoders-in-keras.html

> 이 문서에서는 autoencoder에 대한 일반적인 질문에 답하며 다음의 모델에 해당하는 코드를 다룹니다.

- a simple autoencoders based on a fully-connected layer 
- a sparse autoencoder
- a deep fully-connected autoencoder 
- a deep convolutional autoencoder
- an image denoising model
- a sequence-to-sequence autoencoder
- a variational autoencoder

Note: 모든 예제 코드는 2017년 3월 14일에 Keras 2.0 API에 업데이트 되었습니다. 예제 코드를 실행하기 위해서는 Keras 버전 2.0 이상이 필요합니다. 



## 오토인코더(autoencoder)는 무엇일까요?

<img src="https://blog.keras.io/img/ae/autoencoder_schema.jpg">

**"Autoencoding"** 은 데이터 압축 알고리즘으로 압축 함수와 압축해제 함수는 다음과 같은 세가지 특징을 갖습니다: 1) ==data-specific==, 2) 손실(lossy), 3) 사람의 조작 없이 예제로 부터 자동으로 학습. 또한, "autoencoder" 가 사용되는 대부분의 상황에서 압축 함수와 압축해제 함수는 신경망으로 구현됩니다. 

1) autoencoder는 date-specific 합니다. 오토인코더는 이제껏 훈련된 데이터와 비슷한 데이터로만 압축될 수 있습니다. 예를 들어 말하자면, 오토인코더는 MPEG-2 Audio Layer III (MP3) 압축 알고리즘과는 다릅니다. MP3 알고리즘은 일반적으로 소리에 관한 압축이지만 특정한 종류의 소리에 관한 것은 아닙니다. 얼굴 사진에 대해 학습된 오토인코더는 나무의 사진을 압축하는 데에는 좋은 성능을 내지 못하는데 그 이유는 오토인코더가 배우는 특징은 얼굴 특유의 것이기 때문입니다. 

2) autoencoder는 손실이 있습니다. 즉, 압축 해제된 결과물은 원본 보다 좋지 않습니다. (ex. MP3, JPEG 압축). 이는 손실없는 산술 압축과는 다릅니다. 

3)  autoencoder는 예제 데이터로부터 자동적으로 학습하는데 이는 유용한 성질입니다: 데이터로부터 자동적으로 학습한다는 의미는 특정 종류의 입력값에 대해 잘 작동하는 특별한 형태의 알고리즘을 쉽게 훈련시킬 수 있다는 말입니다. 이는 새로운 공학적 방법 필요 없이 단지 데이터를 적절히 훈련시키면 됩니다. 

오토인코더를 만들기 위해서는 세 가지가 필요합니다

- 인코딩 함수 (encoding function)
- 디코딩 함수 (decoding function )
- 원본에 대해 압축된 표현과 압축 해제된 표현(representation) 간 정보 손실량 간의 거리 함수 (즉, 손실 함수)

인코더와 디코더는 parametic 함수 (일반적으로 신경망) 로 선택되고 거리 함수와 관련하여 차별화되므로 인코딩/디코딩 함수의 매개변수를 확률적 경사하강법(Stochastic gradient descent)을 사용하여 재구성 손실을 최소화하도록 최적화 할 수 있습니다.

간단! 이러한 단어를 모른다고 걱정하지 마세요. 이 실습 예제는 이러한 단어를 몰라도 시작할 수 있습니다. 



## 오토인코더는 데이터 압축에 좋을까요?

일반적으로는 그렇지 않습니다. 사진 압축에서 JPEG와 같은 기본 알고리즘보다 나은 작업을 수행하는 오토인코더를 개발하는 것은 꽤 어렵습니다. 일반적으로 JPEG와 같은 성능을 달성할 수 있는 유일한 방법은 사진을 매우 특정한 유형의 사진으로 제한하는 것입니다. 오토인코더가 data-specific 하다는 점 때문에 오토인코더는 실제 데이터 압축 문제에 적용하기에 비실용적입니다. 따라서 오토인코더는 훈련된 것과 비슷한 데이터에서만 사용될 수 있고 오토인코더를 일반적인 데이터에 대해 사용하기 위해서는 많은 훈련 데이터가 필요합니다. 하지만 미래에는 바뀔 수도 있습니다, 누가 알겠어요?



## 오토인코더는 어디에 좋을까요? 

 오토인코더는 실제 응용에서는 거의 사용되지 않습니다. 2012년, 오토인코더를 응용할 수 있는 방법이deep convolutional neural network에 대한 greedy layer-wise pretraining 에서 발견되었습니다   [1].  그러나 random weight initialization schemes가 처음부터 deep network를 훈련하기에 충분하다는 것을 알게되면서 오토인코더는 빠르게 유행에서 사라졌습니다. 2014년, batch normalization[2]은 훨씬 더 깊은 network를 허용하기 시작했고, 2015년 말부터 residual learning을 사용하여 임의적으로 deep network를 훈련시킬 수 있었습니다 [3].



오늘날 오토인코더의 두 가지 흥미로운 실제 응용분야는 data denosing 과 데이터 시각화를 위한 차원 축소입니다. 적절한 dimensionality와 sparsity contraints를 사용하면, 오토인코더는 PCA나 다른 기법들보다 더 흥미로운 data projection을 배울 수 있습니다. 



특히 2차원 시각화에 대하여, t-SNE는 거의 최고의 알고리즘입니다. 하지만 이는 상대적으로 낮은 차원의 데이터를 요구합니다. 따라서 높은 차원의 데이터에서 유사(similarity) 관계를 시각화하는 좋은 전략은 먼저 오토인코더를 사용하여 데이터를 낮은 차원으로 압축합니다. 그리고나서 압축된 데이터를 t-SNE를 사용하여 2차원 평면으로 매핑합니다. 이미 케라스의 휼륭한 parametric implementation이 Kyle McDonald에 의해 개발되어있고 [github](https://github.com/kylemcdonald/Parametric-t-SNE/blob/master/Parametric%20t-SNE%20(Keras).ipynb) 에서 볼 수 있습니다. 그 밖에도, [scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) 에도 간단하고 실용적이게 구현되어 있습니다. 



## 그렇다면 오토 인코더는 무엇이 중요할까요?

오토인코더가 유명해진 주된 이유는 온라인에서 이용할 수있는 많은 머신러닝 수업에 특집으로 등장하기 때문입니다. 결과적으로, 머신러닝 분야의 많은 신입들은 오토인코더를 매우 좋아합니다. 이것이 이 튜토리얼이 존재하는 이유죠! 

오토인코더가 수많은 연구와 집중을 끌어들이는 또 다른 이유는 오토인코더가 비지도 학습(unsupervised learning)의 문제를 풀어낼 잠재적인 수단으로 오랜동안 생각되어왔기 때문입니다. 다시 한번 말하자면, 오토인코더는 진정한 비지도 학습 기술이 아니고, self-supervised(?) 기술입니다. 이는 지도 학습(supervised learning)의 일종으로 입력 데이터로부터 target을 만들어냅니다. 흥미로운 특징(feature)들을 학습하는 self-supervised model을 얻으려면 흥미로운 합성 목표 및 손실 함수를 제공해야 합니다. 문제는 여기서 발생합니다. ==merely learning to reconstruct your input in minute detail might not be the right choice here. At this point there is significant evidence that focusing on the reconstruction of a picture at the pixel level, for instance, is not conductive to learning interesting, abstract features of the kind that label-supervized learning induces (where targets are fairly abstract concepts "invented" by humans such as "dog", "car"...). In fact, one may argue that the best features in this regard are those that are the *worst* at exact input reconstruction while achieving high performance on the main task that you are interested in (classification, localization, etc).==

In self-supervized learning applied to vision, a potentially fruitful alternative to autoencoder-style input reconstruction is the use of toy tasks such as jigsaw puzzle solving, or detail-context matching (being able to match high-resolution but small patches of pictures with low-resolution versions of the pictures they are extracted from). The following paper investigates jigsaw puzzle solving and makes for a very interesting read: Noroozi and Favaro (2016) [Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles](http://arxiv.org/abs/1603.09246). Such tasks are providing the model with built-in assumptions about the input data which are missing in traditional autoencoders, such as *"visual macro-structure matters more than pixel-level details"*.

<img src="https://blog.keras.io/img/ae/jigsaw-puzzle.png">



## 매우 간단한 오토인코더를 만들어봅시다.

간단한 것에서 시작합시다.  인코더와 디코더로 single fully-connected neural layer를 사용합니다. 

```python
from keras.layers import Input, Dense
from keras.models import Model

# 인코딩될 표현(representation)의 크기
encoding_dim = 32 # 32 floats -> 24.5의 압축으로 입력이 784 float라고 가정 

# 입력 플레이스홀더
input_img = Input(shape=(784,))
# "encoded"는 입력의 인코딩된 표현
encoded = Dense(encdoing_dim, activation='relu')(input_img)
# "decoded"는 입력의 손실있는 재구성 (lossy reconstruction)
decoded = Dense(784, activation='sigmoid')(encoded)

# 입력을 입력의 재구성으로 매핑할 모델
autoencoder = Model(input_img, decoded)
```

분리된 인코더 모델도 만듭시다. 

```python
# 이 모델은 입력을 입력의 인코딩된 입력의 표현으로 매핑
encoder = Model(input_img, encoded)
```

디코더 모델도 만듭니다. 

```python
# 인코딩된 입력을 위한 플레이스 홀더
encoded_input = Input(shape=(encoding_dim,))
# 오토인코더 모델의 마지막 레이어 얻기
decoder_layer = autoencoder.layers[-1]
# 디코더 모델 생성
decoder = Model(encoded_input, decoder_layer(encoded_input))
```



이제 우리의 오토인코더로 MNIST 숫자를 재구성해봅시다. 



먼저, 픽셀 당 바이너리 크로스엔트로피 손실을 사용하도록 모델을 구성하고, optimizer로 Adadelta를 사용합시다. 

```python
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
```



입력 데이터를 준비합시다. MNIST 숫자를 사용할 것이고, 라벨은 버리도록 하겠습니다. 입력 이미지를 인코딩하고 디코딩하는 데에만 관심이 있기 때문이죠. 

```python
from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()
```

모든 값을 0~1 사이로 정규화하고 28x28 이미지를 크기 784의 벡터로 만들겠습니다. 

```python
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print x_train.shape
print x_test.shape
```

이제 오토인코더를 50세대(epochs) 동안 훈련시키죠.

```python
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))
```

50세대 이후, 오토인코더는 약 0.11의 안정적인 train/test 손실 값에 도달하였습니다. 재구성된 입력과 인코딩된 표현(representation)을 시각화 해봅시다. Matplotlib을 이용하겠습니다. 

```python
# 숫자들을 인코딩 / 디코딩
# test set에서 숫자들을 가져왔다는 것을 유의
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)
```

```python
# Matplotlib 사용
import matplotlib.pyplot as plt

n = 10  # 몇 개의 숫자를 나타낼 것인지
plt.figure(figsize=(20, 4))
for i in range(n):
    # 원본 데이터
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 재구성된 데이터
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```

아래에 결과가 있습니다. 위의 줄은 원래의 숫자이고 아랫 줄은 재구성된 숫자입니다. 지금 사용한 간단한 접근 방법으로 꽤 많은 비트 손실이 있었다는 것을 알 수 있습니다. 

<img src="https://blog.keras.io/img/ae/basic_ae_32.png">



## Adding a sparsity constraint on the encoded representations

이전 예제에서, 표현(representation)은 은닉층의 크기(32)에만 제약을 받았습니다. 이러한 상황에서, 전형적으로 발생하는 일은 은닉층이 PCA(principal component analysis) 의 근사값을 학습한다는 것입니다. 표현을 더 간결하게 제한하는 다른 방법은 숨겨진 표현의 활동에 sparsity를 부여하는 것입니다. 이는 주어진 시간에 더 적은 유닛이 "실행"될 수 있도록 합니다. Keras에서는 activity_regularizer를 Dense layer에 추가하여 수행할 수 있습니다. 

```python
from keras import regularizers

encoding_dim = 32

input_img = Input(shape=(784,))
# L1 activity regularizer를 Dense layer에 추가 
encoded = Dense(encoding_dim, activation='relu',
                activity_regularizer=regularizers.l1(10e-5))(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)
```

우리 모델을 100세대(epochs) 동안 훈련시킵시다. 모델은 0.11의 train loss와 0.10의 test loss로 끝납니다. 이 둘의 차이가 발생하는 이유는 대부분 훈련 중 손실에 추가되는 정규화때문입니다. 

새로운 결과를 보시죠. 

<img src="https://blog.keras.io/img/ae/sparse_ae_32.png">

이전 모델과 거의 비슷해보입니다. 유일한 차이라면 인코딩된 표현의 sparsity입니다. encoded_imgs.mean()은 10,000개의 테스트 이미지에서 3.33의 값을 얻었지만, 이전 모델에서는 7.30이었습니다. 따라서, 우리의 모델은 두 배 더 sparser한 인코딩 된 표현을 만들어냈습니다. 



## Deep autoencoder

이제 더이상 인코더나 디코더 같은 단일 layer에 제한을 가질 필요가 없습니다. 이제 다음과 같은 layer의 스택을 사용합시다. 

```python
input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)
```

다음과 같이 시도해보세요:

```python
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))
```

100세대(epochs) 이후, train/test loss는 ~0.097까지 도달합니다. 이전 모델보다 조금 더 나아졌죠. 재구성된 숫자 또한 더 나아보입니다. 

<img src="https://blog.keras.io/img/ae/deep_ae_32.png">



## Convolutional autoencoder

우리가 사용하는 입력은 이미지이기 때문에, convolutional neural networks(convnet)을 인코더/디코더로 사용하는 것이 가능합니다. 실제로, 이미지에 적용되는 오토인코더는 항상 convolutional autoencoder입니다. 왜냐하면 성능이 더 좋기 때문이죠. 



한 번 구현해 봅시다. 인코더는 Conv2D와 MaxPooling2D layer의 층으로 구성되고, 디코더는 Conv2D와 UpSampling2D layer의 층으로 구성됩니다. 

```python
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
```

훈련을 위해서, 원본 MNIST 숫자의 shape (samples, 3, 28, 28)을 사용하고, 픽셀 값만 0~1로 정규화하도록 하겠습니다. 

```python
from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format
```

50세대(epochs) 동안 훈련시키죠. 모델 훈련 과정을 시각화하여 설명하기 위해, Tensorflow 백엔드와 TensworBoard 콜백을 사용할 것입니다. 



먼저, 터미널을 열고 /tmp/autoencoder에 저장된 로그를 읽는 서버인 TensorBoard를 시작합시다. 

```bash
tensorboard --logdir=/tmp/autoencoder
```

이제 모델을 훈련시킵시다. callback 리스트에서 TensorBoard callboack 인스턴스를 전달합니다. 매 세대(epoch) 이후, 이 콜백은 TensorBoard 서버에서 읽을 수 있는 /tmp/autoencoder에 로그를 씁니다. 

```python
from keras.callbacks import TensorBoard

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
```

TensorBoard 웹 인터페이스(http://0.0.0.0:6006)에서 훈련 과정을 모니터링할 수 있습니다. 

<img src="https://blog.keras.io/img/ae/tb_curves.png">

이 모델은 0.094의 손실 값으로 수렴합니다. 이는 명백히 이전 모델보다 나아졌지요(this is in large part due to the higher entropic capacity of the encoded representation, 128 dimensions vs. 32 previously). 재구성된 숫자를 한 번 봅시다. 

```python
decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```

<img src="https://blog.keras.io/img/ae/deep_conv_ae_128.png">

또한, 128차원의 인코딩된 표현으로도 볼 수 있습니다. 이 표현은 8x4x4이기 때문에 그레이스케일 이미지에서 나타내기 위해 4x32로 reshape해야합니다. 

```python
n = 10
plt.figure(figsize=(20, 8))
for i in range(n):
    ax = plt.subplot(1, n, i)
    plt.imshow(encoded_imgs[i].reshape(4, 4 * 8).T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```

<img src="https://blog.keras.io/img/ae/encoded_representations.png">

## Applicatoin to image denosing



[1]: dfdfd

[2]: dfd





