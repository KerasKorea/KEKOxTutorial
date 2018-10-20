## Keras를 이용한 CNN Ensemble 기법 (Ensembling ConvNets using Keras)
[원문](https://towardsdatascience.com/ensembling-convnets-using-keras-237d429157eb)
> 이 글은 머신 러닝과 통계에서 주로 사용되는 앙상블(Ensemble)을 어떻게 사용하는지 CNN 모델들을 통해서 알려줍니다. 앙상블 기법을 적용해 설계한 모델 성능을 한층 강화 시켜봅시다.

* keras
* CNN
* Ensemble
* convnet
* machine learning


![intro img](https://raw.githubusercontent.com/KerasKorea/KEKOxTutorial/master/media/16_1.jpeg)

### 개요
> 통계 및 머신 러닝에서 앙상블 기법은 단일 학습 알고리즘에서 얻는 결과보다 더 나은 성능을 위해 여러 학습 알고리즘을 사용합니다. 일반적으로 무한한,통계학에서 통계적인 앙상블과는 달리 머신 러닝 앙상블 기법은 오직 확실하고 유한한 대안 모델들로 구성되 있습니다. 하지만 이러한 모델들 사이에서 더 유연한 구조가 존재할 수 있습니다. [1](https://en.wikipedia.org/wiki/Ensemble_learning)

앙상블 사용에 주요 동기는 생성된 모델들의 가정 공간에 포함되지 않은 가정(hypothesis)를 찾는 것입니다. 경험적으로, 앙상블은 모델들간 많은 다양성이 존재할 때, 더 좋은 결과를 보여주는 경향이 있습니다. [2](http://jair.org/papers/paper614.html)

#### 동기
만약 큰 머신 러닝 대회의 결과를 본다면, 아마도 단일 모델보단 여러 모델들의 앙상블의 결과가 더 좋은 걸 확인할 수 있습니다. 예를 들어, [ILSVRC2015](http://www.image-net.org/challenges/LSVRC/2015/results)에서 최고 점수를 획득한 단일 모델은 13등에 머물렀습니다. 1등부터 12등까지는 다양한 모델들의 앙상블이 차지했습니다.

앙상들로 다양한 신경망을 사용하는 방법에 관한 문서나 튜토리얼을 본 적이 없기에 이번 튜토리얼을 만들게 되었습니다.

[Keras](https://keras.io/), 특히 [실용 API](https://keras.io/models/model/), 를 사용해 상대적으로 잘 알려진 논문에서 3가지 작은 CNN(ResNet50, Inception과 비교해) 모델을 새로 만들 것입니다. 각 모델들은 [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) 학습 데이터 세트를 기반으로 학습을 진행할 겁니다. [3](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf) 그 때 각 모델들은 테스트 세트를 사용해서 평가될 겁니다. 그 후에, 3가지 모델을 앙상블해 평가를 진행할 겁니다. 앙상블이 어떤 단일 모델들보다 테스트 세트에서 더 좋은 성능을 보여줄 거라 기대됩니다.

여러 앙상블 기법 중 한가지로 **스택(Stacking)**이 있습니다. 이는 일반적인 앙상블 기법으로 다른 앙상블 기법을 대표할 수 있습니다. 스택은 여러 학습 알고리즘의 예측을 결합하는 학습 알고리즘을 학습하는걸 포함하고 있다.[1](https://en.wikipedia.org/wiki/Ensemble_learning#Stacking) 예시를 위해, 앙상블에서 모델 결과값의 평균을 취하는 가장 단순한 스택을 사용할 것입니다. 평균화에 매개변수가 필요하지 않으므로, 해당 앙상블을 학습할 필요가 없습니다.

![This post’s ensemble in a nutshell](https://raw.githubusercontent.com/KerasKorea/KEKOxTutorial/master/media/16_2.png)


#### 데이터 준비
첫 번째, 필요한 라이브러리를 불러옵니다.

```python
from keras.models import Model, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, Average, Dropout
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras.datasets import cifar10

import numpy as np
```

CIFAR-10에서 잘 작동하는 구조를 설명하는 논문을 찾는 것이 상대적으로 쉽기 때문에 CIFAR-10을 사용하고 있습니다. 유명한 데이터 세트를 사용하면 쉽게 재현할 수 있기도 합니다.

아래는 데이터 세트를 불러오는 과정입니다. 학습용, 테스트용 데이터 둘다 정규화합니다. 학습 레이블 벡터는 **one-hot 행렬**로 변환됩니다. 테스트 레이블 벡터를 변환할 필요는 없습니다. 이는 학습단계에서 사용되지 않기 때문이죠.

```python
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.
x_test = x_test / 255.
y_train = to_categorical(y_train, num_classes=10)
```

데이터 세트는 10개의 클래스인 6만장의 32X32 RGB 이미지로 구성되 있습니다. 5만장은 학습과 확인용으로 사용되고 나머지 1만장은 테스트용으로 사용됩니다.

```python
print('x_train shape: {} | y_train shape: {}\nx_test shape : {} | y_test shape : {}'.format(x_train.shape, y_train.shape,                                                                                      x_test.shape, y_test.shape))
```

*>>>* *x_train shape: (50000, 32, 32, 3) | y_train shape: (50000, 10)*  
*>>>* *x_test shape: (10000, 32, 32, 3) | y_test shape: (10000, 10)*

3가지 모델 전부 동일한 형태의 데이터로 작업을 하기에, 모든 모델에서 사용될 단일 입력 레이어를 정의하는게 편리합니다.

```python
input_shape = x_train[0,:,:,:].shape
model_input = Input(shape=input_shape)
```

#### 첫번째 모델 : ConvPool-CNN-C

첫번째로 학습하려는 모델은 ConvPool-CNN-C입니다. 자세한 설명은 [해당 논문](https://arxiv.org/abs/1412.6806)의 4쪽에서 보실수 있습니다. 

첫번째 모델은 꽤나 간단합니다. 몇 개의 컨볼루션 레이어(convolution layer)가 풀링 레이어(pooling layer) 뒤에 연결되는 일반적인 패턴을 띄고 있습니다. 다만, 마지막 레이어에서 익숙하지 않을 수 있습니다. 몇 개의 완전 연결 레이어(FC layer)를 사용하는 대신, 전역 평균 풀링 레이어를 사용했습니다.

어떻게 전역 풀링 레이어가 작동하는지 전반적은 흐름을 알아봅시다. 마지막 컨볼루션 레이어 `Conv2D(10, (1,1))`는 10가지 출력 클래스에 따라서 10가지 피쳐맵을 출력합니다. 그때 `GolobalAveragePooling2D()`레이어는 10가지 피쳐맵의 공간적 평균을 계산합니다. 이는 출력값이 단지 길이가 10인 벡터임을 의미합니다. 그 후, 소프트맥스(softmax) 활성 함수를 적용합니다. 여기서 볼 수 있듯이, 이 방법은 모델 상단에 완전 연결 레이어를 사용하는 것과 동일한 방법입니다. 신경망 논문에서 이 모델의 장점과 전역 풀링 레이어에 대해 좀 더 알아볼 수 있습니다.[5](https://arxiv.org/abs/1312.4400)

여기서 중요 요점 한가지로, 마지막 `Conv2D(10, (1,1))` 레이어의 출력에는 활성 함수를 사용하지 않았다는 점입니다. 이는, 해당 레이어의 출력이 `GlobalAveragePoling2D()`의 앞부분으로 연결되기 때문입니다.

```python
def conv_pool_cnn(model_input):
    
    x = Conv2D(96, kernel_size=(3, 3), activation='relu', padding = 'same')(model_input)
    x = Conv2D(96, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(96, (3, 3), activation='relu', padding = 'same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides = 2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides = 2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (1, 1), activation='relu')(x)
    x = Conv2D(10, (1, 1))(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='softmax')(x)
    
    model = Model(model_input, x, name='conv_pool_cnn')
    
    return model
```

아래 코드는 첫번째 모델을 인스턴스하는 부분입니다.

```python
conv_pool_cnn_model = conv_pool_cnn(model_input)
```

단순화를 위해, 각 모델들은 동일한 매개변수를 사용해서 학습되고 컴파일됩니다. 크기가 32인 배치(batch)로 20에폭(epoch) 진행하면 세가지 모델 중 어떤 것이 되더라도 로컬 최소값(local minima)를 찾기엔 충분해 보입니다. 학습 데이터 세트의 20%를 임의로 선정하여 확인용으로 사용하게 됩니다.

```python
def compile_and_train(model, num_epochs): 
    
    model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['acc']) 
    filepath = 'weights/' + model.name + '.{epoch:02d}-{loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_weights_only=True, save_best_only=True, mode='auto', period=1)
    tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=32)
    history = model.fit(x=x_train, y=y_train, batch_size=32, epochs=num_epochs, verbose=1, callbacks=[checkpoint, tensor_board], validation_split=0.2)
    return history
```

단일 테슬라 k80 gpu를 사용할 시, 1에폭 동안 첫번째와 두번째 모델을 학습하는데 1분정도 걸립니다. cpu를 사용할 경우 학습에 시간이 약간 걸릴겁니다.

```python
_ = compile_and_train(conv_pool_cnn_model, num_epochs=20)
```

첫번째 모델에서 확인용 데이터 세트에서 ~79% 정확도가 나왔습니다.

이제 테스트 세트기반으로 에러율을 계산해 첫번째 모델을 평가합니다.

![ConvPool-CNN-C validation accuracy and loss](https://raw.githubusercontent.com/KerasKorea/KEKOxTutorial/master/media/16_3.png)

```python
def evaluate_error(model):
    pred = model.predict(x_test, batch_size = 32)
    pred = np.argmax(pred, axis=1)
    pred = np.expand_dims(pred, axis=1) # make same shape as y_test
    error = np.sum(np.not_equal(pred, y_test)) / y_test.shape[0]  
  
    return error
evaluate_error(conv_pool_cnn_model)
```

*>>>* 0.2414

#### 두번째 모델 : ALL-CNN-C

![ALL-CNN-C validation accuracy and loss](https://raw.githubusercontent.com/KerasKorea/KEKOxTutorial/master/media/16_4.png)

#### 세번째 모델 : Network in Network CNN

![NIN-CNN validation accuracy and loss](https://raw.githubusercontent.com/KerasKorea/KEKOxTutorial/master/media/16_5.png)

#### 세가지 모델을 앙상블

#### 가능한 앙상블 형태

### 결론


### 참고문서
* [Ensemble Learning](https://en.wikipedia.org/wiki/Ensemble_learning)
* [Popular Ensemble Methods: An Empirical Study](http://jair.org/papers/paper614.html)
* [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
* [Striving for Simplicity: The All Convolutional Net](https://arxiv.org/abs/1412.6806v3)
* [Network In Network](https://arxiv.org/abs/1312.4400v3)


> 이 글은 2018 컨트리뷰톤에서 [`Contributue to Keras`](https://github.com/KerasKorea/KEKOxTutorial) 프로젝트로 진행했습니다.  
> Translator : [mike2ox](https://github.com/mike2ox) (Moonhyeok Song)  
> Translator Email : <firefinger07@gmail.com>