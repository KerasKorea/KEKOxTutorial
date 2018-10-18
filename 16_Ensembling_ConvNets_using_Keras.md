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

#### 첫번째 모델 : ConvPool-CNN-C

![ConvPool-CNN-C validation accuracy and loss](https://raw.githubusercontent.com/KerasKorea/KEKOxTutorial/master/media/16_3.png)

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