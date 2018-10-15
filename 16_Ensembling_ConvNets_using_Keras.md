## Keras를 이용한 CNN Ensemble 기법 (Ensembling ConvNets using Keras)
[원문](https://towardsdatascience.com/ensembling-convnets-using-keras-237d429157eb)
> 이 글은

* keras
* CNN
* Ensemble
* convnet


![intro img](https://raw.githubusercontent.com/KerasKorea/KEKOxTutorial/master/media/16_1.jpeg)

### 개요
> 

#### 동기

![This post’s ensemble in a nutshell](https://raw.githubusercontent.com/KerasKorea/KEKOxTutorial/master/media/16_2.png)


#### 데이터 준비

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