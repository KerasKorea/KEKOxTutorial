# GANs을 이용한 Single Image Super Resolution — Keras

원문: (Medium) [**Single Image Super Resolution Using GANs — Keras**](https://medium.com/@birla.deepak26/single-image-super-resolution-using-gans-keras-aca310f33112), Deepak Birla

## 문서 소개

> 이 문서는 Super Resolution을 기존의 보간법(interpolation)과 같은 방법이 아닌 GANs을 사용하여 성능을 올린 SRGAN의 내용과 함께 keras코드로 쉽게 설명합니다.

- Super Resolution
- GANs
- Perceptual loss


**Image Super Resolution**:
Image super resolution은 화질 저하를 최소로 유지하면서 작은 이미지의 크기를 증가시키거나, 저해상도 이미지에서 얻은 풍부한 디테일에서 고해상도 이미지를 복원하는 것으로 정의할 수 있습니다. 이 문제는 주어진 저해상도 이미지에 대한 여러 솔루션이 있기 때문에 상당히 복잡합니다. 이것은 위성 및 항공 이미지 분석, 의료 이미지 처리, 압축 이미지/비디오 강화 등과 같은 수많은 응용 프로그램을 가지고 있습니다.

**Problem Statement(문제점)**:
저해상도 이미지에서 고해상도 이미지를 복구하거나 복원하기 위해서 다음과 같이 할 수 있습니다. 노이즈 감소(noise-reduction), 업스케일링 이미지(up-scaling image) 및 색상 조정(color adjustments)을 포함하는 이미지 개선과 같은 많은 방법이 있습니다. 이 글에서는 우리는 저해상도의 이미지에 딥 네트워크와 adversarial network (Generative Adversarial Networs)를 적용하여 고해상도의 이미지로 만드는 것에 대해 이야기해 볼 것입니다.

우리의 주요 목표는 재구성된 SR 영상의 텍스쳐 디테일이 손실되지 않도록 저해상도의 영상을 업스케일링해서 초해상도 영상(Super Resolution)이나 고해상도 영상으로 재구성하는 것입니다.

**Why Deep Learning?**
이미지 화질을 높이는 방법은 다양합니다. 흔히 써온 방법 중 하나는 보간법(interpolation)입니다. 보간법은 사용하기는 쉽지만 이미지가 왜곡되거나 시각적 품질을 떨어뜨리는 단점이 있습니다. 대부분의 보간법, bi-cubic과 같은 방법은 흐릿한 된 이미지를 만들어냅니다. 조금 더 정교한 방법들은 주어진 이미지의 유사성을 이용하거나, 저해상도 이미지와 고해상도 이미지의 데이터셋을 사용하여 이 두 이미지 간의 맵핑을 효과적으로 학습합니다. Example-Based SR 알고리즘 중, Sparse‑Coding‑Based 방법이 가장 인기 있는 방법 중 하나입니다.

딥러닝은 최적화된 이미지를 얻기 위한 더 나은 솔루션을 제공합니다. 최근 몇 년 많은 방법들이 Image Super Resolution을 위해 제안 되었습니다. 우리는 SRGAN을 이야기 해 볼 것입니다. 다른 딥러닝 방법들을 한 번 봅시다.

* SRCNN : SRCNN은 전통적인 방법들을 뛰어넘는 첫 번째 딥러닝 방식이었습니다. 이것은 3개의 컨볼루션 레이어로만 구성된 Convolutional Neural Network입니다: patch extraction과 representation, 비선형(non-linear) 맵핑 및 재구성(Reconstruction). 더 자세한 정보는 SRCNN의 [논문](https://arxiv.org/pdf/1501.00092.pdf)에서 확인하세요.

> 번역자의 부가 설명
> `Patch extraction과 representation`은 저해상도의 이미지에서 patch를 추출해서 high dimensional feature vector로 표현하는 단계
> `Non-linear mapping`은High dimensional feature space (e.g. LR patch space)의 patch들을 또 다른 high dimensional feature space (e.g. HR patch space)로 mapping해주는 단계
> `Reconstruction`는 mapping된 feature vector를 바탕으로 HR 이미지를 만들어내는 단계

* VDSR : Very Deep Super Resolution은 SRCNN과 유사한 구조를 사용하지만 더 높은 정확도를 얻기 위해 더 깊어진 구조입니다. SRCNN과 VDSR 모두 입력단계에서 bi-cubic 업샘플링 적용하였고 피쳐 맵을 출력과 동일한 scale로 처리합니다. 더 자세한 정보는 VDSR의 [논문](https://arxiv.org/pdf/1511.04587.pdf)에서 확인하세요.
 
**SRGAN — Super Resolution Generative Adversarial Network**
SRGAN을 이야기하기 전에 우리는 GANs(Generative Adversarial Networks)에 대해 알아야 합니다:

**GANs** : GANs는 비지도 학습(Unsupervised Machine Learning)에서 사용되는 AI 알고리즘의 종류입니다. GANs은 두 개의 네트워크(Generative와 Discriminator)로 구성된 딥러닝 아키텍처를 가지고 있고, 두 네트워크는 서로 반대되는(적대적인; adversarial) 성질을 가지고 있습니다. GANs은 초상화를 그리거나 교향곡을 작곡하는 것과 같은 창작에 관한 것입니다. GANs의 요점은 처음부터 데이터를 생성해 내는 것입니다.

GANs을 이해하기 위해서, 우리는 제일 먼저 generative 모델이 무엇인지 이해해야 합니다. 머신러닝에서, 두 가지 주요 모델의 종류는 생성적이고 차별적입니다. Discriminaive 모델은 두 개 이상의 서로 다른 데이터 클래스를 구별하는 모델입니다 - 예를 들어 자동차의 이미지일 때는 1을 출력하고, 자동차가 아닐 때는 0을 출력하는 convolutional neural network와 같습니다. Generative 모델은 데이터에 관해서 아무것도 모릅니다. 대신, generative 모델의 목적은 학습 데이터의 분포에 맞는 새로운 데이터를 생성하는 것입니다.

GANs은 Generator와 Discriminator로 구성되어 있습니다. Generator가 확률분포에서 데이터를 생성하려고 하고 Discriminator가 생성된 데이터를 심판처럼 행동하는 게임처럼 생각해보세요. Discriminator는 가짜 생성 데이터가 실제 학습 데이터에서 나온 것인지 아닌지 판별합니다. Generators는 실제 학습 데이터와 일치하도록 데이터를 최적화하려고 시도합니다. 또는 우리는 discriminator가 generator를 현실적인 데이터를 만들어내도록 이끈다고 할 수 있습니다. 이들은 인코더와 디코더처럼 일합니다.

예를 통해 쉽게 이해해봅시다.

애니메이션 캐릭터 얼굴을 만들고 싶다고 해봅시다. 그렇다면 우리는 애니메이션 캐릭터 얼굴의 이미지로 구성된 학습데이터를 준비할 것입니다. 그리고 가짜 데이터(fake data)는 랜덤 노이즈(random noise)로 구성되어 있을 것입니다. 이제 generator는 랜덤 노이즈로부터 dicriminator에게 판단될 이미지를 만들기 시작할 것 입니다. Generator와 discriminator 모두 학습을 계속하여 발전기가 실제 학습 데이터와 일치하는 이미지를 생성할 수 있도록 할 것 입니다. 한 가지 흥미로운 점은, generator에 의해 생성된 이미지는 원래의 학습 데이터 이미지에서 나온 특징을 가질 것이지만, 진짜 데이터와 동일하거나 아닐 수도 있다는 것입니다. 이와 같이 우리는 학습 데이터로부터 서로 다른 기능의 혼합으로 애니메이션 얼굴을 만들 수 있습니다.

![Figure 1: GANs basic architecture]('./media/200_0.jpeg')

dircriminator와 generator는 동시에 학습하며, 일단 generator를 학습하면 학습 데이터 샘플의 분포에 대해 충분히 알 수 있으므로 이제 매우 유사한 특성을 공유하는 새로운 샘플을 생성할 수 있게됩니다.

**SRGAN :**
SRGAN의 아이디어: 우리는 단일 이미지(Single image) super resolution에 대한 여러가지 방법을 훑어 보았습니다. 이 방법들은 꽤 빠르고 정확했습니다. 하지만 아직도 한 가지 풀지못한 문제점이 있습니다. 그것은, 우리가 이미지가 왜곡되지 않도록 해상도가 낮은 이미지에서 보다 미세한 텍스처 디테일을 어떻게 복구할 수 있는가 하는 것입니다. 최근의 논문들은 mean squared reconstruction 에러를 감소시키는데 초점을 두었습니다. 그 결과로 high peak signal-to-noise ratios(PSNR)을 가지는 것은 우리가 좋은 영상 품질 결과를 가지고 있다는 것을 의미하지만, 고주파 세부 정보가 부족하고 고해상도 영상에서 기대되는 충실도를 충족시키지 못하기 때문에 사실상 만족스럽지 못한 경우가 많다는 것을 알게되었습니다. 이전의 방법들은 픽셀 공간(픽셀 사이의)의 유사성을 확인하려고 시도하는데, 이는 지각적으로 만족스럽지 못한 결과를 초래하거나 흐릿한 이미지를 생성하게 됩니다. 그래서 우리는 모델의 출력 결과 이미지와 실제 이미지 사이의 지각적 차이를 포착할 수 있는 안정적인 모델이 필요합니다.

위에서 말한 안정적인 모델을 위해 우리는 Content loss 와 Adversarial loss로 구성된 Perceptual 손실 함수를 사용할 것 입니다. 그 외에 SRGAN은 딥 뉴럴 네트워크에 residual 블록을 사용합니다.

![SRGAN Architecture]('./media/200_1.jpeg')

이제 SRGAN의 디테일한 부분들을 살펴봅시다: Super-resolution GAN은 더 높은 해상도의 이미지를 생성하기 위해 adversary 네트워크와 결합시켜 딥 네트워크를 만듭니다.

학습 과정은 다음의 과정과 같습니다:
* HR(High Resolution) 영상을 처리하여 다운샘플링해서 LR(Low Resolution) 영상을 얻습니다. 이제 교육 데이터 세트를 위한 HR 영상과 LR 영상을 모두 확보하였습니다.
* LR이미지를 generator에게 넘겨 SR(Super Resolution) 이미지로 업샘플링합니다.
* Generator가 만든 이미지와 HR이미지를 구분하고 구한 GAN 로스(loss)를 backpropagation으로 dicriminator를 학습합니다.

![Figure 3: Generator and Discriminator Network]('./media/200_2.jpeg')

위는 generator와 discriminator에 대한 네트워크 설계입니다. 주로 컨볼루션 레이어, 배치 노말라이제이션, Parameterized RELU(PReLU)로 구성됩니다. generator는 또한 ResNet과 유사한 스킵 연결이 필요합니다.

네트워크 아키텍처에 대한 몇 가지 노트:
* Residual 블록: 네트워크가 더 깊어짐에 따라 학습은 더 어려워집니다. Residual 학습은 프레임워크는 이런 네트워크들의 학습을 쉽게 만들어주고, 훨씬 더 심층적으로 작업할 수 있게 해주며, 이를 통해 성능 향상을 이끌어 낼 수 있습니다. Residual 블록과 Deep Residual 학습에 대한 내용은 아래에 주어진 논문을 통해 더 알아보실 수 있습니다. 16개의 residual 블록이 generator에 쓰였습니다.
* PixelShuffler x2: 이건 피쳐 맵 업스케일러입니다. 2 sub-pixel CNN이 generator에 쓰입니다. 업스케일링 혹은 업샘플링과 같습니다. 이를 위한 방법은 여러가지입니다. 케라스 코드로 짜여있습니다.
* PRelu(Parameterized Relu): 우리는 Relu 또는 LeakyRelu대신 PRelu를 사용할 것입니다. PRelu는 음의 coefficient를 학습할 수 있도록 하는 학습 가능한 파라미터를 도입합니다.
* k3n64s1의 의미는 kernel 3 channels 64 strides 1을 의미합니다
* 손실 함수: 가장 중요한 부분입니다. 우리가 이미 이야기 했듯이 Perceptual loss를 사용할 것입니다. 이것은 Content(Reconstruction) loss 와 Adversarial loss로 구성되어 있습니다.

![Perceptual Loss]('./media/200_3.jpeg')

* Adversarial loss: 이것은 우리의 솔루션을 super-resolved 이미지와 원래의 이미지의 구별을 위해 훈련된 판별기 네트워크를 이용하여 이미지 매니폴드로 가도록 해줍니다.


* 참고자료
http://jaejunyoo.blogspot.com/2019/05/deep-learning-for-SISR-survey-1.html

> 2020 새마음 새뜻으로 이든이가 해쪄욤 
> Translator: [박정현](https://github.com/parkjh688)  
> Translator email : parkjh688@gmail.com
> Translator twitter : @edensuperb
