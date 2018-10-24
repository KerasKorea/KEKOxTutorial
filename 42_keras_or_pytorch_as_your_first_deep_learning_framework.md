## Keras vs PyTorch 어떤 플랫폼을 선택해야 할까?(keras or pytorch as your first deep learning framework)
[원문](https://deepsense.ai/keras-or-pytorch/)
> 본 글은 딥러닝을 배우는, 가르치는 입장에서 어떤 프레임워크가 좋은지를 Keras와 PyTorch를 비교하며 독자가 선택을 할 수 있게 내용을 전개하고 있다. 원 작성자인 Piotr Migdal과 Rafal Jakubanis은 자신들의 경험을 바탕으로 글을 설명하고 있으므로 더 정확한 선택이 있으리라 생각한다.

* Keras
* PyTorch
* framework

![Keras_vs_PyTorch](https://github.com/KerasKorea/KEKOxTutorial/blob/issue_42/media/42_0.png)  

> 본 글을 읽고 있는 그대, 딥러닝을 배우고 싶나요? 그대가 당신 비즈니스에 적용할 지, 다음 프로젝트에 적용할 지, 아니면 그저 시장성 있는 기술을 갖고 싶은 것인지가 중요하다. 배우기 위해 적절한 프레임워크를 선택하는건 그대 목표에 도달하기 위해 중요한 첫 단계이다. 

우리는 강력하게 당신이 Keras나 PyTorch를 선택하길 추천합니다. 그것들은 배우기도, 실험하기도 재밌는 강력한 도구들입니다. 우리는 교사나 학생의 입장에서 둘 다 알고 있습니다. Piotr는 두 프레임워크로 워크숍을 진행했고, Rafal은 현재 배우고 있는 중입니다.  

([Hacker News](https://news.ycombinator.com/item?id=17415321)와 [Reddit](https://www.reddit.com/r/MachineLearning/comments/8uhqol/d_keras_vs_pytorch_in_depth_comparison_of/)에서 논의한 것을 참조하세요.)

### 소개
Keras와 PyTorch는 데이터 과학자들 사이에서 인기를 얻고있는 딥러닝용 오픈 소스 프레임워크입니다.  

- [Keras](https://keras.io/)는 Tensorflow, CNTK, Theano, MXNet(혹은 Tensorflow안의 tf.contrib)의 상단에서 작동할 수 있는 고수준 API입니다. 2015년 3월에 첫 배포를 한 이래로, 쉬운 사용법과 간단한 문법, 빠른 설계 덕분에 인기를 끌고 있습니다. 이 도구는 구글에서 지원받고 있습니다.
- [PyTorch](https://pytorch.org/)는 2016년 10월에 배포된, 배열 표현식으로 직접 작업 저수준 API입니다. 작년에 큰 관심을 끌었고, 학술 연구에서 선호되는 솔루션, 맞춤 표현식으로 최적화하는 딥러닝 어플리케이션이 되어가고 있습니다. 이 도구는 페이스북에서 지원받고 있습니다.

우리가 두 프레임워크([참조](https://www.reddit.com/r/MachineLearning/comments/6bicfo/d_keras_vs_PyTorch/))의 핵심 상세 내용을 논의하기 전에 당신을 실망시키고자 합니다. - '어떤 툴이 더 좋은가?'에 대한 정답은 없습니다. 선택은 절대적으로 당신의 기술적 지식, 필요성 그리고 기대에 달렸습니다. 본 글은 당신이 처음으로 두 프레임워크 중 한가지를 선택할 때 도움이 될 아이디어를 제공해주는데 목적을 두고 있습니다.

#### 요약하자면
Keras는 플러그 & 플레이 정신에 맞게, 표준 레이어로 실험하고 입문하기 쉬울 겁니다.

PyTorch는 수학적으로 연관된 더 많은 사용자들을 위해 더 유연하고 저수준의 접근성을 제공합니다. 


### 좋아, 근데 다른 프레임워크는 어때?
Tensorflow는 대중적인 딥러닝 프레임워크입니다. 그러나, 원시 Tensorflowsms 계산 그래프 구축을 장황하고 모호하게 추상화하고 있습니다. 일단 딥러닝의 기초 지식을 알고 있다면 문제가 되지 않습니다. 하지만, 새로 입문하는 사람에겐 공식적으로 지원되는 인터페이스로써 Keras를 사용하는게 더 쉽고 생산적일 겁니다.  

[수정 : 최근, Tensorflow에서 [Eager Execution](https://www.tensorflow.org/versions/r1.9/programmers_guide/keras)를 소개했는데 이는 모든 python 코드를 실행하고 초보자에게 보다 직관적으로 모델을 학습시킬 수 있게 해줍니다.(특히 tf.keras API를 사용할 때!)]

그대가 어떤 Theano 튜토리얼을 찾았지만, 이는 더 이상 활발한 개발이 이뤄지지지 않습니다. Caffe는 유연성이 부족하지만, Torch는 Lua를 사용합니다. MXNet, Chainer 그리고 CNTK는 현재 대중적이지 않습니다.


### Keras vs PyTorch : 쉬운 사용법과 유연성
Keras와 PyTorch는 작동에 대한 추상화 단게에서 다릅니다.  

Keras는 딥러닝에 사용되는 레이어와 연산자들을 neat(레코 크기의 블럭)로 감싸고, 데이터 과학자의 입장에서 딥러닝 복잡성을 추상화하는 고수준 API입니다. 

PyTorch는 유저들에게 맞춤 레이어를 작성하고 수학적 최적화 작업을 볼 수 있게 자율성을 주도록 해주는 저수준 환경을 제공합니다. 더 복잡한 구조 개발은 python의 모든 기능을 사용하고 모든 기능의 내부에 접근하는 것보다 간단합니다.

어떻게 Keras와 PyTorch로 간단한 컨볼루션 신경망을 정의할 지를 head-to-head로 비교해보자. 

#### Keras

```python
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPool2D())
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
```

#### PyTorch

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
    
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 10) 
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 6 * 6)
        x = F.log_softmax(self.fc1(x), dim=-1)
 
        return x
 
model = Net()
```

위 코드 블럭은 두 프레임워크의 차이를 약간 맛보게 해줍니다. 모델을 학습하기 위해, PyTorch는 20줄의 코드가 필요한 반면, Keras는 단일 코드만 필요했습니다. GPU 가속화 사용은 Keras에선 암묵적으로 처리되지만, PyTorch는 CPU와 GPU간 데이터 전송할 때 요구합니다.  

만약 초보자라면, Keras는 명확한 이점을 보일 것입니다. Keras는 실제로 읽기 쉽고 간결해 구현 단계에서의 세부 사항을 건너뛰는 동시에 그대의 첫번째 end-to-end 딥러닝 모델을 빠르게 설계하도록 해줄겁니다. 그러나, 이런 세부 사항을 뛰어넘는 건 당신의 딥러닝 작업에서 계산이 필요한 블럭의 내부 작업 탐색에 제한이 됩니다. PyTorch를 사용하는 건 당신에게 역전파처럼 핵심 딥러닝 개념과 학습 단계의 나머지 부분에 대해 생각할 것들을 제공합니다.

PyTorch보다 간단한 Keras는 더이상 장난감을 의미하진 않는다. 이는 초심자들이 사용하는 중요한 딥러닝 도구이다. 능숙한 데이터 과학자들에게도 마찬가지다. 예를 들면, Kaggle에서 열린 `the Dstl Satellite Imagery Feature Detection`에서 상위 3팀이 그들의 솔루션에 Keras를 사용하였다. 반면, 4등인 [우리](https://blog.deepsense.ai/deep-learning-for-satellite-imagery-via-image-segmentation/#_ga=2.53479528.114026073.1540369751-2000517400.1540369751)는 PyTorch와 Keras를 혼합해서 사용하였다.  

당신의 딥러닝 어플리케이션이 Keras가 제공하는 것 이상의 유연성을 필요하는 지 파악하는 건 가치가 있다. 그대의 필요에 따라, Keras는 [가장 적은 힘의 규칙](https://en.wikipedia.org/wiki/Rule_of_least_power)에 입각하는 좋은 방법이 될 수 있다.

#### 요약
- Keras : 좀 더 간결한 API
- PyTorch : 더 유연하고, 딥러닝 개념을 깁게 이해하는데 도움을 줌

### Keras vs PyTorch : 인기와 학습자료 접근성
프레임워크의 인기는 단지 유용성의 대리만은 아니다. 작업 코드가 있는 튜토리얼, 리포지토리 그리고 단체 토론 등 커뮤니티 지원도 중요합니다. 2018년 6월 현재, Keras와 PyTorch는 GitHub과 arXiv 논문에서 인기를 누리고 있습니다.(Keras를 언급한 대부분의 논문들은 Tensorflow 백엔드 또한 언급하고 있습니다.) KDnugget에 따르면, Keras와 PyTorch는 가장 빠르게 성장하는 [데이터 과학 도구들](https://www.kdnuggets.com/2018/05/poll-tools-analytics-data-science-machine-learning-results.html)입니다.

![Percentof ML papers that mention...](https://github.com/KerasKorea/KEKOxTutorial/blob/issue_42/media/42_1.png)  

> 지난 6년간 43k개의 ML논문을 기반으로, arxiv 논문들에서 딥러닝 프레임워크에 대한 언급에 대한 자료입니다. Tensorflow는 전체 논문의 14.3%, PyTorch는 4.7%, Keras 4.0%, Caffe 3.8%, Theano 2.3%, Torch 1.5. MXNet/chainer/cntk는 1% 이하로 언급되었습니다. [참조](https://t.co/YOYAvc33iN) - Andrej Karpathy (@karpathy) 

두 프레임워크는 만족스러운 참고문서를 갖고 있지만, PyTorch는 강력한 커뮤니티 지원을 제공합니다. 


#### 요약

### Keras vs PyTorch : 디버깅과 introspection

#### 요약

### Keras vs PyTorch : 모델을 추출하고 다른 플랫폼과의 호환성

#### 요약

### Keras vs PyTorch : 성능

![Tesla p100](https://github.com/KerasKorea/KEKOxTutorial/blob/issue_42/media/42_2.png)  

![Tesla K80](https://github.com/KerasKorea/KEKOxTutorial/blob/issue_42/media/42_3.png)  

#### 요약

### Keras vs PyTorch : 결론

### 참고문서
* [참고 사이트 1]()
* [참고 사이트 2]()


> 이 글은 2018 컨트리뷰톤에서 [`Contribute to Keras`](https://github.com/KerasKorea/KEKOxTutorial) 프로젝트로 진행했습니다.
> Translator: [mike2ox](https://github.com/mike2ox)(Moonhyeok Song)
> Translator email : <firefinger07@gmail.com>
