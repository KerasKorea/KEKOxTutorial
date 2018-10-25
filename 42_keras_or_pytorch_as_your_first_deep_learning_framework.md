## Keras vs PyTorch 어떤 플랫폼을 선택해야 할까?(keras or pytorch as your first deep learning framework)
[원문](https://deepsense.ai/keras-or-pytorch/)
> 본 글은 딥러닝을 배우는, 가르치는 입장에서 어떤 프레임워크가 좋은지를 Keras와 PyTorch를 비교하며 독자가 선택을 할 수 있게 내용을 전개하고 있다. 원 작성자인 Piotr Migdal과 Rafal Jakubanis은 자신들의 경험을 바탕으로 글을 설명하고 있으므로 더 정확한 선택이 있으리라 생각한다.

* Keras
* PyTorch
* framework

![Keras_vs_PyTorch](https://github.com/KerasKorea/KEKOxTutorial/blob/issue_42/media/42_0.png)  

> 본 글을 읽고 있는 당신, 딥러닝을 배우고 싶나요? 딥러닝을 당신의 사업에 적용하고 싶든, 다음 프로젝트에 적용하고 싶든, 아니면 그저 시장성 있는 기술을 갖고 싶든, 배우기에 적절한 프레임워크를 선택하는 것이 당신의 목표에 도달하기 위해 중요한 첫 단계입니다. 

우리는 강력하게 당신이 Keras나 PyTorch를 선택하길 추천합니다. 그것들은 배우기도, 실험하기도 재밌는 강력한 도구들입니다. 우리는 교사나 학생의 입장에서 둘 다 알고 있습니다. Piotr는 두 프레임워크로 워크숍을 진행했고, Rafal은 현재 배우고 있는 중입니다.  

([Hacker News](https://news.ycombinator.com/item?id=17415321)와 [Reddit](https://www.reddit.com/r/MachineLearning/comments/8uhqol/d_keras_vs_pytorch_in_depth_comparison_of/)에서 논의한 것을 참조하세요.)

### 소개
Keras와 PyTorch는 데이터 과학자들 사이에서 인기를 얻고있는 딥러닝용 오픈 소스 프레임워크입니다.  

- [Keras](https://keras.io/)는 Tensorflow, CNTK, Theano, MXNet(혹은 Tensorflow안의 tf.contrib)의 상단에서 작동할 수 있는 고수준 API입니다. 2015년 3월에 첫 배포를 한 이래로, 쉬운 사용법과 간단한 문법, 빠른 설계 덕분에 인기를 끌고 있습니다. 이 도구는 구글에서 지원받고 있습니다.
- [PyTorch](https://pytorch.org/)는 2016년 10월에 배포된, 배열 표현식으로 직접 작업하는 저수준 API입니다. 작년에 큰 관심을 끌었고, 학술 연구에서 선호되는 솔루션이자, 맞춤 표현식으로 최적화하는 딥러닝 어플리케이션이 되어가고 있습니다. 이 도구는 페이스북에서 지원받고 있습니다.

우리가 두 프레임워크([참조](https://www.reddit.com/r/MachineLearning/comments/6bicfo/d_keras_vs_PyTorch/))의 핵심 상세 내용을 논의하기 전에 당신을 실망시키고자 합니다. - '어떤 툴이 더 좋은가?'에 대한 정답은 없습니다. 선택은 절대적으로 당신의 기술적 지식, 필요성 그리고 기대에 달렸습니다. 본 글은 당신이 처음으로 두 프레임워크 중 한가지를 선택할 때 도움이 될 아이디어를 제공해주는데 목적을 두고 있습니다.

#### 요약하자면
Keras는 플러그 & 플레이 정신에 맞게, 표준 레이어로 실험하고 입문하기 쉬울 겁니다.

PyTorch는 수학적으로 연관된 더 많은 사용자들을 위해 더 유연하고 저수준의 접근성을 제공합니다. 


### 좋아, 근데 다른 프레임워크는 어때?
Tensorflow는 대중적인 딥러닝 프레임워크입니다. 그러나, 원시 Tensorflow는 계산 그래프 구축을 장황하고 모호하게 추상화하고 있습니다. 일단 딥러닝의 기초 지식을 알고 있다면 문제가 되지 않습니다. 하지만, 새로 입문하는 사람에겐 공식적으로 지원되는 인터페이스로써 Keras를 사용하는게 더 쉽고 생산적일 겁니다.  

[수정 : 최근, Tensorflow에서 [Eager Execution](https://www.tensorflow.org/versions/r1.9/programmers_guide/keras)를 소개했는데 이는 모든 python 코드를 실행하고 초보자에게 보다 직관적으로 모델을 학습시킬 수 있게 해줍니다.(특히 tf.keras API를 사용할 때!)]

그대가 어떤 Theano 튜토리얼을 찾았지만, 이는 더 이상 활발한 개발이 이뤄지지지 않습니다. Caffe는 유연성이 부족하지만, Torch는 Lua를 사용합니다. MXNet, Chainer 그리고 CNTK는 현재 대중적이지 않습니다.


### Keras vs PyTorch : 쉬운 사용법과 유연성
Keras와 PyTorch는 작동에 대한 추상화 단계에서 다릅니다.  

Keras는 딥러닝에 사용되는 레이어와 연산자들을 neat(레코 크기의 블럭)로 감싸고, 데이터 과학자의 입장에서 딥러닝 복잡성을 추상화하는 고수준 API입니다. 

PyTorch는 유저들에게 맞춤 레이어를 작성하고 수학적 최적화 작업을 볼 수 있게 자율성을 주도록 해주는 저수준 환경을 제공합니다. 더 복잡한 구조 개발은 python의 모든 기능을 사용하고 모든 기능의 내부에 접근하는 것보다 간단합니다.

어떻게 Keras와 PyTorch로 간단한 컨볼루션 신경망을 정의할 지를 head-to-head로 비교해봅시다. 

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

위 코드 블럭은 두 프레임워크의 차이를 약간 맛보게 해줍니다. 모델을 학습하기 위해, PyTorch는 20줄의 코드가 필요한 반면, Keras는 단일 코드만 필요했습니다. GPU 가속화 사용은 Keras에선 암묵적으로 처리되지만, PyTorch는 CPU와 GPU간 데이터를 전송할 때 요구합니다.  

만약 초보자라면, Keras는 명확한 이점을 보일 것입니다. Keras는 실제로 읽기 쉽고 간결해 구현 단계에서의 세부 사항을 건너뛰는 동시에 그대의 첫번째 end-to-end 딥러닝 모델을 빠르게 설계하도록 해줄겁니다. 그러나, 이런 세부 사항을 뛰어넘는 건 당신의 딥러닝 작업에서 계산이 필요한 블럭의 내부 작업 탐색에 제한이 됩니다. PyTorch를 사용하는 건 당신에게 역전파처럼 핵심 딥러닝 개념과 학습 단계의 나머지 부분에 대해 생각할 것들을 제공합니다.

PyTorch보다 간단한 Keras는 더이상 장난감을 의미하진 않습니다. 이는 초심자들이 사용하는 중요한 딥러닝 도구입니다. 능숙한 데이터 과학자들에게도 마찬가지입니다. 예를 들면, Kaggle에서 열린 `the Dstl Satellite Imagery Feature Detection`에서 상위 3팀이 그들의 솔루션에 Keras를 사용하였습니다. 반면, 4등인 [우리](https://blog.deepsense.ai/deep-learning-for-satellite-imagery-via-image-segmentation/#_ga=2.53479528.114026073.1540369751-2000517400.1540369751)는 PyTorch와 Keras를 혼합해서 사용하였습니다.  

당신의 딥러닝 어플리케이션이 Keras가 제공하는 것 이상의 유연성을 필요하는 지 파악하는 건 가치가 있습니다. 그대의 필요에 따라, Keras는 [가장 적은 힘의 규칙](https://en.wikipedia.org/wiki/Rule_of_least_power)에 입각하는 좋은 방법이 될 수 있습니다.

#### 요약
- Keras : 좀 더 간결한 API
- PyTorch : 더 유연하고, 딥러닝 개념을 깁게 이해하는데 도움을 줌

### Keras vs PyTorch : 인기와 학습자료 접근성
프레임워크의 인기는 단지 유용성의 대리만은 아닙니다. 작업 코드가 있는 튜토리얼, 리포지토리 그리고 단체 토론 등 커뮤니티 지원도 중요합니다. 2018년 6월 현재, Keras와 PyTorch는 GitHub과 arXiv 논문에서 인기를 누리고 있습니다.(Keras를 언급한 대부분의 논문들은 Tensorflow 백엔드 또한 언급하고 있습니다.) KDnugget에 따르면, Keras와 PyTorch는 가장 빠르게 성장하는 [데이터 과학 도구들](https://www.kdnuggets.com/2018/05/poll-tools-analytics-data-science-machine-learning-results.html)입니다.

![Percentof ML papers that mention...](https://github.com/KerasKorea/KEKOxTutorial/blob/issue_42/media/42_1.png)  

> 지난 6년간 43k개의 ML논문을 기반으로, arxiv 논문들에서 딥러닝 프레임워크에 대한 언급에 대한 자료입니다. Tensorflow는 전체 논문의 14.3%, PyTorch는 4.7%, Keras 4.0%, Caffe 3.8%, Theano 2.3%, Torch 1.5. MXNet/chainer/cntk는 1% 이하로 언급되었습니다. [참조](https://t.co/YOYAvc33iN) - Andrej Karpathy (@karpathy) 

두 프레임워크는 만족스러운 참고 문서를 갖고 있지만, PyTorch는 강력한 커뮤니티 지원을 제공합니다. 해당 커뮤니티 게시판은 당신이 난관에 부딪쳤거나 참고 문서나 스택오버플로우는 당신이 필요로 하는 정답이 없다면 방문하기 좋은 곳이다.

한 일화를 들자면, 우리는 특정 신경망 구조에서 초심자 수준의 딥러닝 코스를 PyTorch보다 Keras로 더 쉽게 접근할 수 있다는 걸 발견했습니다. Keras에서 제공하는 코드의 가독성과 실험을 쉽게 해주는 장점으로 인해, Keras는 딥러닝 열광자, 튜터, 고수준의 kaggle 우승자들에 의해 많이 쓰이게 될 겁니다.

Keras 자료와 딥러닝 코스의 예시로, ["Starting deep learning hands-on: image classification on CIFAR-10"](https://blog.deepsense.ai/deep-learning-hands-on-image-classification/#_ga=2.52232937.114026073.1540369751-2000517400.1540369751)와 ["Deep Learning with Rython"](https://www.manning.com/books/deep-learning-with-python)를 참조하십시오. PyTorch 자료로는, 신경망의 내부 작업을 학습하는데 더 도던적이고 포괄적인 접근법을 제공하는 공식 튜토리얼을 추천합니다. PyTorch에 대한 전반적인 내용을 보려면, 이 [문서](http://www.goldsborough.me/ml/ai/python/2018/02/04/20-17-20-a_promenade_of_pytorch/)를 참조하세요.

#### 요약
- Keras : 튜토리얼이나 재사용 가능한 코드로의 접근성이 좋음
- PyTorch : 뛰어난 커뮤니티와 활발한 개발

### Keras vs PyTorch : 디버깅과 코드 복기(introspection)
추상화에서 많은 계산 조각들을 묶어주는 Keras는 문제를 발생시키는 외부 코드 라인을 고정시키는 게 어렵습니다. 좀 더 장황하게 구성된 프레임워크인 PyTorch는 우리의 스크립트 실행을 따라갈 수 있게 해줍니다. 이건 Numpy를 디버깅하는 것과 유사합니다. 우리는 쉽게 코드안의 모든 객체들에 접근할 수 있고, 어디서 오류가 발생하는 지 알려 주는 상태(혹은 기본 python식 디버깅)를 출력할 수 있습니다.  
Keras로 기본 신경망을 만든 사용자들은 PyTorch 사용자들보다 잘못된 방향으로 갈 가능성이 적습니다. 하지만 일단 잘못되기 시작하면, 많이 힘들고 종종 막힌 코드 라인을 찾기 힘듭니다. PyTorch는 모델의 목잡성과 관련없이 보다 직접적이고 컨볼루션이 아닌 디버깅 경험을 제공합니다. 또한, 의심스러운 경우 PyTorch 레포를 쉽게 조회해 코드를 읽어볼 수 있습니다.

#### 요약
- PyTorch : 더 좋은 디버깅 기능을 제공
- Keras : (잠재적으로) 단순 신경망 디버깅 빈도수 감소

### Keras vs PyTorch : 모델을 추출하고 다른 플랫폼과의 호환성

생산에서 학습된 모델을 내보내고 배포하는 옵션은 무엇인가요?

PyTorch는 python기반으로 휴대할 수 없는 pickle에 모델을 저장하지만, Keras는 JSON + H5 파일을 사용하는 안전한 접근 방식의 장점을 활용합니다.(일반적으로 Keras에 저장하는게 더 어렵습니다.) 또한 [R에도 Keras](https://keras.rstudio.com/)가 있습니다. 이 경우, R을 사용하여 데이터 분석팀과 협력해야 할 수도 있습니다. 

Tensorflow에서 실행되는 Keras는 [모바일용 Tensorflow](https://www.tensorflow.org/mobile/mobile_intro)(혹은 [Tensorflow Lite](https://www.tensorflow.org/mobile/tflite/index))를 통해 모바일 플랫폼에 구축할 수 있는 다양한 솔리드 옵션을 제공합니다. [Tensorflow.js](https://js.tensorflow.org/) 혹은 [Keras.js](https://github.com/transcranial/keras-js)를 사용하여 멋진 웹 애플리케이션을 배포할 수 있습니다. 예를 들어, Piotr와 그의 학생들이 만든, [시험 공포증 유발 요소를 탐지하는 딥러닝 브라우저 플러그인](https://github.com/cytadela8/trypophobia)을 보세요.

PyTorch 모델을 추출하는 건 python 코드때문에 더 부담되기에, 현재 많이 추천하는 접근방식은 [ONNX](https://pytorch.org/docs/master/onnx.html)를 사용하여 PyTorch 모델을 Caffe2로 변환하는 것입니다.

#### 요약
- Keras : (Tensorflow backend를 통해) 더 많은 개발 옵션을 제공하고, 모델을 쉽게 추출할 수 있음. 

### Keras vs PyTorch : 성능
> 미리 측정된 최적화는 프로그래밍에서 모든 악의 근원입니다. - Donald Knuth

대부분의 인스턴스에서, 속도 측정에서의 차이는 프레임워크 선택을 위한 주요 요점은 아닙니다.(특히, 학습할 때) GPU 시간은 데이터 과학자의 시간보다 더 인색합니다. 게다가, 학습하는 동안 발생하는 성능의 병목현상은 실패한 실험이나, 최적화하지 않은 신경망이나 데이터 로딩(loading)이 원인일 수 있습니다. 완벽을 위해, 여전히 우리는 해당 주제를 다뤄야할 compel을 느낍니다. 우리는 두 가지 비교사항을 제안합니다.  

- [Tensorflow, Keras 그리고 PyTorch를 비교](https://wrosinski.github.io/deep-learning-frameworks/) by Wojtek Rosinski
- [딥러닝 프레임워크들에 대한 비교 : 로제타 스톤식 접근](https://github.com/ilkarman/DeepLearningFrameworks/) by Microsoft
> 더 상세한 multi-GPU 프레임워크 비교를 보려면, [이 글](https://medium.com/@iliakarmanov/multi-gpu-rosetta-stone-d4fa96162986)을 참조하세요

PyTorch는 Tensorflow만큼 빠르며, RNN에선 잠재적으로 더 빠릅니다. Keras는 지속적으로 더 느립니다. 위의 첫 번째 비교를 작성한 저자가 지적했듯이, 고성능 프레임워크의 연산 효율성 향상(대부분 PyTorc와 Tensorflow)은 빠른 개발 환경과 Keras가 제공하는 실험의 용이성보다 더 중요할 것입니다.

![Tesla p100](https://github.com/KerasKorea/KEKOxTutorial/blob/issue_42/media/42_2.png)  

![Tesla K80](https://github.com/KerasKorea/KEKOxTutorial/blob/issue_42/media/42_3.png)  

#### 요약
- 학습 속도에 대한 걱정과 달리, PyTorch가 Keras를 능가

### Keras vs PyTorch : 결론
Keras와 PyTorch는 배우기위한 첫번째 딥러닝 프레임워크로 좋은 선택입니다.  
만약 당신이 수학자, 연구자, 혹은 당신의 모델이 실제로 어떻게 작동하는지 알고 싶다면, PyTorch를 선택하길 권장합니다. 고급 맞춤형 알고리즘(그리고 디버깅)이 필요한 경우(ex. [YOLOv3](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/) 혹은 [LSTM](https://medium.com/huggingface/understanding-emotions-from-keras-to-pytorch-3ccb61d5a983)을 사용한 객체 인식) 또는 신경망 이외의 배열 식을 최적화해야 할 경우(ex. [행렬 분해](http://blog.ethanrosenthal.com/2017/06/20/matrix-factorization-in-pytorch/) 혹은 [word2vec](https://adoni.github.io/2017/11/08/word2vec-pytorch/) 알고리즘)에 빛을 발합니다.  

 plug & play 프레임워크를 원한다면, Keras는 확실히 더 쉬울 겁니다. 즉, 수학적 구현의 세부 사항들에 많은 시간을 들이지 않고도 모델을 신속하게 제작, 학습 그리고 평가할 수 있습니다.  

수정 : 실제 사례에 대해 코드를 비교하려면, 이 [기사](https://deepsense.ai/keras-vs-pytorch-avp-transfer-learning)를 참조하세요  

딥러닝의 핵심 개념에 대한 지식은 유동성이 있습니다. 어떤 환경에서 기본사항을 숙지하고나면, 다른 곳에 적용하고 새로운 딥러닝 라이브러리로 전환할 때 이를 시행할 수 있다는 점입니다.  

Keras와 PyTorch에서 간단한 딥러닝 방법을 사용해 보는 것을 권장합니다. 당신이 가장 좋아하고 가장 덜 좋아하는 요소는 무엇입니까? 어떤 프레임워크 경험이 더 마음에 드시나요? 

Keras, Tensorflow 그리고 PyTorch의 딥러닝에 대해 자세히 알고 싶은가요? [맞춤형 교육 서비스](https://deepsense.ai/tailored-team-training-tracks/)를 확인하세요.  


### 참고문서
* [케라스 공식 홈페이지](https://keras.io/)
* [파이토치 공식 홈페이지](https://pytorch.org/)  


> 이 글은 2018 컨트리뷰톤에서 [`Contribute to Keras`](https://github.com/KerasKorea/KEKOxTutorial) 프로젝트로 진행했습니다.
> Translator: [mike2ox](https://github.com/mike2ox)(Moonhyeok Song)
> Translator email : <firefinger07@gmail.com>
