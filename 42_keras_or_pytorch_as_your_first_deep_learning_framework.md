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

- [Keras](https://keras.io/)는 Tensorflow, CNTK, Theano, MXNet(혹은 Tensorflow안의 tf.contrib)의 상단에서 작동할 수 있는 고급 API입니다. 2015년 3월에 첫 배포를 한 이래로, 쉬운 사용법과 간단한 문법, 빠른 설계 덕분에 인기를 끌고 있습니다. 구글에서 지원하고 있습니다.
- [PyTorch](https://pytorch.org/)

### 좋아, 근데 다른 프레임워크는 어때?

### Keras vs PyTorch : 쉬운 사용법과 유연성

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
#### 요약

### Keras vs PyTorch : 대중성과 학습자료 접근성


![Percentof ML papers that mention...](https://github.com/KerasKorea/KEKOxTutorial/blob/issue_42/media/42_1.png)


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
