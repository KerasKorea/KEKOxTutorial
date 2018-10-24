## Keras vs PyTorch 어떤 플랫폼을 선택해야 할까?(keras or pytorch as your first deep learning framework)
[원문](https://deepsense.ai/keras-or-pytorch/)
> 문서 간략 소개

* Keras
* PyTorch
* framework

### 소개
![Keras_vs_PyTorch](https://github.com/KerasKorea/KEKOxTutorial/blob/issue_42/media/42_0.png)

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

![Tesla p100](https://github.com/KerasKorea/KEKOxTutorial/blob/issue_42/media/42_0.png)  

![Tesla K80](https://github.com/KerasKorea/KEKOxTutorial/blob/issue_42/media/42_0.png)  

#### 요약

### Keras vs PyTorch : 결론

### 참고문서
* [참고 사이트 1]()
* [참고 사이트 2]()


> 이 글은 2018 컨트리뷰톤에서 [`Contribute to Keras`](https://github.com/KerasKorea/KEKOxTutorial) 프로젝트로 진행했습니다.
> Translator: [mike2ox](https://github.com/mike2ox)(Moonhyeok Song)
> Translator email : <firefinger07@gmail.com>
