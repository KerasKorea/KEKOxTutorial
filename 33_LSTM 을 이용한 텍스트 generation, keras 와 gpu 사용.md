# An applied introduction to LSTMs for text generation — using Keras and GPU-enabled Kaggle Kernels

원문: https://medium.freecodecamp.org/applied-introduction-to-lstms-for-text-generation-380158b29fb3

Kaggle은 최근 데이터 과학자들에게 Kernel (Kaggle의 클라우드 기반 notebook 플랫폼)에 GPU를 적용할 수 있도록 하였습니다. 저는 이것이 더 많은 intensive model을 구축하고 학습하는 방법을 배울 수 있는 절호의 기회라고 생각했습니다. 

 [Kaggle Learn](https://www.kaggle.com/learn/deep-learning), [Keras documentation](https://keras.io/) 그리고 [freeCodeCamp](https://www.freecodecamp.org/)의 좋은 자연어 데이터와 함께라면, [random forests](https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic)를  recurrent neural network로 발전시킬 수 있습니다. 


![img](https://cdn-images-1.medium.com/max/2000/1*msHP2gE21HCHibUquIAB1A.png)

​				<freeCodeCamp’s dataset on Kaggle Datasets>

이 블로그 포스트에서, Kaggle 데이터셋에 게시 된 [freeCodeCamp의 Gitter 채팅 로그 데이터셋](https://www.kaggle.com/freecodecamp/all-posts-public-main-chatroom)에서 새로운 텍스트 출력을 생성하는 LSTM network를 학습하는 방법에 대해 설명하겠습니다. 

 [Python notebook kernel](https://www.kaggle.com/mrisdal/intro-to-lstms-w-keras-gpu-for-text-generation/notebook)에서 제 코드를 볼 수 있습니다. 

이제 6시간의 실행시간 동안 Kernels-Kaggle의 클라우드 기반 호스팅 노트북 플랫폼에서 GPU를 사용할 수 있으므로, Kaggle에서 이전보다 훨씬 많은 intensive 모델을 학습시킬 수 있습니다.

```python
import tensorflow as tf
print(tf.test.gpu_device_name())
# 참조 https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
```

![img](https://cdn-images-1.medium.com/max/2000/1*DAx_AO-YQPr1A0N9YZM7Cw.png)



## Part 1: 데이터 준비

파트 1에서는, 먼저 데이터를 읽어보고 작업 내용을 충분히 이해해보겠습니다. 비상호적인 튜토리얼(예를 들어, GitHub의 코드 공유)를 따라하는 것에서 큰 어려움 중 하나는 작업하고 싶은 데이터가 샘플 코드와 어떻게 다른지 알기 어렵다는 것입니다. 직접 다운받아 비교해보아야합니다. 

Kernel을 사용해서 이 튜토리얼을 따라하는 것이 좋은 2가지 점은 다음과 같습니다. 1) 모든 중요 단계에서 데이터를 훑어볼 수 있습니다. 2) 언제든지 이 notebook을 fork할 수 있으며, 제 환경, 데이터, Docker 이미지 및 필요한 모든 것을 다운로드나 설치 없이 얻을 수 있습니다. 특히, 딥러닝을 위해 GPU를 사용하는 CUDA 환경을 설치한 경험이 있다면, 이러한 환경이 이미 준비되어 있는 것이 얼마나 좋은지 경험하셨을 것입니다. 

### Read in the data

```python
import pandas as pd
import numpy as np

# 필요한 두 컬럼만 읽습니다
chat = pd.read_csv('../input/freecodecamp_casual_chatroom.csv', usecols = ['fromUser.id', 'text'])

# CamperBot의 user id 삭제
chat = chat[chat['fromUser.id'] != '55b977f00fc9f982beab7883'] 

chat.head()
```

![img](https://cdn-images-1.medium.com/max/1600/1*-69tMaz_X9Gz9G3ZPswv6Q.png)

잘 되었습니다!



### Explore the data

밑의 그림에서 freeCodeCamp의 Gitter에서 사용자 id별로 가장 활동적인 채팅 참가자 상위 10명의 게시물 수를 볼 수 있습니다. 

```python
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

f, g = plt.subplots(figsize=(12, 9))
chat['fromUser.id'].value_counts().head(10).plot.bar(color="green")
g.set_xticklabels(g.get_xticklabels(), rotation=25)
plt.title("Most active users in freeCodeCamp's Gitter channel")
plt.show(g)
```

![img](https://cdn-images-1.medium.com/max/1600/1*JxNsHuuN9ESnlzDntuBqBQ.png)

사용자id  `55a7c9e08a7b72f55c3f991e`는 채널에서 140,000개가 넘는 메시지를 남긴 가장 활동적인 사용자입니다. 우리는 이들의 메시지를 사용하여 사용자  `55a7c9e08a7b72f55c3f991e` 의 메시지같은 문장을 생성해내는 LSTM을 학습시키겠습니다. 그러나 먼저,  `55a7c9e08a7b72f55c3f991e`의 메시지를 보면서 무엇에 대해 대화하고 있는지 알아보겠습니다. 

```python
chat[chat['fromUser.id'] == "55a7c9e08a7b72f55c3f991e"].text.head(20)
```

![img](https://cdn-images-1.medium.com/max/1600/1*JxNsHuuN9ESnlzDntuBqBQ.png)

![img](https://cdn-images-1.medium.com/max/2000/1*xM5SOB2YS0oJLzBluhILmg.png)

"documentation", "pair coding", "BASH", "Bootstrap", "CSS" 와 같은 단어와 구문을 볼 수 있습니다. 그리고 "With all of the various frameworks…"으로 시작되는 문장은 JavaScript를 JavaScript를 가리킨다고 가정할수 있습니다. 맞습니다, freeCodeCamp에서 있을만한 주제들입니다. 따라서 만약 우리의 결과가 성공적이라면 이러한 문장들이 생성될 것으로 기대할 수 있습니다. 

### LSTM의 입력값에 대한 연속적인 데이터 준비 

현재, 우리는 사용자id와 메시지 텍스트에 해당하는 컬럼을 가진 데이터 프레임이 있습니다. 여기서 각 행은 전송된 하나의 메시지입니다. 이는 LSTM network의 입력 레이어에 필요한 3차원 형태와는 거리가 멉니다: `model.add(LSTM(batch_size, input_shape=(time_steps, features)))` where `batch_size` is the number of sequences in each sample (can be one or more), `time_steps` is the size of observations in each sample, and `features` is the number of possible observable features (i.e., characters in our case).

어떻게 데이터 프레임을 올바른 형태인 연속 데이터로 만들 수 있을까요? 3단계로 해볼 수 있습니다. 

1. 말뭉치로부터 데이터를 뽑아냅니다. 
2. #1의 말뭉치를 균일한 길이를 가지고 다음 문자와 일부 겹치는 시퀀스의 행렬로 만듭니다. 
3. #2로의 시퀀스를 sparse boolean tensor로 표현합니다. 



### 말뭉치로부터 데이터 뽑기 

다음의 두 셀에서, `55a7c9e08a7b72f55c3f991e` (`'fromUser.id' == '55a7c9e08a7b72f55c3f991e'`)의 메시지만 가져와 데이터를 뽑아내고, 문자열 벡터를 단일 문자열로 만들겠습니다. 우리 모델이 생성하는 텍스트가 올바른 대문자화를 하는지 신경쓰지 않을 것이기 때문에, `tolower()`함수를 사용하여 모두 소문자로 바꾸겠습니다. 이는 학습해야할 차원을 하나 축소시켜줍니다. 

또한, 데이터의 처음 20%를 샘플로 사용하겠습니다. 중간 길이 정도의 텍스트를 생성하는데 그 이상은 필요없기 때문입니다. 이 kernel을 fork하여 원한다면 더 많은(혹은 적은) 데이터로 실험해볼 수 있습니다. 

```python
user = chat[chat['fromUser.id'] == '55a7c9e08a7b72f55c3f991e'].text

n_messages = len(user)
n_chars = len(' '.join(map(str, user)))

print("55a7c9e08a7b72f55c3f991e accounts for %d messages" % n_messages)
print("Their messages add up to %d characters" % n_chars)
```

![img](https://cdn-images-1.medium.com/max/2000/1*B4JTLCSrjbmUCBg6e32qew.png)

```python
sample_size = int(len(user) * 0.2)

user = user[:sample_size]
user = ' '.join(map(str, user)).lower()

user[:100] # Show first 100 characters
```



### Format the corpus into arrays of semi-overlapping sequences of uniform length and next characters

나머지 코드는 LSTM을 학습하기 위한 올바른 형식의 데이터를 준비하는 것으로, François Chollet (Keras 및 Kaggler 작성자)이 작성한  [예제 스크립트](https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py)를 참고하여 작성하였습니다. 문자 수준의 모델을 학습시키므로, 셀에서 문자들("a", "b", "c", …)을 숫자 인덱스에 연결합니다.  [“Fork Notebook”](https://www.kaggle.com/mrisdal/intro-to-lstms-w-keras-gpu-for-text-generation/) 를 클릭하여 이 코드를 다시 실행하면, 사용된 모든 문자를 출력할 수 있습니다. 

```python
chars = sorted(list(set(user)))
print('Count of unique characters (i.e., features):', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
```

![img](https://cdn-images-1.medium.com/max/2000/1*3vGVqm30QNs8SqNNTKae6g.png)

This next cell step gives us an array, `sentences`, made up of `maxlen` (40) character sequences chunked in steps of 3 characters from our corpus `user`, and `next_chars`, an array of single characters from `user` at `i + maxlen` for each `i`. I've printed out the first 10 strings in the array so you can see we're chunking the corpus into partially overlapping, equal length "sentences."

```python
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(user) - maxlen, step):
    sentences.append(user[i: i + maxlen])
    next_chars.append(user[i + maxlen])
print('Number of sequences:', len(sentences), "\n")

print(sentences[:10], "\n")
print(next_chars[:10])
```

![img](https://cdn-images-1.medium.com/max/2000/1*T_gKrW3LTAJpTne4gT4t4g.png)



 `'hi folks. just doing the new signee stuf'`의 다음 문자가 단어 "stuff"의 마지막 글자인 `f`임을 알 수 있습니다.  그리고 시퀀스  `'folks. just doing the new signee stuff. '` 의 다음 문자는 "hello" 단어의 `h`입니다. 이런 방식으로, `next_chars`가 `sentences` 내 시퀀스에 대한 "데이터 라벨" 또는 ground truth가 되도록 해야하고, 이 labled data에 대해 학습된 모델은 주어진 시퀀스 입력에 대한 *새로운 다음 문자* 를 예측으로 생성해낼 수 있습니다. 

#### Represent the sequence data as sparse boolean tensors

다음 셀은 [kernael에서 상호적으로 따라갈 때](https://www.kaggle.com/mrisdal/intro-to-lstms-w-keras-gpu-for-text-generation/), 몇 초 정도 걸립니다. 우리가 훈련시키는 모델에 대한 입력으로 사용하기 위해 `sentence`와 `next_chars`로 부터 문자수준의 특징(feature)들을 인코딩하는 sparse boolean tenser `x`와 `y`를 생성합니다. 마지막 shape는 다음과 같습니다:`input_shape=(maxlen, len(chars)) ` 로 여기서 `maxlen`은 40이며 `len(chars)`는 특징(feature)의 수(즉, 말뭉치의 고유한 문자 수)입니다. 

```
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
```

### Part 2: Modeling

파트 2에서는, 실제 모델을 훈련시키고 텍스트를 생성하겠습니다. 우리는 이미 데이터를 탐색하고 reshape하여 LSTM 모델에 올바른 입력값으로 사용할 수 있도록 만들었습니다. 이 파트에는 2가지 섹션이 있습니다.

1. LSTM network 모델 정의
2. 모델 훈련 및 예측 생성 

#### LSTM network 모델 정의

라이브러리를 읽는 것 부터 시작하겠습니다. Tensorflow backend에 대해 대중적이고 사용하기 쉬운 인터페이스인 Keras를 사용하겠습니다. [Keras를 딥러닝 프레임 워크로 사용하는 이유](https://keras.io/why-use-keras/)를 읽어보세요. 아래에서 우리가 사용할 모델, 레이어, 최적화도구 및 콜백을 볼 수 있습니다. 

```
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback, ModelCheckpoint
import random
import sys
import io
```

![img](https://cdn-images-1.medium.com/max/2000/1*AMS1StiZxxRR64Zg1f3v9g.png)

아래 셀에서 모델을 정의합니다. 우리는 sequential model로 시작하여 입력 레이어로 LSTM을 추가합니다.  입력에 대해 정의한 shape은 우리가 필요로하는 이 시점의 데이터와 동일합니다. 저는 `batch_size` 로 128을 선택하였습니다. 이는 샘플 수, 또는 sequence이며, 우리 모델은 . 원하는 경우 다른 숫자로 실험해볼 수 있습니다. 또한, dense output layer를 추가할 것입니다. 마지막으로, sequence 다음 문자를 예측하기 위해 다범주 분류가 필수적이기 때문에, 활성화 함수로 `softmax` 를 활성화 레이어에 추가합니다. 

```
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
```

이제 모델을 컴파일할 수 있습니다. 모델의 가중치를 최적화하기 위해 학습속도(learning rate)가 0.1인 `RMSprop`을 사용하고(여기서 다른 학습 속도로도 실험할 수 있습니다),`categorical_crossentropy`는 손실 함수로 사용합니다. 교차 엔트로피(cross entropy)는 Kaggle에서 이진 분류 competition에서 사용하는 로그 손실과 동일합니다(우리의 경우에는 두 가지 이상의 가능한 결과가 있는 경우는 제외). 

```
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
```

이제 모델이 준비되었습니다. 데이터를 넣어보기 전에 아래의 셀은 [이 스크립트에서 수정된 코드](https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py)로 몇가지 헬퍼(helper) 함수를 정의합니다. 첫 번째 함수인 `sample()`은 일정한 `temperature` 의 확률 배열에서 인덱스를 샘플링합니다. 잠깐 질문시간입니다, `temperature`은 정확히 무엇일까요? 

> **Temperature** *은 활성화 함수인* * `*softmax*`*를 적용하기 전에 dense layer의 출력에 적용되는 scaling factor 입니다. 간단히 말해서, 시퀀스의 다음 문자에 대한 모델의 추측이 얼마나 보수적인지 또는 "창의적인지" 정의합니다.* `*temperature*`*의 낮은 값(예*. `*0.2*`*)는 "안전한" 추측값을 생성하지만* `*1.0*`*을 넘는*  `*temperature*`*의 값은 "위험한" 추측을 생성하기 시작합니다.* *"st"로 시작하는 영어단어와 "sg"로 시작하는 영어단어를 보았을 때 추측의 양을 생각해봅시다. temperature가 낮으면 많은 "the"와 "and"를 얻을 것입니다; 하지만 temperature가 높다면 결과는 예측할 수 없게 됩니다.  

두 번째 함수는 LSTM에 의해 생성된 예상 텍스트를 출력하는 콜백 함수입니다. 

Anyway, so the second is defining a callback function to print out predicted text generated by our trained LSTM at the first and then every subsequent fifth epoch with five different settings of `temperature` each time (see the line `for diversity in [0.2, 0.5, 1.0, 1.2]:` for the values of `temperature`; feel free to tweak these, too!). This way we can fiddle with the `temperature` knob to see what gets us the best generated text ranging from conservative to creative. Note that we're using our model to predict based on a random sequence, or "seed", from our original subsetted data, `user`: `start_index = random.randint(0, len(user) - maxlen - 1)`.

Finally, we name our callback function `generate_text` which we'll add to the list of callbacks when we fit our model in the cell after this one.

```
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def on_epoch_end(epoch, logs):
    # Function invoked for specified epochs. Prints generated text.
    # Using epoch+1 to be consistent with the training epochs printed by Keras
    if epoch+1 == 1 or epoch+1 == 15:
        print()
        print('----- Generating text after Epoch: %d' % epoch)

        start_index = random.randint(0, len(user) - maxlen - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print('----- diversity:', diversity)

            generated = ''
            sentence = user[start_index: start_index + maxlen]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for i in range(400):
                x_pred = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.

                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()
    else:
        print()
        print('----- Not generating text after Epoch: %d' % epoch)

generate_text = LambdaCallback(on_epoch_end=on_epoch_end)
```

#### 모델 트레이닝 및 예측값 생성

드디어 해냈습니다! 데이터가 준비되었어요(`x`는 시퀀스, `y`는 다음 문자).  

Finally we’ve made it! Our data is ready (`x` for sequences, `y` for next characters), we've chosen a `batch_size` of `128`, and we've defined a callback function which will print generated text using `model.predict()` at the end of the first epoch followed by every fifth epoch with five different `temperature` setting each time. We have another callback, `ModelCheckpoint`, which will save the best model at each epoch if it's improved based on our loss value (find the saved weights file `weights.hdf5` in the "Output" tab of the kernel).

Let’s fit our model with these specifications and `epochs = 15` for the number of epochs to train. And of course, let's not forget to put our GPU to use! This will make training/prediction much faster than if we used a CPU. In any case, you will still want to grab some lunch or go for a walk while you wait for the model to train and generate predictions if you're running this code interactively.

P.S. If you’re running this interactively in your own notebook on Kaggle, you can click the blue square “Stop” button next to the console at the bottom of your screen to interrupt the model training.

```
# define the checkpoint
filepath = "weights.hdf5"
checkpoint = ModelCheckpoint(filepath, 
                             monitor='loss', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='min')

# fit model using our gpu
with tf.device('/gpu:0'):
    model.fit(x, y,
              batch_size=128,
              epochs=15,
              verbose=2,
              callbacks=[generate_text, checkpoint])
```



![img](https://cdn-images-1.medium.com/max/2000/1*kxL8tbTxe3wFsKW7FU_Tiw.png)

<Example output after the first epoch>

### 결론

다했습니다! [Kaggle Kernels](https://www.kaggle.com/mrisdal/intro-to-lstms-w-keras-gpu-for-text-generation/)에서 이 노트북을 실행시키면, 생성된 텍스트를 문자별 출력하는 극적인 모델을 얻어낼 수 있습니다. 

텍스트 행을 갖는 데이터 프레임으로부터 LSTM 모델을 사용하여 GPU의 힘으로 새로운 문장을 생성하는 방법을 배운 과정이 즐거웠기를 바랍니다. 우리의 모델이 처음 epoch에서 마지막 epoch까지 어떻게 개선되었는지 볼 수 있습니다. 첫 번째 epoch에서 생성된 텍스트는 실제 영어와 비슷하지 않습니다. 전반적으로, 낮은 수준의 다양성은 많은 반복을 통하여 텍스트를 생성하는 반면, 높은 수준의 다양성은 알아듣기 힘든 말을 생성해내죠. 

더 나은 텍스트를 생성하기 위해 모델 또는 hyperparameter를 조정할 수 있을까요? 다음을 클릭하여 직접해보세요!  [forking this notebook kernel](https://www.kaggle.com/mrisdal/intro-to-lstms-w-keras-gpu-for-text-generation/) (상단의 "Fork Notebook" 클릭)



#### 다음 단계로의 영감

배운 것을 활용하여 확장하는 방법에 대한 아이디어입니다. 

1. 훈련 데이터 양, 에폭(epoch), 배치(batch) 크기, `temperature`등 다른(hyper)-매개변수를 사용하여 실험해보세요.
2. 동일한 코드를 다른 데이터로 시도해보세요; 이 노트북을 fork하고 "Data" 탭으로 이동합니다. freeCodeCamp 데이터 소스를 제거하고 다른 데이터 셋([좋은 예제들]((https://www.kaggle.com/datasets?sortBy=hottest&group=public&page=1&pageSize=20&size=all&filetype=all&license=all&tagids=11208)))을 넣어봅시다. 
3. dropout 레이어 추가와 같은 복잡한 network 아키텍처를 사용해보세요.
4. Kernel에서 비디오 및 실습 노트북 튜토리얼이 있는 [Kaggle Learn](https://www.kaggle.com/learn/deep-learning) 에서 딥러닝에 대해 더 공부해보세요.
5. "출력"에서  `weights.hdf5`를 사용하여 이 튜토리얼의 사용자가 다른 사람의 문장을 완성하면 새로운 Kernel의 다른 데이터를 기반으로 예측할 수 있습니다. 
6. 최소한의 예제에서 CPU 대 GPU의 속도 향상 효과를 비교해보세요. 

