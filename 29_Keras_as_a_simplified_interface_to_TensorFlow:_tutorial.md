케라스(Keras) 튜토리얼 - 텐서플로우의 간소화된 인터페이스로서

----
만약 텐서플로우가 당신의 주요 프레임워크이고 당신의 삶을 간단하게 만들 간단한 고레벨(high-level) 모델 인터페이스를 찾고 있다면, 이 튜토리얼은 당신을 위한 것입니다.

케라스 레이어와 모델은 순수 텐서플로우 텐서들과 완전 호환되므로, Keras는 TensorFlow를 위한 훌륭한 모델 정의 부가 기능(add-on)이며 다른 텐서플로우 라이브러리와 함께 사용할 수도 있습니다. 
어떻게 이렇게 되는지 볼까요.

이 튜토리얼에서는 텐서플로우 벡엔드로 (Theano 대신) keras를 구성했다고 가정합니다. [여기 이를 어떻게 구성하는지에 대한 설명입니다.](https://keras.io/backend/#switching-from-one-backend-to-another)

우리는 다음 사항을 다룰 것입니다. 
[I. 텐서플로우 텐서에서 Keras 레이어 호출하기]()
[II. 텐서플로우에서 Keras 모델 사용]()
[III. 멀티 GPU 분산 트레이닝]()
[IV. TensorFlow-serving 사용해서 모델 내보내기]()

---------
## I. 텐서플로우 텐서에서 Keras 레이어 호출하기
간단한 MNIST 분류 예제로 시작해봅시다. 케라스 `Dense` 레이어(fully-connected layers 완전 연결 레이어) 스택을 사용한 텐서플로우 숫자 분류기를 빌드해보겠습니다.

텐서플로우 세션을 만들고, 케라스에 등록해서 시작합니다. 이것은 케라스가 우리가 내부적으로 만들어진 모든 변수를 초기화하여 등록한 세션을 사용한다는 의미입니다.

```python
import tensorflow as tf
sess = tf.Session()

from keras import backend as K
K.set_session(sess)
```

이제 MNIST를 시작할 차례입니다. 우리는 텐서플로우에서 하는대로 분류기 빌딩을 시작할 수 있습니다. 

```python
# 이 플레이스홀더(placeholder)는 우리의 플랫 벡터로서 입력 숫자를 포함합니다.
img = tf.placeholder(tf.float32, shape=(None, 784))
```

그러고나서 케라스 레이어들을 사용해 모델 결정 프로세스의 속도를 높일 수 있습니다. 

```python 
from keras.layers import Dense

# 케라스 레이어는 텐서플로의 텐서로 호출할 수 있습니다. 
x = Dense(128, activation='relu')(img)  # 128unit을 가진 완전 연결 레이어 와 ReLU 활성화
x = Dense(128, activation='relu')(x)
preds = Dense(10, activation='softmax')(x)  # 10unit의 output layer와  softmax 활성화
```

우리가 사용할 라벨과 loss 펑션을 위한 플래이스홀더를 정의합니다.

```python 
labels = tf.placeholder(tf.float32, shape=(None, 10))

from keras.objectives import categorical_crossentropy
loss = tf.reduce_mean(categorical_crossentropy(labels, preds))
```

텐서플로 옵티마이저로 모델을 훈련시킵시다

```python
from tensorflow.examples.tutorials.mnist import input_data
mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# 모든 변수 초기화
init_op = tf.global_variables_initializer()
sess.run(init_op)

# 트레이닝 루프 실행
with sess.as_default():
    for i in range(100):
        batch = mnist_data.train.next_batch(50)
        train_step.run(feed_dict={img: batch[0],
                                  labels: batch[1]})

```

이제 모델을 평가할 수 있습니다.

```python
from keras.metrics import categorical_accuracy as accuracy

acc_value = accuracy(labels, preds)
with sess.as_default():
    print acc_value.eval(feed_dict={img: mnist_data.test.images,
                                    labels: mnist_data.test.labels})

```

이 경우에, 케라스를 오직 일부 텐서 입력을 일부 텐서 출력으로 맵핑하는 연산을 생성하는 구문 단축키(syntactical shortcut)로만 사용합니다. 이 최적화는 케라스 옵티마이저가 아닌 기본 텐서플로우 옵티마이저를 통해 수행됩니다. 케라스 `model`을 전혀 사용하지 않았습니다!

참고사항 - 기본 텐서플로우 옵티마이저와 케라스 옵티마이저의 상대성능 : 텐서플로우 옵티마이저와 비교해볼떄 '케라스 방식'의 모델 최적화와는 약간의 속도 차이가 있습니다. 약간 반-직관적(counter-intuitively)으로는 케라스는 5~10% 정도 속도가 빨라보입니다. 하지만 모델 최적화에 케라스 옵티마이저를 쓰든지 기본 TF 옵티마이저를 쓰든지 결국엔 크게 차이가 없습니다.

### 훈련과 테스팅동안 다른 동작
일부 Keras 레이어 (예를 들면, `Dropout`, `BatchNormalization`)는 훈련 시간 및 테스트 시간에 다르게 동작합니다.  레이어가 "학습단계(learning phase)" (train/test)에서 `layer.uses_learning_phase`를 출력하여 그 값(boolean)이 'True'라면, 레이어가 트레이닝 모드와 테스트 모드에서 다르게 동작하고 있다는 것이고 'False'라면 같게 동작한다는 것입니다.

모델에 이러한 레이어가 포함된 경우, 모델에 dropout/etc 적용 여부를 알 수 있도록 `feed_dict`의 파트의 학습단계 값을 지정해야합니다.

Keras 백엔드를 통해 Keras 학습 단계(TensorFlow 텐서 스칼라값)에 접근할 수 있습니다. 

```python
from keras import backend as K
print K.learning_phase()
```
학습 단계를 사용하려면 `feed_dict`에 "1"(훈련 모드) 또는 "0"(테스트 모드) 값을 전달하세요.

```
# 훈련 모드
train_step.run(feed_dict={x: batch[0], labels: batch[1], K.learning_phase(): 1})
```
**역자 주: 해당 파라미터값을 1로 설정했기때문에 훈련모드입니다**

예를 들어, 이전 MNIST 예제에 Dropout layers를 추가하는 것은 아래와 같습니다.

```python
from keras.layers import Dropout
from keras import backend as K

img = tf.placeholder(tf.float32, shape=(None, 784))
labels = tf.placeholder(tf.float32, shape=(None, 10))

x = Dense(128, activation='relu')(img)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
preds = Dense(10, activation='softmax')(x)

loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
with sess.as_default():
    for i in range(100):
        batch = mnist_data.train.next_batch(50)
        train_step.run(feed_dict={img: batch[0],
                                  labels: batch[1],
                                  K.learning_phase(): 1})

acc_value = accuracy(labels, preds)
with sess.as_default():
    print acc_value.eval(feed_dict={img: mnist_data.test.images,
                                    labels: mnist_data.test.labels,
                                    K.learning_phase(): 0})
```

### 네임 스코프(name scopes)와 디바이스 스코프(device scopes)의 호환성

Keras 레이어와 모델은 TensorFlow 네임 스코프와 완벽하게 호환됩니다. 예를 들어 다음 코드를 보십시오.
```python
x = tf.placeholder(tf.float32, shape=(None, 20, 64))
with tf.name_scope('block1'):
    y = LSTM(32, name='mylstm')(x)
```

LSTM 레이어의 가중치는 `block1 / mylstm_W_i, block1 / mylstm_U_i` 등으로 지정됩니다.

디바이스 스코프도 유사하게 동작합니다.

```python
with tf.device('/gpu:0'):
    x = tf.placeholder(tf.float32, shape=(None, 20, 64))
    y = LSTM(32)(x)  # LSTM 레이어에서 모든 연산과 변수는 GPU:0 에 살아있습니다. 
```

### 그래프 스코프에서의 호환성
TensorFlow 그래프 스코프 내에서 정의하는 Keras 레이어나 모델은, 특정 그래프의 일부로서 생성된 모든 변수와 연산을 갖습니다. 예를 들면, 다음과 같이 동작합니다.

```python
from keras.layers import LSTM
import tensorflow as tf

my_graph = tf.Graph()
with my_graph.as_default():
    x = tf.placeholder(tf.float32, shape=(None, 20, 64))
    y = LSTM(32)(x)  # LSTM 레이어에서 모든 연산과 변수들은 그래프의 일부로서 생성됩니다.
```

### 변수 스코프(variable scopes)에서의 호환성 
변수 공유는 **텐서플로우 변수 범위가 아닌** 같은 케라스 레이어(또는 모델) 인스턴스에서 호출될때 가능합니다. 텐서플로우 변수 범위는 케라스 레이어나 모델에 영향을 미치지 않습니다. *케라스의 가중치 공유에 대한 더 자세한 정보는 API의 [가중치 공유](https://keras.io/getting-started/functional-api-guide/#shared-layers) 에서 확인하세요.*

가중치 공유가 케라스에서 어떻게 동작하는지 짧게 요약:  같은 레이어 인스턴스나 모델 인스턴스를 재사용함으로써 가중치를 사용할 수 있습니다. 아래 간단한 예제가 있습니다.

```python
# 케라스 레이어 인스턴스화
lstm = LSTM(32)

# 두 TF placeholder 인스턴스화
x = tf.placeholder(tf.float32, shape=(None, 20, 64))
y = tf.placeholder(tf.float32, shape=(None, 20, 64))

# '같은' LSTM 가중치 두 tensor 인코딩
x_encoded = lstm(x)
y_encoded = lstm(y)
```

### 학습 가능한 가중치 및 상태 업데이트 수집
일부 Keras 레이어 (Stateful RNN 및 BatchNormalization 레이어)에는 각 트레이닝 단계에서 실행하야하는 내부 업데이트가 있습니다. 이것들은 텐서 튜플, `layer.updates`의 리스트로 저장됩니다. 각 훈련 단계에서 실행되도록 할당 작업을 생성해야합니다. 아래가 그 예입니다.

```python
from keras.layers import BatchNormalization

layer = BatchNormalization()(x)

update_ops = []
for old_value, new_value in layer.updates:
    update_ops.append(tf.assign(old_value, new_value))
```

Keras 모델 (Model 인스턴스 또는 Sequential 인스턴스)을 사용하는 경우 `model.udpates`는 동일한 방식으로 동작하며 모델의 모든 기본 레이어에 대한 업데이트를 수집합니다.

또한 `layer.trainable_weights` (또는 `model.trainable_weights`), TensorFlow 변수 인스턴스의 목록을 통해 레이어의 훈련 가중치를 명시적으로 수집할 수 있습니다.

```python
from keras.layers import Dense

layer = Dense(32)(x)  # 인스턴스화와 레어어 호출 
print layer.trainable_weights  # TensorFlow 변수 리스트
```

이를 알면 TensorFlow 옵티마이저를 기반으로 자신만의 트레이닝 루틴을 구현할 수 있습니다.

---------

## II : TensorFlow에서 Keras 모델 사용

### TensorFlow 워크플로우에서 사용할 Keras Sequential 모델 변환
당신의 TensorFlow 프로젝트에서 재사용하려는 Keras Sequential 모델을 발견했다면(예를 들어, [사전 훈련된 가중치가 있는 VGG16 이미지 분류기](https://gist.github.com/auth/github/callback?return_to=https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3&browser_session_id=fc1be80a01c8f9891f2b3014603d4540d13c7cca&code=0454cedef462695746de&state=73cf1fe9c2b34855e963de4e898c49a7dd64c2cc205fa16fc5ff8f86c2cbe335)라면), 어떻게 진행해야할까요?

우선, Theano로 훈련된 컨볼루션을 포함한 사전훈련된 가중치(`Convolution2D` 또는 `Convolution1D` 레이어)가 포함되어 있다면 가중치를 로드할 때 컨볼루션 커널을 뒤집어야합니다. 이것은 Theano와 TensorFlow가 서로 다른 방식으로 컨볼루션을 구현하기 때문입니다 (실제로 TensorFlow는 Caffe와 같은 방식으로 상관 관계를 구현합니다). [이 경우, 당신이 해야할일에 대한 간단한 가이드가 있습니다](https://github.com/keras-team/keras/wiki/Converting-convolution-kernels-from-Theano-to-TensorFlow-and-vice-versa).

다음 Keras 모델에서 특정 TensorFlow 텐서 인 `my_input_tensor`를 입력값으로 사용하도록 수정한다고 가정해봅시다. 이 입력 텐서는 예를 들어 데이터 피더 연산이거나 이전의 TensorFlow 모델의 출력일 수 있습니다.

```
# 원본 케라스 모델 
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=784))
model.add(Dense(10, activation='softmax'))
```

`keras.layers.InputLayer`를 사용하여 사용자 정의 TensorFlow 플레이스홀더 위에 Sequential 모델을 작성한 다음 나머지 모델을 빌드하면됩니다. 

from keras.layers import InputLayer

```
#  수정된 케라스 모델
model = Sequential()
model.add(InputLayer(input_tensor=custom_input_tensor,
                     input_shape=(None, 784)))

# 위와 같이 남은 모델 빌드
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
```

이 단계에서 `model.load_weights (weights_file)`를 호출하여 사전 훈련된 가중치를 로드할 수 있습니다.

그런 다음 Sequential 모델의 출력 텐서를 수집할 것입니다.
```python
output_tensor = model.output
```

이제 `output_tensor` 과 기타등등 위에 새로운 TensorFlow 작업을 추가 할 수 있습니다.

### TensorFlow 텐서에서 Keras 모델 호출하기

Keras 모델은 레이어와 동일한 역할을 하므로 TensorFlow 텐서에서 호출할 수 있습니다.
```python
from keras.models import Sequential

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=784))
model.add(Dense(10, activation='softmax'))

# 이렇게도 동작합니다!
x = tf.placeholder(tf.float32, shape=(None, 784))
y = model(x)
```

참고 : Keras 모델을 호출하면 아키텍처와 가중치를 모두 재사용합니다. 텐서의 모델을 호출할떄, 입력 텐서 위에 새로운 TF연산을 생성하고 이 연산은 이미 모델에 있는 TF 변수 인스턴스를 재사용합니다.

---------
## III. 멀티 GPU 및 분산 트레이닝
### Keras 모델의 일부를 다른 GPU에 할당
TensorFlow 디바이스 스코프는 Keras 레이어 및 모델과 완벽하게 호환되므로, 이를 사용하여 그래프의 특정 부분을 다른 GPU에 할당할 수 있습니다. 다음은 간단한 예입니다.

```python
with tf.device('/gpu:0'):
    x = tf.placeholder(tf.float32, shape=(None, 20, 64))
    y = LSTM(32)(x)  # 이 LSTM 레이어의 모든 연산은 GPU:0에 저장될 것임

with tf.device('/gpu:1'):
    x = tf.placeholder(tf.float32, shape=(None, 20, 64))
    y = LSTM(32)(x)  # 이 LSTM 레이어의 모든 연산은 GPU:1에 저장될 것임
```
LSTM 레이어에서 생성된 변수는 GPU에 저장되지 않습니다. 모든 TensorFlow 변수는 생성된 디바이스 스코프와 관계없이 항상 CPU에 올라가 있습니다. TensorFlow는 백그라운드에서 장치간 변수 전송(device-to-device variable transfer)을 처리합니다. 

동일한 모델의 여러 복제본을 서로 다른 GPU에서 교육하고 다른 복제본간에 동일한 가중치를 공유하려는 경우,
먼저 하나의 디바이스 스코프에서 해당 모델(또는 레이어)를 인스턴스화 한 다음, 다른 GPU 디바이스 스코프에 있는 같은 모델의 인스턴스를 여러 번 호출하면 됩니다. 다음과 같습니다.

```python
with tf.device('/cpu:0'):
    x = tf.placeholder(tf.float32, shape=(None, 784))

    # CPU:0에 있는 공유된 모델
    # 트레이닝 도중에는 실제로 실행되지 않음
    # 연산 템플릿이나  연산 템플릿 및 공유된 변수의 저장소로서 동작함
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=784))
    model.add(Dense(10, activation='softmax'))

# 복제본 0
with tf.device('/gpu:0'):
    output_0 = model(x)  # 이 복제본의 모든 연산은 GPU:0에 있음

# 복제본 1
with tf.device('/gpu:1'):
    output_1 = model(x)  #  이 복제본의 모든 연산은 GPU:1에 있음

# CPU에서 결과 병합(merge)
with tf.device('/cpu:0'):
    preds = 0.5 * (output_0 + output_1)

# `preds` 텐서만 실행하여 GPU의 두 복제본만 실행되도록 함
# (CPU에 병합 연산 추가)
output_value = sess.run([preds], feed_dict={x: data})
```

### 분산 트레이닝
클러스터에 링크된 TF 세션을 Keras에 등록하여 TensorFlow 분산 트레이닝을 쉽게 활용할 수 있습니다.

```python
server = tf.train.Server.create_local_server()
sess = tf.Session(server.target)

from keras import backend as K
K.set_session(sess)
```

분산 설정에서 TensorFlow를 사용하는 방법에 대한 자세한 내용은 [이 튜토리얼](https://www.tensorflow.org/guide/)를 참조하십시오.

------
## IV.TensorFlow-serving을 사용하여 모델 내보내기

TensorFlow Serving은 Google에서 개발한 프로덕션 환경에서 TensorFlow 모델을 제공하기위한 라이브러리입니다.

모든 Keras 모델은 TensorFlow 워크플로우로서 트레이닝 여부에 관계없이 'TensorFlow-serving'(TF-serving의 제한사항때문에 하나의 입출력값만 있을때)으로 내보낼 수 있습니다. 실제로 Theano로 Keras 모델을 트레이닝한 다음, TensorFlow Keras 백엔드로 전환하고 모델을 내보낼 수도 있습니다. 

어떻게 동작하는지에 대한 설명입니다.

그래프가 Keras 학습 단계 (학습 시간과 테스트 시간에 다른 동작)를 사용하는 경우, 모델을 내보내기(export) 전에 가장 먼저해야 할 일은 그래프에 학습 단계 값을 설정하는 것입니다. (예 : test 모드인 경우 0) 이것은 1) Keras 백엔드에 지속적인 학습 단계를 등록하고 2) 나중에 모델을 다시 빌드하여 수행됩니다.

다음은 이 간단한 두 단계에 대한 실행입니다.

```python
from keras import backend as K

K.set_learning_phase(0)  # 모든 새로운 연산은 이제 테스트 모드임

# 빠른 리빌딩을 위해, 모델을 직렬화(serialize)하고 가중치를 얻기
config = previous_model.get_config()
weights = previous_model.get_weights()

# 학습단계에서 0으로 하드 코딩된 모델 리빌딩
from keras.models import model_from_config
new_model = model_from_config(config)
new_model.set_weights(weights)
```

이제 [공식 Tutorial에 있는 지침](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/serving_basic.md)에 따라 TensorFlow-serving를 사용하여 모델을 내보낼 수 있습니다.

```python
from tensorflow_serving.session_bundle import exporter

export_path = ... # where to save the exported graph
export_version = ... # version number (integer)

saver = tf.train.Saver(sharded=True)
model_exporter = exporter.Exporter(saver)
signature = exporter.classification_signature(input_tensor=model.input,
                                              scores_tensor=model.output)
model_exporter.init(sess.graph.as_graph_def(),
                    default_graph_signature=signature)
model_exporter.export(export_path, tf.constant(export_version), sess)
```

------------

이 가이드에서 다루는 새로운 주제를 보고 싶습니까? [Twitter](https://twitter.com/fchollet)를 참고하세요. 


