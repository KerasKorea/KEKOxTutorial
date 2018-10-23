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

이제 모델을 evaluate 합시다

```python
from keras.metrics import categorical_accuracy as accuracy

acc_value = accuracy(labels, preds)
with sess.as_default():
    print acc_value.eval(feed_dict={img: mnist_data.test.images,
                                    labels: mnist_data.test.labels})

```

이 경우에, 케라스를 오직 일부 텐서 입력을 일부 텐서 출력으로 맵핑하는 연산을 생성하는 구문 단축키(syntactical shortcut)로만 사용합니다. 이 최적화는 기본 텐서플로우 옵티마이저를 통해 수행됩니다.








