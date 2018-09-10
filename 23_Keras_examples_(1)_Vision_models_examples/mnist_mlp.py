'''MNIST 데이터 셋으로 간단한 심층 신경망 학습하기
원문: https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py
> 이 스크립트는 케라스와 MNIST 데이터 셋을 이용해 손글씨 숫자 인식 분류기를 학습하는 예제입니다. 

* keras
* mnist
* convolutional neural network

20 에폭 학습 후 테스트 셋의 정확도가 98.40%이 됩니다
(여기엔 아직 *많은* 파라미터 조정의 여지가 있습니다). 
K520 GPU 사용 시 에폭 당 2초가 소요됩니다. 
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

batch_size = 128
num_classes = 10
epochs = 20

# 데이터를 불러오고, 학습 셋과 테스트 셋으로 분리합니다
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# 클래스 벡터들을 이진수 클래스 행렬로 바꿉니다
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])