## 케라스를 이용한 멀티 GPU 학습(How-To: Multi-GPU training with Keras, Python, and deep learning)
[원문 링크](https://www.pyimagesearch.com/2017/10/30/how-to-multi-gpu-training-with-keras-python-and-deep-learning/)

> 본 튜토리얼에서는 케라스로 멀티 GPU를 이용한 모델 학습을 하는 방법을 알아봅니다. 

* Keras
* CNN
* Multi-GPU 
* Classification

----

저는 처음 케라스(Keras)를 사용하기 시작했을 때 케라스의 API와 사랑에 빠지게 되었습니다. 
케라스의 API는 scikit-learn처럼 간단하고 우아하지만, 동시에 최첨단의 심층 신경망(Deep Neural Network, DNN)들을 구현하고 학습시킬 수 있을 만큼 강력하기 때문입니다. 

하지만, 케라스를 사용하며 한 가지 아쉬운 점이 있다면 바로 멀티 GPU 환경에서 케라스를 사용하는 것이 까다로울 수 있다는 것입니다.   

만약 테아노(Theano)를 사용하는 유저들이라면, 이 주제는 테아노에서는 불가능한 것에 대해 다루기 때문에 신경쓰지 않으셔도 됩니다.
텐서플로우(TensorFlow)를 사용한다면 가능성은 있지만, 멀티 GPU를 사용하여 네트워크를 학습시키려면 수 많은 보일러플레이트 코드와 
수정이 필요할 수 있다는 단점이 있죠. 

그래서 저는 지금까지 케라스로 멀티 GPU 학습을 할 때는 MXNet 백엔드를 사용하는 것(심지어 MXNet 라이브러리를 직접 사용하는 것)을 선호했었지만, 
이 역시 여전히 많은 환경설정 작업이 필요했습니다.

그러나 이 모든 것은 **케라스 v2.0.9**에서 멀티 GPU를 텐서플로우 백엔드에서 지원한다는
[François Chollet의 발표](https://twitter.com/fchollet/status/918205049225936896)와 함께 바뀌었습니다 
([@kuza55](https://twitter.com/kuza55)와 그의 [keras-extras](https://github.com/kuza55/keras-extras) 리포지토리 덕분입니다).

저는 [@kuza55](https://twitter.com/kuza55)의 멀티 GPU 기능을 근 1년간 사용해오고 있었고,
이제 이 기능이 케라스에 공식적으로 포함되었다는 소식을 듣게 되어 매우 기쁩니다!

이번 블로그 포스트의 나머지 부분에서는 케라스, 파이썬(Python) 그리고 딥러닝을 사용하여 
이미지를 분류하기 위해 합성곱 신경망(Convolutional Neural Network, CNN)을 학습하는 법에 대해 알아보도록 하겠습니다. 

### MiniGoogLeNet 딥러닝 아키텍처

![MiniGoogLeNet architecture](https://www.pyimagesearch.com/wp-content/uploads/2017/10/miniception_architecture.png)      

**Figure 1**: MiniGoogLeNet 아키텍처는 GoogLeNet/Inception의 축소된 버전입니다. 
이미지 크레딧 [@ericjang11](https://twitter.com/ericjang11), [@pluskid](https://twitter.com/pluskid).

**Figure 1**에서는 합성곱(Convolution, 좌측), 인셉션(Inception, 중앙) 그리고 다운 샘플(Downsample, 우측)에 해당하는 각 모듈들을 확인할 수 있고,
하단에서그 모듈들의 조합으로 만들어진 MiniGoogLeNet 아키텍처를 볼 수 있습니다. 
해당 모델은 포스트 후반부의 다중 GPU 실험에 사용될 예정입니다. 

MiniGoogLeNet에서 사용된 인셉션 모듈은 [Szegedy et al.](https://arxiv.org/abs/1409.4842)이 설계한 인셉션 모듈의 변형입니다. 

저는 [@ericjang11](https://twitter.com/ericjang11)과 [@pluskid](https://twitter.com/pluskid)가 MiniGoogLeNet과 관련 모듈들을
아름답게 시각화한 트윗을 통해 이 "Miniception"모듈을 처음 알게 되었습니다. 

그리고 약간의 조사 후에, 저는 이 그림이 Zhang et al.이 2017년에 출판한 논문 [*Understanding Deep Learning Requires Re-Thinking Generalization*](https://arxiv.org/abs/1611.03530)에 나온다는 것을 발견했습니다. 

그런 다음 저는 MiniGoogLeNet 아키텍처를 케라스와 파이썬을 이용해 구현하였고, 이 내용을 제 책인
[*Deep Learning for Computer Vision with Python*](https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/)에 싣기도 하였습니다!

케라스로 MiniGoogLeNet을 구현하는 것에 대한 자세한 내용은 이번 포스트에서 다루는 내용의 범위를 벗어나기 때문에,
해당 모델의 원리(그리고 구현하는 법)에 관심이 있으시다면 제 책을 참고하시길 바랍니다. 

그렇지 않다면 원문의 하단에 있는 ***"다운로드"*** 섹션에서 소스 코드를 다운로드할 수 있습니다. 

### 케라스와 멀티 GPU로 심층 신경망 학습하기

이제 케라스와 멀티 GPU를 사용하여 심층 신경망을 학습해 보겠습니다.  

해당 튜토리얼을 진행하려면, 먼저 가상 환경에 설치된 **케라스의 버전이 2.0.9 이상인지 확인**해야 합니다 
(제 책에서는 `dl4cv`라는 이름의 가상 환경을 사용합니다). 

```
$ workon dl4cv
$ pip install --upgrade keras
```

이제 `train.py`라는 새 파일을 만들고, 아래와 같이 코드를 작성합니다. 

```python
# set the matplotlib backend so figures can be saved in the background
# (uncomment the lines below if you are using a headless server)
# import matplotlib
# matplotlib.use("Agg")
 
# import the necessary packages
from pyimagesearch.minigooglenet import MiniGoogLeNet
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.utils.training_utils import multi_gpu_model
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import argparse
```

헤드리스 서버를 사용하는 경우, 3, 4번째 행의 코드로 `matplotlib`의 백엔드를 설정해야 합니다. 
이렇게 하면 `matplotlib`의 그림을 디스크에 저장할 수 있게됩니다.
헤드리스 서버를 사용하지 않는다면(즉, 키보드와 마우스 그리고 모니터가 시스템에 연결되어 있는 경우) 위의 코드를 그대로 사용하셔도 됩니다. 

백엔드 설정이 끝나면, 이 스크립트에 필요한 패키지들을 가져옵니다. 

7행에서는 MiniGoogLeNet을 제 `pyimagesearch`모듈에서 가져옵니다 (원문의 ***"다운로드"*** 섹션에서 받으실 수 있습니다).

또 한가지 주목할 만한 점은, 13행에서 [CIFAR-10 데이터 세트](https://www.cs.toronto.edu/~kriz/cifar.html)를 가져오는 부분입니다.
케라스를 사용하면 단 한줄의 코드만으로 CIFAR-10 데이터 세트를 디스크에서 로드할 수 있습니다. 

이제, 스크립트에 필요한 인자들을 파싱하기 위한 명령줄 인터페이스를 작성해 보겠습니다.  

```python
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to output plot")
ap.add_argument("-g", "--gpus", type=int, default=1,
	help="# of GPUs to use for training")
args = vars(ap.parse_args())
 
# grab the number of GPUs and store it in a conveience variable
G = args["gpus"]
```

파싱에는 `argparse` 모듈을 사용하며, 하나의 *필수* 인자와 하나의 *선택적* 인자를 파싱해올 것입니다.

* `--output` : 학습이 완료된 후 관련 플롯들을 저장할 경로입니다.
* `--gpus` : 학습에 사용될 GPU의 개수입니다.

명령줄 인자들을 로드한 후에는, 편의를 위해 GPU의 개수를 변수 `G`에 저장합니다.  

이제 학습 프로세스를 구성하는데 사용되는 두 가지 중요한 변수를 초기화하고,
이어서 [학습률을 다항적으로 감소시키는](https://stackoverflow.com/questions/30033096/what-is-lr-policy-in-caffe) `poly_decay` 학습률 스케줄러를 정의합니다.

```python
# definine the total number of epochs to train for along with the
# initial learning rate
NUM_EPOCHS = 70
INIT_LR = 5e-3
 
def poly_decay(epoch):
	# initialize the maximum number of epochs, base learning rate,
	# and power of the polynomial
	maxEpochs = NUM_EPOCHS
	baseLR = INIT_LR
	power = 1.0
 
	# compute the new learning rate based on polynomial decay
	alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
 
	# return the new learning rate
	return alpha
```

에폭(Epoch)은 `NUM_EPOCHS = 70`로 설정되며, 이는 네트워크가 학습 데이터 전체를 총 몇 번 학습하는지 결정합니다.

또한 초기 학습률은 실험을 통해 찾은 `INIT_LR = 5e-3`로 설정합니다. 

그 후에 정의되는 `poly_decay`는, 학습 도중 매 에폭 이후에 효과적으로 학습률을 감소시키는 역할을 합니다. 
만약 `power = 1.0`로 설정하면, 학습률은 다항적이 아닌 선형적으로 감소하게 됩니다. 

다음은 학습 및 테스트 데이터 세트를 로드한 후, 이미지 데이터를 정수형에서 실수형으로 변환하는 작업입니다. 

```python
# load the training and testing data, converting the images from
# integers to floats
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")
testX = testX.astype("float")
```

그 후, 데이터의 각 원소에서 데이터 전체의
[평균값을 빼줍니다](http://ufldl.stanford.edu/wiki/index.php/Data_Preprocessing#Per-example_mean_subtraction). 

```python
# apply mean subtraction to the data
mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean
```

위 코드 블록의 두 번째 행에서 학습 데이터 세트 전체의 평균값을 구해주며, 
세 번째, 네 번째 행에서 각각 학습과 테스트 세트 이미지에서 위에서 구한 평균값을 빼줍니다. 

그 다음은 원핫 인코딩(One-hot encoding)을 할 차례입니다.
원핫 인코딩에 대한 자세한 설명은 제 책에 나와있습니다. 

```python
# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
```

원핫 인코딩은 단일 정수인 범주형 라벨을 벡터로 변환해주어
범주형 교차 엔트로피(Categorical cross-entropy) 손실 함수(Loss function)를 사용할 수 있게 해줍니다. 

다음은 데이터 오그멘테이션(Augmentation)을 위한 함수와, 기타 콜백 함수들을 정의하는 부분입니다.

```python
# construct the image generator for data augmentation and construct
# the set of callbacks
aug = ImageDataGenerator(width_shift_range=0.1,
	height_shift_range=0.1, horizontal_flip=True,
	fill_mode="nearest")
callbacks = [LearningRateScheduler(poly_decay)]
```

블록의 세 번째 행에서는 데이터 오그멘테이션을 위한 이미지 제너레이터(Generator)를 구성합니다.

데이터 오그멘테이션에 대한 자세한 내용은
[*Deep Learning for Computer Vision with Python*](https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/)
실무자 버전에서 다루고 있지만, 간단히 설명하자면 오그멘테이션은 학습에 사용되는 이미지들을
임의로 변형하여 새로운 이미지를 생성하는 방법론 입니다. 

이러한 변형은 네트워크가 지속적으로 새로운 이미지를 학습할 수 있게 하고, 
검증 데이터 세트에 대해 일반화가 더 잘 이루어지게 합니다. 간혹 학습 데이터 세트에 대한 
성능이 저하될 수도 있지만, 대부분의 경우 이 정도는 타협할 수 있는 수준입니다. 

블록의 마지막 행에서는 매 에폭마다 학습률을 감소시켜주기 위한 콜백 함수를 정의합니다.

다음으로 GPU 변수를 살펴보겠습니다.

```python
# check to see if we are compiling using just a single GPU
if G <= 1:
	print("[INFO] training with 1 GPU...")
	model = MiniGoogLeNet.build(width=32, height=32, depth=3,
		classes=10)
```

만약 GPU의 수가 하나이거나 없으면, `.build` 함수를 통해 `model` 변수를 초기화합니다. 
GPU가 두 개 이상인 경우, 학습 중에 모델을 병렬화 할 것입니다.

```python
# otherwise, we are compiling using multiple GPUs
else:
	print("[INFO] training with {} GPUs...".format(G))
 
	# we'll store a copy of the model on *every* GPU and then combine
	# the results from the gradient updates on the CPU
	with tf.device("/cpu:0"):
		# initialize the model
		model = MiniGoogLeNet.build(width=32, height=32, depth=3,
			classes=10)
	
	# make the model parallel
	model = multi_gpu_model(model, gpus=G)
```

케라스에서 멀티 GPU 모델을 만들려면 약간의 코드를 추가해야 하지만, *그 양이 그리 많지는 않습니다!*

먼저, `with tf.device("/cpu:0"):`를 통해 네트워크 컨텍스트로 CPU(GPU가 아닌)를 사용하도록 한 것이 보이실 겁니다.

왜 멀티 GPU 모델에서 CPU가 필요할까요?

그 이유는, GPU가 무거운 연산들을 처리하는 동안,
CPU가 그에 필요한 오버헤드를(예를 들면, 학습 이미지를 GPU 메모리에 올리는 작업) 처리해주기 때문입니다. 

여기서는 CPU가 기본 모델을 초기화하는 작업을 처리하며,
초기화가 완료된 후에 `multi_gpu_model`을 호출하게 됩니다. 
`multi_gpu_model` 함수는 CPU에서 모든 GPU로 모델을 복제하여 단일 시스템, 다중 GPU 데이터 병렬 처리를 가능하게 해줍니다. 

학습이 시작되면 학습 이미지들은 배치 단위로 각 GPU에 할당됩니다. 
그 후 CPU는 각 GPU에서 계산한 기울기(Gradients)들을 바탕으로 모델의 가중치를 업데이트 합니다. 

이제 우리는 모델을 컴파일하고, 학습을 시작할 수 있습니다.

```python
# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=INIT_LR, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
 
# train the network
print("[INFO] training network...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=64 * G),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // (64 * G),
	epochs=NUM_EPOCHS,
	callbacks=callbacks, verbose=2)
```

위 코드 블록의 세 번째 행에서는
확률적 경사 하강법(Stochastic Gradient Descent, SGD) 알고리즘을 정의합니다.

그 후 모델의 옵티마이저(Optimizer)와 손실 함수의 인자로 SGD와 범주형 교차 엔트로피를 주어 컴파일 합니다.  

이제 우리는 모델을 학습시킬 준비가 되었습니다!

학습을 시작하기 위해서는 `model.fit_generator` 함수를 필요한 인자들과 함께 호출해야 합니다.

각 GPU에 대한 배치 크기를 64로 설정하기 위해 `batch_size=64 * G`로 지정합니다. 

학습은 전에 설정한 바와 같이 70 에폭동안 수행됩니다. 

학습이 진행되는 동안 CPU는 각 GPU에서 계산된 기울기들을 바탕으로 모델의 가중치를 업데이트 합니다.
그 후 업데이트된 모델은 각 GPU에 다시 반영됩니다. 

이제 학습과 테스트 과정은 완료되었으니, 다음은 손실 및 정확도를 그래프로 나타내어
학습 과정을 시각화할 차례입니다.

```python
# grab the history object dictionary
H = H.history
 
# plot the training loss and accuracy
N = np.arange(0, len(H["loss"]))
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H["loss"], label="train_loss")
plt.plot(N, H["val_loss"], label="test_loss")
plt.plot(N, H["acc"], label="train_acc")
plt.plot(N, H["val_acc"], label="test_acc")
plt.title("MiniGoogLeNet on CIFAR-10")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
 
# save the figure
plt.savefig(args["output"])
plt.close()
```

위의 마지막 코드 블록은 `matplotlib`을 사용하여 학습과 테스트 데이터 세트에 대한
손실 및 정확도를 그래프로 나타낸 후, 해당 플롯을 디스크에 저장하는 과정입니다. 

만약 학습 과정이 내부적으로 어떻게 작동하는지 자세히 알고 싶으시다면, 제 책
[*Deep Learning for Computer Vision with Python*](https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/)을 
참고해 보시는 것을 추천합니다.


### 멀티 GPU 학습 결과

이제 우리의 노력의 결과를 확인해볼 차례입니다.

먼저 단일 GPU를 사용하여 기준을 구하는 과정입니다.

```shell
$ python train.py --output single_gpu.png
[INFO] loading CIFAR-10 data...
[INFO] training with 1 GPU...
[INFO] compiling model...
[INFO] training network...
Epoch 1/70
 - 64s - loss: 1.4323 - acc: 0.4787 - val_loss: 1.1319 - val_acc: 0.5983
Epoch 2/70
 - 63s - loss: 1.0279 - acc: 0.6361 - val_loss: 0.9844 - val_acc: 0.6472
Epoch 3/70
 - 63s - loss: 0.8554 - acc: 0.6997 - val_loss: 1.5473 - val_acc: 0.5592
...
Epoch 68/70
 - 63s - loss: 0.0343 - acc: 0.9898 - val_loss: 0.3637 - val_acc: 0.9069
Epoch 69/70
 - 63s - loss: 0.0348 - acc: 0.9898 - val_loss: 0.3593 - val_acc: 0.9080
Epoch 70/70
 - 63s - loss: 0.0340 - acc: 0.9900 - val_loss: 0.3583 - val_acc: 0.9065
Using TensorFlow backend.
 
real    74m10.603s
user    131m24.035s
sys     11m52.143s
```

![single gpu plot](https://www.pyimagesearch.com/wp-content/uploads/2017/10/keras_single_gpu.png)

**Figure 2**: 케라스로 단일 GPU로 MiniGoogLeNet 네트워크를 CIFAR-10 데이터 세트에 학습시킨 결과입니다.

이 실험을 위해, 저는 먼저 제 NVIDIA DevBox에서 하나의 Titan X GPU를 사용해 학습을 진행하였습니다.
매 에폭에는 63초 가량이 소요되었으며, 총 학습에는 74분 10초의 시간이 걸렸습니다. 

그 후 Titan X GPU 4개를 모두 사용하여 훈련을 진행해 보았습니다. 

```shell
$ python train.py --output multi_gpu.png --gpus 4
[INFO] loading CIFAR-10 data...
[INFO] training with 4 GPUs...
[INFO] compiling model...
[INFO] training network...
Epoch 1/70
 - 21s - loss: 1.6793 - acc: 0.3793 - val_loss: 1.3692 - val_acc: 0.5026
Epoch 2/70
 - 16s - loss: 1.2814 - acc: 0.5356 - val_loss: 1.1252 - val_acc: 0.5998
Epoch 3/70
 - 16s - loss: 1.1109 - acc: 0.6019 - val_loss: 1.0074 - val_acc: 0.6465
...
Epoch 68/70
 - 16s - loss: 0.1615 - acc: 0.9469 - val_loss: 0.3654 - val_acc: 0.8852
Epoch 69/70
 - 16s - loss: 0.1605 - acc: 0.9466 - val_loss: 0.3604 - val_acc: 0.8863
Epoch 70/70
 - 16s - loss: 0.1569 - acc: 0.9487 - val_loss: 0.3603 - val_acc: 0.8877
Using TensorFlow backend.
 
real    19m3.318s
user    104m3.270s
sys     7m48.890s
```

![multi gpu plot](https://www.pyimagesearch.com/wp-content/uploads/2017/10/keras_multi_gpu.png)

**Figure 3**: 케라스로 멀티 GPU로(4 Titan X GPUs) MiniGoogLeNet 네트워크를 CIFAR-10 데이터 세트에 학습시킨 결과입니다.
모델의 성능은 유지하면서, 학습 시간은 75% 가량 단축되었습니다. 

4개의 GPU를 사용하여 매 에폭에 걸리는 학습 시간을 16초 가량으로 줄일 수 있었고, 
총 학습에는 19분 3초 밖에 소요되지 않았습니다.

보시다시피, 케라스로 멀티 GPU 학습을 하는 것은 쉬울 뿐만 아니라, 효율적이기도 합니다!

참고: 이 경우 단일 GPU를 사용해 학습한 모델의 정확도가 멀티 GPU를 사용한 경우보다 좋았습니다.
이는 기계 학습의 확률적인 부분에 기인한 것으로, 만약 이러한 과정을 수백 번 반복하여 평균을 산출하면, 
거의 동일한 성능을 보일 것입니다.

### 요약

In today’s blog post we learned how to use multiple GPUs to train Keras-based deep neural networks.

Using multiple GPUs enables us to obtain quasi-linear speedups.

To validate this, we trained MiniGoogLeNet on the CIFAR-10 dataset.

Using a single GPU we were able to obtain 63 second epochs with a total training time of 74m10s.

However, by using multi-GPU training with Keras and Python we decreased training time to 16 second epochs with a total training time of 19m3s.

Enabling multi-GPU training with Keras is as easy as a single function call — I recommend you utilize multi-GPU training whenever possible. In the future I imagine that the multi_gpu_model  will evolve and allow us to further customize specifically which GPUs should be used for training, eventually enabling multi-system training as well.

### 딥러닝을 더 깊게 배워볼 준비가 되셨나요? 저만 따라오세요.

If you’re interested in learning more about deep learning (and training state-of-the-art neural networks on multiple GPUs), be sure to take a look at my new book, Deep Learning for Computer Vision with Python.

Whether you’re just getting started with deep learning or you’re already a seasoned deep learning practitioner, my new book is guaranteed to help you reach expert status.

> 이 글은 2018 컨트리뷰톤에서 Contribute to Keras 프로젝트로 진행했습니다.  
> Translator: [정연준](https://github.com/fuzzythecat)  
> Translator email : fuzzy0427@gmail.com  
