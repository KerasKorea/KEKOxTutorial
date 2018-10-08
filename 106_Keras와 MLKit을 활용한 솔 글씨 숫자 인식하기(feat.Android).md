## Keras와 ML Kit을 활용한 손 글씨 숫자 인식하기(feat.Android)
[From Keras to ML Kit 원문 바로가기](https://proandroiddev.com/from-keras-to-ml-kit-eeaf578a01df)
> 문서 간략 소개

* 케라스
* ML Kit
* TensorFlow Lite
* Android

### From Keras to ML Kit
> 간략 소개


### How can I use my Keras model with ML Kit?
Keras는 텐서플로 위에서 작동할 수 있는 파이썬 기반의 신경망 라이브러리 오픈소스 입니다. 신경망을 쉽게 구현할 수 있는 라이브러리 입니다. TensorFlow와 완벽하게 호환되면서 TensorFlow의 세부사항을 추상화합니다. 신경망  공부를 시작하게 되면 더 많이 사용하게 될 것입니다.
<br>
이 튜토리얼에서는 [Keras repository](https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py)에 있는 기본적인 예를 제 [Jupyter Notebook](https://github.com/miquelbeltran/deep-learning/blob/master/android-mlkit-sample/Keras%20Sample.ipynb)을 통해 살펴볼 것 입니다.
<br>
이번 예제에서 Keras 원작자는 MNIST 데이터 셋으로부터 손 글씨 숫자를 읽을 수 있는 모델을 만들었습니다. MNIST 데이터 셋은 머신 러닝에서 많이 사용되는 데이터 셋이며, 저는 이것을 이용하여 [myFace Generator 프로젝트](https://proandroiddev.com/deep-learning-nd-face-generator-fa92ddbb8c4a)를 진행했습니다.
<br>
![](./media/106_1.png)


Keras로 모델을 생성하는 부분입니다.
```
model = Sequential()

model.add(Dense(512, activation='relu', input_dim=784))
#model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()
```

이 모델은 세 개의 레이어로 단순하게 구성되어 있습니다. 입력 차원은 784차원이며, 출력 차원은 10개의 클래스에 해당하는 10차원이고, 각각 완전 연결 레이어로 구성되어 있습니다.
참고로 원본 샘플의 드롭 아웃은 주석 처리했는데 그 이유는 나중에 설명하겠습니다.
> MNIST데이터 셋의 이미지 크기는 28X28인 2차원입니다. 이를 완전 연결 레이어에 적용시키기 위해선 1X784인 1차원으로 리사이징 해야합니다.

### Training the model
학습 과정은 주피터 노트북에서 확인할 수 있습니다. [케라스 샘플](https://github.com/miquelbeltran/deep-learning/blob/master/android-mlkit-sample/Keras%20Sample.ipynb)
<br>
결과는 다음과 같습니다.

![](./media/106_2.png)

이 모델의 정확도는 0.98이며 우수하지는 않습니다. 저는 오직 5에폭만 학습시켰고, 드롭 아웃 레이어를 뺏기 때문에 정확도가 원래의 예시에 비해 약간 떨어졌습니다.

### Exporting a Keras model
모델 훈련을 마친 후, Keras 모델을 TF Lite로 변환해야 합니다. [텐서플로 모델을 ML Kit으로 내보내기](https://proandroiddev.com/exporting-tensorflow-models-to-ml-kit-bce13b914f31)와 같은 과정이지만 추가적인 과정이 필요합니다. <br>

> 코드 설명은 아래에 있습니다.

```
from keras import backend as K

custom_input_tensor = tf.placeholder(tf.float32, shape=(1, 784))
output_tensor = model(custom_input_tensor)

frozen_graph = freeze_session(K.get_session(), output_names[output_tensor.op.name])

tflite_model = tf.contrib.lite.toco_convert(frozen_graph,[custom_input_tensor], [output_tensor])
open("app/src/main/assets/nmist_mlp.tflite", "wb").write(tflite_model)
```

Keras모델을 입력 텐서로 감싸고, 출력 텐서를 구해야 합니다. 우리는 이 텐서들을 ML Kit의 입력과 출력으로 사용합니다.

<br>```freeze_session``` 전에, 입력 텐서로 ```(1, 784)```벡터의 TensorFlow의 ```placeholder```를 정의하였습니다. 그리고 나서 Keras로 만든 ```model```을 가지고 입력 텐서를 인자값으로 넣어줍니다. 결과 값은 ```output_tensor```입니다.

<br>두번째로, ```freeze_session```을 호출합니다. 이것은 링크로 걸어놓은 나의 예전 글에 나와있습니다. 하지만 이번에는 Keras 뒷단으로 부터 TensorFlow세션을 반환하는 ```K.get_session()```을 호출합니다.

<br>마지막으로, 방금 만든 입력과 출력 텐서를 ```toco_convert```메소드에 전달하고, 변수를 고정시켜서 모델을 ```tflite```파일에 저장합니다.

<br>그러나, 원래 모델은 잠시동안 **TF Lite가 지원하지 않는**```Dropout```을 사용하였습니다. 저는 원래의 모델을 내보내는 과정에서 문제가 생겼었는데, 드롭아웃 레이어를 제거하니깐 해결되었습니다. 저는 TensorFlow의 다음 버전에서 이 문제가 해결되길 기대합니다.

### Running on a Google Colab


### Running the exported model on Android

### 참고문서
* [참고 사이트 1]()
* [참고 사이트 2]()

> 이 글은 2018 컨트리뷰톤에서 [Contribute to Keras](https://github.com/KerasKorea/KEKOxTutorial) 프로젝트로 진행했습니다. <br>
> Translator : [김수정](https://github.com/SooDevv) <br>
> Translator email : [soojung.dev@gmail.com](soojung.dev@gmail.com)
