# Caffe 모델 Keras 모델로 바꾸기

[원문 링크](https://nicolovaligi.com/converting-deep-learning-model-caffe-keras.html)


> 이 문서는 딥러닝 프레임워크인 Caffe로 학습된 모델을 Keras에서 사용하기 위해 Keras 모델로 바꾸는 내용에 대한 튜토리얼입니다. Pre-trained 모델을 사용하는데 도움이 될 것입니다. 설명, 원활한 번역을 위해 원문과 조금 다를 수 있습니다.

* Keras

* Caffe 프레임워크

* Pre-trained 모델

</br>

![197_0.jpg](./media/197_0.jpg)

</br>

많은 딥러닝 연구자들은 새로운 딥러닝 네트워크와 모델을 개발하기 위해 [**Caffee 프레임워크**](http://caffe.berkeleyvision.org/)를 사용합니다. 내가 그렇게 생각하는 이유는 [**Model Zoo**](https://github.com/albertomontesg/keras-model-zoo)에 있는 pre-trained 모델(학습되어 있는 모델)들 때문입니다. Pre-trained 가중치(weight)를 사용하는 것은 몇 가지의 장점이 있습니다:

* 데이터를 충분히 구하는 것은 굉장히 어려운 일입니다.
* 모델을 처음부터 학습시키는 것은 오래 걸리고, 소모적입니다.

However, I really can't get behind Caffe's heavy use of protobufs for network definition (code is data, after all). This lack of flexibility forces everybody to fork the codebase for minor details, like image preprocessing. And you get to write that in C++, of all things.

그러나, Caffe가 너무 무겁기 때문에 계속 사용하기 어렵고, Caffe의 부족한 유연성 때문에 이미지 처리와 같은 사소한 세부 사항들을 위해 모든 사람들이 코드 베이스에 포크할 수 밖에 없습니다. 그리고 여러분은 C++와 같은 언어를 사용할 수 밖에 없겠죠.

나는 TensorFlow와 그와 비슷한 친구들의 접근을 훨씬 더 선호하는데, 이러한 프레임워크들은 Python 함수의 힘을 빌려 계산 그래프를 작성할 수 있습니다. 이러한 유연성은 프레임워크 코드 밖에서 스크립트 언어로 작성되었을 때 재사용 가능한 사전 처리 단계로 확장됩니다. 나는 보통 케라스로 일하는 것을 즐깁니다. 왜냐하면 그것은 쉬운 일을 쉽게, 힘든 일을 가능하게 하기 때문입니다. 

In this post I will go through the process of converting a pre-trained Caffe network to a Keras model that can be used for inference and fine tuning on different datasets. You can see the end result here: Keras DilatedNet. I will assume knowledge of Python and Keras.

이 포스트에서는 pre-trianed Caffe 네트워크를 다른 데이터 셋 대한 학습 및 정밀한 튜닝에 사용할 수 있는 Keras 모델로 전환하는 과정을 살펴볼 것입니다. 최종 결과(코드)는 여기서 확인하세요 : [Keras DilatedNet](https://github.com/nicolov/segmentation_keras). Python과 Keras로 만들어져 있습니다.

### Picking a model for image segmentation

최근에, 나는 이미지 세그멘테이션에 대한 최신 기술을 연구해왔고, 이해하고 연구하고 싶은 몇 가지 잠재적인 모델들을 선택해보았고 아래와 같습니다.

* [**SegNet**](http://mi.eng.cam.ac.uk/projects/segnet/)은 표준 인코더-디코더 네트워크를 사용하며, 또한 흥미로운 Bayesian extension에 대한 내용도 가지고 있습니다.. CamVid로 weight를 학습.

* [**DirmedNet**](https://github.com/fyu/dilation)은 픽셀 단위 라벨링에 더 좋은 dilated 컨볼루션들을 사용해서 오토인코더 구조를 제거합니다.

* [**Semantic Segmentation**](https://arxiv.org/pdf/1411.4038.pdf)을 위한 Fully Convolutional Networks는 출력에 대해 fully connected layer를 가진 분류 네트워크와 유사합니다.

확장된 컨볼루션 덕분에 더 좋은 결과를 얻었고, 매우 깨끗한 Caffe 코드가 제공되기 때문에 두 번째로 말한 네트워크로 진행해보기로 했습니다.

### Converting the weights

Caffe는 직렬화 된 프로토콜 버퍼인 * .caffemodel 파일에 가중치를 저장합니다. caffe-tensorflow를 사용하여 이들을 numpy에 쉽게 로드 할 수 있는 HD5 파일로 변환 할 것입니다. 스크립트는 .prototxt (네트워크 정의) 및 .caffemodel 파일을 변환하여 가중치 및 TensorFlow 그래프를 생성합니다.
