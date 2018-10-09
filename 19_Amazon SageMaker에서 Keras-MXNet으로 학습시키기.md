# Amazon SageMaker에서 Keras-MXNet으로 학습시키기

이전에 소개한 바와 같이, [Apache MXNet](https://mxnet.apache.org/) 은 [Keras-MXNet](https://github.com/awslabs/keras-apache-mxnet)으로 알려진 [Keras 2](https://keras.io/)의 백앤드로써 사용가능하다. 

이 글에선, [Amazon SageMaker](https://aws.amazon.com/ko/sagemaker/) 에서 Keras-MXNet을 학습시키는 방법을 소개합니다  :

- GPU 와 GPU 학습을 위해 커스텀 Docker 컨테이너 만들기
- 다중 GPU 학습 환경 설정하기
- 케라스 스크립트로 파라미터 넘겨주기
- Keras, MXNet 형식으로 학습된 모델 저장하기

평소처럼, GitHub에서 코드를 찾아볼 수 있습니다.

## MXNET을 위해 Keras 환경 설정하기

필요한것은 *.keras/keras.json* 안의 *'mxnet'*으로 *'backend'*를 설정하는 것이다, 하지만 *'image_data_format'*을 *'channels_first'*로 설정하는 것은 MXNet 학습을 더 빠르게 만들어 줄 것이다.

이미지 데이터를 가지거 작업을 할 때, 입력 형태는 *'channels_first'* (i.e. 채널의 갯수, 너비, 높이) 또는 *'channels_last'*(i.e. 높이, 너비, 채널의 갯수) 가 가능하다. MNIST 에서는 입력의 형태는 (1, 28, 28) 또는 (28, 28, 1) 모두 가능하다 : 이것은 1개의 채널(흑백의 사진), 28X28 픽셀을 의미한다. ImageNet에서는 입력의 형태는 (3, 224, 224) 또는 (224, 224, 3) 두가지가 가능하다 : 이것은 3개의 채널(RGB), 224X224 픽셀을 의미한다.

컨테이너로 사용할 configuration file은 이렇다.

```json
{
    "epsilon": 1e-07, 
    "floatx": "float32", 
    "image_data_format": "channels_first", 
    "backend": "mxnet"
}
```



## 사용자 지정 컨테이너를 빌드하기

SageMaker는 TensorFlow나 MXNet의 환경과 내재된 알고리즘의 집합을 제공하지만 Keras에는 제공하지 않는다. 하지만, 개발자들은 학습과 예측을 위한 사용자 지정 컨테이너를 빌드할 수 있는 선택지가 있다.

당연하게도, SageMaker에서 사용자 지정 컨테이너를 성공적으로 호출하기 위해선 몇 가지 규칙을 정의해야 한다.

- **학습과 예측  스크립트의 이름:** 디폴트로, 두가지의 스크립트는 각각 *'train'* 과 *'serve'*로 명명되어야하고, 실행가능해야하면서, 확장자가 없어야한다. SageMaker는 *'docker run your_container train'*를 실행하면서 학습을 시작할 수 있다.
- 컨테이너 안에서 **하이퍼 파라미터들의 위치**: */opt/ml/input/config/hyperparameters.json.*
- 컨테이너 안에서 **입력 데이터 파라미터의 위치**: */opt/ml/input/data*.

위 규칙들을 정의하기 위해선 <u>간단한 CNN으로 MNIST 배우기</u>의 한 예제의 Keras 스크립트에서 몇 가지를 바꿔야한다. 당신도 볼 수 있듯이, 몇 가지 보완점들은 마이너한 것들이고 당신의 코드에 넣는다고 해서 아무런 문제도 발생하지 않을 것이다.



## CPU 기반 Docker 컨테이너 빌드하기

Docker 파일은 여기서 확인 할 수 있다.

```ubuntu
FROM ubuntu:16.04

RUN apt-get update && \
    apt-get -y install build-essential libopencv-dev libopenblas-dev libjemalloc-dev libgfortran3 \
    python-dev python3-dev python3-pip wget curl

COPY mnist_cnn.py /opt/program/train
RUN chmod +x /opt/program/train

RUN mkdir /root/.keras
COPY keras.json /root/.keras/

RUN pip3 install mxnet --upgrade --pre && \
    pip3 install keras-mxnet --upgrade --pre

RUN rm -rf /var/lib/apt/lists/*
RUN rm -rf /root/.cache

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

ENV PATH="/opt/program:${PATH}"

WORKDIR /opt/program

```

**우분투 16.04 이미지파일**에서 시작하여 다음 파일들을 설치합니다  :

- **파이썬 3** 뿐만 아니라 MXNet 의 **기본 종속성**.
- **MXNet**과 **Keras-MXNet**의 최신 패키지

> 사전 배포 버전을 설치하지 않아도 됩니다.

이것이 완료 된다면, 컨테이너 사이즈를 조금 줄이기위해 다양한 캐시를 비워야합니다. 그 후,  :

- */opt/program*에 적절한 이름(*'train'*)으로 **Keras 스크립트**를 복사하시고, 실행가능하게 만들어야 합니다.

*더 나은 유연성을 위해, 우리는 하이퍼 파라미터로 전달된 S3 위치로부터 실제 학습 스크립트를 가져오는 포괄적인 런쳐를 사용할 수 있다. 하지만 이번 문서에는 다루지 않는다*

- *root/.keras/keras.json*로 **Keras 속성 파일**을 복사합니다.

마지막으로, 스크립트의 디렉토리를 **work directory**로 설정하고, **PATH**에 추가합니다.

긴 파일은 아니지만, 여느때처럼 이런 세부사항들은 중요합니다.



## GPU 기반 Docker 컨테이너를 빌드하기

이제, GPU를 구축하겠습니다. 오직 두가지 방법만 다릅니다.

- Ubuntu 16.04를 기반으로 한 **CUDA 9.0 이미지**에서 시작합니다. 이것은 MXNet이 필요한 모든 CUDA 라이브러리를 모두 가지고 있습니다. ``
- **CUDA 9.0-enabled MXNet** 패키지를 설치합니다.

다른 모든 과정은 CPU 기반 Docker 컨테이너를 빌드 하는 것과 같습니다.

```
FROM nvidia/cuda:9.0-runtime

RUN apt-get update && \
    apt-get -y install build-essential libopencv-dev libopenblas-dev libjemalloc-dev libgfortran3 \
    python-dev python3-dev python3-pip wget curl

COPY mnist_cnn.py /opt/program/train
RUN chmod +x /opt/program/train

RUN mkdir /root/.keras
COPY keras.json /root/.keras/

RUN pip3 install mxnet-cu90 --upgrade --pre && \
    pip3 install keras-mxnet --upgrade --pre

RUN rm -rf /var/lib/apt/lists/*
RUN rm -rf /root/.cache

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

ENV PATH="/opt/program:${PATH}"

WORKDIR /opt/program
```



## Amazon ECR에서 Docker 저장소 생성하기

SageMaker를 사용하기 위해선 가져온 컨테이너가 Amazon ECR에서 호스트되어야 합니다.

저장소를 만들고 로그인 해보겠습니다.

```sh
aws ecr describe-repositories --repository-names $repo_name > /dev/null 2>&1
if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name $repo_name > /dev/null
fi

$(aws ecr get-login --region $region --no-include-email)
```



## 컨테이너를 빌드하고 ECR에 푸시하기

이제 두 컨테이너를 빌드하고 저장소로 푸시하겠습니다. GPU와 GPU 버전에 대해 나누어 진행합니다. 

```sh
docker build -t $image_tag -f $dockerfile .
docker tag $image_tag $account.dkr.ecr.$region.amazonaws.com/$repo_name:latest
docker push $account.dkr.ecr.$region.amazonaws.com/$repo_name:latest
```

작업을 완료하면, 두개의 컨테이너가 ECR안에 들어가 있는 것을 볼 수 있을 것입니다.

![img](https://cdn-images-1.medium.com/max/1000/1*1sXM6ChcKtg6zLKdplBqag.png)

Docker 부분이 모두 끝났습니다. 이제 SageMaker에서 학습 작업을 구성하겠습니다.



## 학습 작업 구성하기

사실 감흥이 덜한 작업입니다, 좋은 소식이기도 합니다 : **내장된 알고리즘으로 학습하는 것과 실제론 다른 점이 없기 때문입니다**.

첫째로, 로컬 머신에서 S3로 **MNIST 데이터 집합을 업로드 해야합니다**. 

```python
local_directory = 'data'
prefix          = repo_name+'/input'

train_input_path      = sess.upload_data(
  local_directory+'/train/', key_prefix=prefix+'/train')
validation_input_path = sess.upload_data(
  local_directory+'/validation/', key_prefix=prefix+'/validation')
```

그리고, 학습 작업을 다음의 작업을 통해 구성합니다 :

- 구축한 컨테이너를 선택하여 SageMaker 추정기에 대해 일반적인 파라미터를 설정합니다.
- Keras 스크립트로 **하이퍼 파라미터**를 전송합니다.
- Keras 스크립트로 **입력받은 데이터**를 전송합니다.

```python
output_path = 's3://{}/{}/output'.format(sess.default_bucket(), repo_name)
image_name  = '{}.dkr.ecr.{}.amazonaws.com/{}:latest'.format(account, region, repo_name)

estimator = sagemaker.estimator.Estimator(
                       image_name=image_name,
                       base_job_name=base_job_name,
                       role=role, 
                       train_instance_count=1, 
                       train_instance_type=train_instance_type,
                       output_path=output_path,
                       sagemaker_session=sess)

estimator.set_hyperparameters(lr=0.01, epochs=10, gpus=gpu_count, batch_size=batch_size)

estimator.fit({'training': train_input_path, 'validation': validation_input_path})
```

이것으로 학습은 끝났습니다. 마지막으로 SageMaker에 맞게 Keras 스크립트를 수정하겠습니다.



## SageMaker에 맞게 Keras 스크립트 수정하기

하이퍼 파라미터, 입력 데이터, 다중-GPU 환경설정하기, 데이터 집합 불러오기/저장하기 등을 고려해야만합니다.

### 하이터 파라미터와 입력데이터 설정 전달하기

전에 설명한 바와 같이, SageMaker는 하이터 파라미터들을***/opt/ml/input/config/hyperparameters.json***에 복사합니다. 우리가 해야하는 일은 이 파일을 읽고, 파라미터를 추출해 필요시 디폴트 값을 설정하는 일입니다.

```python
prefix     = '/opt/ml/'
param_path = os.path.join(prefix, 'input/config/hyperparameters.json')

with open(param_path, 'r') as params:
    hyperParams = json.load(params)
    
lr = float(hyperParams.get('lr', '0.1'))
batch_size = int(hyperParams.get('batch_size', '128'))
epochs = int(hyperParams.get('epochs', '10'))
gpu_count = int(hyperParams.get('gpu_count', '0'))
```

비슷한 방법으로, SageMaker는 입력 데이터 설정을 */opt/ml/input/data*에 복사합니다. 위와 같은 방법으로 해결할 수 있습니다.

> 이번 예제에서는, 환경설정 정보는 필요하지 않지만, 이러한 방법으로 우리는 하이퍼 파라미터와 입력 데이터를 읽어올 수 있습니다.

### 학습 데이터와 검증 데이터 집합을 불러오기

이번 예제와 같이 파일 모드에서 학습을 진행할 때, SageMaker는 **자동적으로** 데이터 집합을  */opt/ml/input/<channel_name>*로 복사한다 : 이곳에서, 우리는 *학습*과 *검증* 채널을 정의했고, 상응하는 디렉토리로부터 MNIST 파일을 간단하게 읽어올 수 있다.

```python
def load_data(input_path):
    # Adapted from https://github.com/keras-team/keras/blob/master/keras/datasets/fashion_mnist.py
    files = ['training/train-labels-idx1-ubyte.gz', 
             'training/train-images-idx3-ubyte.gz',
             'validation/t10k-labels-idx1-ubyte.gz', 
             'validation/t10k-images-idx3-ubyte.gz']
    # Load training labels
    with gzip.open(input_path+files[0], 'rb') as lbpath:
        y_train = np.frombuffer(
            lbpath.read(), np.uint8, offset=8)
    # Load training samples
    with gzip.open(input_path+files[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
    # Load validation labels
    with gzip.open(input_path+files[2], 'rb') as lbpath:
        y_test = np.frombuffer(
            lbpath.read(), np.uint8, offset=8)
    # Load validation samples
    with gzip.open(input_path+files[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)
    print("Files loaded")
    return (x_train, y_train), (x_test, y_test)
```

### 다중-GPU 학습 환경설정하기

Keras_MXNet은 다중-GPU 학습을 매우 쉽게 설정할 수 있게 해준다.(previous post) *gpu_count* 하이퍼 파라미터에 의존하여, 우리는 단지 컴파일링 전에 맞춤형 Keras API로 우리의 모델을 포장할 필요가 있다.

```python
from keras.utils import multi_gpu_model

model = Sequential()
model.add(...)
...
if gpu_count > 1:
    model = multi_gpu_model(model, gpus=gpu_count)
model.compile(...)
```

### 모델들을 저장하기

학습이 끝났을 시에 우리가 해야하는 진짜 마지막 일은 모델을 ***/opt/ml/model***에 저장하는 것이다 : SageMaker는 해당 디렉토리에 존재하는 모든 품목들을 가지고, *model.tar.gz*를 구축하고 구축한 파일을 학습에 쓰였던 S3 박스에 복사합니다.

사실, 우리는 학습된 모델을 두개의 다른 형태로 저장할 것입니다. :**Keras 포멧**  (i.e. HDF5 파일) 과 **MXNet 형식** (i.e. JSON 파일과 *.params* 파일) 입니다. 이렇게 함으로서 우리는 두가지 라이브러리를 모두 사용할 수 있게 됩니다.

```python
from keras.models import save_mxnet_model

prefix      = '/opt/ml/'
model_path  = os.path.join(prefix, 'model')

model_name = 'mnist-cnn-'+str(epochs)
model.save(model_path+'/'+model_name+'.hd5') # Keras model
save_mxnet_model(model=model, prefix=model_path+'/'+model_name) # MXNet model
```

모두 끝났습니다. 지금까지 우리가 한 일은 SageMaker의 입력과 출력에 맞게 스크립트의 인터페이스를 수정한 것입니다. Keras 코드는 어떠한 수정도 필요 없습니다.



## 스크립트를 실행하기

GPU 버전을 실행해봅시다. p3.8xlarge 환경에서 2개의 GPU를 학습 시킬 것입니다.

```verilog
Using MXNet backend
Hyper parameters: {'lr': '0.01', 'batch_size': '256', 'epochs': '10', 'gpus': '2'}
Input parameters: {'validation': {'S3DistributionType': 'FullyReplicated', 'TrainingInputMode': 'File', 'RecordWrapperType': 'None'}, 'training': {'S3DistributionType': 'FullyReplicated', 'TrainingInputMode': 'File', 'RecordWrapperType': 'None'}}
Files loaded
x_train shape: (60000, 1, 28, 28)
60000 train samples
10000 test samples
Train on 60000 samples, validate on 10000 samples
Epoch 1/10
/usr/local/lib/python3.5/dist-packages/mxnet/module/bucketing_module.py:408: UserWarning: Optimizer created manually outside Module but rescale_grad is not normalized to 1.0/batch_size/num_workers (1.0 vs. 0.00390625). Is this intended?
  force_init=force_init)
[17:43:09] src/operator/nn/./cudnn/./cudnn_algoreg-inl.h:107: Running performance tests to find the best convolution algorithm, this can take a while... (setting env variable MXNET_CUDNN_AUTOTUNE_DEFAULT to 0 to disable)

  256/60000 [..............................] - ETA: 35:17 - loss: 2.3051 - acc: 0.0938
 2048/60000 [>.............................] - ETA: 4:18 - loss: 1.8998 - acc: 0.3926 

<output removed>

56320/60000 [===========================>..] - ETA: 0s - loss: 0.0284 - acc: 0.9905
58624/60000 [============================>.] - ETA: 0s - loss: 0.0284 - acc: 0.9905
60000/60000 [==============================] - 2s 26us/step - loss: 0.0283 - acc: 0.9905 - val_loss: 0.0257 - val_acc: 0.9916

Test loss: 0.025707698954685294
Test accuracy: 0.9916
Saved Keras model
MXNet Backend: Successfully exported the model as MXNet model!
MXNet symbol file -  /opt/ml/model/mnist-cnn-10-symbol.json
MXNet params file -  /opt/ml/model/mnist-cnn-10-0000.params

Model input data_names and data_shapes are: 
data_names :  ['/conv2d_1_input1']
data_shapes :  [DataDesc[/conv2d_1_input1,(256, 1, 28, 28),float32,NCHW]]

Note: In the above data_shapes, the first dimension represent the batch_size used for model training. 
You can change the batch_size for binding the module based on your inference batch_size.
Saved MXNet model

===== Job Complete =====
Billable seconds: 121
```

**S3 박스를 확인해보겠습니다.**

```
$ aws s3 ls $BUCKET/keras-mxnet-gpu/output/keras-mxnet-mnist-cnn-2018-05-30-17-39-50-724/output/
2018-05-30 17:43:34    8916913 model.tar.gz
$ aws s3 cp $BUCKET/keras-mxnet-gpu/output/keras-mxnet-mnist-cnn-2018-05-30-17-39-50-724/output/model.tar.gz .
$ tar tvfz model.tar.gz
-rw-r--r-- 0/0   4822688 2018-05-30 17:43 mnist-cnn-10.hd5
-rw-r--r-- 0/0   4800092 2018-05-30 17:43 mnist-cnn-10-0000.params
-rw-r--r-- 0/0      4817 2018-05-30 17:43 mnist-cnn-10-symbol.json
```

이제 이 모델을 우리가 사용하고 싶은 곳에는 어디든 사용할 수 있습니다.

