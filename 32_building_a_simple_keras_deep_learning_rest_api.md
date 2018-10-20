## Keras 모델을 REST API로 배포해보기(Building a simple Keras + deep learning REST API)
[원문](https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html)
> 이 글은 Adrian Rosebrock이 작성한 안내 게시글로 Keras 모델을 REST API로 제작하는 간단한 방법을 안내하고 있습니다.

* Keras
* REST API
* flask

### 개요
이번 튜토리얼에서, 우리는 케라스 모델을 가지고 REST API로 베포하는 간단한 방법을 설명합니다.

이 글에 있는 예시들은 자체 딥러닝 API를 구축하는 템플릿/스타트 포인트 역할을 합니다. 코드를 확장하고 API 엔드포인트의 확장성과 견고성이 얼마나 필요한지에 따라 코드를 맞춤화할 수 있습니다.

특히, 아래 항목들을 배울 수 있습니다 :
- 인퍼런스(inference)에 효율적으로 사용되도록 Keras 모델을 메모리에 불러오는(혹은 불러오지 않는) 방법
- Flask 웹 프레임워크를 사용하여 API 엔드포인트를 만드는 방법
- JSON-ify 모델을 사용하여 예측하고 클라이언트에게 결과를 반환하는 방법
- cURL과 python을 사용하여 Keras REST API를 호출하는 방법

이번 튜토리얼이 끝날 때, Keras REST API를 만드는데 사용되는 구성 요소들을 잘 이해할 것입니다.

이번 튜토리얼에서 제시한 코드를 자신만의 딥러닝 REST API의 스타트 포인트로 자유롭게 사용하세요.

**참고 : 이번 글에서 다루는 방법은 교육용입니다. 고로 생산 수준이나 과부하 상태에서 확장할 수 있는 건 아닙니다. 만약 메세지 큐와 배치 기능을 활용한 Keras REST API를 해보고 싶다면 [이 튜토리얼](https://www.pyimagesearch.com/2018/01/29/scalable-keras-deep-learning-rest-api/)을 참조하세요.**  

---
#### 개발 환경 구축
우선 Keras가 컴퓨터에 이미 구성 / 설치되어 있다고 가정하려 합니다. 만약 아닐 경우, [공식 설치 지침](https://keras.io/#installation)에 따라 Keras를 설치하세요.

여기서부터, python 웹 프레임워크인 [Flask](http://flask.pocoo.org/)를 설치해야 API 엔드포인트를 구축할 수 있습니다. 또한, API도 사용할 수 있도록 [요청](http://docs.python-requests.org/en/master/)이 필요합니다.

관련 pip 설치 명령어는 다음과 같습니다. 

```bash
    $ pip install flask gevent requests pillow
```

#### Keras REST API 설계
우리의 Keras REST API는 `run_keras_server.py`라는 단일 파일에 자체적으로 포함되어 있습니다. 단순화를 위해 단일 파일안에 설치하도록 했습니다. 구현도 쉽게 모듈화 할 수 있습니다.

`run_keras_server.py`에서 3가지 함수를 발견하실 수 있습니다 :
- `load_model` : 학습된 Keras 모델을 불러오고 인퍼런스를 위해 준비하는데 사용합니다.  
- `prepare_image` : 이 함수는 예측을 위해 입력 이미지를 신경망을 통해 전달하기 전에 작동합니다. 만약 이미지 데이터로 작업하지 않는다면, 파일 이름이 더 일반적인 `prepare_datapoint`로 변경하고 필요한 경우 스케일링/정규화를 적용하는 것이 좋습니다.  
- `predict` : 요청에서 수신 데이터를 분류하고 결과를 클라이언트에게 반환할 API의 실제 엔트포인트입니다.  

이번 튜토리얼의 전체 코드는 [이곳](https://github.com/jrosebr1/simple-keras-rest-api)에서 보실 수 있습니다.

```python
# 필수 패키지를 import합니다.
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io

# Flask 애플리케이션과 Keras 모델을 초기화합니다.
app = flask.Flask(__name__)
model = None
```  

첫 번째 코드 조각은 필요한 패키지를 가져오고 Flask 애플리케이션과 Keras 모델을 초기화합니다.  

아래는 `load_model` 함수 정의입니다.

```python
def load_model():
    # 미리 학습된 Keras 모델을 불러옵니다(여기서 우리는 ImageNet으로 학습되고 
    # Keras에서 제공하는 모델을 사용합니다. 하지만 쉽게 하기위해
    # 당신이 설계한 신경망으로 대체할 수 있습니다.)
    global model
    model = ResNet50(weights="imagenet")
```
함수 이름에서 알 수 있듯이, 이 함수는 신경망을 인스턴스화하고 디스크에서 가중치를 불러오는 역할을 띕니다.

단순하게 하기 위해, ImageNet 데이터 세트로 미리 학삽된 ResNet50 구조를 활용하려 합니다.  

클라이언트로부터 오는 데이터를 예측하기 전에, 사전에 데이터를 준비하고 처리하는 과정이 필요합니다.

```python
def prepare_image(image, target):
    # 만약 이미지가 RGB가 아니라면, RGB로 변환해줍니다.
    if image.mode != "RGB":
        image = image.convert("RGB")

    # 입력 이미지 사이즈를 재정의하고 사전 처리를 진행합니다.
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # 처리된 이미지를 반환합니다.
    return image
```
이 함수는 :
- 입력 이미지를 받고
- (필요하다면) RGB로 이미지를 변환하고
- 224x224 픽셀 사이즈로 이미지를 재정의하고 (ResNet의 입력 차원에 맞게)
- 평균 감산과 스케일링을 통해 배열 사전 처리합니다.

다시 말해, 모델을 통해 입력 데이터를 전달하기 전에 필요한 사전 처리, 스케일링, 정규화를 기반으로 함수를 수정해야 합니다.

이제 `predict` 함수를 정의할 준비가 됐습니다. 이 함수는 `/predict` 엔트포인트로 어떤 요청들을 처리합니다.

```python
@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(224, 224))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict(image)
            results = imagenet_utils.decode_predictions(preds)
            data["predictions"] = []

            # loop over the results and add them to the list of
            # returned predictions
            for (imagenetID, label, prob) in results[0]:
                r = {"label": label, "probability": float(prob)}
                data["predictions"].append(r)

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)
```
`data` 딕셔너리는 클라이언트에게 반환하길 희망하는 데이터를 저장하는데 사용합니다. 이 함수엔 예측의 성공 여부를 나타내는 부울을 가지고 있습니다. 또한, 이 딕셔너리를 사용하여 들어오는 데이터에 대한 예측 결과를 저장합니다.

들어오는 데이터를 승인하기위해 다음 사항을 확인해야 합니다:

- 요청 방법은 POST(이미지, JSON, 인코딩된 데이터 등을 포함하여 엔트포인트로 임의의 데이터를 보낼수 있도록 함)입니다.
- POST를 통해 이미지가 파일 속성으로 전달되었습니다. 

그런 다음 그 데이터를 가지고 다음을 진행합니다:
- PIL 형식으로 읽어옵니다.
- 사전 처리를 처리합니다.
- 신경망을 통해 데이터를 전달합니다.
- 결과를 반복하고 그 결과들을 각각 `data["predictions"]`에 추가합니다.
- JSON 형태으로 클라이언트에게 응답을 반환합니다.

```python
# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    app.run()
```

#### REST API에서 Keras 모델을 불러오지 않는 방법

```python
# ensure an image was properly uploaded to our endpoint
if request.method == "POST":
    if request.files.get("image"):
        # read the image in PIL format
        image = request.files["image"].read()
        image = Image.open(io.BytesIO(image))

        # preprocess the image and prepare it for classification
        image = prepare_image(image, target=(224, 224))

        # load the model
        model = ResNet50(weights="imagenet")

        # classify the input image and then initialize the list
        # of predictions to return to the client
        preds = model.predict(image)
        results = imagenet_utils.decode_predictions(preds)
        data["predictions"] = []
```


#### Keras REST API를 시작하기

```bash
$ python run_keras_server.py
Using TensorFlow backend.
 * Loading Keras model and Flask starting server...please wait until server has fully started
...
 * Running on http://127.0.0.1:5000
```



![Not Found](https://raw.githubusercontent.com/KerasKorea/KEKOxTutorial/master/media/32_0.png)


![Method Not Allowed](https://raw.githubusercontent.com/KerasKorea/KEKOxTutorial/master/media/32_1.png)

#### cURL을 사용해서 Keras REST API 테스트하기


![beagle](https://raw.githubusercontent.com/KerasKorea/KEKOxTutorial/master/media/32_2.jpg)

```bash
$ curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
{
  "predictions": [
    {
      "label": "beagle",
      "probability": 0.9901360869407654
    },
    {
      "label": "Walker_hound",
      "probability": 0.002396771451458335
    },
    {
      "label": "pot",
      "probability": 0.0013951235450804234
    },
    {
      "label": "Brittany_spaniel",
      "probability": 0.001283277408219874
    },
    {
      "label": "bluetick",
      "probability": 0.0010894243605434895
    }
  ],
  "success": true
}
```

#### Keras REST API 프로그래밍 방식 사용

```python
# import the necessary packages
import requests

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://localhost:5000/predict"
IMAGE_PATH = "dog.jpg"

# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

# submit the request
r = requests.post(KERAS_REST_API_URL, files=payload).json()

# ensure the request was successful
if r["success"]:
    # loop over the predictions and display them
    for (i, result) in enumerate(r["predictions"]):
        print("{}. {}: {:.4f}".format(i + 1, result["label"],
            result["probability"]))

# otherwise, the request failed
else:
    print("Request failed")
```

```bash
$ python simple_request.py
1. beagle: 0.9901
2. Walker_hound: 0.0024
3. pot: 0.0014
4. Brittany_spaniel: 0.0013
5. bluetick: 0.0011
```

---

### 참고
* [PyImageSearch](https://www.pyimagesearch.com/)
* [Flask 웹 프레임워크](http://docs.python-requests.org/en/master/)

> 이 글은 2018 컨트리뷰톤에서 [`Contributue to Keras`](https://github.com/KerasKorea/KEKOxTutorial) 프로젝트로 진행했습니다.  
> Translator : [mike2ox](https://github.com/mike2ox) (Moonhyeok Song)  
> Translator Email : <firefinger07@gmail.com>