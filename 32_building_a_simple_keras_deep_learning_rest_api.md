## Keras 모델을 REST API로 배포해보기(Building a simple Keras + deep learning REST API)
[원문](https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html)
> 이 글은 Adrian Rosebrock이 작성한 안내 게시글로 Keras 모델을 REST API로 제작하는 간단한 방법을 안내하고 있습니다.

* Keras
* REST API
* flask

### 개요
이번 튜토리얼에서, 우리는 케라스 모델을 REST API로 배포하는 간단한 방법을 설명합니다.

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
    # view로부터 반환될 데이터 딕셔너리를 초기화합니다.
    data = {"success": False}

    # 이미지가 엔트포인트에 올바르게 업로드 되었는디 확인하세요
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # PIL 형식으로 이미지를 읽어옵니다.
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # 분류를 위해 이미지를 사전 처리합니다.
            image = prepare_image(image, target=(224, 224))

            # 입력 이미지를 분류하고 클라이언트로부터 반환되는 예측치들의 리스트를 초기화 합니다.
            preds = model.predict(image)
            results = imagenet_utils.decode_predictions(preds)
            data["predictions"] = []

            # 결과를 반복하여 반환된 예측 목록에 추가합니다.
            for (imagenetID, label, prob) in results[0]:
                r = {"label": label, "probability": float(prob)}
                data["predictions"].append(r)

            # 요청이 성공했음을 나타냅니다.
            data["success"] = True

    # JSON 형식으로 데이터 딕셔너리를 반환합니다.
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

만약 이미지가 아닌 데이터로 작업한다면, `request.files` 코드를 삭제하고 원본 입력 데이터를 직접 구문 분석하거나 `request.get_json()`로 입력 데이터를 python 딕셔너리/객체에 자동으로 구문 분석되도록 해야합니다.

추가로, [참조한 튜토리얼](https://scotch.io/bar-talk/processing-incoming-request-data-in-flask)은 Flask의 요청 객체의 기본 요소를 설명하는 내용을 읽어 보세요.

이제 서비스를 시작하겠습니다.

```python
# 실행에서 메인 쓰레드인 경우, 먼저 모델을 불러온 뒤 서버를 시작합니다.
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    app.run()
```

우선 `load_model`로 디스크에서 Keras 모델을 불러옵니다.

`load_model` 호출은 차단 작업이며 모델을 완전히 불러올 때까지 웹서비스가 시작되지 않도록 합니다. 웹 서비스를 시작하기 전에 모델을 메모리로 완전히 불러오고 인퍼런스를 할 준비가 되지 않았다면 아래와 같은 상황이 발생할 수 있습니다:

1. POST형식으로 서버에 요청합니다.
2. 서버가 요청을 받고 데이터를 사전 처리하고 그 데이터를 모델을 통해 전달하도록 시도합니다.
3. *만약 모델을 완전히 불러오지 않았다면, 에러가 발생할 겁니다.*

당신만의 Keras REST API를 설계할 때, 요청들을 승인하기 전에 인퍼런스를 위한 준비가 되었는지 모델이 불러와졌는지를 보장하는 논리가 들어가 있는지 확인하셔야 합니다.

#### REST API에서 Keras 모델을 불러오지 않는 방법

예측 함수에 있는 당신의 모델을 불러오는 걸 시도할 수 있습니다. 아래 코드를 참조하세요:

```python
# 이미지가 엔트포인트에 올바르게 업로드되었는지 확인하세요
if request.method == "POST":
    if request.files.get("image"):
        # PIL 형태로 이미지를 읽어옵니다.
        image = request.files["image"].read()
        image = Image.open(io.BytesIO(image))

        # 분류를 위해 이미지를 사전 처리합니다.
        image = prepare_image(image, target=(224, 224))

        # 모델을 불러옵니다.
        model = ResNet50(weights="imagenet")

        # 입력 이미지를 분류하고 클라이언트로부터 반환되는 예측치들의 리스트를 초기화 합니다.
        preds = model.predict(image)
        results = imagenet_utils.decode_predictions(preds)
        data["predictions"] = []
```
위 코드는 새로운 요청이 들어올 때마다 모델을 불러온다는 의미를 가집니다. 이는 믿기 힘들 정도로 비효율적이고 여러분 시스템의 메모리가 부족해질 수도 있습니다.

만약 위 코드를 실행하려고 하면 API 실행속도가 상당히 느릴겁니다.(특히 모델이 큰 경우) 이는 각 모델을 불러오는데 사용된 I/O 및 CPU 작업의 상당한 오버헤드(overhead)때문에 발생합니다.

어떻게 당신 서버의 메모리를 쉽게 압도하는지 알아보기 위해, 동시에 서버로 N개의 입력 요청이 있다고 가정해 봅시다. 이는 N개의 모델을 메모리로 불러오는 것을 의미합니다. 만약 ResNet처럼 큰 모델일 경우, 모델의 N개 사본을 RAM에 저장하면 시스템 메모리가 쉽게 소진될 수 있습니다.

이를 해결하기 위해, 매우 구체적이고 정당한 이유가 없는 한 새로 들어오는 요청에 대해 새 모델 인스턴스를 로드하지 않도록 시도하십시오.

**경고** : 단일 스레드인 기본 Flask 서버를 사용한다고 가정합니다. 멀티 스레드 서버에서 배포할 경우, 이 글에서 앞서 설명한 "정확한" 방법을 사용하더라도 여러 모델을 메모리에 로드하는 상황에 놓일 수 있습니다. 만약 Apache나 nginx같은 서버를 사용하려면, [이곳](https://www.pyimagesearch.com/2018/01/29/scalable-keras-deep-learning-rest-api/)에서 설명하는 대로 파이프라인을 더 확장하는게 좋습니다.

#### Keras REST API를 시작하기

Keras REST API 서비스를 시작하는건 쉽습니다.

터미널을 열어서 아래를 실행해보세요:

```bash
$ python run_keras_server.py
Using TensorFlow backend.
 * Loading Keras model and Flask starting server...please wait until server has fully started
...
 * Running on http://127.0.0.1:5000
```

결과물에서 볼 수 있듯이, 모델을 먼저 불러옵니다. 그런 수, Flask 서버를 시작할 수 있습니다.

이제 http://127.0.0.1:5000을 통해 서버에 엑세스할 수 있습니다.

그러나, IP 주소 + 포트를 복사하여 브라우저에 붙여넣으려면 다음 이미지가 표시됩니다.

![Not Found](https://raw.githubusercontent.com/KerasKorea/KEKOxTutorial/master/media/32_0.png)  

그 이유는 Flask URL 경로에 색인/홈페이지 세트가 없기 때문입니다.

대신에, 브라우저를 통해 `/predict` 엔트포인트에 액세스해 보세요.

![Method Not Allowed](https://raw.githubusercontent.com/KerasKorea/KEKOxTutorial/master/media/32_1.png)  

그리고 "Method Not Allowed"(방법 허용되지 않음) 오류가 표시됩니다. 해당 오류는 브라우저에서 GET 요청을 수행하지만 `/predict`는 POST만 허용하기 때문에 발생합니다. (다음 섹션에서 수행하는 방법을 보여드리려 합니다.)

#### cURL을 사용해서 Keras REST API 테스트하기
Keras REST API를 테스트하고 디버깅할 때는 [cURL](https://curl.haxx.se/)을 사용하는 것을 고려하세요.(사용법을 배우기에 좋은 툴입니다.)

아래에서 분류하고 싶은 이미지(ex. *개*)보다 구체적으로 *비글*을 보실 수 있을 겁니다.


![beagle](https://raw.githubusercontent.com/KerasKorea/KEKOxTutorial/master/media/32_2.jpg)  

*curl*을 사용해서 API로 이미지를 전달할 수 있고 ResNet이 생각하는 이미지 내용을 확인할 수 있습니다.

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

`-X` 플래그와 `POST` 값은 POST 요청을 수행하고 있음을 말해줍니다.

`-F image=@dog.jpg`를 제공하여 인코딩된 데이터를 제출합니다. 그 후, `image`는 `dog.jpg`파일로 설정됩니다. `dog.jpg`보다 먼저 `@`를 제공하는 것은 cURL이 이미지의 내용을 로드하고 요청에 데이터를 전달하기를 원한다는 것을 의미합니다.

끝으로, 엔트포인트를 얻게 됩니다 : `http://localhost:5000/predict`

어떻게 입력 이미지가 99.01%의 신뢰도로 *"비글"*을 올바르게 분류하는지 보세요. 나머지 상위 5개 예측치 및 관련 확률도 Keras API의 응답에 포함됩니다.

#### Keras REST API 프로그래밍 방식 사용

아마도, Keras REST API에 데이터를 *제출*한 다음 반환된 예측치를 출력합니다. 이렇게 하려면 서버로부터 응답을 프로그래밍 방식으로 처리해야 합니다.

이는 python 패키지인 [`requests`](http://docs.python-requests.org/en/master/)를 이용하면 간단한 프로세스 입니다.

```python
# 필수 패키지를 불어옵니다.
import requests

# Keras REST API 엔드포인트의 URL를 입력 이미지 경로와 같이 초기화 합니다.
KERAS_REST_API_URL = "http://localhost:5000/predict"
IMAGE_PATH = "dog.jpg"

# 입력 이미지를 불러오고 요청에 맞게 페이로드(payload)를 구성합니다.
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

# 요청을 보냅니다.
r = requests.post(KERAS_REST_API_URL, files=payload).json()

# 요청이 성공했는지 확인합니다.
if r["success"]:
    # 예측을 반복하고 이를 표시합니다.
    for (i, result) in enumerate(r["predictions"]):
        print("{}. {}: {:.4f}".format(i + 1, result["label"],
            result["probability"]))

# 그렇지 않다면 요청은 실패합니다.
else:
    print("Request failed")
```

`KERAS_REST_API_URL`은 엔드포인트를 지정하는 반면 `IMAGE_PATH`는 디스크에 상주하는 입력 이미지의 경로입니다.

`IMAGE_PATH`를 사용하여 이미지를 불러온 다음 요청에 대한 `페이로드`를 구성합니다.

`페이로드`가 있을 경우 `requests.post`를 호출하여 데이터를 엔드포인트에 POST할 수 있습니다. 호출 끝에 `.json()`을 추가하면 다음과 같은 `requests`가 표시됩니다.

1. 서버로부터 온 응답이 JSON에 있어야 합니다.
2. JSON 객체가 자동으로 구문 분석 및 추상화되기를 원합니다.

요청 결과가 `r`이면 분류가 성공(혹은 실패)인지 확인한 다음 `r["predictions"]`을 반복할 것입니다.

`simple_request.py`를 실행하기 위해, 먼저 `run_keras_server.py`(즉 Flask 웹서버)가 현재 실행되고 있는지 확인하세요. 여기서, 별도의 셸에서 다음 명령을 실행합니다.

```bash
$ python simple_request.py
1. beagle: 0.9901
2. Walker_hound: 0.0024
3. pot: 0.0014
4. Brittany_spaniel: 0.0013
5. bluetick: 0.0011
```

성공적으로 Keras REST API를 불러왔고 python을 통해 모델의 예측치들도 얻었습니다.  

---

이 글에서 아래 항목들을 수행하는 방법을 배웠습니다:

- Keras 모델을 Flask 웹 프레임워크를 사용해 REST API로 감싸는 법
- cURL을 활용하여 데이터를 API로 전송하는 법
- python과 `requests`패키지를 사용하여 엔드포인트로 데이터를 보내고 결과를 출력하는 법.

이번 튜토리얼에 쓰인 코드는 [여기](https://github.com/jrosebr1/simple-keras-rest-api)에서 보실 수 있고, 고유한 Keras REST API용 템플릿으로 사용하실 수 있습니다.(원하는 대로 수정할 수 있습니다.)

명심하세요. 이 글에 있는 코드는 교육용입니다. 대량의 호출 및 수신 요청에 따라 규모를 확장할 수 있는 생산 수준이 아닙니다.

이 방법은 다음 경우에 가장 적합합니다:

1. Keras 딥러닝 모델을 위한 REST API를 빠르게 설정할 필요가 있을 때
2. 엔드포인트가 크게 타격을 받지 않을 때

메세지 큐 및 배치 기능을 활용하는 고급 Keras REST API에 관심이 있을 경우, [이 글](https://www.pyimagesearch.com/2018/01/29/scalable-keras-deep-learning-rest-api/)을 참조하세요. 

만약 질문이나 의견이 있다면 [PyImageSearch](https://www.pyimagesearch.com/)에서 Adrian에게 연락하세요. 앞으로 다뤄야할 주제에 대한 제안일 경우 Twitter에서 [Francois](https://twitter.com/fchollet)를 찾아보세요.

### 참고
* [PyImageSearch](https://www.pyimagesearch.com/)
* [Flask 웹 프레임워크](http://docs.python-requests.org/en/master/)

> 이 글은 2018 컨트리뷰톤에서 [`Contributue to Keras`](https://github.com/KerasKorea/KEKOxTutorial) 프로젝트로 진행했습니다.  
> Translator : [mike2ox](https://github.com/mike2ox) (Moonhyeok Song)  
> Translator Email : <firefinger07@gmail.com>
