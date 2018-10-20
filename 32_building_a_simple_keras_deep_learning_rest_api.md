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



#### REST API에서 Keras 모델을 불러오지 않는 방법

#### Keras REST API를 시작하기

#### cURL을 사용해서 Keras REST API 테스트하기

#### Keras REST API 프로그래밍 방식 사용


### 참고
* [참고 사이트 1]()
* [참고 사이트 2]()

> 이 글은 2018 컨트리뷰톤에서 [`Contributue to Keras`](https://github.com/KerasKorea/KEKOxTutorial) 프로젝트로 진행했습니다.  
> Translator : [mike2ox](https://github.com/mike2ox) (Moonhyeok Song)  
> Translator Email : <firefinger07@gmail.com>