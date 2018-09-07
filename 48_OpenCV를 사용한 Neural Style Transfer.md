## OpenCV를 사용한 Neural Style Transfer(Neural Style Transfer with OpenCV)
[원문 링크](https://www.pyimagesearch.com/2018/08/27/neural-style-transfer-with-opencv/)
> 이 문서는 Neural Style Transfer 를 하는 방법을 `Keras` 와 `OpenCV` 를 이용해서 보여줍니다. 많은 예제들이 content 이미지에 style 이미지의 style 을 합치지만, 이 튜토리얼에서는 OpenCV 를 사용해서 content 이미지 뿐만 아니라 실시간으로 촬영되는 비디오에도 style 이미지의 style 을 합칩니다.

* 케라스
* Neural Style transfer

<br>![](https://s3-us-west-2.amazonaws.com/static.pyimagesearch.com/opencv-neural-style/neural_style_transfer_animation.gif)</br>

### Introduction
이 튜토리얼에서, 당신은 neural style transfer 를 OpenCV, 파이썬, 딥러닝을 이용해서 이미지 뿐만 아니라 실시간으로 촬영되는 비디오에도 적용해볼 수 있을거에요. 튜토리얼이 끝날 때 쯤, 당신은 neural style transfer 를 이용한 아주 아름다운 작품을 만들 수 있을 겁니다.

오리지널 neural style transfer 알고리즘은 2015년에 Gatys 와 몇몇에 의해 그들의 논문인 [`A Neural Algorithm of Artistic Style`](https://arxiv.org/abs/1508.06576) 소개되었습니다.

2016 년에 Johnson 과 몇몇이 실시간 Perceptual Losses for Real-Time Style Transfer and Super- Resolution (Style Trasfer 및 Super-Resolution 를 위한 perceptual 손실)을 발표했는데, 이는 perceptual 손실을 사용하는 Super-Resolution 문제를 Neural Style Transfer 에 적용한 것입니다.
결과는  Gatys 등이 발표했던 neural style transfer 알고리즘 방법보다 최대 3배 정도 빠르다는 것입니다(그러나 몇 가지 단점이 있으며, 이 가이드에서 나중에 논의할 예정입니다).

이 포스트의 마지막엔 당신은 neural style transfer 알고리즘은 당신의 이미지와 비디오 스트림에 어떻게 적용하는지 알 수 있을 것입니다.

[Nueral style transfer with OpenCV 데모 영상](https://youtu.be/DRpydtvjGdE)

> 위의 데모 영상은 이 튜토리얼이 끝난 후에 우리가 어떤 것을 배우게 되었는지에 대해 잘 보여주는 영상입니다. 한 번 보시는 것을 추천합니다!
오늘 가이드의 나머지 부분에서는 OpenCV 및 Python을 사용하여 자신의 예술 작품을 생성하기 위해 신경 스타일 전송 알고리즘을 적용하는 방법을 시연합니다.

오늘 가이드의 나머지 부분에서는 OpenCV 및 Python을 사용하여 자신의 예술 작품을 생성하기 위해 Neural style transfer 알고리즘을 적용하는 방법을 시연합니다.

제가 오늘 여기서 논의하는 방법은 CPU에서 거의 실시간으로 실행될 수 있으며 GPU에서 완전히 실시간 성능을 얻을 수 있습니다.

우리는 Neural style transfer 에 대해서 그것이 무엇이고 어떻게 작동하는지를 포함하는 간단한 논의를 시작할 것입니다.

여기서부터 우리는 OpenCV와 Python을 이용하여 실제로 Neural style transfer 를 적용할 것입니다.

<br></br><br></br>

### Neural style transfer 란 무엇일까?
![Figure 1](https://www.pyimagesearch.com/wp-content/uploads/2018/08/neural_style_transfer_example.jpg)
<center>figure1:OpenCV 를 사용한 Neural style transfer 의 예. content 이미지 (왼쪽). Style 이미지 (중앙). 스타일화 된 결과(Stylized output) (오른쪽). </center>

<br></br>

Neural style transfer 는 다음의 프로세스입니다:

1. 어떤 한 이미지의 스타일을 가져온다.
2. 그리고 그 스타일을 다른 이미지에 적용한다.

Neural style transfer 의 프로세스는 **Figure1** 에서 확인할 수 있습니다. **왼쪽** 사진은 우리의 content 이미지입니다. 독일의 Black Forest 의 산 저상에서 맥주를 즐기고 있는 나의 모습입니다.

**중앙** 에 위치한 사진은 우리의 스타일 이미지입니다. [빈센트 반 고흐](https://en.wikipedia.org/wiki/Vincent_van_Gogh)의 유명한 그림인 별이 빛나는 밤이죠.

그리고 **오른쪽** 은 반 고흐의 별이 빛나는 밤의 스타일을 content 이미지인 내 사진에 적용한 결과입니다. 어떻게 언덕, 숲, 나, 그리고 심지어 맥주의 내용까지 보존했는지 보세요. 저것들을 보존하면서도 별의 빛나는 밤의 스타일을 적용되었습니다. 마치 반 고흐가 그의 뛰어난 페인트 스트로크를 산 위에서의 경치에 바친 것 같습니다!

그래서 질문은 흠.. 우리가 Neural style transfer 를 하는 뉴럴 네트워크를 어떻게 정의할까요?

가능한 일이긴 한걸까요?

물론 가능합니다! 그리고 우리는 다음 섹션에서 Neural style transfer 가 어떻게 가능한지에 대해 토론할 것입니다.

<br></br><br></br>

### Neural style transfer 는 어떻게 동작할까?

![Figure2](https://www.pyimagesearch.com/wp-content/uploads/2018/08/neural_style_transfer_gatys.jpg)
<center>Figure 2: Neural Style Transfer with OpenCV possible (Figure 1 of Gatys et. al. 2015).</center>

<br></br>

이 시점에서 여러분은 아마 머리를 긁적이며 "우리가 어떻게 신경망을 정의해서 스타일 전달을 할 수 있을까?"라는 생각을 하고 있을 것입니다.

흥미롭게도, 2015년에 [Gatys 등이 작성한 논문](https://arxiv.org/abs/1508.06576)은 새로운 구조를 전혀 필요로 하지 않는 Neural style transfer 알고리즘을 제안했습니다! 대신 미리 학습된 네트워크( pre-trained network, 일반적으로 ImageNet)를 사용하고 스타일 전송의 최종 목표를 달성하기 위해 필요한 손실 함수를 정의합니다.

**그러면 질문은 "어떤 뉴럴 네트워크를 우리가 써야할까" 가 아니라 "어떤 손실 함수를 우리가 써야할까?" 겠네요.**

그에 대한 대답은 세가지 구성요소로 이야기할 수 있습니다.

1. Content loss
2. Style loss
3. Total-variation loss

각각의 구성요소는 개별적으로 계산이 된 후 한 개의 meta 손실 함수로 합쳐집니다. meta 손실 함수값을 최소화 시키기 위해서 우리는 content, style, total-variation 들의 손실을 최소화 시켜야 합니다.

Gatys 등은 아름다운 결과를 만들어냈지만 문제는 그것이 꽤 느리다는 것이었습니다.

Johnson 외 연구진 등(2016)은 Gatys 외 연구진(Gatys et al., Gatys 등)의 연구를 기반으로 했고, 최대 3배 까지 빠른 Neural style transfer 알고리즘을 제안하였습니다. Johnson 외 연구진들의 방법은 perceptual loss 함수를 기반으로하는 super-resolution 문제로 Neural style transfer 를 프레임화합니다.

Johnson 외 연구진들의 방법이 확실히 빠르지만 가장 큰 단점은 Gatys 외 연구진들의 방법에서와 같이 스타일 이미지를 임의로 선택할 수 없다는 것입니다.

대신 먼저 원하는 이미지의 스타일을 재현하기 위해 네트워크를 명시적으로 학습해야 합니다. 네트워크가 학습이 되면 당신이 원하는 어떠한 content 이미지도 네트워크에 적용할 수 있습니다. You should see the Johnson et al. method as a more of an “investment” in your style image — you better like your style image as you’ll be training your own network to reproduce its style on content images.

Johnson 외 연구진들은 그들이 어떻게 Neural style transfer 모델을 학습시켰는지에 대한 문서를 그들의 [GitHub 페이지](https://github.com/jcjohnson/fast-neural-style)에서 제공합니다.

마지막으로, 2017 년에 발표한 Ulyanov 외 연구진들의 논문인 [ Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022) 역시 주목할 가치가 있습니다. 배치 정규화를 instance normalization 으로 대체 함으로서(instance normalization 학습과 테스트 모두에 적용하였습니다.) 실시간으로 더욱 빠른 퍼포먼스와 이론적으로 더 만족스러운 결과를 이끌어 냈습니다.

나는 Johnson 외 연구진이 사용한 두 가지 모델을 ECCV 논문에 Ulyanov 외 연구진들의 모델들과 함께 이 게시물의 "다운로드" 섹션에 포함시켰다.

<br></br><br></br>

### 프로젝트 구조

프로젝트는 몇 개의 파일을 가지고 있는데, 이 프로젝트는 <strong>*"Downloads"*</strong> 섹션에서 다운로드 받을 수 있습니다.

scripts + models + images 들을 다운로드 받은 후에 `tree` 커맨드를 입력하면 아래와 같은 디렉토리 및 파일 구조를 확인할 수 있습니다.

```python
$ tree --dirsfirst
.
├── images
│   ├── baden_baden.jpg
│   ├── giraffe.jpg
│   ├── jurassic_park.jpg
│   └── messi.jpg
├── models
│   ├── eccv16
│   │   ├── composition_vii.t7
│   │   ├── la_muse.t7
│   │   ├── starry_night.t7
│   │   └── the_wave.t7
│   └── instance_norm
│       ├── candy.t7
│       ├── feathers.t7
│       ├── la_muse.t7
│       ├── mosaic.t7
│       ├── starry_night.t7
│       ├── the_scream.t7
│       └── udnie.t7
├── neural_style_transfer.py
├── neural_style_transfer_examine.py
└── neural_style_transfer_video.py

4 directories, 18 files
```

<strong>*"Downloads"*</strong> 섹션에서 .zip 파일을 다운받으면, 당신은 이 프로젝트를 위해서 온라인의 그 어떤곳에서 다른 것을 다운로드 받을 필요가 없습니다. 제가 test 에 도움이 될 이미지들을 `images/` 에, 모델들은 `models/` 에 준비를 해놓았습니다. 이 모델들은 Johnson 외 연구진들이 미리 학습시켜놓은 것입니다.
당신은 또한 세개의 파이썬 스크립트를 찾을 수 있을 것입니다.

<br></br><br></br>

### Neural Style Transfer 구현하기

이제 OpenCV와 Python으로 Neural Style Transfer 를 구현해 보겠습니다.

`neural_style_transfer.py` 파일을 열고, 아래의 코드를 넣어보세요.


```python
# 필요한 패키지들 import
import argparse
import imutils
import time
import cv2

# argument parser 를 정의하고, argument 를 파싱합니다.
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="neural style transfer model")
ap.add_argument("-i", "--image", required=True,
	help="input image to apply neural style transfer to")
args = vars(ap.parse_args())
```

첫 번째, 우리는 우리가 필요로하는 패키지들을 import 하고 커맨드 라인 arguement 를 파싱합니다.

우리가 import 할 것은 아래와 같습니다.
* [imutils](https://github.com/jrosebr1/imutils): 이 패키지는 `pip install --upgrade imutils` 로 설치가 가능합니다. 최근에 imutils==0.5.1 버전이 배포되었으니 업그레이드 하는 것을 잊지마세요!
* [OpenCV](https://opencv.org): 이 튜토리얼을 위해서 OpenCV 3.4 또는 그 이상의 버전이 필요합니다. 제가 업로드한 또 다른 튜토리얼을 이용해서 [Ubuntu](https://www.pyimagesearch.com/2018/08/15/how-to-install-opencv-4-on-ubuntu/)와 [macOS](https://www.pyimagesearch.com/2018/08/17/install-opencv-4-on-macos/) 를 위한 OpenCV 4 를 설치할 수도 있을 거에요.

우리는 또한 두 줄의 커맨드 라인 arguments 가 필요합니다.
* `--model` : Neural style transfer 모델의 path(위치) 입니다. 11 개의 미리 학습된 모델들이 <strong>*"Downloads"*</strong> 에 있습니다.
* `--image` : 우리가 스타일을 적용할 입력 이미지입니다. 이미 4 개의 샘플 이미지를 준비해뒀습니다. 튜토리얼을 진행하는데 부담갖지 마세요!

당신은 커맨드 라인 arguments 코드를 바꿀 필요가 없습니다. - arguments 는 실행시간동안 처리될 것 입니다. 만약 이런 방식이 익숙하지 않다면, [커맨드 라인 arguments + argparse](https://www.pyimagesearch.com/2018/03/12/python-argparse-command-line-arguments/) 에 대한 블로그 포스트를 한 번 읽어보세요.

이제는 재미있는 파트입니다. - 우리는 우리의 이미지와 모델을 가져올 것이고, 그 다음엔 neural style transfer 를 해볼 것입니다.
