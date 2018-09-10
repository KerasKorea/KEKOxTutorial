# 딥러닝(CycleGAN)을 이용해 Fornite 를 PUBG 로 바꾸기
(Turning Fortnite into PUBG with Deep Learning (CycleGAN))
[원문 링크](https://towardsdatascience.com/turning-fortnite-into-pubg-with-deep-learning-cyclegan-2f9d339dcdb0)
> 이 문서는 `CycleGAN` 을 이용해 Image Style Trasfer 를 게임 배경에 적용해봅니다. 원작자의 튜토리얼에 대한 부가설명은 `인용구` 를 이용해서 표현할 것입니다.

* Keras
* GANs
* CycleGAN
* Style Transfer

### introduction
Image Style Trasfer 를 위한 CycleGAN 이해 및 게임용 그래픽 모듈에 대한 적용 탐색을 할 것 입니다.

![intro_gif](https://cdn-images-1.medium.com/max/1600/1*8h--ozdvpCR3__lOwUkAtw.gif)
신경망은 PUBG 의 시각적 스타일로 Fortnite 를 재창조하려고 시도합니다.

<br></br>
만약 여러분이 게이머라면, 여러분은 미친듯한 인기를 누리고 있는 Battle Royale 게임인 Fortnite 와 PUBG에 대해 들어봤을 것입니다. 두 게임 모두 100명의 선수들이 단지 한 명의 생존자가 남아있을 때까지 작은 섬에서 경기를 하는 방식이 매우 유사합니다.

저는 Fortnite 의 게임 플레이를 좋아하지만 PUBG 의 더 현실적인 시각화를 더 좋아합니다. 이것이 저를 생각하게 만들었죠. 게임 개발자들이 우리에게 그 옵션을 제공할 필요 없이 우리가 좋아하는 시각 효과를 선택할 수 있는 게임용 그래픽 모드를 가질 수 있을까? 만약 PUBG 의 비주얼을 Fortnite 의 프레임 렌더링할 수 있는 방법이 있다면 어떨까요?

그 때 저는 딥러닝이 도움이 될 수 있는지 알아보기로 결심했습니다. 그리고 저는 CycleGAN 이라고 불리는 신경 네트워크를 찾게 되었습니다. 이 글에서는 CycleGANs 의 작동 방식을 검토하고 Fortnite 를 PUBG 의 스타일로 시각적인 변환을 시도해보겠습니다.


#### 소제목

#### 소제목

### 참고문서
* [참고 사이트 1]()
* [참고 사이트 2]()

----
