## Keras 와 Gym 과 함께하는 Deep Q-Learning 을 향한 여행 🛫(My Journey Into Deep Q-Learning with Keras and Gym
[원문 링크](https://medium.com/@gtnjuvin/my-journey-into-deep-q-learning-with-keras-and-gym-3e779cc12762)
> 이 튜토리얼은

* Keras
* Deep Reinforcement Learning
* Deep Q-Learning
* Gym

### Introduction
![CartPole Game](./media/133_1.gif)
*CartPole 게임*

<br></br>

이 글은 오래된 게임인 **CartPole** 을 해보기위해 적용된 **Deep Reinforcement Learning(Deep Q-Learning)** 을 구현하는 방법을 보여줍니다.

작업을 용이하게 하기 위해 두 가지 도구를 사용했습니다.

* [OpenAI Gym](https://gym.openai.com): 많은 오래된 비디오 게임 환경과의 상호작용을 위한 간단한 인터페이스를 제공합니다(Atari 게임 콜렉션은 아주 훌륭합니다).
* [Keras](https://keras.io): 인기절정의 Deep Learning 라이브러리

결국 우리는 100줄 미만의 코드로 스스로 학습하는 "AI"를 만들게 될 것입니다. 🤖

또한 나는 독자들이 Deep Reinforcement Learning에 대한 사전 요구 조건을 갖추지 않아도 되도록 설명해 줄 것 입니다.

이 튜토리얼에 있는 코드는 [*Github*](https://github.com/GaetanJUVIN/Deep_QLearning_CartPole)에서 가져왔습니다.

<br></br>
<br></br>

### 강화학습은 무엇일까? (What is Reinforcement Learning?)

강화학습(Reinforcement Learning)은 기계학습(Machine Learning)의 한 종류입니다. 이를 통해 환경(입력/출력)에서 상호 작용하여 학습하는 AI 에이전트(Agent)를 만들 수 있습니다. AI 에이전트는 [시행착오](https://en.wikipedia.org/wiki/Trial_and_error)를 통해 배울 것입니다. 많은 노력 끝에, 환경에서 성공할 수 있는 충분한 경험을 갖게 될 것입니다.

이런 종류의 기계 학습은 인간의 학습 방식과 매우 유사합니다. 예를 들어, 이것은 우리가 걷는 법을 배울 때와 같습니다. 우리는 한 발을 다른 발 앞에 놓으려고 여러 번 시도하지만, 우리가 걷는 데 성공하는 것은 많은 실패와 관찰 후에만 가능합니다.

<br></br>

![강화학습](./media/134_2.png)
*figure1 : 강화학습이란?*

<br></br>
<br></br>

### Deep Reinforcement Learning 이란? (What is Deep Reinforcement Learning?)
[구글의 딥마인드(DeepMind)](https://deepmind.com/blog/deep-reinforcement-learning/)는 [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)이라는 유명한 논문을 발표했습니다.

<br></br>

![deepmind_logo](./media/134_3.jpeg)
*figure2 : DeepMind 로고*

<br></br>

2013년 말, 구글은 **Deep Q Network(DQN)** 라는 새로운 알고리즘을 선보였습니다. AI 에이전트가 화면을 관찰하는 것만으로 어떻게 게임을 배울 수 있는지를 보여줬습니다. AI 에이전트는 게임에 대한 사전 정보를 받지 않고도 게임을 어떻게 하는지 배울 수 있습니다.

그것은 상당히 인상적이었으며 이 논문은 **딥러닝** 과 **강화학습** 이 혼합된 **'Deep Reinforcement Learning'** 이라는 새로운 시대를 열게 되었습니다.

[클릭해서 보세요: <U>DeepMind의 Atari Player</U>](https://www.youtube.com/watch?v=V1eYniJ0Rnk)

Deep Q Network 알고리즘에서 신경망은 환경을 기반으로 최고의 동작을 수행하는 데 사용됩니다(일반적으로 "State"라고 합니다).

우리는 **Q function** 이라 불리는 function을 가지고, 이 funtion은 State를 기반으로 잠재적인 보상을 추정하는 데 사용됩니다. 우리는 그것을 Q(State, Action)라고 부릅니다. 여기서 Q는 `State` 및 `Action`을 기준으로 예상되는 미래 값을 계산하는 function 입니다.

<br></br>
<br></br>

### 어린시절 하던 오래된 게임 CartPole (An old game from our childhood: CartPole)
이 게시물에서는 에이전트에게 복잡한 게임을 교육하는 데 시간이 좀 걸릴 수 있기 때문에 "단순한" 게임을 선택했습니다(몇 시간에서 하루 종일).

**CartPole** 의 목표는 움직이는 카트 위에 있는 폴의 균형을 맞추는 것입니다.

나는 게임 시뮬레이터인 `OpenAI Gym`이라는 도구를 사용할 것입니다. 따라서 픽셀 정보를 제공하는 대신 사용 가능한 변수(State, 폴 각도, 카트 위치 등)를 제공합니다.

게임과의 상호작용을 위해, 우리 에이전트는 카트에 0 또는 1의 일련의 동작을 수행하여 카트를 왼쪽이나 오른쪽으로 밀어서 옮길 수 있습니다.

Gym은 게임 환경과의 모든 상호작용을 단순화함으로써 AI 에이전트의 "두뇌"에 초점을 맞추도록 합니다.

```python
# INPUT
# action은 0 또는 1

# OUTPUT
# 다음 State, 보상, 정보 : 우리가 무엇을 위해서 학습하는지에 대한 것
# done : 게임이 끝났는지 아닌지에 대한 boolean 타입의 값
next_state, reward, done, info = env.step(action)
```

<br></br>
<br></br>

### 단순 신경망을 사용하기 위한 Keras 사용 (Using Keras To Implement a Simple Neural Network)

![케라스자랑](./media/134_4.jpeg)
*figure3 : 인기 많은 TensorFlow, CNTK or Theano 처럼 Keras.io 는 high-level 의 신경망 API 입니다.*

이 글은 **딥러닝** 이나 **신경망** 에 관한 것이 아닙니다. 따라서 **신경망** 은 입력을 출력에 맵핑하는 블랙 박스 알고리즘으로 간주할 것입니다.

**신경망** 은 기본적으로 데이터 쌍(입력 및 출력 데이터)을 기반으로 학습하고, 특정 유형의 패턴을 탐지하고, 또 다른 입력 데이터를 기반으로 출력을 예측하는 알고리즘입니다.

<br></br>

![신경망](./media/134_5.png)
*figure4 : 3개의 입력, 1개의 히든 레이어, 2개의 출력*

<br></br>

우리가 이 포스트에서 사용할 **신경망** 은 figure4와 유사합니다. 그것은 4개의 정보를 받는 입력 레이어와 3개의 히든 레이어를 가질 것입니다. 그리고 게임 버튼이 2개(0과 1)이므로 출력 레이어에 노드가 2개 있을 것입니다.

**Keras** 는 Python으로 작성되었으며 TensorFlow, CNTK 또는 Theano의 위에서 실행할 수 있는 높은 수준의 신경 네트워크 API입니다. "이것은 빠른 실험을 가능하게 하는 것에 초점을 맞추어 개발되었습니다. 아이디어에서 최소의 지연으로 결과를 얻을 수 있는 것이 좋은 연구를 하는 열쇠입니다."

**Keras** 는 기본적인 **신경망** 을 구현하는 것을 정말 간단하게 만듭니다.

<br></br>
<br></br>

### 0# Initialization
