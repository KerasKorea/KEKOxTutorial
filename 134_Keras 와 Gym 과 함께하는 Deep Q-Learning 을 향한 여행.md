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
