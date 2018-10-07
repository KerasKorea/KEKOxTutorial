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

###
