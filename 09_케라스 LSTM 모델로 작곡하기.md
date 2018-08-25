## 케라스 LSTM 모델로 작곡하기(How to Generate Music using a LSTM Neural Network in Keras)
[원문 링크](https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5)
> 이 문서는 순환신경망(RNN)인 `LSTM` 과 Python 음악 툴킷인 `music21` 을 이용해서 작곡을 해보는 것에 대해 설명합니다.
2018년 8월을 기준으로, 동작하지 않는 코드는 동작하지 않는 부분을 동작하도록 변형하였기 때문에 코드는 원문과 같지 않을 수 있습니다. 또한 그대로 번역한 것이 아닌 필요한 설명과 합쳐서 다시 쓴 글이기 때문에 원문과 다를 수 있습니다.
원문에서 나온 코드들을 이해를 돕기 위해 jupyter notebook 파일을 첨부합니다.

* 케라스
* LSTM
* Neural Netwroks layer

<br></br>
![intro image](https://cdn-images-1.medium.com/max/2000/1*evQj8gukICFrnBICeJvY0w.jpeg)

### Introduction
인공 신경망은 우리의 삶을 개선시키는데 이용됩니다. 그것들은 우리가 사고 싶은 물건에 대한 추천을 제공하고, [작가의 스타일을 기반으로 텍스트를 생성](http://www.cs.utoronto.ca/~ilya/pubs/2011/LANG-RNN.pdf)하며, 심지어 [이미지의 예술 스타일을 바꾸는 데 사용](https://arxiv.org/pdf/1508.06576.pdf)되기도 합니다. 최근 몇년에는, 신경 네트워크를 사용하여 텍스트를 생성하는 방법에 대한 여러 튜토리얼이 있었지만 음악을 만드는 방법에 대한 튜토리얼이 부족했습니다. 이 글에서는 Keras 라이브러리를 사용하여 파이썬으로 순환신경망을 사용하여 음악을 만드는 방법에 대해 설명합니다.

참을성이 부족한 사람들을 위해 이 튜토리얼 끝에 Github 저장소 링크가 있습니다!

### Background
구현을 하기 전에 우리가 알아야할 몇 가지가 있습니다.

#### Recurrent Neural Networks (RNN)
순환 신경망은 순차적인 정보를 이용하는 인공 신경망의 종류 중 하나입니다. 순환(반복, Recurrent)라고 하는 이유는 시퀀스의 모든 단일 요소에 대해 동일한 기능을 수행하기 때문이고, 각 결과는 이전 연산에 따라 달라집니다. 반면에 기존의 인공 신경망들은 이전의 입력에 대해 출력이 독립적입니다.

이 튜토리얼에서 우리는 Long Short-Term Memory(LSTM) 네트워크를 사용할 것 입니다. LSTM 은 순환신경망의 하나로 경사 하강을 통해 효율적으로 학습합니다. 게이트 메커니즘을 사용하여 LSTM는 장기 패턴을 인식하고 인코딩할 수 있습니다. LSTM는 음악 및 텍스트 생성의 경우와 같이 네트워크가 오랜 시간 동안 정보를 기억해야 하는 문제를 해결하는 데 매우 유용합니다.

#### Music21
[Music21](http://web.mit.edu/music21/)은 컴퓨터 지원 음악학에 사용되는 Python 툴킷입니다. 우리가 음악 이론의 기초를 가르치고, 음악의 예를 만들고, 음악을 공부할 수 있게 해줍니다. 이 툴킷은 MIDI 파일의 음악 표기법을 습득할 수 있는 간단한 인터페이스를 제공합니다. 또한 노트 및 코드 객체를 만들어 쉽게 MIDI 파일을 만들 수 있습니다.

이 튜토리얼에서 우리는 Music21 을 데이터 셋의 콘텐츠를 추출하고, 음악 표기법으로부터 인공 신경망을 통해 결과값을 얻는데 사용할 것입니다.

> 이 문서에서는 Anaconda 패키지를 사용하여 가상환경을 만들어서 사용했고, Anaconda 를 사용하면 music21 설치할 필요가 없습니다.
>
> Anaconda 가상환경에서 music21 을 설치하려고 한다면 아래와 같이 이미 설치되어 있다는 메세지가 나올 것입니다.
>
>(issue_09)  junghyun@MacBook-Pro-4/~/KEKOxTutorial/issue_09/  pip install music21
Requirement already satisfied: music21 in /anaconda3/lib/python3.6/site-packages (5.1.0)

#### Keras
케라스는 텐서플로우와의 상호작용을 단순화하는 고수준 인공 신경망 API 입니다. 빠르게 실험을 할 수 있도록 초점을 맞춰서 개발되었습니다.

이 튜토리얼에서는 케라스 라이브러리를 이용해서 LSTM 모델을 만들고 학습시킬 것입니다. 모델이 학습되면 우리는 모델을 음악 표기법으로부터 우리의 음악을 만드는데 사용할 것입니다.

### Training
이 섹션에서는 우리는 모델의 데이터를 수집하는 방법, LSTM 모델과 모델의 아키텍처에서 사용할 수 있도록 데이터를 준비하는 방법에 대해 이야기할 것입니다.

#### Data
우리의 Github 저장소에서 주로 Final Fantasy 사운드트랙 음악으로 구성된 피아노 음악을 사용했습니다. 우리는 대부분의 곡들이 가지고 있는 독특하고 아름다운 멜로디와 엄청난 양의 곡이 있기 때문에 Final Fantasy 음악을 데이터로 결정했습니다. 하지만 한 개의 악기로 이루어진 모든 MIDI 파일은 데이터로 사용가능합니다.

인공 신경망을 구현하는 첫 번째 스텝은 사용할 데이터를 검사해보는 것입니다.

아래에서 Music21을 사용하여 Midi 파일로부터 얻은 결과를 볼 수 있습니다.

<br></br>

```
...
<music21.note.Note F>
<music21.chord.Chord A2 E3>
<music21.chord.Chord A2 E3>
<music21.note.Note E>
<music21.chord.Chord B-2 F3>
<music21.note.Note F>
<music21.note.Note G>
<music21.note.Note D>
<music21.chord.Chord B-2 F3>
<music21.note.Note F>
<music21.chord.Chord B-2 F3>
<music21.note.Note E>
<music21.chord.Chord B-2 F3>
<music21.note.Note D>
<music21.chord.Chord B-2 F3>
<music21.note.Note E>
<music21.chord.Chord A2 E3>
...
```

데이터는 두가지 타입으로 구분됩니다: `노트`와 `코드`입니다. 노트에는 노트의 **피치**, **옥타브** 및 **오프셋** 에 대한 정보가 포함되어 있습니다.

* **피치** 는 소리의 주파수 또는 소리의 높낮이를 의미하고, "A, B, C, D, E, F, G"와 같이 표현됩니다.

* **옥타브** 는 피아노에서 어떤 피치를 사용하는지를 나타냅니다.

* **오프셋** 은 노트가 피스에서 위치한 위치를 나타냅니다.

그리고 코드는 동시에 연주되는 노트 세트를 위한 컨테이너입니다.

이제 우리는 정확하게 음악을 만들어내기 위해서는 <U>우리의 신경 네트워크가 다음에 어떤 음이 될지 예측할 수 있어야 한다는 것</U>을 알 수 있습니다. 그 말은 우리의 예측 값이 우리가 트레인 셋에서 본 노트와 코드 개체를 포함하고 있다는 것입니다. Github 페이지에 있는 트레인 셋에는 총 개수가 다른 노트와 코드가 352개 있습니다. 네트워크가 처리할 수 있는 많은 가짓수의 예측값이 있는 것처럼 보이지만, LSTM 네트워크는 쉽게 처리할 수 있습니다.

다음으로 우리가 노트를 어디에 두어야 할지 생각해봐야 합니다. 대부분의 사람들이 음악을 들었을 때처럼, 노트들 사이에는 다양한 간격이 있습니다. 우리는 많은 노트를 빠르게 연주하다가 노트 없이 잠깐 멈추도록 연주할 수도 있습니다.


빌로우 위 해브 어너더 엑섶트 프럼 어 미디 파일 댓 해즈 빈 레드 유징 뮤직 트웬티 원. 오운리 디스 타임 위 해브 애디드 디 오프셋 어브 디 아브젝트 비하인드 잇 디스 얼라우즈 어스 투 시 디 인터벌 비트윈 이치 노우트 에이 ...더보기

아래에는 Music21을 사용하여 읽은 Midi 파일에서 발췌한 것이 있습니다. 다만 이번에는 뒤에 있는 개체의 오프셋을 추가했습니다. 이렇게 하면 각 노트와 코드 사이의 간격을 볼 수 있습니다.

<br></br>

```
...
<music21.note.Note B> 72.0
<music21.chord.Chord E3 A3> 72.0
<music21.note.Note A> 72.5
<music21.chord.Chord E3 A3> 72.5
<music21.note.Note E> 73.0
<music21.chord.Chord E3 A3> 73.0
<music21.chord.Chord E3 A3> 73.5
<music21.note.Note E-> 74.0
<music21.chord.Chord F3 A3> 74.0
<music21.chord.Chord F3 A3> 74.5
<music21.chord.Chord F3 A3> 75.0
<music21.chord.Chord F3 A3> 75.5
<music21.chord.Chord E3 A3> 76.0
<music21.chord.Chord E3 A3> 76.5
<music21.chord.Chord E3 A3> 77.0
<music21.chord.Chord E3 A3> 77.5
<music21.chord.Chord F3 A3> 78.0
<music21.chord.Chord F3 A3> 78.5
<music21.chord.Chord F3 A3> 79.0
...

```

위의 midi 파일에서 읽은 노트, 코드들을 보면 노트 사이에 가장 일반적인 간격은 0.5 입니다. 그러므로 우리는 출력이 가능한 다양한 오프셋을 무시함으로써 데이터와 모델을 단순화할 수 있습니다. 이렇게 단순화 시킨다고 해서 네트워크가 생성하는 음악의 멜로디에 너무 큰 영향을 미치지 않을 것입니다. 따라서 본 자습서에서는 오프셋을 무시하고 가능한 출력 개수를 352로 유지합니다.

#### Preparing Data

데이터를 살짝 보고 LSTM 네트워크의 입력 및 출력이 노트와 코드라는 것을 확인했으므로 네트워크를 위한 데이터를 준비할 때입니다.

먼저 아래 코드 조각에서 볼 수 있는 것처럼 데이터를 배열에 로드합니다.

```
from music21 import converter, instrument, note, chord
import glob     # 원문에는 없지만 아래에서 사용하기 때문에 glob 을 import 해줘야합니다.

from music21 import converter, instrument, note, chord
import glob    # 원문에는 없지만 아래에서 사용하기 때문에 glob 을 import 해줘야합니다.

notes = []
for file in glob.glob("midi_songs/*.mid"):
    midi = converter.parse(file)
    notes_to_parse = None
    try:       # 학습 데이터 중 TypeError 를 일으키는 파일이 있어서 해놓은 예외처리
        parts = instrument.partitionByInstrument(midi)
    except TypeError:
        print('## 1 {} file occur error.'.format(file))
    if parts: # file has instrument parts
        print('## 2 {} file has instrument parts'.format(file))
        notes_to_parse = parts.parts[0].recurse()
    else: # file has notes in a flat structure
        print('## 3 {} file has notes in a flat structure'.format(file))
        notes_to_parse = midi.flat.notes
    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))
```
<br></br>

위의 코드를 어느 정도 이해하기 위해서는 코드 안에서 사용하는 music21 라이브러리의 함수들을 알아야합니다. 그래서 [music21 document](http://web.mit.edu/music21/doc/index.html) 를 보고 함수들을 최대한 설명했으나 음악과 음악 관련 소프트웨어에 대해서 잘 알지 못하기 때문에 어려울 수 있으니, 직접 한 번 각 함수의 설명을 읽어보는 것을 추천합니다.

* music21.converter : 다양한 음악 파일 포맷으로부터 음악을 로드하는 툴을 가지고 있습니다.
* music21.converter.parse() : 파일의 아이템을 파싱하고 스트림에 넣어줍니다.
* instrument.partitionByInstrument() : 단일 스트림, 스코어, 멀티 파트 구조인 경우에 각각의 악기별로 파티션을 나누고 다른 파트들을 하나로 합쳐줍니다.
* Stream.recurse : 스트림안에 존재하는 Music21 객체가 가지고 있는 값들의 리스트를 반복(iterate) 할 수 있는 iterator 를 리턴해줍니다.

먼저 convert.parse(file) 기능을 사용하여 각 파일을 Music21 스트림 개체로 로드합니다. 그 스트림 개체를 사용하면 파일에 있는 모든 노트와 코드 목록이 나옵니다. 노트의 가장 중요한 부분은 피치의 문자열 표기법을 사용하여 다시 만들 수 있으므로 모든 노트 객체의 피치를 문자열 표기법으로 추가합니다. 그리고 우리는 각 음을 점으로 나누어서 현 안에 있는 모든 음들의 id를 하나의 문자열로 인코딩함으로써 모든 화음을 추가합니다. 이러한 인코딩을 통해 네트워크에 의해 생성된 출력을 올바른 노트와 코드로 쉽게 디코딩할 수 있습니다.

이제 모든 노트와 코드를 순차적 목록에 넣었으므로 네트워크의 입력으로 사용될 시퀀스를 만들 수 있습니다.

![Figure1](https://cdn-images-1.medium.com/max/1600/1*sM3FeKwC-SD66FCKzoExDQ.jpeg)
<center>Figure 1: 우리가 데이터를 'apple', 'orange' 와 같은 카테고리 형식에서 0, 1 과 같은 숫자 형태로 변환할 때, 데이터는 범주가 고유한 값 집합에 있는 위치를 나타내는 정수 인덱스로 변환됩니다. 예를 들어, 사과는 첫 번째 뚜렷한 값이기 때문에 0으로 매핑됩니다. 오렌지는 리스트에서 인덱스가 1인 값이고, 그래서 1이 됩니다. 파인애플은 리스트에서 인덱스가 3이지만 'apple' 이라는 이미 매핑된 값이 있는 엘레먼트가 있으므로 2에 매핑됩니다</center>

<br></br>

먼저 Figure1 에서 설명한 것처럼 문자열 데이터를 숫자 형식으로 나타내는 일을 진행할 것입니다. 이렇게 하는 이유는 문자열 기반 데이터보다 숫자 기반 데이터를 신경망이 더 잘 학습하기 때문입니다.

다음으로, 우리는 네트워크와 각각의 출력에 대한 입력 순서를 만들어야 합니다. 각 입력 시퀀스의 출력은 노트 리스트의 입력 시퀀스에서 노트의 시퀀스 뒤에 오는 첫 번째 노트 또는 코드입니다.
