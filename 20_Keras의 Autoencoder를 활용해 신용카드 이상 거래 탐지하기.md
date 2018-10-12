# Keras의 Autoencoder를 활용해 신용카드 이상 거래 탐지하기



신용카드 이상 거래 탐지는 어떻게 이루어질까?



일요일 아침, 조용함 속에서 당신은 입가에 미소를 띄우며 일어난다. *오늘은 좋은 날이 될거야!* 당신의 핸드폰이, 그것도 상당히 '국제적으로' 울리는 일만 아니었다면. 당신은 핸드폰을 천천히 집어들었고, 무언가 특이한 것을 듣게 되었다 - "Bonjour, je suis Michele, 아, 이런. 죄송합니다. 저는 미셸이예요. 당신이 쓰고 있는 은행의 직원인데요." 대체 뭐가 그렇게 급하길래 스위스에 있는 누군가가 이 시간에 당신에게 전화를 걸게 했을까? "혹시 디아블로 3 100장을 사기 위해 3,358.65 달러를 지불하신 적이 있나요?" 그 말을 듣자마자, "아뇨, 그런 적 없는데요!?" 미셸의 대답은 빠르고 간결했다-"감사합니다, 저희가 처리하겠습니다." 휴, 하마터면 어쩔 뻔했어! 그런데 미셸이 어떻게 이 거래가 이상하다는 것을 알게 되었을까? 사실 지난 주에 같은 은행 계좌를 통해 10개의 새로운 스마트폰을 구매하기도 했는데, 그 때는 미셸이 전화를 걸지 않았다.



[Nilson Report](https://www.nilsonreport.com/upload/content_promo/The_Nilson_Report_10-17-2016.pdf)에 따르면, 신용카드 사기로 인한 연간 피해액은 국제적으로 218억 달러에 달했다(2015년 기준). 당신이 사기꾼이라면 아마 매우 운이 좋다고 생각할지도 모르겠다. 같은 해에 미국에서는 100달러당 12센트 정도가 도둑맞았다. 우리의 친구 미셸은 여기서 해결해야 할 심각한 문제가 있을지도 모른다.



이 글에서, 우리는 Keras로 구현된 Autoencoder 신경망을 비지도학습(unsupervised) 또는 반지도학습(semi-supervised) 방식으로 훈련하여 신용카드 거래에서 이상 탐지하는 방법을 배울 것이다. 학습된 모델은 미리 정답을 알고 있는, 익명화된 데이터셋을 기반으로 평가될 것이다.



[이 글에서 쓴 코드와 학습된 모델은 이 GitHub Repository에서 확인할 수 있다.](https://github.com/curiousily/Credit-Card-Fraud-Detection-using-Autoencoders-in-Keras)



## 기본 설정(Setup)

우리는 Tensorflow 1.2와 Keras 2.0.4를 사용할 것이다. 그럼 이제 시작해보자:

```python
# 필요한 라이브러리 로드
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers

# 시각화 라이브러리 설정
%matplotlib inline

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 14, 8

# RANDOM_SEED와 LABELS 설정
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]
```



## 데이터 가져오기(Loading the data)

우리가 사용하려는 데이터는 [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud/version/3)에서 다운로드 받을 수 있다. 이것은 이틀 동안 발생한 신용 카드 거래 데이터이며, 총 284,807개의 거래 기록과 그 중 492개의 이상 거래 기록이 포함되어 있다.



데이터셋에 있는 모든 변수는 수치 데이터이다. 데이터는 개인 정보보호 문제로 인해 PCA 변환 과정을 거쳤다. 두 가지 변하지 않은 것은 시간(Time)과 거래액(Amount)이다. 시간은 각 거래가 이루어진 시점과 데이터셋 내의 첫번째 거래가 이루어진 시점 사이의 시간을 초(second) 단위로 기록했다.



```python
df = pd.read_csv("data/creditcard.csv")
```



## 데이터 둘러보기(Exploration)

```python

```

=> `(204808,31)`

데이터는 31개의 열(column)로 구성되어 있으며, 그 중 2개는 각각 시간과 거래액이다. 다른 데이터들은 PCA 변환을 통해 나온 결과값이다. 



그럼 결측값(missing value)이 있는지 확인해보자.

```python
df.isnull().values.any()
```

=> `False`



```python
count_classes = pd.value_counts(df['Class'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.title("Transaction class distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency");
```

![](https://cdn-images-1.medium.com/max/800/1*ZkAPbHr4-K4E8LB7XUkjDQ.png)

우리는 상당히 불균형한 데이터를 가지고 있다. 정상 거래가 이상 거래에 비해 압도적으로 많다. 



두 거래 유형에 대해 살펴보자. 

```python
frauds = df[df.Class == 1]
normal = df[df.Class == 0]
frauds.shape
```

=> `(492, 31)`

```python
normal.shape
```

=> `(284315, 31)`



각 거래 유형의 거래액은 어떻게 다를까?

```python
frauds.Amount.describe()
```

=>

```
count     492.000000
mean      122.211321
std       256.683288
min         0.000000
25%         1.000000
50%         9.250000
75%       105.890000
max      2125.870000
Name: Amount, dtype: float64
```



```python
normal.Amount.describe()
```

=>

```
count    284315.000000
mean         88.291022
std         250.105092
min           0.000000
25%           5.650000
50%          22.000000
75%          77.050000
max       25691.160000
Name: Amount, dtype: float64
```



좀 더 그래픽적인 요소를 더해보자.

```python
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per transaction by class')

bins = 50

ax1.hist(frauds.Amount, bins = bins)
ax1.set_title('Fraud')

ax2.hist(normal.Amount, bins = bins)
ax2.set_title('Normal')

plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
plt.show();
```

![](https://cdn-images-1.medium.com/max/800/1*ReLu4N0WJEkRLGkm34Z2gQ.png)



이상 거래가 특정 시간에 더 자주 발생하기도 할까?

```python
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')

ax1.scatter(frauds.Time, frauds.Amount)
ax1.set_title('Fraud')

ax2.scatter(normal.Time, normal.Amount)
ax2.set_title('Normal')

plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()
```

![](https://cdn-images-1.medium.com/max/800/1*paXtez2VD4FESWFkHPtFkA.png)

거래 시간이 그렇게 중요한 것 같지는 않아 보인다.



## Autoencoder

Autoencoder는 처음에는 이상해보일 수도 있다. 이러한 모델들이 하는 일은 입력값을 받아서, 입력값을 예측하는 것이다. 어리둥절한가? 적어도 내가 처음 이것을 들었을 때는 그랬다.

좀 더 자세히, Autoencoder 신경망에 대해 살펴보자. 이 Autoencoder는 다음과 같은 항등 함수

![](https://cdn-images-1.medium.com/max/800/1*aj1coSJRGTt7XwiqjCUGDA.png)