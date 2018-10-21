## Tensorflow와 Keras를 활용한 가상화폐 가격 예측하기 💸
[원문 링크](https://medium.com/@gtnjuvin/my-journey-into-deep-q-learning-with-keras-and-gym-3e779cc12762)
> 이 튜토리얼은 Tensorflow와 Keras를 활용해서 가상화폐 가격을 예측해봅니다.

* Keras
* CNN
* LSTM, GRU

### Introduction

가상화폐, 특히 Bitcoin은 최근 소셜 미디어와 검색 엔진에서 가장 큰 인기를 얻고 있는 것 중 하나입니다. 지능적인 발명 전략을 취한다면 그들(Bitcoin)의 높은 변동성은 높은 수익으로 이어질 수 있습니다! 이제는 전 세계 모든 사람들이 갑자기 가상화폐에 대해 이야기하기 시작한 것 같습니다. 불행히도, index가 부족하기 때문에 기존의 금융상품과 비교할 때 가상화폐는 상대적으로 예측 불가능하죠. 이 튜토리얼은 가상화폐의 미래 추세를 파악하기 위해 Bitcoin을 예로 들어 딥러닝(Deep Learning)으로 이러한 가상화폐의 가격을 예측하는 것을 목표로 합니다.

<br></br>
<br></br>

### Getting Started

아래 코드를 실행하려면 다음 환경과 라이브러리를 설치해야 합니다.

1. Python 2.7
2. Tensorflow=1.2.0
3. Keras=2.1.1
4. Pandas=0.20.3
5. Numpy=1.13.3
6. h5py=2.7.0
7. sklearn=0.19.1

<br></br>
<br></br>

### Data Collection

예측 데이터는 `Kaggle` 또는 `Poloniex`에서 수집할 수 있습니다. 일관성을 유지하기 위해 `Poloniex`에서 수집된 데이터의 열 이름이 `Kaggle`의 열 이름과 일치하도록 변경됩니다.

<br></br>

```python
import json
import numpy as np
import os
import pandas as pd
import urllib2

# poloniex's API에 연결합니다.
url = 'https://poloniex.com/public?command=returnChartData&currencyPair=USDT_BTC&start=1356998100&end=9999999999&period=300'

# API를 통해 얻은 json을 파싱하고, pandas의 DataFrame으로 바꿔줍니다.
openUrl = urllib2.urlopen(url)
r = openUrl.read()
openUrl.close()
d = json.loads(r.decode())
df = pd.DataFrame(d)

original_columns=[u'close', u'date', u'high', u'low', u'open']
new_columns = ['Close','Timestamp','High','Low','Open']
df = df.loc[:,original_columns]
df.columns = new_columns
df.to_csv('data/bitcoin2015to2017.csv',index=None)
view raw
```

<br></br>
<br></br>

### Data Preparation

예측을 위해 소스에서 수집된 데이터를 파싱해야 합니다. 이 [블로그](https://nicholastsmith.wordpress.com/2017/11/13/cryptocurrency-price-prediction-using-deep-learning-in-tensorflow/)의 PastSampler class를 이용하여 데이터를 분할하여 데이터 리스트와 라벨 리스트를 얻을 수 있습니다. 입력 크기(N)는 256이고 출력 크기(K)는 16입니다. Poloniex에서 수집된 데이터는 5분 단위로 체크 표시됩니다. 이는 입력이 1280분 동안 지속되고 출력이 80분 이상임을 나타냅니다.

<br></br>

```Python
import numpy as np
import pandas as pd

class PastSampler:
    '''
    학습 데이터(training samples)를 과거 데이터를 이용해서 미래를 예측할 수 있도록 형태를 갖춰줍니다.
    '''

    def __init__(self, N, K, sliding_window = True):
        '''
        N개의 과거 데이터를 이용해 K개의 미래를 예측합니다.
        '''
        self.K = K
        self.N = N
        self.sliding_window = sliding_window

    def transform(self, A):
        M = self.N + self.K     #한개의 row당 데이터 개수 (sample + target)
        #indexes
        if self.sliding_window:
            I = np.arange(M) + np.arange(A.shape[0] - M + 1).reshape(-1, 1)
        else:
            if A.shape[0]%M == 0:
                I = np.arange(M)+np.arange(0,A.shape[0],M).reshape(-1,1)

            else:
                I = np.arange(M)+np.arange(0,A.shape[0] -M,M).reshape(-1,1)

        B = A[I].reshape(-1, M * A.shape[1], A.shape[2])
        ci = self.N * A.shape[1]    #한 데이터당 feature 개수
        return B[:, :ci], B[:, ci:] #학습 matrix, 타겟 matrix

#데이터 파일 위치(path)
dfp = 'data/bitcoin2015to2017.csv'

# 가격 데이터 컬럼(열, columns)
columns = ['Close']
df = pd.read_csv(dfp)
time_stamps = df['Timestamp']
df = df.loc[:,columns]
original_df = pd.read_csv(dfp).loc[:,columns]
```
