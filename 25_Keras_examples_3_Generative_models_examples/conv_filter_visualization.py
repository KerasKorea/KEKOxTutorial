'''입력공간의 기울기를 통해 VGG16 필터들을 시각화

이 글은 CPU환경에서 몇분이면 실행할 수 있습니다.

결과 예시 : http://i.imgur.com/4nj4KjN.jpg
'''
from __future__ import print_function

import numpy as np
import time
from keras.preprocessing.image import save_img
from keras.applications import vgg16
from keras import backend as K

# 각 필터들을 위한 생성 이미지의 차원 설정
img_width = 128
img_height = 128

# 시각화하고 싶은 레이어의 이름 설정
# (모델에 대한 정의는 keras/applications/vgg16.py에서 볼 수 있습니다.)
layer_name = 'block5_conv1'

# 텐서(tensor)를 확인된 이미지로 변환해주는 함수

def deprocess_image(x):

    # 텐서를 정규화한다 : 중심은 0, 편차는 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # RGB 배열로 변환
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# ImageNet의 가중치를 VGG16에 적용, 설계한다. 
model = vgg16.VGG16(weights='imagenet', include_top=False)
print('Model loaded.')

model.summary()

# 이미지를 입력받기 위한 placeholder 설정
input_img = model.input

# (앞서 이름을 지정한)각 핵심 레이어의 출력들을 가져옴.
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])


def normalize(x):
    # L2 norm으로 텐서를 정규화 해주는 함수
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


kept_filters = []
for filter_index in range(200):
   
    # 실제론 512개의 필터가 있지만 처음 200개의 필터만 스캔합니다.
    print('Processing filter %d' % filter_index)
    start_time = time.time()

    # 관심을 두고 있는 레이어의 n번째 필터의 활성화를 최대치로 하는 손실 함수를 설계합니다.
    layer_output = layer_dict[layer_name].output
    if K.image_data_format() == 'channels_first':
        loss = K.mean(layer_output[:, filter_index, :, :])
    else:
        loss = K.mean(layer_output[:, :, :, filter_index])

    # 손실 함수를 통해 입력 이미지의 기울기를 계산합니다
    grads = K.gradients(loss, input_img)[0]

    # 정규화 기법 : 기울기를 정규화 합니다.
    grads = normalize(grads)

    # 입력 이미지의 손실과 기울기를 반환합니다.
    iterate = K.function([input_img], [loss, grads])

    # 기울기 상승을 위해 스탭 크기 지정.
    step = 1.

    # 몇 개의 임의의 노이즈와 같이 회색 이미지부터 시작합니다.
    if K.image_data_format() == 'channels_first':
        input_img_data = np.random.random((1, 3, img_width, img_height))
    else:
        input_img_data = np.random.random((1, img_width, img_height, 3))
    input_img_data = (input_img_data - 0.5) * 20 + 128

    # 20 스텝동안 기울기 상승을 시도합니다.
    for i in range(20):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

        print('Current loss value:', loss_value)
        if loss_value <= 0.:
            # 몇가지 필터가 0을 가질 때는 넘어갑니다.
            break

    # 입력 이미지의 결과물을 디코드(decode)합니다.
    if loss_value > 0:
        img = deprocess_image(input_img_data[0])
        kept_filters.append((img, loss_value))
    end_time = time.time()
    print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

# 8 X 8 격자인 64개의 필터들을 사용할 겁니다.
n = 8

# 가장 큰 손실값을 가진 필터는 더 잘보일 것입니다.
# 상위 64개의 필터는 유지시킬 겁니다.
kept_filters.sort(key=lambda x: x[1], reverse=True)
kept_filters = kept_filters[:n * n]

# build a black picture with enough space for
# our 8 x 8 filters of size 128 x 128, with a 5px margin in between
# 128 x 128 크기의 8 x 8 필터를 저장할 수 있는 충분한 공간이 있는 검정 이미지를 만듭니다. 
# 5px의 여유공간도 둬야합니다.
margin = 5
width = n * img_width + (n - 1) * margin
height = n * img_height + (n - 1) * margin
stitched_filters = np.zeros((width, height, 3))

# 필터와 이미지를 저장합니다.
for i in range(n):
    for j in range(n):
        img, loss = kept_filters[i * n + j]
        width_margin = (img_width + margin) * i
        height_margin = (img_height + margin) * j
        stitched_filters[
            width_margin: width_margin + img_width,
            height_margin: height_margin + img_height, :] = img

# 결과를 디스크에 저장합니다.
save_img('stitched_filters_%dx%d.png' % (n, n), stitched_filters)