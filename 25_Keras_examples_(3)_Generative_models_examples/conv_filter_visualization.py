'''Visualization of the filters of VGG16, via gradient ascent in input space.
VGG16의 필터들의 시각화,

이 스크립트는 cpu기반에서 몇분안으로 작동할 수 있습니다.
결과 예시: http://i.imgur.com/4nj4KjN.jpg
'''
from __future__ import print_function

import numpy as np
import time
from keras.preprocessing.image import save_img
from keras.applications import vgg16
from keras import backend as K

# dimensions of the generated pictures for each filter.
# 각각의 필터에 맞게 생성되는 이미지의 차원들
img_width = 128
img_height = 128

# 우리가 시각화 하고 싶은 레이어의 이름을 지정합니다.
# 모델 정의는 keras/applications/vgg16.py에서 보실 수 있으십니다.
layer_name = 'block5_conv1'

# util function to convert a tensor into a valid image
# 텐서(tensor)가 검증된 이미지로 변환되도록 해주는 활용 함수입니다.

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    # 텐서를 일반화 해줍니다. : 중심을 0.으로, 편차를 0.1로
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # RGB 배열로 변환해 줍니다.
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# ImageNet 가중치를 가지고 있는 VGG16 네트워크를 설계합니다.
model = vgg16.VGG16(weights='imagenet', include_top=False)
print('Model loaded.')

model.summary()

# this is the placeholder for the input images
# 입력 이미지를 위한 placeholder입니다.
input_img = model.input

# get the symbolic outputs of each "key" layer (we gave them unique names).
# 각 핵심 레이어(layer)의 중요 결과물들을 가져옵니다.(그것들에게 고유 이름을 줄 수 있습니다.)
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    # L2 norm을 통해 텐서를 일반화 해주는 활용 함수 입니다.
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


kept_filters = []
for filter_index in range(200):
    # we only scan through the first 200 filters,
    # 우리는 오직 200개의 필터들로 훑어 봅니다.
    # but there are actually 512 of them
    # 하지만 실제론 512개 입니다.
    print('Processing filter %d' % filter_index)
    start_time = time.time()

    # we build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    # 관심있는 계층의 n번째 필터의 활성화를 최대로 해주는 손실 함수를 설계합니다.
    layer_output = layer_dict[layer_name].output
    if K.image_data_format() == 'channels_first':
        loss = K.mean(layer_output[:, filter_index, :, :])
    else:
        loss = K.mean(layer_output[:, :, :, filter_index])

    # we compute the gradient of the input picture wrt this loss
    # TODO : 앞의 손실 함수를 통해 입력 이미지의 기울기(*)를 계산합니다.
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    # 기울기를 우리는 일반화 해줍니다.
    grads = normalize(grads)

    # this function returns the loss and grads given the input picture
    # 해당 함수는 주어진 입력 이미지로 손실과 기울기를 반환합니다.
    iterate = K.function([input_img], [loss, grads])

    # step size for gradient ascent(*)
    # TODO : 기울기 감소에서 step의 크기
    step = 1.

    # we start from a gray image with some random noise
    # 우리는 임의의 노이즈가 있는 회색 이미디부터 시작합니다.
    if K.image_data_format() == 'channels_first':
        input_img_data = np.random.random((1, 3, img_width, img_height))
    else:
        input_img_data = np.random.random((1, img_width, img_height, 3))
    input_img_data = (input_img_data - 0.5) * 20 + 128

    # we run gradient ascent for 20 steps
    # 20 스텝동안 우리는 기울기 감소를 진행합니다.
    for i in range(20):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

        print('Current loss value:', loss_value)
        if loss_value <= 0.:
            # some filters get stuck to 0, we can skip them
            # TODO : 약간의 필터들은 0으로 
            break

    # decode the resulting input image
    # 입력이미지의 결과를 디코딩(decode)한다.
    if loss_value > 0:
        img = deprocess_image(input_img_data[0])
        kept_filters.append((img, loss_value))
    end_time = time.time()
    print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

# we will stich the best 64 filters on a 8 x 8 grid.
# TODO : 우리는 8 X 8 격자인 64개의 최적의 필터들을 stich 할수 있습니다.
n = 8

# the filters that have the highest loss are assumed to be better-looking.
# 가장 높은 손실을 가지고 있는 필터들은 더 잘 보이는 거로 추정됩니다.
# we will only keep the top 64 filters.
# 우리는 64개의 필터들을 유지할 수 있습니다.
kept_filters.sort(key=lambda x: x[1], reverse=True)
kept_filters = kept_filters[:n * n]

# build a black picture with enough space for
# our 8 x 8 filters of size 128 x 128, with a 5px margin in between
# TODO : 2번째 줄이 이해가 안됨.
margin = 5
width = n * img_width + (n - 1) * margin
height = n * img_height + (n - 1) * margin
stitched_filters = np.zeros((width, height, 3))

# fill the picture with our saved filters
# 저장된 필터들과 같이 이미지들을 채웁니다.
for i in range(n):
    for j in range(n):
        img, loss = kept_filters[i * n + j]
        width_margin = (img_width + margin) * i
        height_margin = (img_height + margin) * j
        stitched_filters[
            width_margin: width_margin + img_width,
            height_margin: height_margin + img_height, :] = img

# 결과를 로컬에 저장합니다.
save_img('stitched_filters_%dx%d.png' % (n, n), stitched_filters)