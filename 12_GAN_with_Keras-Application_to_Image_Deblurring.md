## 케라스로 설계한 GAN으로 이미지 흐림 제거 (GAN with Keras: Application to Image Deblurring)
[원문 링크](https://blog.sicara.com/keras-generative-adversarial-networks-image-deblurring-45e3ab6977b5)
> 이 글은 케라스 라이브러리를 이용해 이미지의 흐린부분을 없애주는 효과에 GAN을 적용해보는 튜토리얼입니다. 
* GAN(Generative Adversarial Networks | 생성적 적대 신경망)
* Image Deblur
* Keras
* Deep Learning
* Tutorial


![deblur](https://cdn-images-1.medium.com/max/2000/1*WFQmmhJM8HMD0D5Ax4vROw.jpeg)

2014년, 이안 굿펠로우가 **GAN**을 소개했었습니다. 이 글에선 **케라스로 설계한 GAN으로 이미지 흐림 제거**를 실행하는데 초점을 두고 있습니다.

본 글에 대한 [원본 논문](https://arxiv.org/pdf/1711.07064.pdf)과 그에 대한 [파이토치 코드](https://github.com/KupynOrest/DeblurGAN/)를 보시기 바랍니다. 본 글의 모든 케라스 코드들은 [이곳](https://github.com/RaphaelMeudec/deblur-gan)에서 이용하실 수 있으십니다.


### GAN 빠르게 복습하기
GAN에서, 두 개의 네트워크가 각각 견제하면서 학습을 합니다. 생성자(generator)는 진짜같은 가짜 입력값을 생성해서 식별자(discriminator)를 속입니다. 식별자는 입력값이 진짜인지 가짜인지 말해줍니다.

![GAN Training Process](https://cdn-images-1.medium.com/max/1000/1*N4oqJsGmH-KZg3Vqrm_uYw.jpeg)  
GAN Training Process - [source](https://www.kdnuggets.com/2017/01/generative-adversarial-networks-hot-topic-machine-learning.html)


### 데이터

### 모델

#### 생성자

```python
from keras.layers import Input, Conv2D, Activation, BatchNormalization
from keras.layers.merge import Add
from keras.layers.core import Dropout

def res_block(input, filters, kernel_size=(3,3), strides=(1,1), use_dropout=False):
    """
    Instanciate a Keras Resnet Block using sequential API.
    :param input: Input tensor
    :param filters: Number of filters to use
    :param kernel_size: Shape of the kernel for the convolution
    :param strides: Shape of the strides for the convolution
    :param use_dropout: Boolean value to determine the use of dropout
    :return: Keras Model
    """
    x = ReflectionPadding2D((1,1))(input)
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    if use_dropout:
        x = Dropout(0.5)(x)

    x = ReflectionPadding2D((1,1))(x)
    x = Conv2D(filters=filters,
                kernel_size=kernel_size,
                strides=strides,)(x)
    x = BatchNormalization()(x)

    # Two convolution layers followed by a direct connection between input and output
    merged = Add()([input, x])
    return merged
```
> 위 코드는 [res_block.py](https://gist.github.com/RaphaelMeudec/ee723dbb0ad429bc73f7641b61043765#file-res_block-py)에서 보실수 있으십니다.

```python
from keras.layers import Input, Activation, Add
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from layer_utils import ReflectionPadding2D, res_block

ngf = 64
input_nc = 3
output_nc = 3
input_shape_generator = (256, 256, input_nc)
n_blocks_gen = 9


def generator_model():
    """Build generator architecture."""
    # Current version : ResNet block
    inputs = Input(shape=image_shape)

    x = ReflectionPadding2D((3, 3))(inputs)
    x = Conv2D(filters=ngf, kernel_size=(7,7), padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Increase filter number
    n_downsampling = 2
    for i in range(n_downsampling):
        mult = 2**i
        x = Conv2D(filters=ngf*mult*2, kernel_size=(3,3), strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    # Apply 9 ResNet blocks
    mult = 2**n_downsampling
    for i in range(n_blocks_gen):
        x = res_block(x, ngf*mult, use_dropout=True)

    # Decrease filter number to 3 (RGB)
    for i in range(n_downsampling):
        mult = 2**(n_downsampling - i)
        x = Conv2DTranspose(filters=int(ngf * mult / 2), kernel_size=(3,3), strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    x = ReflectionPadding2D((3,3))(x)
    x = Conv2D(filters=output_nc, kernel_size=(7,7), padding='valid')(x)
    x = Activation('tanh')(x)

    # Add direct connection from input to output and recenter to [-1, 1]
    outputs = Add()([x, inputs])
    outputs = Lambda(lambda z: z/2)(outputs)

    model = Model(inputs=inputs, outputs=outputs, name='Generator')
    return model
```
> 위 코드는 [generator.py](https://gist.github.com/RaphaelMeudec/39b85509f9d8f41caffaf83525adced8#file-generator-py)에서 보실 수 있으십니다.

#### 식별자
```python
from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model

ndf = 64
output_nc = 3
input_shape_discriminator = (256, 256, output_nc)


def discriminator_model():
    """Build discriminator architecture."""
    n_layers, use_sigmoid = 3, False
    inputs = Input(shape=input_shape_discriminator)

    x = Conv2D(filters=ndf, kernel_size=(4,4), strides=2, padding='same')(inputs)
    x = LeakyReLU(0.2)(x)

    nf_mult, nf_mult_prev = 1, 1
    for n in range(n_layers):
        nf_mult_prev, nf_mult = nf_mult, min(2**n, 8)
        x = Conv2D(filters=ndf*nf_mult, kernel_size=(4,4), strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

    nf_mult_prev, nf_mult = nf_mult, min(2**n_layers, 8)
    x = Conv2D(filters=ndf*nf_mult, kernel_size=(4,4), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filters=1, kernel_size=(4,4), strides=1, padding='same')(x)
    if use_sigmoid:
        x = Activation('sigmoid')(x)

    x = Flatten()(x)
    x = Dense(1024, activation='tanh')(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x, name='Discriminator')
    return model
```
> 위 코드는 [discriminator.py](https://gist.github.com/RaphaelMeudec/d1a37d2543c3b223cf16643b5bd06cc2#file-discriminator-py)에서 보실 수 있으십니다.

```python
from keras.layers import Input
from keras.models import Model

def generator_containing_discriminator_multiple_outputs(generator, discriminator):
    inputs = Input(shape=image_shape)
    generated_images = generator(inputs)
    outputs = discriminator(generated_images)
    model = Model(inputs=inputs, outputs=[generated_images, outputs])
    return model
```
> 위 코드는 [gan.py](https://gist.github.com/RaphaelMeudec/15d1dde98739a2a22fd6bb2c4de5d17b#file-gan-py)에서 보실 수 있으십니다.

### 학습

#### 손실들

```python
import keras.backend as K
from keras.applications.vgg16 import VGG16
from keras.models import Model

image_shape = (256, 256, 3)

def perceptual_loss(y_true, y_pred):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))
```
> 위 코드는 [perceptual_loss.py](https://gist.github.com/RaphaelMeudec/18c560ff875cfbe32cc0ff38d3374f1b#file-perceptual_loss-py)에서 보실 수 있으십니다.

```python
import keras.backend as K

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)
```
> 위 코드는 [wasserstein_loss.py](https://gist.github.com/RaphaelMeudec/4fa6a6fe8fb341a1c410c5e2dad087de#file-wasserstein_loss-py)에서 보실 수 있으십니다.

#### 학습 과정

```python
# Load dataset
data = load_images('./images/train', n_images)
y_train, x_train = data['B'], data['A']

# Initialize models
g = generator_model()
d = discriminator_model()
d_on_g = generator_containing_discriminator_multiple_outputs(g, d)

# Initialize optimizers
g_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
d_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
d_on_g_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

# Compile models
d.trainable = True
d.compile(optimizer=d_opt, loss=wasserstein_loss)
d.trainable = False
loss = [perceptual_loss, wasserstein_loss]
loss_weights = [100, 1]
d_on_g.compile(optimizer=d_on_g_opt, loss=loss, loss_weights=loss_weights)
d.trainable = True
```
> 위 코드는 [gan_training_initialization.py](https://gist.github.com/RaphaelMeudec/1c0ed2dbdef394d2ab854d63161487d9#file-gan_training_initialization-py)에서 보실 수 있으십니다.

```python
for epoch in range(epoch_num):
  print('epoch: {}/{}'.format(epoch, epoch_num))
  print('batches: {}'.format(x_train.shape[0] / batch_size))

  # Randomize images into batches
  permutated_indexes = np.random.permutation(x_train.shape[0])

  for index in range(int(x_train.shape[0] / batch_size)):
      batch_indexes = permutated_indexes[index*batch_size:(index+1)*batch_size]
      image_blur_batch = x_train[batch_indexes]
      image_full_batch = y_train[batch_indexes]
```
> 위 코드는 [gan_training_batches.py](https://gist.github.com/RaphaelMeudec/68b997ba5dfb3abf66f74fd67baa4e3d#file-gan_training_batches-py)에서 보실 수 있으십니다.

```python
for epoch in range(epoch_num):
  for index in range(batches):
    # [Batch Preparation]

    # Generate fake inputs
    generated_images = g.predict(x=image_blur_batch, batch_size=batch_size)
    
    # Train multiple times discriminator on real and fake inputs
    for _ in range(critic_updates):
        d_loss_real = d.train_on_batch(image_full_batch, output_true_batch)
        d_loss_fake = d.train_on_batch(generated_images, output_false_batch)
        d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

    d.trainable = False
    # Train generator only on discriminator's decision and generated images
    d_on_g_loss = d_on_g.train_on_batch(image_blur_batch, [image_full_batch, output_true_batch])

    d.trainable = True
```
> 위 코드는 [gan_training_fit.py](https://gist.github.com/RaphaelMeudec/c9c07f61981dda4d4ff742e020b6c4ba#file-gan_training_fit-py)를 통해 보실 수 있으십니다.


#### 작업 환경

### 이미지 흐림 제거 결과

### 참고문서
* [NIPS 2016 : 생성적 적대적 신경망](https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Generative-Adversarial-Networks) by 이안 굿펠로우
* [ICCV 2017 : GAN 튜토리얼](https://sites.google.com/view/iccv-2017-gans/schedule)
* [케라스를 이용해 GAN 구현](https://github.com/eriklindernoren/Keras-GAN) by [에릭 린더-노렌](http://www.eriklindernoren.se/)
* [GAN 목록](https://deeplearning4j.org/generative-adversarial-network) by deeplearning4j
* [진짜 놀라운 GAN](https://github.com/nightrome/really-awesome-gan) by [호거 캐서](http://www.it-caesar.com/)

저자 : 당신이 이 글(케라스로 설계한 GAN으로 이미지 흐림 제거)을 즐겼으면 합니다! 밑에 팔로우 버튼 잊지마세요!
