import numpy as np 				# linear algebra
import keras.backend as K
from keras import layers
from keras import models, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from capsule.layers import CapsuleLayer,Mask,Length,PrimaryCap
from get_training_data import get_training_data


def get_model(input_shape, n_class, num_routing):
    x = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='valid', activation='relu', name='conv1')(x)
    primarycaps = PrimaryCap(conv1, dim_vector=8, n_channels=32, kernel_size=3, strides=2, padding='valid')
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_vector=16, num_routing=num_routing, name='digitcaps')(primarycaps)
    out_caps = Length(name='out_caps')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked = Mask()([digitcaps, y])
    x_recon = layers.Dense(512, activation='relu')(masked)
    x_recon = layers.Dense(1024, activation='relu')(x_recon)
    x_recon = layers.Dense(np.prod(input_shape), activation='sigmoid')(x_recon)
    x_recon = layers.Reshape(target_shape=input_shape, name='out_recon')(x_recon)

    return models.Model([x, y], [out_caps, x_recon])

def get_cnn_model(input_shape,n_class):
	model=models.Sequential()
	#model.add(layers.Input(shape=input_shape))
	model.add(layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='valid', activation='relu', name='conv1'))
	model.add(layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='valid', activation='relu', name='conv2'))
	model.add(layers.Flatten())
	model.add(layers.Dense(512, activation='relu'))
	model.add(layers.Dense(n_class, activation='relu'))
	return model




def margin_loss(y_true, y_pred):
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
    return K.mean(K.sum(L, 1))


def train_generator(x, y, batch_size, shift_fraction=0., samplewise_std_normalization=False, zoom_range=0.,
                        horizontal_flip=False, vertical_flip=False, rotation_range=0):
    train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction,
                                           samplewise_std_normalization=samplewise_std_normalization,
                                           zoom_range=zoom_range,
                                           horizontal_flip=horizontal_flip,
                                           vertical_flip=vertical_flip,
                                           rotation_range=rotation_range)  
    generator = train_datagen.flow(x, y, batch_size=batch_size)
    while 1:
      x_batch, y_batch = generator.next()
      yield ([x_batch, y_batch], [y_batch, x_batch])
      
            
def train(model, data,batch_size,epochs):

    (x_train, y_train), (x_test, y_test) = data

    model.compile(optimizer=optimizers.Adam(lr=0.001),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., 0.392],
                  metrics={'out_caps': 'accuracy'})
    model.fit_generator(generator=train_generator(x_train, y_train,batch_size,0.2),
                        steps_per_epoch=int(y_train.shape[0] / batch_size),
                        epochs=epochs,
                        validation_data=[[x_test, y_test], [y_test, x_test]])
    return model



x_train,x_test=get_training_data('./data')



#comment after it runs well
model=get_cnn_model((140,140,3),5)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,x_test,epochs=2)
model.summary()
# upto

'''
model=get_model((280,280,3),5,1)
model.compile(optimizer=optimizers.Adam(lr=0.001),loss=[margin_loss, 'mse'],loss_weights=[1., 0.392],metrics={'out_caps': 'accuracy'})
model.summary()
#model.fit(xm_train,ym_train)   # do not uncomment this line
model=train(model,((x_train,y_train),(x_train,y_train)),5,2)
'''

model.save_weights('./data/model_weights.h5')
model.to_json('./data/model.json')
