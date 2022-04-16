from keras.layers import Conv2D,MaxPool2D,Reshape,ConvLSTM2D,UpSampling2D
from keras.models import Model
from keras.layers import Input
from keras.backend import binary_crossentropy
import keras.backend as K
from tensorflow.keras.optimizers import Adam,SGD
from plotsAccLoss import plot_acc, plot_loss

smooth = 1e-12
def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)

def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)

def jaccard_coef_int(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)

def jaccard_coef_loss(y_true, y_pred):
    return -K.log(jaccard_coef(y_true, y_pred)) + binary_crossentropy(y_pred, y_true)
shape = (512,512,3)
ipt = Input(shape=shape)
x = Reshape((1,shape[0],shape[1],shape[2]))(ipt)
x = ConvLSTM2D(68,kernel_size=3,padding='same',return_sequences=False,activation='relu')(x)
x = Conv2D(128,kernel_size=3,padding='same',activation='relu')(x)
x = MaxPool2D(pool_size=2,padding='same')(x)
#x = Conv2DTranspose(128,kernel_size=3,padding='same',strides=2,activation='relu')(x)
x = UpSampling2D(size=(2,2))(x)
x = Conv2D(256,kernel_size=3,padding='same',activation='relu')(x)
x = Conv2D(1,kernel_size=1,padding='same',activation='sigmoid')(x)
rg = Model(ipt,x)
rg.compile(loss='binary_crossentropy',optimizer=Adam(learning_rate=0.0001),metrics=[jaccard_coef_int,'accuracy'])
history = rg.fit(train_loader,validation_data = test_loader,epochs=epochs)
plot_acc(history)
plot_loss(history)
