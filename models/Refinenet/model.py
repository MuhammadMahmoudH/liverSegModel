from keras.models import Model
from keras.layers import Input, concatenate, ZeroPadding2D, Conv2D, MaxPooling2D, AveragePooling2D, \
    BatchNormalization, Activation, UpSampling2D
from keras.initializers import he_uniform
from keras.layers.merge import add, Concatenate


# smooth =1e-12
# def dice_loss(y_true, y_pred):
#     smooth = 1.
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
#
# def jaccard_coef(y_true, y_pred):
#     intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
#     sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
#
#     jac = (intersection + smooth) / (sum_ - intersection + smooth)
#
#     return K.mean(jac)
#
#
# def jaccard_coef_int(y_true, y_pred):
#     y_pred_pos = K.round(K.clip(y_pred, 0, 1))
#
#     intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
#     sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])
#
#     jac = (intersection + smooth) / (sum_ - intersection + smooth)
#
#     return K.mean(jac)
#
#
# def jaccard_coef_loss(y_true, y_pred):
#     return -K.log(jaccard_coef(y_true, y_pred)) + binary_crossentropy(y_pred, y_true)
# def unit_1(in_layer, n1=64, n2=64, n3=256, s2=1, p2=1, d2=1):
#     '''
#     Two-Brach Unit
#     :param in_layer:
#     :return:
#     '''
#     # branch 1
#     x1 = Conv2D(n1, (1, 1), strides=(1, 1), padding='valid', kernel_initializer=he_uniform(), use_bias=False)(in_layer)
#     x1 = BatchNormalization(momentum=0.95)(x1)
#     x1 = Activation('relu')(x1)
#
#     x1 = ZeroPadding2D(padding=(p2, p2))(x1)
#     x1 = Conv2D(n2, (3, 3), strides=(s2, s2), padding='valid', dilation_rate=(d2, d2), kernel_initializer=he_uniform(), use_bias=False)(x1)
#     x1 = BatchNormalization(momentum=0.95)(x1)
#     x1 = Activation('relu')(x1)
#
#     x1 = Conv2D(n3, (1, 1), strides=(1, 1), padding='valid', kernel_initializer=he_uniform(), use_bias=False)(x1)
#     x1 = BatchNormalization(momentum=0.95)(x1)
#
#     # branch 2
#     x2 = Conv2D(n3, (1, 1), strides=(s2, s2), padding='valid', kernel_initializer=he_uniform(), use_bias=False)(in_layer)
#     x2 = BatchNormalization(momentum=0.95)(x2)
#
#     x = add([x1, x2])
#     x = Activation('relu')(x)
#     return x
#
# def unit_2(in_layer, n1=64, n2=64, n3=256, p2=1, d2=1):
#     '''
#     Shortcut Unit
#     :param in_layer:
#     :return:
#     '''
#     x = Conv2D(n1, (1, 1), strides=(1, 1), padding='valid', kernel_initializer=he_uniform(), use_bias=False)(in_layer)
#     x = BatchNormalization(momentum=0.95)(x)
#     x = Activation('relu')(x)
#
#     x = ZeroPadding2D(padding=(p2, p2))(x)
#     x = Conv2D(n2, (3, 3), strides=(1, 1), padding='valid', dilation_rate=(d2, d2), kernel_initializer=he_uniform(), use_bias=False)(x)
#     x = BatchNormalization(momentum=0.95)(x)
#     x = Activation('relu')(x)
#
#     x = Conv2D(n3, (1, 1), strides=(1, 1), padding='valid', kernel_initializer=he_uniform(), use_bias=False)(x)
#     x = BatchNormalization(momentum=0.95)(x)
#
#     x = add([in_layer, x])
#     x = Activation('relu')(x)
#     return x
#
# def unit_3(in_layer):
#     '''
#     Pyramid Pooling
#     :param in_layer:
#     :return:
#     '''
#     def pyramid(pool_size, stride):
#         x = AveragePooling2D(pool_size=pool_size, strides=stride, padding='valid')(in_layer)
#         x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid', kernel_initializer=he_uniform(), use_bias=False)(x)
#         x = UpSampling2D(stride)(x)
#         return x
#
#     x1 = pyramid(60, 60)
#     x2 = pyramid(30, 30)
#     x3 = pyramid(20, 20)
#     x4 = pyramid(10, 10)
#     return concatenate([in_layer, x1, x2, x3, x4])

def refinenet(input_shape=(512, 512, 3), num_classes=1):
    def shortcut(in_layer, n1, p1, p2, s1, s2, d1=1, d2=1):
        x = ZeroPadding2D(padding=(p1, p1))(in_layer)
        x = Conv2D(n1, (3, 3), strides=(s1, s1), padding='valid', kernel_initializer=he_uniform(), dilation_rate=d1)(x)
        # x = BatchNormalization(momentum=0.95)(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D(padding = (p2, p2))(x)
        x = Conv2D(n1, (3, 3), strides=(s2, s2), padding='valid', kernel_initializer=he_uniform(), dilation_rate=d2)(x)
        x = add([in_layer, x])
        x = Activation('relu')(x)
        return x

    def atrousConv(in_layer, n1, p1, p2, p3, s1, s2, s3, d1=1, d2=1, d3=1):
        x1 = ZeroPadding2D(padding=(p1, p1))(in_layer)
        x1 = Conv2D(n1, (3, 3), strides=(s1, s1), padding='valid', kernel_initializer=he_uniform(), dilation_rate=d1)(x1)
        x1 = Activation('relu')(x1)
        x1 = ZeroPadding2D(padding=(p2, p2))(x1)
        x1 = Conv2D(n1, (3, 3), strides=(s2, s2), padding='valid', kernel_initializer=he_uniform(), dilation_rate=d2)(x1)

        x2 = ZeroPadding2D(padding=(p3, p3))(in_layer)
        x2 = Conv2D(n1, (1, 1), strides=(s3, s3), padding='valid', kernel_initializer=he_uniform(), dilation_rate=d3)(x2)
        x = add([x1, x2])
        x = Activation('relu')(x)
        return x

    inputs = Input(shape=input_shape)

    x = ZeroPadding2D(padding=3)(inputs)
    x = Conv2D(16, (7, 7), strides=2, padding='valid', kernel_initializer=he_uniform(), activation='relu')(x)
    x = ZeroPadding2D(padding=1)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)

    x = shortcut(x, n1=16, p1=1, p2=1, s1=1, s2=1)
    x = shortcut(x, n1=16, p1=1, p2=1, s1=1, s2=1)

    x = atrousConv(x, n1=32, p1=1, p2=1, p3=0, s1=2, s2=1, s3=2)
    x = shortcut(x, n1=32, p1=1,p2=1,s1=1,s2=1)

    x = atrousConv(x, n1=64, p1=2, p2=2, p3=0, s1=1, s2=1, s3=1, d1=2, d2=2)
    x = shortcut(x, n1=64, p1=2, p2=2, s1=1, s2=1, d1=2, d2=2)

    x = atrousConv(x, n1=128, p1=4, p2=4, p3=0, s1=1, s2=1, s3=1, d1=4, d2=4)
    x = shortcut(x, n1=128, p1=4, p2=4, s1=1, s2=1, d1=4, d2=4)

    x = Conv2D(128, (1, 1), strides=(1, 1), padding='valid', activation='relu', kernel_initializer=he_uniform())(x)

    x = Conv2D(num_classes, (1, 1), strides=(1, 1), padding='valid', activation='sigmoid')(x)

    x = UpSampling2D((8, 8))(x)

    model = Model(inputs=inputs, outputs=x)
    # model.compile(loss='binary_crossentropy',optimizer=Adam(learning_rate=0.001),metrics=[jaccard_coef_int])
    # model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss=bce_dice_loss, metrics=[dice_loss])
    return model

# refinenet_model = refinenet((512, 512, 3),1)
# history = refinenet_model.fit(train_loader,validation_data = test_loader,epochs=epochs)
# summarize history for acc
# def plot_jaccard_coef_int(history):
#     plt.plot(history.history['jaccard_coef_int'])
#     plt.plot(history.history['val_jaccard_coef_int'])
#     plt.title('accuracy')
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.show()
#
# # summarize history for loss
# def plot_loss(history):
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.show()
# plot_jaccard_coef_int(history)
# plot_loss(history)