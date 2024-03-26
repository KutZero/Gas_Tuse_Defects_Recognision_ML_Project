from common_dependencies import *

def get_model_v2():
    # создание архитектуры модели
    CONV_DROP_PERCENT = 0.10
    DENSE_DROP_PERCENT = 0.10
    # 1 подсеть //////////////////////////////////////////////////
    input_time = Input((16,16,32), name = 'input_time')
    
    conv_1_1 = Conv2D(512, (4,4), activation='relu', name='conv_1_1')(input_time)
    pool_1_1 = MaxPooling2D((2,2), strides=2, name='pool_1_1')(conv_1_1)
    
    conv_1_2 = Conv2D(1024, (5,5), activation='relu', name='conv_1_2')(pool_1_1)
    drop_1_1 = Dropout(CONV_DROP_PERCENT, name='drop_1_1')(conv_1_2)
    pool_1_2 = MaxPooling2D((2,2), strides=2, name='pool_1_2')(drop_1_1)
    
    # 2 подсеть //////////////////////////////////////////////////
    input_amp = Input((16,16,32), name = 'input_amp')
    
    conv_2_1 = Conv2D(512, (4,4), activation='linear', name='conv_2_1')(input_amp)
    pool_2_1 = MaxPooling2D((2,2), strides=2, name='pool_2_1')(conv_2_1)
    
    conv_2_2 = Conv2D(1024, (5,5), activation='linear', name='conv_2_2')(pool_2_1)
    drop_2_1 = Dropout(CONV_DROP_PERCENT, name='drop_2_1')(conv_2_2)
    pool_2_2 = MaxPooling2D((2,2), strides=2, name='pool_2_2')(drop_2_1)
    
    # выходная подсеть //////////////////////////////////////////////////
    
    conc_3_1 = concatenate([pool_1_2,pool_2_2], axis=3, name='conc_3_1')
    flat_3_1 = Flatten(name='flat')(conc_3_1)
    
    d_3_1 = Dense(4096, activation='linear', name='d_3_1')(flat_3_1)
    drop_3_1 = Dropout(DENSE_DROP_PERCENT, name='drop_3_1')(d_3_1)
    d_3_2 = Dense(2048, activation='linear', name='d_3_2')(drop_3_1)
    drop_3_2 = Dropout(DENSE_DROP_PERCENT, name='drop_3_2')(d_3_2)
    d_3_3 = Dense(512, activation='linear', name='d_3_3')(drop_3_2)
    d_3_4 = Dense(128, activation='linear', name='d_3_4')(d_3_3)
    d_3_5 = Dense(32, activation='linear', name='d_3_5')(d_3_4)
    d_3_6 = Dense(8, activation='linear', name='d_3_6')(d_3_5)
    
    output_3_1 = Dense(1, activation='sigmoid', name='output_3_1')(d_3_6)
    
    return keras.Model([input_time, input_amp], output_3_1, name='model')

def get_model_v3():
    # создание архитектуры модели
    #DROP_PERCENT = 0
    # 1 подсеть //////////////////////////////////////////////////
    input_time = Input((8,8,32), name = 'input_time')
    
    conv_1_1 = Conv2D(1024, (3,3), activation='relu', name='conv_1_1')(input_time)
    conv_1_2 = Conv2D(1024, (3,3), activation='relu', name='conv_1_2')(conv_1_1)
    conv_1_3 = Conv2D(1024, (3,3), activation='relu', name='conv_1_3')(conv_1_2)
    pool_1_1 = MaxPooling2D((2,2), strides=2, name='pool_1_1')(conv_1_3)
    
    # 2 подсеть //////////////////////////////////////////////////
    input_amp = Input((8,8,32), name = 'input_amp')
    
    conv_2_1 = Conv2D(1024, (3,3), activation='relu', name='conv_2_1')(input_amp)
    conv_2_2 = Conv2D(1024, (3,3), activation='relu', name='conv_2_2')(conv_2_1)
    conv_2_3 = Conv2D(1024, (3,3), activation='relu', name='conv_2_3')(conv_2_2)
    pool_2_1 = MaxPooling2D((2,2), strides=2, name='pool_2_1')(conv_2_3)
    
    # выходная подсеть //////////////////////////////////////////////////
    
    conc_3_1 = concatenate([pool_1_1, pool_2_1], axis=3, name='conc_3_1')
    flat_3_1 = Flatten(name='flat')(conc_3_1)
    
    #d_3_1 = Dense(8192, activation='linear', name='d_3_1')(flat_3_1)
    d_3_1 = Dense(4096, activation='linear', name='d_3_1')(flat_3_1)
    d_3_2 = Dense(2048, activation='linear', name='d_3_2')(d_3_1)
    d_3_3 = Dense(512, activation='linear', name='d_3_3')(d_3_2)
    d_3_4 = Dense(128, activation='linear', name='d_3_4')(d_3_3)
    d_3_5 = Dense(32, activation='linear', name='d_3_5')(d_3_4)
    d_3_6 = Dense(8, activation='linear', name='d_3_6')(d_3_5)
    
    output_3_1 = Dense(1, activation='sigmoid', name='output_3_1')(d_3_6)
    
    return keras.Model([input_time, input_amp], output_3_1, name='model')

def get_model_v4():
    # создание архитектуры модели
    # 1 подсеть //////////////////////////////////////////////////
    input_time = Input((16,16,32), name = 'input_time')
    
    # 12
    dconv_1_1 = Conv2D(64, (3,3), dilation_rate=(2, 2), activation='relu', name='dconv_1_1')(input_time)
    
    # 6
    dconv_1_2 = Conv2D(64, (3,3), dilation_rate=(5, 5), activation='relu', name='dconv_1_2')(input_time)
    up_1_2 = UpSampling2D(2, interpolation='bilinear', name='up_1_2') (dconv_1_2)
    
    # 4
    dconv_1_3 = Conv2D(64, (3,3), dilation_rate=(6, 6), activation='relu', name='dconv_1_3')(input_time)
    up_1_3 = UpSampling2D(3, interpolation='bilinear', name='up_1_3') (dconv_1_3)
    
    # 2
    dconv_1_4 = Conv2D(64, (3,3), dilation_rate=(7, 7), activation='relu', name='dconv_1_4')(input_time)
    up_1_4 = UpSampling2D(6, interpolation='bilinear', name='up_1_4') (dconv_1_4)
    
    conc_1_1 = concatenate([dconv_1_1, up_1_2, up_1_3, up_1_4],axis=3, name='conc_1_1')
    
    conv_1_1 = Conv2D(512, (3,3), activation='relu', name='conv_1_1')(conc_1_1)
    conv_1_2 = Conv2D(512, (3,3), activation='relu', name='conv_1_2')(conv_1_1)
    pool_1_1 = MaxPooling2D((2,2), strides=2, name='pool_1_1')(conv_1_2)
    
    conv_1_3 = Conv2D(1024, (3,3), activation='relu', name='conv_1_3')(pool_1_1)
    pool_1_2 = MaxPooling2D((2,2), strides=2, name='pool_1_2')(conv_1_3)
    
    # 2 подсеть //////////////////////////////////////////////////
    input_amp = Input((16,16,32), name = 'input_amp')
    
    # 12
    dconv_2_1 = Conv2D(64, (3,3), dilation_rate=(2, 2), activation='linear', name='dconv_2_1')(input_amp)
    
    # 6
    dconv_2_2 = Conv2D(64, (3,3), dilation_rate=(5, 5), activation='linear', name='dconv_2_2')(input_amp)
    up_2_2 = UpSampling2D(2, interpolation='bilinear', name='up_2_2') (dconv_2_2)
    
    # 4
    dconv_2_3 = Conv2D(64, (3,3), dilation_rate=(6, 6), activation='linear', name='dconv_2_3')(input_amp)
    up_2_3= UpSampling2D(3, interpolation='bilinear', name='up_2_3') (dconv_2_3)
    
    # 2
    dconv_2_4 = Conv2D(64, (3,3), dilation_rate=(7, 7), activation='linear', name='dconv_2_4')(input_amp)
    up_2_4 = UpSampling2D(6, interpolation='bilinear', name='up_2_4') (dconv_2_4)
    
    conc_2_1 = concatenate([dconv_2_1, up_2_2, up_2_3, up_2_4],axis=3, name='conc_2_1')
    
    conv_2_1 = Conv2D(512, (3,3), activation='linear', name='conv_2_1')(conc_2_1)
    conv_2_2 = Conv2D(512, (3,3), activation='linear', name='conv_2_2')(conv_2_1)
    pool_2_1 = MaxPooling2D((2,2), strides=2, name='pool_2_1')(conv_2_2)
    
    conv_2_3 = Conv2D(1024, (3,3), activation='linear', name='conv_2_3')(pool_2_1)
    pool_2_2 = MaxPooling2D((2,2), strides=2, name='pool_2_2')(conv_2_3)
    
    # выходная подсеть //////////////////////////////////////////////////
    
    conc_3_1 = concatenate([pool_1_2, pool_2_2], axis=3, name='conc_3_1')
    flat_3_1 = Flatten(name='flat')(conc_3_1)
    
    d_3_1 = Dense(4096, activation='linear', name='d_3_1')(flat_3_1)
    d_3_2 = Dense(1024, activation='linear', name='d_3_2')(d_3_1)
    d_3_3 = Dense(128, activation='linear', name='d_3_3')(d_3_2)
    d_3_4 = Dense(32, activation='linear', name='d_3_4')(d_3_3)
    
    output_3_1 = Dense(1, activation='sigmoid', name='output_3_1')(d_3_4)
    
    return keras.Model([input_time, input_amp], output_3_1, name='model')

def get_model_v5():
    # создание архитектуры модели
    # 1 подсеть //////////////////////////////////////////////////
    input_time = Input((16,16,32), name = 'input_time')
    
    # 12
    dconv_1_1 = Conv2D(128, (3,3), dilation_rate=(2, 2), activation='relu', name='dconv_1_1')(input_time)
    
    # 6
    dconv_1_2 = Conv2D(128, (3,3), dilation_rate=(5, 5), activation='relu', name='dconv_1_2')(input_time)
    up_1_2 = UpSampling2D(2, interpolation='bilinear', name='up_1_2') (dconv_1_2)
    
    # 4
    dconv_1_3 = Conv2D(128, (3,3), dilation_rate=(6, 6), activation='relu', name='dconv_1_3')(input_time)
    up_1_3 = UpSampling2D(3, interpolation='bilinear', name='up_1_3') (dconv_1_3)
    
    # 2
    dconv_1_4 = Conv2D(128, (3,3), dilation_rate=(7, 7), activation='relu', name='dconv_1_4')(input_time)
    up_1_4 = UpSampling2D(6, interpolation='bilinear', name='up_1_4') (dconv_1_4)
    
    conc_1_1 = concatenate([dconv_1_1, up_1_2, up_1_3, up_1_4],axis=3, name='conc_1_1')
    
    conv_1_1 = Conv2D(512, (3,3), activation='relu', name='conv_1_1')(conc_1_1)
    conv_1_2 = Conv2D(512, (3,3), activation='relu', name='conv_1_2')(conv_1_1)
    pool_1_1 = MaxPooling2D((2,2), strides=2, name='pool_1_1')(conv_1_2)
    
    conv_1_3 = Conv2D(1024, (3,3), activation='relu', name='conv_1_3')(pool_1_1)
    pool_1_2 = MaxPooling2D((2,2), strides=2, name='pool_1_2')(conv_1_3)
    
    # 2 подсеть //////////////////////////////////////////////////
    input_amp = Input((16,16,32), name = 'input_amp')
    
    # 12
    dconv_2_1 = Conv2D(128, (3,3), dilation_rate=(2, 2), activation='linear', name='dconv_2_1')(input_amp)
    
    # 6
    dconv_2_2 = Conv2D(128, (3,3), dilation_rate=(5, 5), activation='linear', name='dconv_2_2')(input_amp)
    up_2_2 = UpSampling2D(2, interpolation='bilinear', name='up_2_2') (dconv_2_2)
    
    # 4
    dconv_2_3 = Conv2D(128, (3,3), dilation_rate=(6, 6), activation='linear', name='dconv_2_3')(input_amp)
    up_2_3= UpSampling2D(3, interpolation='bilinear', name='up_2_3') (dconv_2_3)
    
    # 2
    dconv_2_4 = Conv2D(128, (3,3), dilation_rate=(7, 7), activation='linear', name='dconv_2_4')(input_amp)
    up_2_4 = UpSampling2D(6, interpolation='bilinear', name='up_2_4') (dconv_2_4)
    
    conc_2_1 = concatenate([dconv_2_1, up_2_2, up_2_3, up_2_4],axis=3, name='conc_2_1')
    
    conv_2_1 = Conv2D(512, (3,3), activation='linear', name='conv_2_1')(conc_2_1)
    conv_2_2 = Conv2D(512, (3,3), activation='linear', name='conv_2_2')(conv_2_1)
    pool_2_1 = MaxPooling2D((2,2), strides=2, name='pool_2_1')(conv_2_2)
    
    conv_2_3 = Conv2D(1024, (3,3), activation='linear', name='conv_2_3')(pool_2_1)
    pool_2_2 = MaxPooling2D((2,2), strides=2, name='pool_2_2')(conv_2_3)
    
    # выходная подсеть //////////////////////////////////////////////////
    
    conc_3_1 = concatenate([pool_1_2, pool_2_2], axis=3, name='conc_3_1')
    flat_3_1 = Flatten(name='flat')(conc_3_1)
    
    d_3_1 = Dense(4096, activation='linear', name='d_3_1')(flat_3_1)
    d_3_2 = Dense(1024, activation='linear', name='d_3_2')(d_3_1)
    d_3_3 = Dense(128, activation='linear', name='d_3_3')(d_3_2)
    d_3_4 = Dense(32, activation='linear', name='d_3_4')(d_3_3)
    
    output_3_1 = Dense(1, activation='sigmoid', name='output_3_1')(d_3_4)
    
    return keras.Model([input_time, input_amp], output_3_1, name='model')

def get_model_v6():
    # создание архитектуры модели
    # 1 подсеть //////////////////////////////////////////////////
    input_time = Input((16,16,32), name = 'input_time')
    
    # 12
    dconv_1_1 = Conv2D(128, (3,3), dilation_rate=(2, 2), activation='relu', name='dconv_1_1')(input_time)
    
    # 6
    dconv_1_2 = Conv2D(128, (3,3), dilation_rate=(5, 5), activation='relu', name='dconv_1_2')(input_time)
    up_1_2 = UpSampling2D(2, interpolation='bilinear', name='up_1_2') (dconv_1_2)
    
    # 4
    dconv_1_3 = Conv2D(128, (3,3), dilation_rate=(6, 6), activation='relu', name='dconv_1_3')(input_time)
    up_1_3 = UpSampling2D(3, interpolation='bilinear', name='up_1_3') (dconv_1_3)
    
    # 2
    dconv_1_4 = Conv2D(128, (3,3), dilation_rate=(7, 7), activation='relu', name='dconv_1_4')(input_time)
    up_1_4 = UpSampling2D(6, interpolation='bilinear', name='up_1_4') (dconv_1_4)
    
    conc_1_1 = concatenate([dconv_1_1, up_1_2, up_1_3, up_1_4],axis=3, name='conc_1_1')
    
    conv_1_1 = Conv2D(512, (3,3), activation='relu', name='conv_1_1')(conc_1_1)
    conv_1_2 = Conv2D(512, (3,3), activation='relu', name='conv_1_2')(conv_1_1)
    pool_1_1 = MaxPooling2D((2,2), strides=2, name='pool_1_1')(conv_1_2)
    
    conv_1_3 = Conv2D(1024, (3,3), activation='relu', name='conv_1_3')(pool_1_1)
    pool_1_2 = MaxPooling2D((2,2), strides=2, name='pool_1_2')(conv_1_3)
    
    # 2 подсеть //////////////////////////////////////////////////
    input_amp = Input((16,16,32), name = 'input_amp')
    
    # 12
    dconv_2_1 = Conv2D(128, (3,3), dilation_rate=(2, 2), activation='linear', name='dconv_2_1')(input_amp)
    
    # 6
    dconv_2_2 = Conv2D(128, (3,3), dilation_rate=(5, 5), activation='linear', name='dconv_2_2')(input_amp)
    up_2_2 = UpSampling2D(2, interpolation='bilinear', name='up_2_2') (dconv_2_2)
    
    # 4
    dconv_2_3 = Conv2D(128, (3,3), dilation_rate=(6, 6), activation='linear', name='dconv_2_3')(input_amp)
    up_2_3= UpSampling2D(3, interpolation='bilinear', name='up_2_3') (dconv_2_3)
    
    # 2
    dconv_2_4 = Conv2D(128, (3,3), dilation_rate=(7, 7), activation='linear', name='dconv_2_4')(input_amp)
    up_2_4 = UpSampling2D(6, interpolation='bilinear', name='up_2_4') (dconv_2_4)
    
    conc_2_1 = concatenate([dconv_2_1, up_2_2, up_2_3, up_2_4],axis=3, name='conc_2_1')
    
    conv_2_1 = Conv2D(512, (3,3), activation='linear', name='conv_2_1')(conc_2_1)
    conv_2_2 = Conv2D(512, (3,3), activation='linear', name='conv_2_2')(conv_2_1)
    pool_2_1 = MaxPooling2D((2,2), strides=2, name='pool_2_1')(conv_2_2)
    
    conv_2_3 = Conv2D(1024, (3,3), activation='linear', name='conv_2_3')(pool_2_1)
    pool_2_2 = MaxPooling2D((2,2), strides=2, name='pool_2_2')(conv_2_3)
    
    conc_2_2 = concatenate([pool_1_2, pool_2_2], axis=3, name='conc_2_2')
    
    # выходная подсеть по наличию дефекта //////////////////////////////////////////////////
    
    flat_3_1 = Flatten(name='flat_3_1')(conc_2_2)
    
    d_3_1 = Dense(4096, activation='linear', name='d_3_1')(flat_3_1)
    d_3_2 = Dense(1024, activation='linear', name='d_3_2')(d_3_1)
    d_3_3 = Dense(128, activation='linear', name='d_3_3')(d_3_2)
    d_3_4 = Dense(32, activation='linear', name='d_3_4')(d_3_3)
    
    output_3_1 = Dense(1, activation='sigmoid', name='output_3_1')(d_3_4)
    
    # выходная подсеть по глубине дефекта //////////////////////////////////////////////////
    
    flat_4_1 = Flatten(name='flat_4_1')(conc_2_2)
    
    d_4_1 = Dense(4096, activation='linear', name='d_4_1')(flat_4_1)
    d_4_2 = Dense(1024, activation='linear', name='d_4_2')(d_4_1)
    d_4_3 = Dense(128, activation='linear', name='d_4_3')(d_4_2)
    d_4_4 = Dense(32, activation='linear', name='d_4_4')(d_4_3)
    
    output_4_1 = Dense(1, activation='tanh', name='output_4_1')(d_4_4)
    
    return keras.Model([input_time, input_amp], [output_3_1, output_4_1], name='model')

def get_model_v7():
    # создание архитектуры модели
    # 1 подсеть //////////////////////////////////////////////////
    input_time = Input((32,32,32), name = 'input_time')
    
    dconv_1_1 = Conv2D(128, (5,5), activation='relu', padding='same', name='dconv_1_1')(input_time)
    
    dconv_1_2 = Conv2D(128, (5,5), dilation_rate=(4, 4), activation='relu', name='dconv_1_2')(input_time)
    up_1_2 = UpSampling2D(2, interpolation='bilinear', name='up_1_2') (dconv_1_2)
    
    dconv_1_3 = Conv2D(128, (5,5), dilation_rate=(6, 6), activation='relu', name='dconv_1_3')(input_time)
    up_1_3 = UpSampling2D(4, interpolation='bilinear', name='up_1_3') (dconv_1_3)
    
    dconv_1_4 = Conv2D(128, (5,5), dilation_rate=(7, 7), activation='relu', name='dconv_1_4')(input_time)
    up_1_4 = UpSampling2D(8, interpolation='bilinear', name='up_1_4') (dconv_1_4)
    
    conc_1_1 = concatenate([dconv_1_1, up_1_2, up_1_3, up_1_4], axis=3, name='conc_1_1')
    
    conv_1_1 = Conv2D(128, (3,3), activation='relu', name='conv_1_1', padding='same')(conc_1_1)
    conv_1_2 = Conv2D(128, (3,3), activation='relu', name='conv_1_2', padding='same')(conv_1_1)
    conv_1_3 = Conv2D(128, (3,3), activation='relu', name='conv_1_3', padding='same')(conv_1_2)
    pool_1_1 = MaxPooling2D((2,2), strides=2, name='pool_1_1')(conv_1_3)
    
    conv_1_4 = Conv2D(256, (3,3), activation='relu', name='conv_1_4', padding='same')(pool_1_1)
    conv_1_5 = Conv2D(256, (3,3), activation='relu', name='conv_1_5', padding='same')(conv_1_4)
    conv_1_6 = Conv2D(256, (3,3), activation='relu', name='conv_1_6', padding='same')(conv_1_5)
    pool_1_2 = MaxPooling2D((2,2), strides=2, name='pool_1_2')(conv_1_6)
    
    # 2 подсеть //////////////////////////////////////////////////
    input_amp = Input((32,32,32), name = 'input_amp')
    
    dconv_2_1 = Conv2D(128, (5,5), activation='relu', padding='same', name='dconv_2_1')(input_amp)
    
    dconv_2_2 = Conv2D(128, (5,5), dilation_rate=(4, 4), activation='relu', name='dconv_2_2')(input_amp)
    up_2_2 = UpSampling2D(2, interpolation='bilinear', name='up_2_2') (dconv_2_2)
    
    dconv_2_3 = Conv2D(128, (5,5), dilation_rate=(6, 6), activation='relu', name='dconv_2_3')(input_amp)
    up_2_3 = UpSampling2D(4, interpolation='bilinear', name='up_2_3') (dconv_2_3)
    
    dconv_2_4 = Conv2D(128, (5,5), dilation_rate=(7, 7), activation='relu', name='dconv_2_4')(input_amp)
    up_2_4 = UpSampling2D(8, interpolation='bilinear', name='up_2_4') (dconv_2_4)
    
    conc_2_1 = concatenate([dconv_2_1, up_2_2, up_2_3, up_2_4], axis=3, name='conc_2_1')
    
    conv_2_1 = Conv2D(128, (3,3), activation='relu', name='conv_2_1', padding='same')(conc_2_1)
    conv_2_2 = Conv2D(128, (3,3), activation='relu', name='conv_2_2', padding='same')(conv_2_1)
    conv_2_3 = Conv2D(128, (3,3), activation='relu', name='conv_2_3', padding='same')(conv_2_2)
    pool_2_1 = MaxPooling2D((2,2), strides=2, name='pool_2_1')(conv_2_3)
    
    conv_2_4 = Conv2D(256, (3,3), activation='relu', name='conv_2_4', padding='same')(pool_2_1)
    conv_2_5 = Conv2D(256, (3,3), activation='relu', name='conv_2_5', padding='same')(conv_2_4)
    conv_2_6 = Conv2D(256, (3,3), activation='relu', name='conv_2_6', padding='same')(conv_2_5)
    pool_2_2 = MaxPooling2D((2,2), strides=2, name='pool_2_2')(conv_2_6)
    
    conc_2_2 = concatenate([pool_1_2, pool_2_2], axis=3, name='conc_2_2')
    
    conv_3_1 = Conv2D(512, (3,3), activation='relu', name='conv_3_1', padding='same')(conc_2_2)
    conv_3_2 = Conv2D(512, (3,3), activation='relu', name='conv_3_2', padding='same')(conv_3_1)
    conv_3_3 = Conv2D(512, (3,3), activation='relu', name='conv_3_3', padding='same')(conv_3_2)
    pool_3_1 = MaxPooling2D((2,2), strides=2, name='pool_3_1')(conv_3_3)
    
    conv_3_4 = Conv2D(512, (3,3), activation='relu', name='conv_3_4', padding='same')(pool_3_1)
    conv_3_5 = Conv2D(512, (3,3), activation='relu', name='conv_3_5', padding='same')(conv_3_4)
    conv_3_6 = Conv2D(512, (3,3), activation='relu', name='conv_3_6', padding='same')(conv_3_5)
    pool_3_2 = MaxPooling2D((2,2), strides=2, name='pool_3_2')(conv_3_6)
    
    conv_3_7 = Conv2D(512, (3,3), activation='relu', name='conv_3_7', padding='same')(pool_3_2)
    conv_3_8 = Conv2D(512, (3,3), activation='relu', name='conv_3_8', padding='same')(conv_3_7)
    conv_3_9 = Conv2D(512, (3,3), activation='relu', name='conv_3_9', padding='same')(conv_3_8)
    pool_3_3 = MaxPooling2D((2,2), strides=2, name='pool_3_3')(conv_3_9)
    
    # выходная подсеть по наличию дефекта //////////////////////////////////////////////////
    
    flat_3_1 = Flatten(name='flat_3_1')(pool_3_3)
    
    d_3_1 = Dense(2048, activation='linear', name='d_3_1')(flat_3_1)
    d_3_2 = Dense(1024, activation='linear', name='d_3_2')(d_3_1)
    d_3_3 = Dense(128, activation='linear', name='d_3_3')(d_3_2)
    d_3_4 = Dense(32, activation='linear', name='d_3_4')(d_3_3)
    d_3_5 = Dense(32, activation='linear', name='d_3_6')(d_3_4)
    d_3_6 = Dense(32, activation='linear', name='d_3_5')(d_3_5)
    
    output_def_bool = Dense(1, activation='sigmoid', name='output_3_1')(d_3_6)
    
    # выходная подсеть по глубине дефекта //////////////////////////////////////////////////
    
    d_4_1 = Dense(2048, activation='linear', name='d_4_1')(flat_3_1)
    d_4_2 = Dense(1024, activation='linear', name='d_4_2')(d_4_1)
    d_4_3 = Dense(512, activation='linear', name='d_4_3')(d_4_2)
    d_4_4 = Dense(128, activation='linear', name='d_4_4')(d_4_3)
    d_4_5 = Dense(64, activation='linear', name='d_4_5')(d_4_4)
    d_4_6 = Dense(16, activation='linear', name='d_4_6')(d_4_5)
    
    output_def_depth = Dense(1, activation='tanh', name='output_4_1')(d_4_6)
    
    return keras.Model([input_time, input_amp], [output_def_bool, output_def_depth], name='model')

def get_model_v8():
    # создание архитектуры модели
    # 1 подсеть //////////////////////////////////////////////////
    input_time = Input((16,16,32), name = 'input_time')
    
    dconv_1_1 = Conv2D(128, (3,3), activation='relu', padding='same', name='dconv_1_1')(input_time)
    
    dconv_1_2 = Conv2D(128, (3,3), dilation_rate=(1, 1), activation='relu', padding='same', name='dconv_1_2')(input_time)
    
    dconv_1_3 = Conv2D(128, (3,3), dilation_rate=(2, 2), activation='relu', padding='same', name='dconv_1_3')(input_time)
    
    dconv_1_4 = Conv2D(128, (3,3), dilation_rate=(3, 3), activation='relu', padding='same', name='dconv_1_4')(input_time)
    
    dconv_1_5 = Conv2D(128, (3,3), dilation_rate=(4, 4), activation='relu', padding='same', name='dconv_1_5')(input_time)
    
    dconv_1_6 = Conv2D(128, (3,3), dilation_rate=(5, 5), activation='relu', padding='same', name='dconv_1_6')(input_time)
    
    dconv_1_7 = Conv2D(128, (3,3), dilation_rate=(6, 6), activation='relu', padding='same', name='dconv_1_7')(input_time)
    
    dconv_1_8 = Conv2D(128, (3,3), dilation_rate=(7, 7), activation='relu', padding='same', name='dconv_1_8')(input_time)
    
    
    conc_1_1 = concatenate([dconv_1_1, dconv_1_2,
                            dconv_1_3, dconv_1_4,
                            dconv_1_5, dconv_1_6,
                            dconv_1_7, dconv_1_8,], axis=3, name='conc_1_1')
    
    conv_1_1 = Conv2D(512, (3,3), activation='relu', name='conv_1_1', padding='same')(conc_1_1)
    conv_1_2 = Conv2D(512, (3,3), activation='relu', name='conv_1_2', padding='same')(conv_1_1)
    conv_1_3 = Conv2D(512, (3,3), activation='relu', name='conv_1_3', padding='same')(conv_1_2)
    pool_1_1 = MaxPooling2D((2,2), strides=2, name='pool_1_1')(conv_1_3)
    
    # 2 подсеть //////////////////////////////////////////////////
    input_amp = Input((16,16,32), name = 'input_amp')
    
    dconv_2_1 = Conv2D(128, (3,3), activation='relu', padding='same', name='dconv_2_1')(input_amp)
    
    dconv_2_2 = Conv2D(128, (3,3), dilation_rate=(1, 1), activation='relu', padding='same', name='dconv_2_2')(input_amp)
    
    dconv_2_3 = Conv2D(128, (3,3), dilation_rate=(2, 2), activation='relu', padding='same', name='dconv_2_3')(input_amp)
    
    dconv_2_4 = Conv2D(128, (3,3), dilation_rate=(3, 3), activation='relu', padding='same', name='dconv_2_4')(input_amp)
    
    dconv_2_5 = Conv2D(128, (3,3), dilation_rate=(4, 4), activation='relu', padding='same', name='dconv_2_5')(input_amp)
    
    dconv_2_6 = Conv2D(128, (3,3), dilation_rate=(5, 5), activation='relu', padding='same', name='dconv_2_6')(input_amp)
    
    dconv_2_7 = Conv2D(128, (3,3), dilation_rate=(6, 6), activation='relu', padding='same', name='dconv_2_7')(input_amp)
    
    dconv_2_8 = Conv2D(128, (3,3), dilation_rate=(7, 7), activation='relu', padding='same', name='dconv_2_8')(input_amp)
    
    
    conc_2_1 = concatenate([dconv_2_1, dconv_2_2,
                            dconv_2_3, dconv_2_4,
                            dconv_2_5, dconv_2_6,
                            dconv_2_7, dconv_2_8,], axis=3, name='conc_2_1')
    
    conv_2_1 = Conv2D(128, (3,3), activation='relu', name='conv_2_1', padding='same')(conc_2_1)
    conv_2_2 = Conv2D(128, (3,3), activation='relu', name='conv_2_2', padding='same')(conv_2_1)
    conv_2_3 = Conv2D(128, (3,3), activation='relu', name='conv_2_3', padding='same')(conv_2_2)
    pool_2_1 = MaxPooling2D((2,2), strides=2, name='pool_2_1')(conv_2_3)
    
    # общая сверточная часть
    
    conc_2_2 = concatenate([pool_1_1, pool_2_1], axis=3, name='conc_2_2')
    
    conv_3_1 = Conv2D(256, (3,3), activation='relu', name='conv_3_1', padding='same')(conc_2_2)
    conv_3_2 = Conv2D(256, (3,3), activation='relu', name='conv_3_2', padding='same')(conv_3_1)
    conv_3_3 = Conv2D(256, (3,3), activation='relu', name='conv_3_3', padding='same')(conv_3_2)
    pool_3_1 = MaxPooling2D((2,2), strides=2, name='pool_3_1')(conv_3_3)
    
    conv_3_4 = Conv2D(512, (3,3), activation='relu', name='conv_3_4', padding='same')(pool_3_1)
    conv_3_5 = Conv2D(512, (3,3), activation='relu', name='conv_3_5', padding='same')(conv_3_4)
    conv_3_6 = Conv2D(512, (3,3), activation='relu', name='conv_3_6', padding='same')(conv_3_5)
    pool_3_2 = MaxPooling2D((2,2), strides=2, name='pool_3_2')(conv_3_6)
    
    conv_3_7 = Conv2D(1024, (3,3), activation='relu', name='conv_3_7', padding='same')(pool_3_2)
    conv_3_8 = Conv2D(1024, (3,3), activation='relu', name='conv_3_8', padding='same')(conv_3_7)
    conv_3_9 = Conv2D(1024, (3,3), activation='relu', name='conv_3_9', padding='same')(conv_3_8)
    pool_3_3 = MaxPooling2D((2,2), strides=2, name='pool_3_3')(conv_3_9)
    
    
    # выходная подсеть по наличию дефекта //////////////////////////////////////////////////
    
    flat_3_1 = Flatten(name='flat_3_1')(pool_3_3)
    
    d_3_1 = Dense(2048, activation='linear', name='d_3_1')(flat_3_1)
    d_3_2 = Dense(1024, activation='linear', name='d_3_2')(d_3_1)
    d_3_3 = Dense(128, activation='linear', name='d_3_3')(d_3_2)
    d_3_4 = Dense(32, activation='linear', name='d_3_4')(d_3_3)
    d_3_5 = Dense(32, activation='linear', name='d_3_6')(d_3_4)
    d_3_6 = Dense(32, activation='linear', name='d_3_5')(d_3_5)
    
    output_def_bool = Dense(1, activation='sigmoid', name='output_3_1')(d_3_6)
    
    # выходная подсеть по глубине дефекта //////////////////////////////////////////////////
    
    d_4_1 = Dense(2048, activation='linear', name='d_4_1')(flat_3_1)
    d_4_2 = Dense(1024, activation='linear', name='d_4_2')(d_4_1)
    d_4_3 = Dense(512, activation='linear', name='d_4_3')(d_4_2)
    d_4_4 = Dense(128, activation='linear', name='d_4_4')(d_4_3)
    d_4_5 = Dense(64, activation='linear', name='d_4_5')(d_4_4)
    d_4_6 = Dense(16, activation='linear', name='d_4_6')(d_4_5)
    
    output_def_depth = Dense(1, activation='tanh', name='output_4_1')(d_4_6)
    
    return keras.Model([input_time, input_amp], [output_def_bool, output_def_depth], name='model')

def get_model_v9():
    # создание архитектуры модели
    # 1 подсеть //////////////////////////////////////////////////
    DROP = 0
    
    input_time = Input((16,16,32), name = 'input_time')
    
    dconv_1_1 = Conv2D(128, (3,3), activation='relu', padding='same', name='dconv_1_1')(input_time)
    
    dconv_1_2 = Conv2D(128, (3,3), dilation_rate=(1, 1), activation='relu', padding='same', name='dconv_1_2')(input_time)
    
    dconv_1_3 = Conv2D(128, (3,3), dilation_rate=(2, 2), activation='relu', padding='same', name='dconv_1_3')(input_time)
    
    dconv_1_4 = Conv2D(128, (3,3), dilation_rate=(3, 3), activation='relu', padding='same', name='dconv_1_4')(input_time)
    
    dconv_1_5 = Conv2D(128, (3,3), dilation_rate=(4, 4), activation='relu', padding='same', name='dconv_1_5')(input_time)
    
    dconv_1_6 = Conv2D(128, (3,3), dilation_rate=(5, 5), activation='relu', padding='same', name='dconv_1_6')(input_time)
    
    dconv_1_7 = Conv2D(128, (3,3), dilation_rate=(6, 6), activation='relu', padding='same', name='dconv_1_7')(input_time)
    
    dconv_1_8 = Conv2D(128, (3,3), dilation_rate=(7, 7), activation='relu', padding='same', name='dconv_1_8')(input_time)
    
    
    conc_1_1 = concatenate([dconv_1_1, dconv_1_2,
                            dconv_1_3, dconv_1_4,
                            dconv_1_5, dconv_1_6,
                            dconv_1_7, dconv_1_8,], axis=3, name='conc_1_1')
    
    conv_1_1 = Conv2D(256, (3,3), activation='relu', name='conv_1_1', padding='same')(conc_1_1)
    conv_1_2 = Conv2D(256, (3,3), activation='relu', name='conv_1_2', padding='same')(conv_1_1)
    conv_1_3 = Conv2D(256, (3,3), activation='relu', name='conv_1_3', padding='same')(conv_1_2)
    bnorm_1_1 = BatchNormalization(name='bnorm_1_1')(conv_1_3)
    pool_1_1 = MaxPooling2D((2,2), strides=2, name='pool_1_1')(bnorm_1_1)
    
    # 2 подсеть //////////////////////////////////////////////////
    input_amp = Input((16,16,32), name = 'input_amp')
    
    dconv_2_1 = Conv2D(128, (3,3), activation='relu', padding='same', name='dconv_2_1')(input_amp)
    
    dconv_2_2 = Conv2D(128, (3,3), dilation_rate=(1, 1), activation='relu', padding='same', name='dconv_2_2')(input_amp)
    
    dconv_2_3 = Conv2D(128, (3,3), dilation_rate=(2, 2), activation='relu', padding='same', name='dconv_2_3')(input_amp)
    
    dconv_2_4 = Conv2D(128, (3,3), dilation_rate=(3, 3), activation='relu', padding='same', name='dconv_2_4')(input_amp)
    
    dconv_2_5 = Conv2D(128, (3,3), dilation_rate=(4, 4), activation='relu', padding='same', name='dconv_2_5')(input_amp)
    
    dconv_2_6 = Conv2D(128, (3,3), dilation_rate=(5, 5), activation='relu', padding='same', name='dconv_2_6')(input_amp)
    
    dconv_2_7 = Conv2D(128, (3,3), dilation_rate=(6, 6), activation='relu', padding='same', name='dconv_2_7')(input_amp)
    
    dconv_2_8 = Conv2D(128, (3,3), dilation_rate=(7, 7), activation='relu', padding='same', name='dconv_2_8')(input_amp)
    
    
    conc_2_1 = concatenate([dconv_2_1, dconv_2_2,
                            dconv_2_3, dconv_2_4,
                            dconv_2_5, dconv_2_6,
                            dconv_2_7, dconv_2_8,], axis=3, name='conc_2_1')
    
    conv_2_1 = Conv2D(256, (3,3), activation='relu', name='conv_2_1', padding='same')(conc_2_1)
    conv_2_2 = Conv2D(256, (3,3), activation='relu', name='conv_2_2', padding='same')(conv_2_1)
    conv_2_3 = Conv2D(256, (3,3), activation='relu', name='conv_2_3', padding='same')(conv_2_2)
    bnorm_2_1 = BatchNormalization(name='bnorm_2_1')(conv_2_3)
    pool_2_1 = MaxPooling2D((2,2), strides=2, name='pool_2_1')(bnorm_2_1)
    
    # общая сверточная часть
    
    conc_2_2 = concatenate([pool_1_1, pool_2_1], axis=3, name='conc_2_2')
    
    conv_3_1 = Conv2D(256, (3,3), activation='relu', name='conv_3_1', padding='same')(conc_2_2)
    conv_3_2 = Conv2D(256, (3,3), activation='relu', name='conv_3_2', padding='same')(conv_3_1)
    conv_3_3 = Conv2D(256, (3,3), activation='relu', name='conv_3_3', padding='same')(conv_3_2)
    bnorm_3_1 = BatchNormalization(name='bnorm_3_1')(conv_3_3)
    pool_3_1 = MaxPooling2D((2,2), strides=2, name='pool_3_1')(bnorm_3_1)
    
    conv_3_4 = Conv2D(512, (3,3), activation='relu', name='conv_3_4', padding='same')(pool_3_1)
    conv_3_5 = Conv2D(512, (3,3), activation='relu', name='conv_3_5', padding='same')(conv_3_4)
    conv_3_6 = Conv2D(512, (3,3), activation='relu', name='conv_3_6', padding='same')(conv_3_5)
    bnorm_3_2 = BatchNormalization(name='bnorm_3_2')(conv_3_6)
    pool_3_2 = MaxPooling2D((2,2), strides=2, name='pool_3_2')(bnorm_3_2)
    
    conv_3_7 = Conv2D(1024, (3,3), activation='relu', name='conv_3_7', padding='same')(pool_3_2)
    conv_3_8 = Conv2D(1024, (3,3), activation='relu', name='conv_3_8', padding='same')(conv_3_7)
    conv_3_9 = Conv2D(1024, (3,3), activation='relu', name='conv_3_9', padding='same')(conv_3_8)
    bnorm_3_3 = BatchNormalization(name='bnorm_3_3')(conv_3_9)
    pool_3_3 = MaxPooling2D((2,2), strides=2, name='pool_3_3')(bnorm_3_3)
    
    
    # выходная подсеть по наличию дефекта //////////////////////////////////////////////////
    
    flat_3_1 = Flatten(name='flat_3_1')(pool_3_3)
    
    d_3_1 = Dense(4096, activation='linear', name='d_3_1')(flat_3_1)
    d_3_2 = Dense(2048, activation='linear', name='d_3_2')(d_3_1)
    d_3_3 = Dense(256, activation='linear', name='d_3_3')(d_3_2)
    d_3_4 = Dense(128, activation='linear', name='d_3_4')(d_3_3)
    d_3_5 = Dense(64, activation='linear', name='d_3_5')(d_3_4)
    d_3_6 = Dense(32, activation='linear', name='d_3_6')(d_3_5)
    d_3_7 = Dense(8, activation='linear', name='d_3_7')(d_3_6)
    
    output_def_bool = Dense(1, activation='sigmoid', name='output_3_1')(d_3_7)
    
    return keras.Model([input_time, input_amp], [output_def_bool], name='model')

def get_model_v10(crop_size: int):
    # создание архитектуры модели
    # 1 подсеть //////////////////////////////////////////////////
    DROP = 0
    
    input_time = Input((crop_size,crop_size,32), name = 'input_time')
    
    dconv_1_1 = Conv2D(128, (3,3), activation='relu', padding='same', name='dconv_1_1')(input_time)
    dconv_1_2 = Conv2D(128, (3,3), dilation_rate=(1, 1), activation='relu', padding='same', name='dconv_1_2')(input_time)
    dconv_1_3 = Conv2D(128, (3,3), dilation_rate=(2, 2), activation='relu', padding='same', name='dconv_1_3')(input_time)
    dconv_1_4 = Conv2D(128, (3,3), dilation_rate=(3, 3), activation='relu', padding='same', name='dconv_1_4')(input_time)
    dconv_1_5 = Conv2D(128, (3,3), dilation_rate=(4, 4), activation='relu', padding='same', name='dconv_1_5')(input_time)
    dconv_1_6 = Conv2D(128, (3,3), dilation_rate=(5, 5), activation='relu', padding='same', name='dconv_1_6')(input_time)
    dconv_1_7 = Conv2D(128, (3,3), dilation_rate=(6, 6), activation='relu', padding='same', name='dconv_1_7')(input_time)
    dconv_1_8 = Conv2D(128, (3,3), dilation_rate=(7, 7), activation='relu', padding='same', name='dconv_1_8')(input_time)
    
    conc_1_1 = concatenate([dconv_1_1, dconv_1_2,
                            dconv_1_3, dconv_1_4,
                            dconv_1_5, dconv_1_6,
                            dconv_1_7, dconv_1_8], axis=3, name='conc_1_1')
    
    conv_1_1 = Conv2D(256, (3,3), activation='relu', name='conv_1_1', padding='same')(conc_1_1)
    conv_1_2 = Conv2D(256, (3,3), activation='relu', name='conv_1_2', padding='same')(conv_1_1)
    
    conc_1_2 = concatenate([conv_1_1, conv_1_2], axis=3, name='conc_1_2')
    conv_1_3 = Conv2D(256, (3,3), activation='relu', name='conv_1_3', padding='same')(conc_1_2)
    
    conv_1_4 = Conv2D(256, (3,3), activation='relu', name='conv_1_4', padding='same')(conv_1_3)
    
    conc_1_3 = concatenate([conv_1_3, conv_1_4], axis=3, name='conc_1_3')
    conv_1_5 = Conv2D(256, (3,3), activation='relu', name='conv_1_5', padding='same')(conc_1_3)
    bnorm_1_1 = BatchNormalization(name='bnorm_1_1')(conv_1_5)
    pool_1_1 = MaxPooling2D((2,2), strides=2, name='pool_1_1')(bnorm_1_1)
    
    # 2 подсеть //////////////////////////////////////////////////
    input_amp = Input((crop_size,crop_size,32), name = 'input_amp')
    
    dconv_2_1 = Conv2D(128, (3,3), activation='relu', padding='same', name='dconv_2_1')(input_amp)
    dconv_2_2 = Conv2D(128, (3,3), dilation_rate=(1, 1), activation='relu', padding='same', name='dconv_2_2')(input_amp)
    dconv_2_3 = Conv2D(128, (3,3), dilation_rate=(2, 2), activation='relu', padding='same', name='dconv_2_3')(input_amp)
    dconv_2_4 = Conv2D(128, (3,3), dilation_rate=(3, 3), activation='relu', padding='same', name='dconv_2_4')(input_amp)
    dconv_2_5 = Conv2D(128, (3,3), dilation_rate=(4, 4), activation='relu', padding='same', name='dconv_2_5')(input_amp)
    dconv_2_6 = Conv2D(128, (3,3), dilation_rate=(5, 5), activation='relu', padding='same', name='dconv_2_6')(input_amp)
    dconv_2_7 = Conv2D(128, (3,3), dilation_rate=(6, 6), activation='relu', padding='same', name='dconv_2_7')(input_amp)
    dconv_2_8 = Conv2D(128, (3,3), dilation_rate=(7, 7), activation='relu', padding='same', name='dconv_2_8')(input_amp)
    
    conc_2_1 = concatenate([dconv_2_1, dconv_2_2,
                            dconv_2_3, dconv_2_4,
                            dconv_2_5, dconv_2_6,
                            dconv_2_7, dconv_2_8], axis=3, name='conc_2_1')
    
    conv_2_1 = Conv2D(256, (3,3), activation='relu', name='conv_2_1', padding='same')(conc_2_1)
    conv_2_2 = Conv2D(256, (3,3), activation='relu', name='conv_2_2', padding='same')(conv_2_1)
    
    conc_2_2 = concatenate([conv_2_1, conv_2_2], axis=3, name='conc_2_2')
    conv_2_3 = Conv2D(256, (3,3), activation='relu', name='conv_2_3', padding='same')(conc_2_2)
    
    conv_2_4 = Conv2D(256, (3,3), activation='relu', name='conv_2_4', padding='same')(conv_2_3)
    
    conc_2_3 = concatenate([conv_2_3, conv_2_4], axis=3, name='conc_2_3')
    conv_2_5 = Conv2D(256, (3,3), activation='relu', name='conv_2_5', padding='same')(conc_2_3)
    bnorm_2_1 = BatchNormalization(name='bnorm_2_1')(conv_2_5)
    pool_2_1 = MaxPooling2D((2,2), strides=2, name='pool_2_1')(bnorm_2_1)
    
    # общая сверточная часть
    
    conc_3_1 = concatenate([pool_1_1, pool_2_1], axis=3, name='conc_3_1')
    
    conv_3_1 = Conv2D(256, (3,3), activation='relu', name='conv_3_1', padding='same')(conc_3_1)
    conv_3_2 = Conv2D(256, (3,3), activation='relu', name='conv_3_2', padding='same')(conv_3_1)
    
    conc_3_2 = concatenate([conv_3_1, conv_3_2], axis=3, name='conc_3_2')
    conv_3_3 = Conv2D(256, (3,3), activation='relu', name='conv_3_3', padding='same')(conc_3_2)
    
    conv_3_4 = Conv2D(256, (3,3), activation='relu', name='conv_3_4', padding='same')(conv_3_3)
    
    conc_3_3 = concatenate([conv_3_3, conv_3_4], axis=3, name='conc_3_3')
    conv_3_5 = Conv2D(256, (3,3), activation='relu', name='conv_3_5', padding='same')(conc_3_3)
    bnorm_3_1 = BatchNormalization(name='bnorm_3_1')(conv_3_5)
    pool_3_1 = MaxPooling2D((2,2), strides=2, name='pool_3_1')(bnorm_3_1)
    
    
    
    conv_3_6 = Conv2D(512, (3,3), activation='relu', name='conv_3_6', padding='same')(pool_3_1)
    conv_3_7 = Conv2D(512, (3,3), activation='relu', name='conv_3_7', padding='same')(conv_3_6)
    
    conc_3_4 = concatenate([conv_3_6, conv_3_7], axis=3, name='conc_3_4')
    conv_3_8 = Conv2D(512, (3,3), activation='relu', name='conv_3_8', padding='same')(conc_3_4)
    
    conv_3_9 = Conv2D(512, (3,3), activation='relu', name='conv_3_9', padding='same')(conv_3_8)
    
    conc_3_5 = concatenate([conv_3_8, conv_3_9], axis=3, name='conc_3_5')
    conv_3_10 = Conv2D(512, (3,3), activation='relu', name='conv_3_10', padding='same')(conc_3_5)
    bnorm_3_2 = BatchNormalization(name='bnorm_3_2')(conv_3_10)
    pool_3_2 = MaxPooling2D((2,2), strides=2, name='pool_3_2')(bnorm_3_2)
    
    
    
    conv_3_11 = Conv2D(1024, (3,3), activation='relu', name='conv_3_11', padding='same')(pool_3_2)
    conv_3_12 = Conv2D(1024, (3,3), activation='relu', name='conv_3_12', padding='same')(conv_3_11)
    
    conc_3_6 = concatenate([conv_3_11, conv_3_12], axis=3, name='conc_3_6')
    conv_3_13 = Conv2D(1024, (3,3), activation='relu', name='conv_3_13', padding='same')(conc_3_6)
    
    conv_3_14 = Conv2D(1024, (3,3), activation='relu', name='conv_3_14', padding='same')(conv_3_13)
    
    conc_3_7 = concatenate([conv_3_13, conv_3_14], axis=3, name='conc_3_7')
    conv_3_15 = Conv2D(1024, (3,3), activation='relu', name='conv_3_15', padding='same')(conc_3_7)
    bnorm_3_3 = BatchNormalization(name='bnorm_3_3')(conv_3_15)
    pool_3_3 = MaxPooling2D((2,2), strides=2, name='pool_3_3')(bnorm_3_3)
    
    
    # выходная подсеть по наличию дефекта //////////////////////////////////////////////////
    
    flat_3_1 = Flatten(name='flat_3_1')(pool_3_3)
    
    d_3_1 = Dense(4096, activation='linear', name='d_3_1')(flat_3_1)
    d_3_2 = Dense(2048, activation='linear', name='d_3_2')(d_3_1)
    d_3_3 = Dense(256, activation='linear', name='d_3_3')(d_3_2)
    d_3_4 = Dense(128, activation='linear', name='d_3_4')(d_3_3)
    d_3_5 = Dense(64, activation='linear', name='d_3_5')(d_3_4)
    d_3_6 = Dense(32, activation='linear', name='d_3_6')(d_3_5)
    d_3_7 = Dense(8, activation='linear', name='d_3_7')(d_3_6)
    
    output_def_bool = Dense(1, activation='sigmoid', name='output_3_1')(d_3_7)
    
    return keras.Model([input_time, input_amp], [output_def_bool], name='model')