import time

import keras
import numpy as np
from keras import backend as K
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from keras.metrics import categorical_accuracy
from keras.optimizers import Adam
from keras.utils.data_utils import get_file
from PIL import Image

from nets.pspnet import pspnet
from nets.pspnet_training import CE, Generator, dice_loss_with_CE
from utils.metrics import Iou_score, f_score

if __name__ == "__main__":     
    #   模型保存地址
    log_dir = "logs/"
    #   图片格式
    inputs_size = [256,256,3]
    #   分类个数（背景也算一个类别）
    #   num_classes=种类数+1（背景）
    num_classes =2
    #   model选择：mobilenet，resnet50
    #   mobilenet（快），resnet50（慢）
    backbone = "mobilenet"
    #   损失函数
    #   种类少（几类）时，设置为True
    #   种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
    #   种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
    dice_loss = False
    #-----------------------------------------------------------#

    aux_branch = True
    downsample_factor = 16

    model = pspnet(num_classes,inputs_size,downsample_factor=downsample_factor,backbone=backbone,aux_branch=aux_branch)
    # 初始model地址
    model_path = "model/mobilenetv2.h5"
    model.load_weights(model_path, by_name=True, skip_mismatch=True)

    # 打开训练集
    with open("dataset_processing/classfication/ImageSets/Segmentation/train.txt","r") as f:
        train_lines = f.readlines()

    # 打开验证集
    with open("dataset_processing/classfication/ImageSets/Segmentation/val.txt","r") as f:
        val_lines = f.readlines()


    checkpoint_period = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    tensorboard = TensorBoard(log_dir=log_dir)

    if backbone=="mobilenet":
        freeze_layers = 146
    else:
        freeze_layers = 172

    for i in range(freeze_layers):
        model.layers[i].trainable = False
    print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model.layers)))


    if True:
        lr = 1e-4
        Init_Epoch = 0
        Freeze_Epoch = 50
        Batch_size = 8

        model.compile(loss = dice_loss_with_CE() if dice_loss else CE(),
                optimizer = Adam(lr=lr),
                metrics = [f_score()])
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(len(train_lines), len(val_lines), Batch_size))

        gen = Generator(Batch_size, train_lines, inputs_size, num_classes,aux_branch).generate()
        gen_val = Generator(Batch_size, val_lines, inputs_size, num_classes,aux_branch).generate(False)

        model.fit_generator(gen,
                steps_per_epoch=max(1, len(train_lines)//Batch_size),
                validation_data=gen_val,
                validation_steps=max(1, len(val_lines)//Batch_size),
                epochs=Freeze_Epoch,
                initial_epoch=Init_Epoch,
                callbacks=[checkpoint_period, reduce_lr,tensorboard])

    for i in range(freeze_layers):
        model.layers[i].trainable = True

    if True:
        lr = 1e-5
        Freeze_Epoch = 50
        Unfreeze_Epoch = 100
        Batch_size = 4

        model.compile(loss = dice_loss_with_CE() if dice_loss else CE(),
                optimizer = Adam(lr=lr),
                metrics = [f_score()])
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(len(train_lines), len(val_lines), Batch_size))

        gen = Generator(Batch_size, train_lines, inputs_size, num_classes,aux_branch).generate()
        gen_val = Generator(Batch_size, val_lines, inputs_size, num_classes,aux_branch).generate(False)

        model.fit_generator(gen,
                steps_per_epoch=max(1, len(train_lines)//Batch_size),
                validation_data=gen_val,
                validation_steps=max(1, len(val_lines)//Batch_size),
                epochs=Unfreeze_Epoch,
                initial_epoch=Freeze_Epoch,
                callbacks=[checkpoint_period, reduce_lr,tensorboard])


