    
def backbone(backbone):
    backbones = {
        1: 'ResNet',
        2: 'VGG16',
        3: 'VGG19',
        4: 'InceptionNetV3',
        5: 'DenseNet121'
    }
    backboneSelected = backbones[backbone]
    return backboneSelected

def modelType(model):
    modelType = {
        1: 'unet',
        2: 'deeplabv3_plus',
        3: 'unet++',
        4: 'LSTM',
        5: 'BCDU_net_D3',
        6: 'R-cnn',
        7: 'pretrained_LSTM_UNet',
    }
    modelSelected = modelType[model]
    return modelSelected