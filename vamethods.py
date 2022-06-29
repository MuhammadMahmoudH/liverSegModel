    
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

# def modelType(model):
#     modelType = {
#         1: 'unet',
#         2: 'deeplabv3_plus',
#         3: 'unet++',
#         4: 'LSTM',
#         5: 'BCDU_net_D3',
#         6: 'R-cnn',
#         7: 'pretrained_LSTM_UNet',
#     }

def modelType(model = 21):
    modelType = {
        1: 'unet_lits',
        2: 'deeplabv3_plus_ResNet50',
        21: 'deeplabv3_plus_ResNet50_relu',
        3: 'unet++',
        4: 'LSTM',
        5: 'BCDU_net_D3',
        6: 'R-cnn',
        7: 'pretrained_LSTM_UNet',
        8: 'pretrained_LSTM_UNet_dcm',
        9: 'deeplabv3_plus_dcm',
        10: 'unet_list',
        11: 'unet',
        12: 'UnetDenseNet121_dcm',
        13: 'UnetDenseNet121',
        14: 'unetplusplus_vgg16_lits',
        15: 'unetplusplus_vgg16_lits_dcm',
        16: 'vggUNet_lits',

    }


    modelSelected = modelType[model]
    return modelSelected