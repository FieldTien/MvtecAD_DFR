import torchvision.models as models
import torch
import torch.nn as nn

'''
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (17): ReLU(inplace=True)
    (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (24): ReLU(inplace=True)
    (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (26): ReLU(inplace=True)
    (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (31): ReLU(inplace=True)
    (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (33): ReLU(inplace=True)
    (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (35): ReLU(inplace=True)
    (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
'''
class VGG_Extract(nn.Module):
    def __init__(self):
      super(VGG_Extract, self).__init__()
      self.model = models.vgg19(pretrained=True) 
      # hierarchy 1 (level 1)
      self.conv1_1 = self.model.features[0]
      self.relu1_1 = self.model.features[1]
      self.conv1_2 = self.model.features[2]
      self.relu1_2 = self.model.features[3]

      # hierarchy 2 (level 2)
      self.pool1 = self.model.features[4]
      self.conv2_1 = self.model.features[5]
      self.relu2_1 = self.model.features[6]
      self.conv2_2 = self.model.features[7]
      self.relu2_2 = self.model.features[8]

      # hierarchy 3 (level 3)
      self.pool2 = self.model.features[9]
      self.conv3_1 = self.model.features[10]
      self.relu3_1 = self.model.features[11]
      self.conv3_2 = self.model.features[12]
      self.relu3_2 = self.model.features[13]
      self.conv3_3 = self.model.features[14]
      self.relu3_3 = self.model.features[15]
      self.conv3_4 = self.model.features[16]
      self.relu3_4 = self.model.features[17]
      
      # hierarchy 4 (level 4)
      self.pool3 = self.model.features[18]
      self.conv4_1 = self.model.features[19]
      self.relu4_1 = self.model.features[20]
      self.conv4_2 = self.model.features[21]
      self.relu4_2 = self.model.features[22]
      self.conv4_3 = self.model.features[23]
      self.relu4_3 = self.model.features[24]
      self.conv4_4 = self.model.features[25]
      self.relu4_4 = self.model.features[26]

      # hierarchy 5 (level 5)
      self.pool4 = self.model.features[27]
      self.conv5_1 = self.model.features[28]
      self.relu5_1 = self.model.features[29]
      self.conv5_2 = self.model.features[30]
      self.relu5_2 = self.model.features[31]
      self.conv5_3 = self.model.features[32]
      self.relu5_3 = self.model.features[33]
      self.conv5_4 = self.model.features[34]
      self.relu5_4 = self.model.features[35]
    def forward(self, x, feature_layers):
        # level 1
       
        conv1_1 = self.conv1_1(x)
        relu1_1 = self.relu1_1(conv1_1)
        conv1_2 = self.conv1_2(relu1_1)
        relu1_2 = self.relu1_2(conv1_2)
        pool1 = self.pool1(relu1_2)

        # level 2
        pool1 = pool1
        conv2_1 = self.conv2_1(pool1)
        relu2_1 = self.relu2_1(conv2_1)
        conv2_2 = self.conv2_2(relu2_1)
        relu2_2 = self.relu2_2(conv2_2)
        pool2 = self.pool2(relu2_2)

        # level 3
        pool2 = pool2
        conv3_1 = self.conv3_1(pool2)
        relu3_1 = self.relu3_1(conv3_1)
        conv3_2 = self.conv3_2(relu3_1)
        relu3_2 = self.relu3_2(conv3_2)
        conv3_3 = self.conv3_3(relu3_2)
        relu3_3 = self.relu3_3(conv3_3)
        conv3_4 = self.conv3_4(relu3_3)
        relu3_4 = self.relu3_4(conv3_4)
        pool3 = self.pool3(relu3_4)

        # level 4
        pool3 = pool3
        conv4_1 = self.conv4_1(pool3)
        relu4_1 = self.relu4_1(conv4_1)
        conv4_2 = self.conv4_2(relu4_1)
        relu4_2 = self.relu4_2(conv4_2)
        conv4_3 = self.conv4_3(relu4_2)
        relu4_3 = self.relu4_3(conv4_3)
        conv4_4 = self.conv4_4(relu4_3)
        relu4_4 = self.relu4_4(conv4_4)
        pool4 = self.pool4(relu4_4)

        # level 5
        pool4 = pool4
        conv5_1 = self.conv5_1(pool4)
        relu5_1 = self.relu5_1(conv5_1)
        conv5_2 = self.conv5_2(relu5_1)
        relu5_2 = self.relu5_2(conv5_2)
        conv5_3 = self.conv5_3(relu5_2)
        relu5_3 = self.relu5_3(conv5_3)
        conv5_4 = self.conv5_4(relu5_3)
        relu5_4 = self.relu5_4(conv5_4)
        # pool5 = self.pool5(relu5_4)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        feat_maps = dict((key, value) for key, value in out.items() if key in feature_layers) 
        '''
        extract_shape = (x.shape[2]//4, x.shape[3]//4)
        features = []
        for _, feat_map in feat_maps.items():
            features.append(nn.functional.interpolate(feat_map, size=extract_shape, mode="nearest"))
        features = torch.cat(features, axis = 1) 
        '''   
        return feat_maps
  
             