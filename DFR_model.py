import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from vgg19 import VGG_Extract
from feat_cae import FeatCAE
from sklearn.decomposition import PCA
class AD_DFR(nn.Module):
    def __init__(self,cfg):
        super(AD_DFR, self).__init__()
        self.avgpool = nn.AvgPool2d((4, 4), stride=(4, 4))
        self.feature = VGG_Extract()
        for param in self.feature.parameters():
             param.requires_grad = False
             
        '''  
        self.extracted_vgg_item = ['relu1_1','relu1_2','relu2_1','relu2_2',
                                   'relu3_1','relu3_2','relu3_3','relu3_4',
                                   'relu4_1','relu4_2','relu4_3','relu4_4',
                                   'relu5_1','relu5_2','relu5_3','relu5_4']
        '''   
        self.extracted_vgg_item = cfg.backbone
        self.device = cfg.device
        self.latent_dim = cfg.latent_dim
        self.is_bn = cfg.is_bn
        #self.CAE = FeatCAE( in_channels=5504, latent_dim=self.latent_dim, is_bn=True)
        
    # TODO: using unfold, fold, then xx.mean(dim=, keepdim=True)
    def extracting(self, input):
        B, C, H, W = input.shape
        feat_maps = self.feature(input,self.extracted_vgg_item)
        features = torch.Tensor().to(self.device)
        for _, feat_map in feat_maps.items():
            feat_map = nn.functional.interpolate(feat_map, size=(H,W),mode='bilinear',align_corners=True)
            feat_map = self.avgpool(feat_map)
            features = torch.cat([features, feat_map], dim=1) 
        #features = features.detach().numpy()      
        return features
    def PCA(self, loader):
        if (self.latent_dim==None):
            feats  = torch.Tensor()
            for index,img in enumerate(loader):
              if(index > 3 ): break
              features = self.extracting(img.to(self.device))
              features = torch.flatten(features ,2).permute(0, 2, 1)
              features = torch.unbind(features, dim=0)
              features = torch.cat(features, dim=0)
              feats  = torch.cat([feats , features.to('cpu')], dim=0) 
            feats  = feats .detach().numpy()
            pca = PCA(n_components=0.90) 
            pca.fit(feats)
            n_dim, in_feat = pca.components_.shape
            self.latent_dim = n_dim
            print("AE Parameter (in_feat, n_dim): ({}, {})".format(in_feat, n_dim))
        else:
            img = next(iter(loader))
            features = self.extracting(img.to(self.device))
            in_feat =  features.shape[1]
        self.CAE = FeatCAE( in_channels=in_feat, latent_dim=self.latent_dim, is_bn=self.is_bn).to(self.device)
    def forward(self, input):
        features = self.extracting(input)
        AE_OUT = self.CAE(features)    
        # output = self.fold(input)
        return AE_OUT , features

    def MSE_loss(self, input):
        y_hat,y = self.forward(input)
        err = y_hat-y
        loss = torch.mean(err**2)
        return loss 
    def model_output_err(self, input):  #1x56x56 
        y_hat,y = self.forward(input)
        err = y_hat-y
        loss = torch.mean(err**2, axis = 1) 
        return loss    
   
if __name__ == "__main__":    
    input = torch.randn(1,3,224,224)
    class config():
        def __init__(self):
            self.backbone = ['relu1_1','relu1_2','relu2_1','relu2_2',
                    'relu3_1','relu3_2','relu3_3','relu3_4',
                    'relu4_1','relu4_2','relu4_3','relu4_4',
                    'relu5_1','relu5_2','relu5_3','relu5_4']
            self.device = torch.device('cuda:0')  
            self.latent_dim  = None   
    cfg = config() 
    model = AD_DFR(cfg).to(cfg.device)