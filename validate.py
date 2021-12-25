from numpy.core.fromnumeric import size
import torch
from sklearn.metrics import roc_curve, auc # roc curve tools
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pickle
from DataLoader import myImageFloder
from DFR_model import AD_DFR
import numpy as np
import os
import pandas as pd
from pathlib import Path   
from PIL import Image
import torchvision.transforms as transforms
def validate(model,loader,device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for step,batch in enumerate(loader):
            batch = batch.to(device)
            loss = model.MSE_loss(batch)   
            epoch_loss  += loss.item()
        step += 1  
    return (epoch_loss/step) 

def inference_testset(item,model,test_good_loader,test_bad_loader,cfg):
    model.eval()
    predicts_good=[]
    input_size=cfg.img_to_size
    for img in test_good_loader:
        with torch.no_grad():
            img = img.to(cfg.device)
            predicts = model.model_output_err(img)
            predicts = F.interpolate(torch.unsqueeze(predicts,0),size=input_size,mode='bilinear',align_corners=True)
            predicts_good.append(torch.squeeze(predicts,0))
    predicts_good = torch.cat(predicts_good,axis=0).to('cpu')
    gt_good = torch.zeros(predicts_good.shape)

    predicts_bad,gt_bad=[],[]
    for img,gt in test_bad_loader:
        with torch.no_grad():
            img = img.to(cfg.device)
            predicts = model.model_output_err(img)
            predicts = F.interpolate(torch.unsqueeze(predicts,0),size=input_size,mode='bilinear',align_corners=True)
            predicts_bad.append(torch.squeeze(predicts,0))
            gt_bad.append(gt[:,0])
    predicts_bad = torch.cat(predicts_bad,axis=0).to('cpu')
    gt_bad = torch.cat(gt_bad,axis=0).to('cpu') 
    predicts = torch.cat([predicts_good,predicts_bad],axis=0).numpy()
    gt = torch.cat([gt_good,gt_bad],axis=0).numpy()
    #normalize predicts 
    #maximun,minimun = np.max(predicts),np.min(predicts)
    #predicts = (predicts-minimun)/(maximun-minimun)
    return gt, predicts


def roc_auc_compute(item,model,test_good_loader,test_bad_loader,cfg):
    gt, predicts = inference_testset(item,model,test_good_loader,test_bad_loader,cfg)
    predict_img_level = np.max(predicts.reshape((predicts.shape[0],predicts.shape[1]*predicts.shape[2])),1)
    gt_img_level = np.max(gt.reshape((gt.shape[0],gt.shape[1]*gt.shape[2])),1)
    fpr_binary, tpr_binary, threshold_binary = roc_curve(gt_img_level,predict_img_level)
    roc_auc_binary = auc(fpr_binary,tpr_binary)
    predicts ,gt =predicts.ravel() , gt.ravel()
    ground_truth_labels = gt # we want to make them into vectors
    score_value = predicts # we want to make them into vectors
    fpr, tpr, threshold = roc_curve(ground_truth_labels,score_value)
    roc_auc = auc(fpr,tpr) 
    fig, ax = plt.subplots(1,1)
    ax.plot(fpr, tpr, label='Segmentation ROC curve (AUC = %0.3f)' % roc_auc)
    ax.plot(fpr_binary, tpr_binary, label='Detection ROC curve (AUC = %0.3f)' % roc_auc_binary)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROCAUC OF %s'%item)
    ax.legend(loc="lower right") 
    folder =  "validate/ROCAUC"
    if not os.path.exists(folder):
        os.makedirs(folder)
    fig.savefig(folder+'/%s.png'%item)
    return roc_auc,roc_auc_binary
def fpr_threshold_compute(model,cfg):
    model.eval()
    data_dir =  "models/"+ cfg.heatmap_item + "/datapath"+'/data_dir.txt'
    with open(data_dir, "rb") as fp:   #Pickling
        data_dir = pickle.load(fp) 
    (train_img,valid_img,test_good,test_bad) = data_dir[0],data_dir[1],data_dir[2],data_dir[3]
    gt_bad = []
    for dir in test_bad:
        new_dir = dir.replace("test","ground_truth")
        new_dir = new_dir.replace(".png","_mask.png")
        gt_bad += [new_dir]
    test_good_loader = torch.utils.data.DataLoader(
        myImageFloder(test_good, False), 
        batch_size= 1, shuffle= True
            , num_workers= 0, drop_last=False
    )
    test_bad_loader = torch.utils.data.DataLoader(
        myImageFloder(test_bad, False,gt=gt_bad), 
        batch_size= 1, shuffle= True
            , num_workers= 0, drop_last=False
    )
    gt, predicts = inference_testset(cfg.heatmap_item,model,test_good_loader,test_bad_loader,cfg)
    predicts ,gt =predicts.ravel() , gt.ravel()
    fpr, tpr, threshold = roc_curve(gt,predicts)
    for index,i in enumerate(fpr):
        if(i>=cfg.fpr):
           break
    threshold = threshold[index-1]
    threshold_Path =  "models/"+ cfg.heatmap_item + "/model"+"/threshold.pkl"  
    
   
    if Path(threshold_Path).is_file():
       print ("Threshold Data exist")
       with open(threshold_Path, 'rb') as handle:
           DB = pickle.load(handle)
    else:
       print ("Threshold Data does not exist")    
       DB={}
    DB[cfg.fpr]= threshold
    with open(threshold_Path, 'wb') as handle:
        pickle.dump(DB, handle)
    return threshold    

def validate_heatmap(cfg):
    ''' 
        cfg.heatmap_path
        cfg.heatmap_item
        cfg.heatmap_gt
    '''   
    
            
    valloader = torch.utils.data.DataLoader(
            myImageFloder([cfg.heatmap_path], True), 
            batch_size= 1, shuffle= True
                , num_workers= 0, drop_last=False
            )
    #### Load model ####
    model_Path =  "models/"+ cfg.heatmap_item + "/model"+"/VGG19_CAE.pkl"   
    result = torch.load(model_Path, map_location=cfg.device)
    cfg.latent_dim  = result["latent_size"]
    model = AD_DFR(cfg).to(cfg.device)
    model.PCA(valloader)
    model.CAE.load_state_dict(result["CAE"])
    threshold_Path =  "models/"+ cfg.heatmap_item + "/model"+"/threshold.pkl" 
    if Path(threshold_Path).is_file():
        with open(threshold_Path, 'rb') as handle:
           DB = pickle.load(handle)
        if (cfg.fpr in DB):
            threshold = DB[cfg.fpr]
        else:  
            threshold = fpr_threshold_compute(model,cfg)       
    else:
        threshold = fpr_threshold_compute(model,cfg)
    model.eval()
    img = next(iter(valloader))
    img = img.to(cfg.device)
    with torch.no_grad():
        predicts = model.model_output_err(img)
    predicts = F.interpolate(torch.unsqueeze(predicts,0),size=(img.shape[2],img.shape[3]))
    heatmap=(predicts[0][0].to('cpu')>=threshold).float()
    invTrans = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],std = [ 1., 1., 1. ])])
    origin_img = invTrans(img)
    
    
    fig = plt.figure(figsize = (7, 3))
    ax = fig.add_subplot(1, 2, 1)
    if(cfg.heatmap_gt!=None):
        gt = Image.open(cfg.heatmap_gt).convert('1')  
        gt = transforms.ToTensor()(gt)
        ax.imshow(gt[0], cmap='viridis')
    ax.imshow(origin_img.squeeze().permute(1,2,0).to("cpu"),alpha=0.75)
    ax.set_title('Ground True',size=15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(heatmap, cmap='viridis')
    ax.imshow(origin_img.squeeze().permute(1,2,0).to("cpu"),alpha=0.75)
    ax.set_title('Ground Predict',size=15)
    ax.set_xticks([])
    ax.set_yticks([])   
    fig.savefig(cfg.heatmap_export)
    #print(heatmap)
def validate_score(cfg):
    record_losses,record_seg_auc,record_det_auc , record_latent_dim= [],[],[],[]
    for item in cfg.classes:
        data_dir =  "models/"+ item + "/datapath"+'/data_dir.txt'
        with open(data_dir, "rb") as fp:   #Pickling
            data_dir = pickle.load(fp) 
        (train_img,valid_img,test_good,test_bad) = data_dir[0],data_dir[1],data_dir[2],data_dir[3]


        gt_bad = []
        for dir in test_bad:
            new_dir = dir.replace("test","ground_truth")
            new_dir = new_dir.replace(".png","_mask.png")
            gt_bad += [new_dir]

        valloader = torch.utils.data.DataLoader(
            myImageFloder(valid_img, True), 
            batch_size= 3, shuffle= True
                , num_workers= 0, drop_last=False
        )
        test_good_loader = torch.utils.data.DataLoader(
            myImageFloder(test_good, False), 
            batch_size= 1, shuffle= True
                , num_workers= 0, drop_last=False
        )
        test_bad_loader = torch.utils.data.DataLoader(
            myImageFloder(test_bad, False,gt=gt_bad), 
            batch_size= 1, shuffle= True
                , num_workers= 0, drop_last=False
        )

        
        model_Path =  "models/"+ item + "/model"+"/VGG19_CAE.pkl"   
        result = torch.load(model_Path, map_location=cfg.device)
        cfg.latent_dim  = result["latent_size"]

        model = AD_DFR(cfg).to(cfg.device)
        model.PCA(test_good_loader)
        print("latnet dim is : ", cfg.latent_dim )
        record_latent_dim.append(cfg.latent_dim)
        model.CAE.load_state_dict(result["CAE"])
        val_loss = validate(model,valloader,cfg.device)
        roc_auc,roc_auc_imglevel = roc_auc_compute(item,model,test_good_loader,test_bad_loader,cfg)
        record_losses.append(val_loss)
        record_seg_auc.append(roc_auc)
        record_det_auc.append(roc_auc_imglevel)
    df = pd.DataFrame({'latnet dim': record_latent_dim,'Valid loss': record_losses,'Segmentation AUC': record_seg_auc,'Detection AUC': record_det_auc}, index=cfg.classes)
    df.loc['Mean'] = df.mean(axis=0).tolist()
    df.to_csv("AUC_summary.csv",index=True)
    print(df)

if __name__ == "__main__": 
    class config():
        def __init__(self):
            self.backbone = ['relu1_1','relu1_2','relu2_1','relu2_2',
                'relu3_1','relu3_2','relu3_3','relu3_4',
                'relu4_1','relu4_2','relu4_3','relu4_4',
                'relu5_1','relu5_2','relu5_3','relu5_4']
            self.device = torch.device('cuda:0')  
            self.latent_dim  = None   
            #self.latent_dim  = 50   
            self.data_dir_head  =  "mvtecad_unsupervise" 
            self.device  =  torch.device('cuda')
            self.img_to_size = (224,224)
            self.BATCH_SIZE = 5
            self.EPOCH = 500
            self.LR = 0.005
            self.is_bn = True
            self.fpr = 0.001
            #self.classes = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
            self.classes = ['bottle']
            self.heatmap_path = 'mvtecad_unsupervise/cable/test/bent_wire/001.png'
            self.heatmap_item = 'cable'
            self.heatmap_gt = 'mvtecad_unsupervise/cable/ground_truth/bent_wire/001_mask.png'
            self.heatmap_export = 'validate/Inferece.png'
    cfg = config() 
  
    #validate_score(cfg)
    print(validate_heatmap(cfg))