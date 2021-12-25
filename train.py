import torch
import os 
from torch._C import Size
import torch.optim as optim
from DFR_model import AD_DFR
from DataLoader import myImageFloder
from data_dir import  mk_data_dir,resize
import pandas as pd
#########################################
#    Define Train and Valid function
#########################################
def train(model, optimizer,trainloader,device):
    model.train()
    epoch_loss = 0
    for step,batch in enumerate(trainloader):
        batch = batch.to(device)
        optimizer.zero_grad()
        loss = model.MSE_loss(batch)
        loss.backward()
        optimizer.step()      
        epoch_loss  += loss.item()
        
    step += 1  
    return (epoch_loss/step)  
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
def train_process(cfg):
    for item in cfg.classes:
        ########################################
        #    Spilt and Resize the data 
        ##########################################
        
        class_dir = mk_data_dir(item,cfg.data_dir_head)
        #resize(class_dir,cfg.img_to_size)

        (train_img,valid_img,test_good,test_bad) = class_dir

        trainloader = torch.utils.data.DataLoader(
            myImageFloder(train_img, True), 
            batch_size= cfg.BATCH_SIZE, shuffle= True
                , num_workers= 0, drop_last=False
        )
        valloader = torch.utils.data.DataLoader(
            myImageFloder(valid_img, False), 
            batch_size= cfg.BATCH_SIZE, shuffle= True
                , num_workers= 0, drop_last=False
        )
        #########################################
        #  Save model path
        #########################################

        folder =  "models/"+ item + "/model"
        if not os.path.exists(folder):
            os.makedirs(folder)    
        Path =  folder+"/VGG19_CAE.pkl"   
        
        #########################################
        #  Train
        #########################################

        model = AD_DFR(cfg).to(cfg.device)
        model.PCA(trainloader)

        #model.CAE.load_state_dict(torch.load(Path, map_location=device))
        params=0
        for param in model.parameters():
            if param.requires_grad:
                params += param.numel()
                
        #model.PCA(trainloader)        
            
        optimizer = optim.Adam(model.CAE.parameters(), lr=cfg.LR)
        t_loss,v_loss = [],[]
        best_loss, best_epoch = 1000, 0


        for epoch in range(cfg.EPOCH):  # loop over the dataset multiple times
            print('\n============================\nEpoch: ', epoch)
            train_loss = train(model, optimizer,trainloader,cfg.device)
            print('Train loss:  {:.4f}'.format(train_loss))
            valid_loss = validate(model, valloader,cfg.device)
            print('Valid loss:  {:.4f}'.format(valid_loss))
            t_loss += [train_loss]
            v_loss += [valid_loss]
            if (best_loss > valid_loss):
                best_loss = valid_loss
                best_epoch += 1
                torch.save({"latent_size":model.latent_dim,"CAE":model.CAE.state_dict()}, Path)
                


        #########################################
        #  Save Loss Record 
        #########################################
        folder =  "models/"+ item + "/validate"
        if not os.path.exists(folder):
            os.makedirs(folder)
        history =  folder+"/history.csv"    
        df = pd.DataFrame(list(zip(list(range(1, cfg.EPOCH+1)),t_loss, v_loss)), columns =['EPOCH','Train Loss', 'Valid Loss']) 
        df.to_csv(history,index=False)
    

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
            self.LR = 0.0001
            self.is_bn = True
            self.classes = ['bottle']
    cfg = config() 
    train_process(cfg)