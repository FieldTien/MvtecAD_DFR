import argparse
import os 
from train import train_process
from validate import validate_score,validate_heatmap
import wget
import tarfile
def config():
    
    parser = argparse.ArgumentParser(description="Settings of DFR")
    parser.add_argument('--mode', type=str, choices=["train", "evaluation","inference", "download"],
                        default="train", help="train or evaluation AUC or inference heatmap or download finetuned CAE")
    parser.add_argument('--data_dir_head', type=str, default="mvtecad_unsupervise" , help="Data Folder head name")
    parser.add_argument('--img_to_size', type=int, nargs="+", default=(224, 224), help="image size (hxw)")
    parser.add_argument('--device', type=str, default="cuda:0", help="device for training and testing")
    parser.add_argument('--BATCH_SIZE', type=int, default=5, help="batch size for training")
    parser.add_argument('--LR', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--EPOCH', type=int, default=500, help="epochs for training")    
    multisclale_feature = ['relu1_1','relu1_2','relu2_1','relu2_2','relu3_1','relu3_2','relu3_3','relu3_4',
                           'relu4_1','relu4_2','relu4_3','relu4_4','relu5_1','relu5_2','relu5_3','relu5_4']
    parser.add_argument('--backbone', type=str,  nargs='+', default=multisclale_feature, help="Each Conv feature map")   
    parser.add_argument('--latent_dim', type=int, default=None, help="Autoencoder reduction size")   
    parser.add_argument('--is_bn', type=bool, default=True, help="if using bn layer in CAE") 
    classes = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    parser.add_argument('--classes', type=str,  nargs='+', default=classes, help="Train the model for each class") 
    
    parser.add_argument('--fpr', type=float, default=0.005, help="False positive rate for Treshold") 
    parser.add_argument('--heatmap_path', type=str, default='mvtecad_unsupervise/cable/test/bent_wire/001.png', help="Inferece heatmap dir") 
    parser.add_argument('--heatmap_item', type=str, default='cable', help="Inferece heatmap class") 
    parser.add_argument('--heatmap_gt', type=str, default=None, help="Inferece heatmap gt")
    parser.add_argument('--heatmap_export', type=str, default='validate/Inferece.png', help="Inferece heatmap out dir") 
    
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    #########################################
    #    On the whole data       ['bottle', 'cable', 'capsule', 'carpet'] ['toothbrush', 'transistor', 'wood', 'zipper']
    #########################################
    cfg = config()
    #cfg.classes = ['leather'] 
    cfg.mode = "inference"
    print(cfg.mode)
    cfg.heatmap_path = 'mvtecad_unsupervise/bottle/test/broken_small/001.png'
    cfg.heatmap_item = 'bottle'
    cfg.heatmap_gt = 'mvtecad_unsupervise/bottle/ground_truth/broken_small/001_mask.png'
    cfg.device = 'cpu'
    cfg.heatmap_export = 'validate/Inferece.png'
    if(cfg.mode == "train"):
        train_process(cfg)
    elif(cfg.mode == "evaluation"):    
        validate_score(cfg)
    elif(cfg.mode == "inference"):    

        validate_heatmap(cfg)  
    elif(cfg.mode == "download"):     
        file_path = "models.tar.gz"
        if os.path.isfile(file_path):
            print("Zip file has already exists")
        else:    
            site_url = 'https://www.dropbox.com/s/n8kc4t0lc90fk02/models.tar.gz?dl=1'
            file_path = wget.download(site_url)

        tar = tarfile.open(file_path, 'r:gz')
        tar.extractall()
   
    