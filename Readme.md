# MvtecAD unsupervised Anomaly Detection
This respository  is the unofficial implementations of [DFR: Deep Feature Reconstruction for Unsupervised Anomaly Segmentation](https://arxiv.org/abs/2012.07122)

# Result of 500 epochs  trained model 

## Selects latent sizes of Autoencoder by PCA
| Classes  |latent size |Segmentation AUC |Detection AUC|
| -------- | -------- | -------- | -------- |
| bottle   |  116     | 97.2771%    |99.8413%|
| cable   |  557     | 95.5101%    |84.8951%|
| capsule  |  162     | 98.8928%    |97.3275%|
| carpet  |  245     | 97.9116%    |90.5297%|
| grid  |  145     | 97.2484    |79.5322%|
| hazelnut  |  459     | 98.5848%    |100%|
| leather  |  325     | 98.8649%    |95.4484%|
| metal_nut  |  380     | 96.127%    |97.263%|
| pill  |  292     | 98.0543%    |94.108%|
| screw  |  283     | 99.3001%    |92.0066%|
| tile  |  557     | 89.4887%    |91.7388%|
| toothbrush  |  243     | 98.6729%    |91.3889%|
| transistor  |  333     | 83.9157%    |89.0833%|
| wood  |  364     | 91.7027%    |98.9474%|
| zipper  |  115     | 95.6663%    |83.2983%|
| mean  |       | 95.8141%    |92.3606|

# How to run 
## requirements
pytorch scikit-learn matplotlib numpy pandas PIL wget
## Train

```
python main.py --mode          train      
               --data_dir_head [Datapath] 
               --BATCH_SIZE    [BATCH_SIZE] 
               --LR            [Learning Rate] 
               --EPOCH         [Epochs] 
               --backbone      [Feature map of Conv in VGG19]
               --latent_dim    [Latent size of CAE] 
               --classes       [Default is all] 
```
## Download 500 Epochs Finetuned Models
Here provide the model of each classes in Drophox
```
python main.py --mode download                   
```
## Evaluate the ROC-AUC of Test Set
```
python main.py --mode        evaluation    
               --classes     [Default is all] 
```
## Inference the model
```
python main.py --mode           inference    
               --heatmap_path   [Input path] 
               --heatmap_item   [Class of input] 
               --heatmap_gt     [GT path Default is None]
               --device         [cpu or cuda]
               --device         [Output path ]         
```
Example run in main.py
```
if __name__ == "__main__":  
    cfg = config()
    cfg.mode = "inference"
    cfg.heatmap_path = 'mvtecad_unsupervise/bottle/test/broken_small/001.png'
    cfg.heatmap_item = 'bottle'
    cfg.heatmap_gt = 'mvtecad_unsupervise/bottle/ground_truth/broken_small/001_mask.png'
    cfg.device = 'cpu'
    cfg.heatmap_export = 'validate/Inferece.png'
```
validate/Inferece.png is 
![](https://i.imgur.com/gAYpVH6.png)

# Code Reference
https://github.com/YoungGod/DFR

https://www.kaggle.com/danieldelro/unsupervised-anomaly-segmentation-of-screw-images




