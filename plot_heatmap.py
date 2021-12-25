import os
import os.path
import random
from pathlib import Path
import pickle
from PIL import Image
from validate import validate_heatmap
from main import config
#ad_class = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

cfg = config()
for item  in  cfg.classes:
    save_dir = 'validate/inferece/'+item
    data_dir_head = "mvtecad_unsupervise" 
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    test = Path(data_dir_head+"/%s/test"%item)
    ground_truth = Path(data_dir_head+"/%s/ground_truth"%item)
    test_good = [str(i) for i in test.glob('*/*') if str(i).split('\\')[3] == 'good' ]
    test_bad = [str(i) for i in test.glob('*/*') if str(i).split('\\')[3] != 'good' ]

    for index,img in enumerate(test_good):
        cfg.heatmap_path =  img
        cfg.heatmap_export = save_dir+'/inference_good_'+ str(index) + ".png"
        cfg.heatmap_item = item
        cfg.heatmap_gt = None
        validate_heatmap(cfg)  
    
        
    for index,img in enumerate(test_bad):    
        cfg.heatmap_path =  img
        cfg.heatmap_export = save_dir+'/inference_bad_'+ str(index) + ".png"
        cfg.heatmap_item = item
        cfg.heatmap_gt = img.replace('test', 'ground_truth').replace('.png', '_mask.png')
        validate_heatmap(cfg)  
