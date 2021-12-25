import os
import os.path
import random
from pathlib import Path
import pickle
from PIL import Image

#ad_class = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

def mk_data_dir(item,data_dir_head):
    train = Path(data_dir_head+"/%s/train"%item)
    test = Path(data_dir_head+"/%s/test"%item)
    ground_truth = Path(data_dir_head+"/%s/ground_truth"%item)
    train = [str(i) for i in train.glob('*/*')]
    test_good = [str(i) for i in test.glob('*/*') if str(i).split('\\')[3] == 'good' ]
    test_bad = [str(i) for i in test.glob('*/*') if str(i).split('\\')[3] != 'good' ]
    #test_gt = [str(i) for i in ground_truth.glob('*/*') ]
    random.shuffle(train)  
    train_img,valid_img = train[:int(0.85*len(train))],train[int(0.85*len(train)):]
    class_dir = (train_img,valid_img,test_good,test_bad)
    folder =  "models/"+ item + "/datapath"
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(folder+'/data_dir.txt', "wb") as fp:   #Pickling
        pickle.dump(class_dir, fp) 
    return  class_dir      
def resize(class_dir,size):
    for item in class_dir:
        for files in item:
             img = Image.open(str(files)).convert('RGB')
             img = img.resize(size)
             img.save(str(files))
            
        
if __name__ == "__main__":  
    
    item = 'bottle'
    data_dir_head = "mvtecad_unsupervise"  
    class_dir = mk_data_dir(item,data_dir_head)
    print(class_dir)
    resize(class_dir,(224,224))
    