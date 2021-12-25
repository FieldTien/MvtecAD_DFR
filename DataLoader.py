from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data as data
def default_loader(path):
    return Image.open(path).convert('RGB')
class myImageFloder(data.Dataset):
    def __init__(self, images, training, loader=default_loader,gt=None):
        self.images = images
        self.loader = loader
        self.training = training
        self.gt = gt
    def __getitem__(self, index):
        images = self.images[index]
        images = self.loader(images)  
        if(self.training==True):
          t_list = [ 
                    
                transforms.Resize(size=224),
                #transforms.RandomRotation(degrees=300),
                #transforms.RandomHorizontalFlip(),
                #transforms.CenterCrop(size=224),  # Image net standards
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225]) ] 
        else :
          t_list = [ transforms.Resize(size=224),
                  transforms.ToTensor(),
                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]         
        processed = transforms.Compose(t_list)
        images = processed(images)
        if(self.gt!=None):
          gt = self.gt[index]
          gt = self.loader(gt)  
          gt_list = [transforms.ToTensor()]
          gt_processed = transforms.Compose(gt_list)
          gt = gt_processed(gt)
          return images, gt
        else:
          return images 
    def __len__(self):
        return len(self.images)