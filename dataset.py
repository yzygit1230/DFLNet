from torch.utils.data import Dataset
from PIL import Image


class BreastData(Dataset):
    def __init__(self, df, transforms=None, img_size= False):
        self.data = df
        self.transform = transforms
        self.img_size = img_size
       
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data['images'].iloc[idx]).convert("RGB")   
        gt = Image.open(self.data['masks'].iloc[idx]).convert('L')

        img_shape = gt.size  
        sample = {'image': img, 'gt': gt}
    
        sample = self.transform(sample)
        path = str(self.data['images'].iloc[idx]).split('/')[-1]

        return sample['image'], sample['gt'], path