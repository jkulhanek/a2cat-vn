from deep_rl import register_trainer
from deep_rl.core import AbstractTrainer
from deep_rl.core import MetricContext
from deep_rl.configuration import configuration
from environments.gym_house.goal import GoalImageCache
import os
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
from models import AuxiliaryBigGoalHouseModel as Model
from torch.optim import Adam
import torch.nn.functional as F
from experiments.ai2_auxiliary.trainer import compute_auxiliary_target

class HouseDataset(Dataset):
    def __init__(self, deconv_cell_size, transform = None):
        self.image_cache = GoalImageCache((174,174), configuration.get('house3d.dataset_path'))
        self.images = list(self.image_cache.all_image_paths(['rgb','depth','segmentation']))
        self.transform = transform
        self.deconv_cell_size = deconv_cell_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, depth, segmentation = self.images[index]
        image = self.image_cache.read_image(image)
        depth = self.image_cache.read_image(depth)
        segmentation = self.image_cache.read_image(segmentation)
        image = torch.from_numpy(np.transpose(image.astype(np.float32), [2,0,1]) / 255.0).unsqueeze(0)
        depth =  torch.from_numpy(np.transpose(depth[:,:,:1].astype(np.float32), [2,0,1]) / 255.0).unsqueeze(0)
        segmentation = torch.from_numpy(np.transpose(segmentation.astype(np.float32), [2,0,1]) / 255.0).unsqueeze(0)

        segmentation = compute_auxiliary_target(segmentation.unsqueeze(0), self.deconv_cell_size, (42, 42)).squeeze(0)
        depth = compute_auxiliary_target(depth.unsqueeze(0), self.deconv_cell_size, (42, 42)).squeeze(0)
        ret = (image, depth, segmentation)

        if self.transform:
            ret = self.transform(ret)
        return ret

@register_trainer(save = True, saving_period = 1)
class SupervisedTrained(AbstractTrainer):
    def __init__(self, name, **kwargs):
        super().__init__(dict(), dict())
        self.name = name
        self.batch_size = 32
        self.main_device = torch.device('cuda')

    def optimize(self, image, depth, segmentation):
        image = image.to(self.main_device)
        depth = depth.to(self.main_device)
        segmentation = segmentation.to(self.main_device)
        zeros1 = torch.rand((image.size()[0], 1, 3,174,174), dtype = torch.float32, device = self.main_device)
        zeros2 = torch.rand((image.size()[0], 1, 3,174,174), dtype = torch.float32, device = self.main_device)
        (r_depth, r_segmentation, _), _ = self.model.forward_deconv(((image, zeros1), None), None, None)
        (_, _, r_goal_segmentation), _ = self.model.forward_deconv(((zeros2, image), None,), None, None)
        loss = F.mse_loss(r_depth, depth) + F.mse_loss(r_segmentation, segmentation) + F.mse_loss(r_goal_segmentation, segmentation)
        loss = loss / 3.0

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, path):
        super().save(path)
        torch.save(self.model.state_dict(), os.path.join(path, 'weights.pth'))
        print('Saving to %s' % os.path.join(path, 'weights.pth'))

    def process(self, mode = 'train', **kwargs):
        assert mode == 'train'
        # Single epoch
        metric_context = MetricContext()
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size,shuffle=True, num_workers=4)
        total_loss = 0
        total_updates = 0
        for i, item in enumerate(dataloader):
            loss = self.optimize(*item)
            print('loss is %s' % loss)
            total_loss += loss
            total_updates += 1

        print('Epoch done with loss=%s' % (total_loss / total_updates))
        return (1, (1, 1), metric_context)

    def create_dataset(self, deconv_cell_size):        
        return HouseDataset(deconv_cell_size)

    def _initialize(self):
        model = Model(3, 6).to(self.main_device)
        self.dataset = self.create_dataset(model.deconv_cell_size)     
        self.optimizer = Adam(model.parameters())
        return model

    def run(self, process, **kwargs):
        self.model = self._initialize()
        for i in range(30):
            print('Starting epoch %s' % (i + 1))
            process()

def default_args():
    return dict()