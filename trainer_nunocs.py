import os.path
import numpy as np
import os,sys,copy,time,cv2
from tqdm import tqdm
from scipy.signal import convolve2d
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchvision
from PIL import Image
from dataset_nunocs import *
from pointnet2 import *
from loss import *
from tensorboardX import SummaryWriter
from Utils import log


class TrainerNunocs:
  def __init__(self,cfg):
    self.cfg = cfg
    self.epoch = 0
    self.iter_idx = -1

    self.best_train = 1e9
    self.best_val = 1e9

    self.train_data = NunocsIsolatedDataset(self.cfg,phase='train')
    self.val_data = NunocsIsolatedDataset(self.cfg,phase='val')

    self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.cfg['batch_size'], shuffle=True, 
                                                    num_workers=self.cfg['n_workers'], pin_memory=False, drop_last=True,
                                                    worker_init_fn=worker_init_fn)
    self.val_loader = torch.utils.data.DataLoader(self.val_data, batch_size=self.cfg['batch_size'], shuffle=True, 
                                                  num_workers=self.cfg['n_workers'], pin_memory=False, drop_last=False,
                                                  worker_init_fn=worker_init_fn)

    self.model = PointNetSeg(n_in=self.cfg['input_channel'],n_out=3*self.cfg['ce_loss_bins'])
    self.model = nn.DataParallel(self.model)
    self.model.cuda()

    start_lr = self.cfg['start_lr']
    # start_lr = self.cfg['start_lr']/64*self.cfg['batch_size']
    if self.cfg['optimizer_type']=='adam':
      self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=start_lr, 
                                  weight_decay=self.cfg['weight_decay'], betas=(0.9, 0.99), amsgrad=False)
    elif self.cfg['optimizer_type']=='sgd':
      self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=start_lr, 
                                  weight_decay=self.cfg['weight_decay'], momentum=0.9)

    self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.cfg['lr_milestones'], gamma=0.1)

    tr_writer = SummaryWriter(log_dir=self.cfg['save_dir'])
    self.tr_writer = tr_writer
    self.checkpoint_file = os.path.join(self.cfg['save_dir'], "checkpoint.pth")
    self.bestmodel_file = os.path.join(self.cfg['save_dir'], "best_model.pth")
    self.log_fn = os.path.join(self.cfg['save_dir'], "console.out")

  def save(self, pt_file):
    """ save models"""
    torch.save({
        "iter_idx": self.iter_idx,
        "model": self.model.state_dict(),
        "optimizer": self.optimizer.state_dict()},
        pt_file)

  def restore(self, pt_file):
    # Read checkpoint file.
    load_res = torch.load(pt_file)

    # Resume iterations
    self.iter_idx = load_res["iter_idx"]
    # Resume model
    self.model.load_state_dict(load_res["model"], strict=False)
    # Resume optimizer
    self.optimizer.load_state_dict(load_res["optimizer"])

  def train(self):
    # initialize parameters
    if not os.path.exists(self.checkpoint_file):
        print("training from scratch")
        cur_epoch = 0
    else:
        print("restoring from {}".format(self.checkpoint_file))
        self.restore(self.checkpoint_file)
        # record cur_epoch
        fn = os.path.join(self.cfg['save_dir'], "cur_epoch.txt")
        if os.path.exists(fn):
            with open(fn, "r") as f:
                cur_epoch = int(f.read()) + 1
            for _ in np.arange(cur_epoch):
                self.scheduler.step()
        else:
            cur_epoch = 0

    for self.epoch in range(cur_epoch, self.cfg['n_epochs']):
      np.random.seed(self.cfg['random_seed']+self.epoch)
      self.train_loop()
      self.val_loop()
      self.scheduler.step()
      # record cur_epoch
      fn = os.path.join(self.cfg['save_dir'], "cur_epoch.txt")
      with open(fn, "w") as f:
          f.write("{}\n".format(self.epoch))

  def train_loop(self):
    tic = time.time()
    self.model.train()
    prefix = "Training: {:3d} / {:3d}".format(self.epoch, self.cfg['n_epochs'])
    all_loss = []
    criteria = NocsMinSymmetryCELoss(self.cfg)
    for batch in tqdm(self.train_loader, prefix):
      self.iter_idx += 1
      input_data = batch['input'].cuda().float()
      cloud_nocs = batch['cloud_nocs'].cuda().float()
      pred, l4_points = self.model(input_data) # pred: (B, N, 3)
      bin_resolution = 1/self.cfg['ce_loss_bins']
      loss = criteria(pred, cloud_nocs)
      all_loss.append(loss.item())
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      
      if self.iter_idx % max(1, len(self.train_loader) // 10) == 0:
        self.tr_writer.add_scalar("train loss per iteration", loss, global_step=self.iter_idx)
        self.save(self.checkpoint_file) # 保存迭代序号，模型参数，优化器参数;
    
    total_time_train_per_epoch = (time.time() - tic) / 3600.0
    avg_loss = np.array(all_loss).mean()
    print_str = "Train Epoch %3d | loss %f | Time %.2fhr | lr %.20f" % \
        (self.epoch, avg_loss, total_time_train_per_epoch, self.optimizer.param_groups[0]['lr']) 
    print(print_str, flush=True)
    log(print_str, self.log_fn)
    self.tr_writer.add_scalar("train loss per epoch", avg_loss, global_step=self.epoch)
    self.save(self.checkpoint_file)

  def val_loop(self):
    tic = time.time()
    self.model.eval()
    all_loss = []
    criteria = NocsMinSymmetryCELoss(self.cfg)
    for batch in tqdm(self.val_loader):
      input_data = batch['input'].cuda().float()
      cloud_nocs = batch['cloud_nocs'].cuda().float()
      pred, l4_points = self.model(input_data) # pred: (B, N, 3)
      bin_resolution = 1/self.cfg['ce_loss_bins']
      loss = criteria(pred, cloud_nocs)
      all_loss.append(loss.item())
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

    avg_loss = np.array(all_loss).mean()
    toal_time_test_per_epoch = (time.time() - tic) / 3600.0
    print_str = "Test Epoch %3d | loss %f | Time %.2fhr" % \
        (self.epoch, avg_loss, toal_time_test_per_epoch)
    print(print_str, flush=True)
    log(print_str, self.log_fn)
    self.tr_writer.add_scalar("test loss per epoch", avg_loss, global_step=self.epoch)
    self.save(self.checkpoint_file)

    if avg_loss < self.best_val:
        self.save(self.bestmodel_file)
        self.best_val = avg_loss
        fn = os.path.join(self.cfg['save_dir'], "best_va_acc.txt")
        with open(fn, "w") as f:
            f.write("{}\n".format(self.best_val))
            f.write("{}\n".format(self.epoch))


class TrainerImplicitNunocsNet:
  def __init__(self,cfg):
    self.cfg = cfg
    self.epoch = 0
    self.iter_idx = -1

    self.best_train = 1e9
    self.best_val = 1e9

    self.train_data = NunocsIsolatedDataset(self.cfg,phase='train')
    self.val_data = NunocsIsolatedDataset(self.cfg,phase='val')

    self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.cfg['batch_size'], shuffle=True, 
                                                    num_workers=self.cfg['n_workers'], pin_memory=False, drop_last=True,
                                                    worker_init_fn=worker_init_fn)
    self.val_loader = torch.utils.data.DataLoader(self.val_data, batch_size=self.cfg['batch_size'], shuffle=True, 
                                                  num_workers=self.cfg['n_workers'], pin_memory=False, drop_last=False,
                                                  worker_init_fn=worker_init_fn)

    self.model = PointNetReg(n_in=self.cfg['input_channel'],n_out=3)
    self.model = nn.DataParallel(self.model)
    self.model.cuda()

    start_lr = self.cfg['start_lr']
    # start_lr = self.cfg['start_lr']/64*self.cfg['batch_size']
    if self.cfg['optimizer_type']=='adam':
      self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=start_lr, 
                                  weight_decay=self.cfg['weight_decay'], betas=(0.9, 0.99), amsgrad=False)
    elif self.cfg['optimizer_type']=='sgd':
      self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                  lr=start_lr, weight_decay=self.cfg['weight_decay'], momentum=0.9)

    self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.cfg['lr_milestones'], gamma=0.1)

    tr_writer = SummaryWriter(log_dir=self.cfg['save_dir'])
    self.tr_writer = tr_writer
    self.checkpoint_file = os.path.join(self.cfg['save_dir'], "checkpoint.pth")
    self.bestmodel_file = os.path.join(self.cfg['save_dir'], "best_model.pth")
    self.log_fn = os.path.join(self.cfg['save_dir'], "console.out")

  def save(self, pt_file):
    """ save models"""
    torch.save({
        "iter_idx": self.iter_idx,
        "model": self.model.state_dict(),
        "optimizer": self.optimizer.state_dict()},
        pt_file)

  def restore(self, pt_file):
    # Read checkpoint file.
    load_res = torch.load(pt_file)

    # Resume iterations
    self.iter_idx = load_res["iter_idx"]
    # Resume model
    self.model.load_state_dict(load_res["model"], strict=False)
    # Resume optimizer
    self.optimizer.load_state_dict(load_res["optimizer"])

  def train(self):
    # initialize parameters
    if not os.path.exists(self.checkpoint_file):
        print("training from scratch")
        cur_epoch = 0
    else:
        print("restoring from {}".format(self.checkpoint_file))
        self.restore(self.checkpoint_file)
        # record cur_epoch
        fn = os.path.join(self.cfg['save_dir'], "cur_epoch.txt")
        if os.path.exists(fn):
            with open(fn, "r") as f:
                cur_epoch = int(f.read()) + 1
            for _ in np.arange(cur_epoch):
                self.scheduler.step()
        else:
            cur_epoch = 0

    for self.epoch in range(cur_epoch, self.cfg['n_epochs']):
      np.random.seed(self.cfg['random_seed']+self.epoch)
      self.train_loop()
      self.val_loop()
      self.scheduler.step()
      # record cur_epoch
      fn = os.path.join(self.cfg['save_dir'], "cur_epoch.txt")
      with open(fn, "w") as f:
          f.write("{}\n".format(self.epoch))

  def train_loop(self):
    tic = time.time()
    self.model.train()
    prefix = "Training: {:3d} / {:3d}".format(self.epoch, self.cfg['n_epochs'])
    all_loss = []
    criteria = nn.SmoothL1Loss(reduction='mean', beta=0.1)
    for batch in tqdm(self.train_loader, prefix):
      self.iter_idx += 1
      input_data = batch['input'].cuda().float()
      cloud_nocs = batch['cloud_nocs'].cuda().float()
      pred = self.model(input_data) # pred: (B, N, 3)
      loss = criteria(pred, cloud_nocs)
      all_loss.append(loss.item())

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      
      if self.iter_idx % max(1, len(self.train_loader) // 10) == 0:
        self.tr_writer.add_scalar("train loss per iteration", loss, global_step=self.iter_idx)
        self.save(self.checkpoint_file)
    
    total_time_train_per_epoch = (time.time() - tic) / 3600.0
    avg_loss = np.array(all_loss).mean()
    print_str = "Train Epoch %3d | loss %f | Time %.2fhr | lr %.20f" % \
        (self.epoch, avg_loss, total_time_train_per_epoch, self.optimizer.param_groups[0]['lr']) 
    print(print_str, flush=True)
    log(print_str, self.log_fn)
    self.tr_writer.add_scalar("train loss per epoch", avg_loss, global_step=self.epoch)
    self.save(self.checkpoint_file)

  def val_loop(self):
    tic = time.time()
    self.model.eval()
    all_loss = []
    criteria = nn.SmoothL1Loss(reduction='mean', beta=0.1)
    for batch in tqdm(self.val_loader):
      input_data = batch['input'].cuda().float()
      cloud_nocs = batch['cloud_nocs'].cuda().float()
      pred = self.model(input_data)
      loss = criteria(pred, cloud_nocs)
      all_loss.append(loss.item())
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

    avg_loss = np.array(all_loss).mean()
    toal_time_test_per_epoch = (time.time() - tic) / 3600.0
    print_str = "Test Epoch %3d | loss %f | Time %.2fhr" % \
        (self.epoch, avg_loss, toal_time_test_per_epoch)
    print(print_str, flush=True)
    log(print_str, self.log_fn)
    self.tr_writer.add_scalar("test loss per epoch", avg_loss, global_step=self.epoch)
    self.save(self.checkpoint_file)

    if avg_loss < self.best_val:
        self.save(self.bestmodel_file)
        self.best_val = avg_loss
        fn = os.path.join(self.cfg['save_dir'], "best_va_acc.txt")
        with open(fn, "w") as f:
            f.write("{}\n".format(self.best_val))
            f.write("{}\n".format(self.epoch))
