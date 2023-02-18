import numpy as np
from transformations import *
import xml.etree.ElementTree as ET
import os,sys,yaml,copy,pickle,time,cv2,socket,argparse,gzip,trimesh
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
from Utils import *
import pybullet as p
import pybullet_data
from PIL import Image
from pybullet_env.env import *
hostname = socket.gethostname()


def generate_clutter_data_worker(class_name,ids,gui,cfg,cfg_dataset,split,gripper):
  def spawn_env():
    env = Env(cfg, gripper, gui=gui)
    env.add_bin(pos=np.array([0.45,-0.5,0.1]))
    p.removeBody(env.robot_id) # 生成杂乱场景 RGBD 不需要机器人，只要相机和箱子
    env.env_body_ids = PU.get_bodies()
    hostname = socket.gethostname()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    env.out_dir = f"{code_dir}/dataset/{class_name}/{split}"
    return env

  env = spawn_env()
  print(f"\n\nSaving to {env.out_dir}\n\n")
  os.system(f'mkdir -p {env.out_dir}')
  with open(f'{env.out_dir}/cfg.yml','w') as ff:
    yaml.dump(cfg,ff)

  cam_in_bin_ori = env.cam_in_bin.copy()
  for i,id in enumerate(ids):
    # 随机一个相机位姿，保证能观察到完整的箱子
    while 1:
      tf = random_uniform_magnitude(max_T=0.05,max_R=10) # 随机一个相机位姿
      env.cam_in_bin = cam_in_bin_ori@tf
      verts = (np.linalg.inv(env.cam_in_bin)@(to_homo(env.bin_verts).T)).T[:,:3] # bin 坐标（相机系）
      projected = env.K@verts.T # bin 投影到相机
      us = projected[:,0]/projected[:,2].reshape(-1,1) # 成像齐次坐标分量 n（像素数） x 1
      vs = projected[:,1]/projected[:,2].reshape(-1,1)
      if us.min()>=0 and us.max()<cfg['W'] and vs.min()>=0 and vs.max()<cfg['H']:
        break

    # 随机一个模型，生成场景用
    name = np.random.choice(cfg_dataset[class_name]['train']) 
    code_dir = os.path.dirname(os.path.realpath(__file__))
    obj_file = f'{code_dir}/data/object_models/{name}'
    
    # 生成场景
    env.generate_one(obj_file=obj_file, data_id=id, scale_range=cfg_dataset['object_scales']
                    , n_ob_range=cfg_dataset['num_pile_objects'])
    
    env.reset()
    if i%10==0:
      del env
      env = spawn_env()


def generate_clutter_data():
  for split in ['train','test']:
    if split=='train':
      ids = np.arange(cfg_dataset['n_train']) # 训练场景数 20000
    else:
      ids = np.arange(cfg_dataset['n_val']) # 测试场景数 1000
    gui = False
    generate_clutter_data_worker(class_name,ids,gui,cfg,cfg_dataset,split,gripper)




if __name__=="__main__":
  from data_reader import DataReader
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser = argparse.ArgumentParser()
  parser.add_argument('--class_name',type=str,default='nut')
  parser.add_argument('--config_dir',type=str,default="{}/config.yml".format(code_dir))
  args = parser.parse_args()

  with open(args.config_dir,'r') as ff:
    cfg = yaml.safe_load(ff)
  with open(f'{code_dir}/config_grasp.yml','r') as ff:
    cfg_grasp = yaml.safe_load(ff)

  class_name = args.class_name
  cfg_dataset = cfg['dataset']
  gripper = RobotGripper.load(gripper_dir=cfg_grasp['gripper_dir'][class_name])

  generate_clutter_data()
