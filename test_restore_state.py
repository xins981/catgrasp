import pybullet as p
from pybullet_env.env_grasp import *
from pybullet_env.env import *
from pybullet_env.utils_pybullet import *
from pybullet_env.env_semantic_grasp import *

#region 恢复测试场景
code_dir = os.path.dirname(os.path.realpath(__file__))

cfg_grasp = YamlConfig("{}/config_grasp.yml".format(code_dir))
gripper = RobotGripper.load(gripper_dir=cfg_grasp['gripper_dir']['screw'])
with open('{}/config.yml'.format(code_dir),'r') as ff:
    cfg = yaml.safe_load(ff)


# 加载环境
obj_path = f"{code_dir}/data/object_models/screw_carr_94323A329_NYLON.obj"
env = Env(cfg,gripper,gui=False)
env.add_bin()
place_dir = obj_path.replace('.obj','_place.obj')
place_id,_ = create_object(place_dir,scale=np.array([1,1,1]),ob_in_world=np.eye(4),mass=0.1,useFixedBase=True,concave=True)
p.changeDynamics(place_id,-1,lateralFriction=0.1,spinningFriction=0.1,collisionMargin=0.0001)
p.changeVisualShape(place_id,-1,rgbaColor=[1,1,1,1])
create_gripper_visual_shape(env.env_grasp.gripper)
env.env_body_ids = PU.get_bodies()

# 添加零件
num_obj = 1
env.make_pile(obj_file=obj_path,scale_range=[1,1],n_ob_range=[num_obj,num_obj+1])

# 恢复场景状态
bullet_run_id = 1
bullet_state_id = 1
state_path = f'/tmp/catgrasp_vis_A329_NYLON/screw_carr_94323A329_NYLON_run{bullet_run_id:0>2d}_state{bullet_state_id:0>2d}.bullet'
p.restoreState(fileName=state_path)

rgb,depth,seg = env.camera.render(env.cam_in_world)
Image.fromarray(rgb).save(f'{code_dir}/screw_carr_94323A329_NYLON_run{bullet_run_id:0>2d}_state{bullet_state_id:0>2d}_rgb.png')
#endregion