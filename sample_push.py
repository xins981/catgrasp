import open3d as o3d
import numpy as np
from Utils import *
from dexnet.grasping.gripper import RobotGripper
from autolab_core import YamlConfig
import my_cpp
import pybullet as p
from pybullet_env.env_grasp import *
from pybullet_env.env import *
from pybullet_env.utils_pybullet import *
from pybullet_env.env_semantic_grasp import *
import pybullet_tools.utils as PU
import trimesh
import json

code_dir = os.path.dirname(os.path.realpath(__file__))

bin_no_ground_dir = 'bin_no_ground.ply'


def sample_push(ob_pts, ob_norms):
    sample_ids = farthest_point_sample(ob_pts, 1000)
    pts = ob_pts[sample_ids]
    norms = ob_norms[sample_ids]

    resolution = compute_cloud_resolution(pts)
    r_ball = resolution * 3
    point_cloud_kdtree = cKDTree(pts)
    
    gripper_dir = f'{code_dir}/../../urdf/robotiq_hande'
    f = open(os.path.join(gripper_dir, 'params.json'), 'r')
    params = json.load(f)
    
    hands = []

    pt_id = 0
    while pt_id < len(pts):
        pt = pts[pt_id]
        norm = norms[pt_id]

        #region local frame
        kd_indices = point_cloud_kdtree.query_ball_point(pt.reshape(1,3),r=r_ball)
        kd_indices = np.array(kd_indices[0]).astype(int).reshape(-1)
        sqr_distances = np.linalg.norm(pt.reshape(1,3)-pts[kd_indices], axis=-1) ** 2

        M = np.zeros((3, 3))
        for _ in range(len(kd_indices)):
            if sqr_distances[_] != 0:
                norm = norms[kd_indices[_]]
                norm = norm.reshape(-1, 1)
                if np.linalg.norm(norm) != 0:
                    norm /= np.linalg.norm(norm)
                M += np.matmul(norm, norm.T)
        
        if sum(sum(M)) == 0:
            print("M matrix is empty as there is no point near the neighbour")
            r_ball *= 2
            print(f"Here is a bug, if points amount is too little it will keep trying and never go outside. Update r_ball to {r_ball} and resample")
            continue

        approach_normal = -norm.reshape(3)
        approach_normal /= np.linalg.norm(approach_normal)
        eigval, eigvec = np.linalg.eig(M)

        def proj(u,v):
            u = u.reshape(-1)
            v = v.reshape(-1)
            return np.dot(u,v)/np.dot(u,u) * u
        
        minor_pc = eigvec[:, np.argmin(eigval)].reshape(3)
        minor_pc = minor_pc-proj(approach_normal,minor_pc)
        minor_pc /= np.linalg.norm(minor_pc)
        major_pc = np.cross(minor_pc, approach_normal)
        major_pc = major_pc / np.linalg.norm(major_pc)
        
        R = np.concatenate((approach_normal.reshape(3,1), major_pc.reshape(3,1), minor_pc.reshape(3,1)), axis=1)
        #endregion

        #region grid search
        hands_local = []
        
        # rot by minor axis
        for minor_rot in np.arange(-90, 90, 30):
            R_inplane = euler_matrix(0,0,minor_rot*np.pi/180,axes='sxyz')[:3,:3]
            rot_origin = R @ R_inplane
            
            # 点云转换到 hand 系
            frame_origin = np.eye(4)
            frame_origin[:3,:3] = rot_origin
            frame_origin[:3,3] = pt
            cloud_in_hand = frame_origin @ pts
            
            # 截取指背内点云
            keep_height = []
            for i in np.arange(len(cloud_in_hand)):
                point = cloud_in_hand[i]
                z_val = point[2]
                if z_val > -params['hand_height'] and z_val < params['hand_height']:
                    keep_height.append(i)
            pts_croped_h = cloud_in_hand[keep_height]

            # translation along major axis
            for major_off in np.linspace(-5, 5, 10):
                is_coll = False
                lfinger_y = -params['finger_width'] + major_off
                rfinger_y = params['finger_width'] + major_off
                
                # 截取指侧内的点云
                pts_croped = [] # 既在指尖内又在指侧内的点云(抵近侦测)
                for pt_in_h in pts_croped_h:
                    x_val = pt_in_h[0]
                    y_val = pt_in_h[1]
                    if y_val > lfinger_y and y_val < rfinger_y:
                        if x_val < 0:
                            is_coll = True
                            break
                        pts_croped.append(pt_in_h)
                if is_coll:
                    continue
                
                # 抵近侦测
                last_depth = 0
                for depth_off in np.arange(0, 50, 5):
                    tip_x = last_depth + depth_off
                    for p in pts_croped:
                        x_val = p[0]
                        if x_val < tip_x:
                            is_coll = True
                            break
                    if is_coll == True:
                        break
                    last_depth = tip_x
                off_vec = [last_depth, major_off, 0]
                frame_hand = frame_origin.copy()
                frame_hand[:3,3] += off_vec
                hands_local.append(frame_hand)
        
        #endregion
        
        hands += hands_local
        pt_id += 1
    
    return hands


def EvalPush(push_candidates, ob_pts, background_pts):
    # 计算零件间，和边界的距离，构造距离图
    dist_origin = np.sum(chamfer_distance(ob_pts, background_pts))

    bin_no_ground = o3d.io.read_point_cloud(bin_no_ground_dir)
    aabb_bin = bin_no_ground.get_axis_aligned_bounding_box()
    max_boun = aabb_bin.get_max_bound()
    min_boun = aabb_bin.get_min_bound()

    # 过滤越界动作
    push_in_bin = []
    push_translations = []
    for push_cand in push_candidates:
        move_actu = push_cand[1]
        if move_actu[2] != 0:
            move_actu[2] = 0
        ob_trfmd = ob_pts + move_actu
        if ob_trfmd[0] < min_boun[0] or ob_trfmd[0] > max_boun[0] or ob_trfmd[1] < min_boun[1] or ob_trfmd[1] > max_boun[1]:
            continue
        push_in_bin.append((push_cand[0], move_actu))
        push_translations.append(move_actu)

    # 过滤碰撞动作
    colli_keep_ids = my_cpp.detectCollisionMove(push_translations, ob_pts, background_pts, octomap_resolution)
    push_colli_free = push_in_bin[colli_keep_ids]
  
    # 打分
    push_filted = []
    for p in push_colli_free:
        ob_trfmd = ob_pts + p[1]
        dist_p = np.sum(chamfer_distance(ob_trfmd, background_pts))
        if dist_p > dist_origin:
            continue
        p.score = dist_p
        push_filted.append(p)

    ret = sorted(push_filted, key=lambda p: -p.score)
    return ret


# debug_dir = '/tmp/catgrasp_vis'
# obj_name = 'screw_carr_97435A401_WEAR'
debug_dir = '/tmp/catgrasp_vis_A329_NYLON'
obj_name = 'screw_carr_94323A329_NYLON'
run_id = 1
state_id = 1
body_id = 6
atp = 0
scene_dir = '{}/{}_run{:0>2d}_state{:0>2d}_scene.ply' .format(debug_dir, obj_name, 7, state_id)
ob_dir = '{}/{}_run{:0>2d}_state{:0>2d}_body{:0>2d}_atp{}_ob.ply' .format(debug_dir, obj_name, run_id, state_id, body_id, atp)
obj_part1_dir = 'screw_carr_94323A329_NYLON_run01_state01_body06_atp0_ob_part1.ply'
obj_part2_dir = 'screw_carr_94323A329_NYLON_run01_state01_body06_atp0_ob_part2.ply'

#region 推移操作定义
def move(pose_in_cam):
  pose_in_world = env.cam_in_world @ pose_in_cam
  gripper_in_world = pose_in_world @ np.linalg.inv(env.env_grasp.grasp_pose_in_gripper_base)
  # set_body_pose_in_world(gripper_vis_id, gripper_in_world)
  command = env.move_arm(link_id=env.gripper_id, link_pose=gripper_in_world, obstacles=[], timeout=5, 
                          use_ikfast=True)
  if command != None:
    command.execute(time_step=0.05)

def push(pose0, pose1, body_id):
  over0 = pose0.copy()
  over1 = pose1.copy()
  over0[2, 3] = 0.61
  over1[2, 3] = 0.61
  save_grasp_pose_mesh(gripper, pose0, out_dir=f'{code_dir}/pose0.obj')
  save_grasp_pose_mesh(gripper,pose0,out_dir=f'{code_dir}/pose0_enclosed.obj',enclosed=True)
  save_grasp_pose_mesh(gripper, pose1, out_dir=f'{code_dir}/pose1.obj')
  save_grasp_pose_mesh(gripper,pose1,out_dir=f'{code_dir}/pose1_enclosed.obj',enclosed=True)

  # Execute push.
  num_arms = len(env.arm_ids)
  PU.set_joint_positions(env.robot_id, env.arm_ids, np.zeros(num_arms))
  env.env_grasp.open_gripper()
  move(over0)
  move(pose0)

  position0 = pose0[:3,3]
  position1 = pose1[:3,3]
  vec = np.float32(position1) - np.float32(position0) # 移动向量
  len_push = np.linalg.norm(vec) # 移动距离
  vec = vec / len_push # 移动方向
  n_push = np.int32(np.floor(np.linalg.norm(position1 - position0) / 0.001))
  target = pose0.copy()

  for i in range(n_push): # 单位毫米为移动单位，完成整个移动过程
    target[:3,3] = position0 + vec * i * 0.001
    move(target)
    
  move(pose1)
  move(over1)
#endregion

#region 构造仿真环境
cfg_grasp = YamlConfig("{}/config_grasp.yml".format(code_dir))
gripper = RobotGripper.load(gripper_dir=cfg_grasp['gripper_dir']['screw'])
with open('{}/config.yml'.format(code_dir),'r') as ff:
  cfg = yaml.safe_load(ff)

# 加载环境
obj_path = f"{code_dir}/data/object_models/screw_carr_94323A329_NYLON.obj"
env = Env(cfg,gripper,gui=True)
env.add_bin()
place_dir = obj_path.replace('.obj','_place.obj')
place_id,_ = create_object(place_dir,scale=np.array([1,1,1]),ob_in_world=np.eye(4),mass=0.1,useFixedBase=True,concave=True)
p.changeDynamics(place_id,-1,lateralFriction=0.1,spinningFriction=0.1,collisionMargin=0.0001)
p.changeVisualShape(place_id,-1,rgbaColor=[1,1,1,1])
gripper_vis_id = create_gripper_visual_shape(env.env_grasp.gripper)
env.env_body_ids = PU.get_bodies()

# 添加零件
num_obj = 1
env.make_pile(obj_file=obj_path,scale_range=[1,1],n_ob_range=[num_obj,num_obj+1])

# 恢复场景状态
bullet_run_id = 1
bullet_state_id = 1
state_path = f'/tmp/catgrasp_vis_A329_NYLON/screw_carr_94323A329_NYLON_run01_state01.bullet'
p.restoreState(fileName=state_path)

rgb,depth,seg = env.camera.render(env.cam_in_world)
Image.fromarray(rgb).save(f'{code_dir}/screw_carr_94323A329_NYLON_run{bullet_run_id:0>2d}_state{bullet_state_id:0>2d}_rgb_befor_push.png')
#endregion

#region 位姿采样
# 目标零件点云
mesh = trimesh.load(ob_dir)
pts, face_ids = trimesh.sample.sample_surface_even(mesh, 1024)
normals = mesh.face_normals[face_ids]
pcd = toOpen3dCloud(pts)
pcd.voxel_down_sample(voxel_size=0.001)
ob_pts = np.asarray(pcd.points).copy() # 采样点云
ob_normals = copy.deepcopy(normals) # 采样法向



scene_pts = o3d.io.read_point_cloud(scene_dir).points # 场景点云

# 碰撞检测，障碍物网格
with open('{}/config.yml'.format(code_dir),'r') as ff:
    cfg = yaml.safe_load(ff)
K = np.array(cfg['K']).reshape(3,3)
resolution = 0.0005
gripper_in_grasp = np.linalg.inv(gripper.get_grasp_pose_in_gripper_base())
octomap_resolution = 0.001
kdtree = cKDTree(ob_pts)
dists,indices = kdtree.query(scene_pts)
gripper_diameter = np.linalg.norm(gripper.trimesh.vertices.max(axis=0)-gripper.trimesh.vertices.min(axis=0))
keep_ids = np.where(dists<=gripper_diameter/2)[0]
background_pts = np.array(scene_pts)[keep_ids]
# background_pts,ids = cloudA_minus_cloudB(background_pts,ob_pts,thres=0.005)
pcd = toOpen3dCloud(background_pts)
pcd = pcd.voxel_down_sample(voxel_size=0.001)
background_pts = my_cpp.makeOccupancyGridFromCloudScan(np.asarray(pcd.points), K, octomap_resolution)

n_sphere_dir = 150
sphere_pts = hinter_sampling(min_n_pts=1000,radius=1)[0]
sphere_pts = sphere_pts/np.linalg.norm(sphere_pts,axis=-1).reshape(-1,1)
higher_mask = sphere_pts[:,2]>=np.cos(60*np.pi/180)
sphere_pts = sphere_pts[higher_mask]
rot_y_180 = euler_matrix(0,np.pi/2,0,axes='sxyz')[:3,:3]
sphere_pts = (rot_y_180@sphere_pts.T).T
if sphere_pts.shape[0]>n_sphere_dir:
    ids = np.random.choice(np.arange(len(sphere_pts)),size=n_sphere_dir,replace=False)
    sphere_pts = sphere_pts[ids]
print('#sphere_pts={}'.format(len(sphere_pts)))

# 零件形心
max_xyz = ob_pts.max(axis=0)
min_xyz= ob_pts.min(axis=0)
centroid = (max_xyz + min_xyz) / 2 

sampler = PushSampler()
push_heads = sampler.sample_pushs()
push_val = 0.03
y_axis = [0, 1, 0]
push_candidate = []
for head in push_heads: # push frame to cam frame
  appro_dir = head[:3,0]
  cos_val = np.cos(appro_dir, [0,0,1])
  if cos_val > np.cos(30):
    for angle_horiz in np.linspace(0, 360, 12):
      rot_z = euler_matrix(0,0,angle_horiz*np.pi/180,axes='sxyz')[:3,:3]
      push_vec = rot_z @ y_axis
      push_candidate.append((head, push_vec * 0.03))
  else:
    head_pos = head[:3,3]
    push_vec = centroid - head_pos
    push_vec = push_vec / np.linalg.norm(push_vec)
    push_candidate.append((head, push_vec * 0.03))

  
 
    




   



#endregion