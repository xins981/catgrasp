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

code_dir = os.path.dirname(os.path.realpath(__file__))
# debug_dir = '/tmp/catgrasp_vis'
# obj_name = 'screw_carr_97435A401_WEAR'
debug_dir = '/tmp/catgrasp_vis_A329_NYLON'
obj_name = 'screw_carr_94323A329_NYLON'
run_id = 1
state_id = 1
body_id = 6
atp = 0
scene_dir = '{}/{}_run{:0>2d}_state{:0>2d}_scene.ply' .format(debug_dir, obj_name, 7, state_id)
bin_no_ground_dir = 'bin_no_ground.ply'
ob_dir = '{}/{}_run{:0>2d}_state{:0>2d}_body{:0>2d}_atp{}_ob.ply' .format(debug_dir, obj_name, run_id, state_id, body_id, atp)
obj_part1_dir = 'screw_carr_94323A329_NYLON_run01_state01_body06_atp0_ob_part1.ply'
obj_part2_dir = 'screw_carr_94323A329_NYLON_run01_state01_body06_atp0_ob_part2.ply'

#region 推移操作定义
def move(pose_in_cam, attachment=None):
  pose_in_world = env.cam_in_world @ pose_in_cam
  gripper_in_world = pose_in_world @ np.linalg.inv(env.env_grasp.grasp_pose_in_gripper_base)
  # set_body_pose_in_world(gripper_vis_id, gripper_in_world)
  if attachment != None:
    # command = env.move_arm(link_id=env.gripper_id, link_pose=gripper_in_world, obstacles=[], timeout=5, 
    #                       attachments=[attachment], use_ikfast=True)
    command = env.move_arm(link_id=env.gripper_id, link_pose=gripper_in_world, obstacles=[], timeout=5, 
                          use_ikfast=True)
  else:
    command = env.move_arm(link_id=env.gripper_id, link_pose=gripper_in_world, obstacles=[], timeout=5, 
                          use_ikfast=True)
  if command != None:
    command.execute(time_step=0.05)

def push(pose0, pose1, body_id):  # pylint: disable=unused-argument
  over0 = pose0.copy()
  over1 = pose1.copy()
  over0[2, 3] = 0.61
  over1[2, 3] = 0.61
  save_grasp_pose_mesh(gripper, pose0, out_dir=f'{code_dir}/pose0.obj')
  save_grasp_pose_mesh(gripper,pose0,out_dir=f'{code_dir}/pose0_enclosed.obj',enclosed=True)
  save_grasp_pose_mesh(gripper, pose1, out_dir=f'{code_dir}/pose1.obj')
  save_grasp_pose_mesh(gripper,pose1,out_dir=f'{code_dir}/pose1_enclosed.obj',enclosed=True)

  # save_grasp_pose_mesh(gripper, over0, out_dir=f'{code_dir}/over0.obj')
  # save_grasp_pose_mesh(gripper, over1, out_dir=f'{code_dir}/over1.obj')

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

  attachment = PU.create_attachment(env.robot_id, env.gripper_id, body_id)
  for i in range(n_push): # 单位毫米为移动单位，完成整个移动过程
    target[:3,3] = position0 + vec * i * 0.001
    move(target, attachment)
    
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

timeStep=1./60.
p.setTimeStep(timeStep)

# 添加零件
num_obj = 1
env.make_pile(obj_file=obj_path,scale_range=[1,1],n_ob_range=[num_obj,num_obj+1])

# 恢复场景状态
bullet_run_id = 1
bullet_state_id = 1
state_path = f'/tmp/catgrasp_vis_A329_NYLON/screw_carr_94323A329_NYLON_run01_state01.bullet'
# state_path = f'/tmp/catgrasp_vis_A329_NYLON/screw_carr_94323A329_NYLON_run{bullet_run_id:0>2d}_state{bullet_state_id:0>2d}.bullet'
p.restoreState(fileName=state_path)

# for body_id in env.ob_ids:
#   p.changeDynamics(body_id,-1,activationState=p.ACTIVATION_STATE_WAKE_UP)
# p.stepSimulation()

rgb,depth,seg = env.camera.render(env.cam_in_world)
Image.fromarray(rgb).save(f'{code_dir}/screw_carr_94323A329_NYLON_run{bullet_run_id:0>2d}_state{bullet_state_id:0>2d}_rgb_befor_push.png')

# tm_i = 0
# while tm_i < 100:
#   p.stepSimulation()
#   time.sleep(timeStep)
#   tm_i += 1
#endregion

#region 相机采集空间信息
scene_pts = o3d.io.read_point_cloud(scene_dir).points
ob_pts = o3d.io.read_point_cloud(ob_dir).points
bin_no_ground = o3d.io.read_point_cloud(bin_no_ground_dir)
obj_part1 = o3d.io.read_point_cloud(obj_part1_dir)
obj_part2 = o3d.io.read_point_cloud(obj_part2_dir)

# 零件一侧的 bin 表面法向
bin_edge = None
dist_edge = float('inf')
for i in range(1,5):
  tmp_edge = o3d.io.read_point_cloud(f'bin_edge_{i}.ply')
  tmp_dist = np.sum(chamfer_distance(obj_part1.points, tmp_edge.points))
  if tmp_dist < dist_edge:
    dist_edge = tmp_dist
    bin_edge = tmp_edge
bin_edge_pts = np.array(bin_edge.points)
mean_edge = np.mean(bin_edge_pts, axis=0)
kdtree_edge = cKDTree(bin_edge_pts)
_,indice = kdtree_edge.query(mean_edge)
edge_normal = np.array(bin_edge.normals[indice])

# 碰撞检测障碍物
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
ob_pts = np.array(ob_pts)
background_pts,ids = cloudA_minus_cloudB(background_pts,ob_pts,thres=0.005)
pcd = toOpen3dCloud(background_pts)
pcd = pcd.voxel_down_sample(voxel_size=0.001)
background_pts = my_cpp.makeOccupancyGridFromCloudScan(np.asarray(pcd.points), K, octomap_resolution)
#endregion

#region 计算采样点（靠近盒子一侧的中间部分）
dist_part1 = np.sum(chamfer_distance(obj_part1.points, bin_edge.points))
dist_part2 = np.sum(chamfer_distance(obj_part2.points, bin_edge.points))
pcd = obj_part1 if dist_part1 <= dist_part2 else obj_part2
sample_pts = np.array(pcd.points).copy()
sample_normals = np.array(pcd.normals).copy()

max_xyz = sample_pts.max(axis=0)
min_xyz= sample_pts.min(axis=0)
center = (max_xyz + min_xyz) / 2
dists = np.linalg.norm(sample_pts - center, axis=1) 
min_dist = dists.min()
dists_mean = np.mean(dists)
centre_keep_id = [i for i, dist in enumerate(dists) if dist < dists_mean]
sample_pts = sample_pts[centre_keep_id]
sample_normals = sample_normals[centre_keep_id]

# centre_pcd = toOpen3dCloud(sample_pts, normals=sample_normals)
# o3d.io.write_point_cloud('centre.ply', centre_pcd)
#endregion

#region 采样候选推移位姿
push_start_poses = []
push_end_poses = []
xy_axis = np.array([
    [1, 0, 0],
    [0, 1, 0],
])
init_bite = 0.002
dist_push = 0.03
dist_finger_centre_to_sufc = 0.03

for i in range(len(sample_pts)):
  point = np.array(sample_pts[i])
  normal = np.array(sample_normals[i])
  normal /= np.linalg.norm(normal)
  normal_pro_xy = normal.copy()
  normal_pro_xy[2] = 0
  normal_pro_xy_vec = normal_pro_xy / np.linalg.norm(normal_pro_xy)

  dot = []
  for axis in xy_axis:
    dot.append(normal_pro_xy_vec.dot(axis))
  axis_id = np.argmax(np.abs(dot))
  finger_adj_vec = xy_axis[axis_id] if dot[axis_id] > 0 else -xy_axis[axis_id]
  push_vec = -finger_adj_vec
  # 过滤：1.手指调整方向和采样点法向相反；2.推移方向和近侧盒边的法向相反；
  if finger_adj_vec.dot(normal) < 0 or push_vec.dot(edge_normal) < 0.5:
    continue
  finger_adj = 0.009 * finger_adj_vec 
  finger_outside = point + finger_adj # 手指外侧
  
  t = finger_outside + push_vec * dist_finger_centre_to_sufc # 根据手指外侧坐标，推算 push 坐标（这里只确认了平面 xy 坐标）
  R = np.diag(np.ones(3))[:, [2,1,0]]
  R[1,1] *= -1
  pose_start = np.eye(4)
  pose_start[:3,3] = t
  pose_start[2,3] = 0.7295
  # pose_start[2,3] = 0.723
  pose_start[:3,:3] = R
  push_start_poses.append(pose_start)

  pose_end = pose_start.copy()
  pose_end[:3,3] += (push_vec * dist_push)
  push_end_poses.append(pose_end)

  # approach_dir = R[:,0] # 启发式探索高度
  # for d in np.arange(-10, 10, 1):
  #   push_origin = t + (init_bite + d * 1e-3) * approach_dir
  #   pose_start = np.eye(4)
  #   pose_start[:3,3] = push_origin
  #   pose_start[:3,:3] = R
  #   push_start_poses.append(pose_start)

  #   pose_end = pose_start.copy()
  #   pose_end[:3,3] += (push_vec * dist_push)
  #   push_end_poses.append(pose_end)
#endregion

#region 过滤推移位姿(检测碰撞)
collision_keep_ids = my_cpp.filterPushPose(push_start_poses, gripper_in_grasp,
                                          gripper.trimesh_enclosed.vertices,
                                          gripper.trimesh_enclosed.faces,
                                          background_pts, resolution)
collision_keep_ids = collision_keep_ids[5:]
push_start_poses = np.array(push_start_poses)[collision_keep_ids]
push_end_poses = np.array(push_end_poses)[collision_keep_ids]
#endregion

#region 选择推移位姿
# finger_meshes = env.env_grasp.finger_meshes
# grip_dirs = env.env_grasp.grip_dirs
# num_contact = []
# for i,_ in enumerate(push_start_poses):
#   start = push_start_poses[i]
#   end = push_end_poses[i]
#   translation_in_cam = end[:3, 3] - start[:3, 3]
#   translation_in_cam = translation_in_cam / np.linalg.norm(translation_in_cam)
#   translation_in_cam_homo = np.append(translation_in_cam, 1)
#   # 平移向量变换到抓取系
#   cam_to_push = np.linalg.inv(start)
#   translation_in_push = (cam_to_push @ translation_in_cam_homo)[:3]
#   # 计算与 y 轴正负方向的夹角
#   dot_y = []
#   for i in range(0, 1):
#     dot_y.append(xy_axis[i].dot(translation_in_push))
#   # 选择与平移向量同向的闭合方向手指，计算其接触点
#   finger_id = np.argmax(dot_y)
#   finger_mesh_in_world = PU.get_link_mesh_pose_matrix(env.robot_id,env.finger_ids[0])
#   gripper_base_in_world = get_link_pose_in_world(env.robot_id,env.gripper_id)
#   finger_mesh_in_gripper_base = np.linalg.inv(gripper_base_in_world)@finger_mesh_in_world
#   grasp_in_gripper_base = gripper.get_grasp_pose_in_gripper_base()
#   finger_mesh_in_grasp = np.linalg.inv(grasp_in_gripper_base)@finger_mesh_in_gripper_base
#   cam_in_finger = np.linalg.inv(finger_mesh_in_grasp)@np.linalg.inv(start)
#   surface_pts, dist_to_finger_surface = get_finger_contact_area(finger_meshes[finger_id],ob_in_finger=cam_in_finger,
#                                                                 ob_pts=sample_pts,
#                                                                 ob_normals=sample_normals,
#                                                                 grip_dir=grip_dirs[finger_id],surface_tol=0.005)
#   if surface_pts is None:
#     num_contact.append(0)
#   else:
#     num_contact.append(len(surface_pts))
  
# 按照接触点的数量排序，选出前五个
# num_contact = np.array(num_contact)
# k = 5 if len(num_contact) > 4 else len(num_contact)
# contact_keep_id = num_contact.argsort()[-k:][::-1]
# push_start_poses = push_start_poses[contact_keep_id]
# push_end_poses = push_end_poses[contact_keep_id]

# 计算距离中心点最近的终点位姿位置
bin_no_ground_pts = np.array(bin_no_ground.points)
bin_no_ground_pts_max = bin_no_ground_pts.max(axis=0)
bin_no_ground_pts_min = bin_no_ground_pts.min(axis=0)
bin_center = (bin_no_ground_pts_max + bin_no_ground_pts_min) / 2
bin_center[2] = 0
dist_center_xy = []
poses_height = []
for end in push_end_poses:
  end_position = end[:3,:3].copy()
  poses_height.append(end[2,3])
  end_position[2] = 0
  dist_center_xy.append(np.linalg.norm(end_position - bin_center))

poses_height = np.array(poses_height)
dist_center_xy = np.array(dist_center_xy)
height_keep_id = poses_height.argsort()[-5:][::-1]
dist_center_xy = dist_center_xy[height_keep_id]
push_start_poses = push_start_poses[height_keep_id]
push_end_poses = push_end_poses[height_keep_id]
exec_keep_id = np.argmin(dist_center_xy)
exec_push = (push_start_poses[exec_keep_id], push_end_poses[exec_keep_id])
#endregion

push(exec_push[0], exec_push[1], env.ob_ids[0])
# push(push_start_poses[0], push_end_poses[0], env.ob_ids[0])

rgb,depth,seg = env.camera.render(env.cam_in_world)
Image.fromarray(rgb).save(f'{code_dir}/screw_carr_94323A329_NYLON_run{bullet_run_id:0>2d}_state{bullet_state_id:0>2d}_rgb_after_push.png')

# while (1):
# 	p.stepSimulation()
# 	time.sleep(timeStep)
        
# ================================================================================================================
#region 移除盒子底部
# 根据法向过滤点云
# pcd = np.array(pcd_scene.points).copy() # 深拷贝
# normals = np.array(pcd_scene.normals).copy()
# inner_prod = normals.dot([0, 0, -1])
# mask = np.logical_and(inner_prod > 0.2, inner_prod < 0.9)
# #  [ for item in inner_prod if item >]  (inner_prod < 0.9)
# pcd = pcd[mask]
# normals = normals[mask]

# 保存新点云
# pcd_bin_no_ground = toOpen3dCloud(pcd, normals=normals)
# o3d.io.write_point_cloud('{}/{}_run{:0>2d}_state{:0>2d}_bin_no_ground.ply'\
#                          .format(debug_dir, obj_name, run_id, state_id), pcd_bin_no_ground)
#endregion