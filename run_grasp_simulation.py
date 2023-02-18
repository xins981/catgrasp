import numpy as np
import open3d as o3d
from transformations import *
import os,sys,yaml,copy,pickle,time,cv2,socket,argparse,inspect,trimesh,operator,gzip,re,random
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
sys.path.append("{}/../".format(code_dir))
sys.path.append("{}/ss-pybullet".format(code_dir))
from dexnet.grasping.grasp import ParallelJawPtGrasp3D
from autolab_core import YamlConfig
from dexnet.grasping.grasp_sampler import *
from dexnet.grasping.gripper import RobotGripper
from Utils import *
import pybullet as p
import pybullet_data
import pybullet_tools.utils as PU
from PIL import Image
from data_reader import *
from pybullet_env.env_base import EnvBase
from pybullet_env.env_grasp import *
from pybullet_env.env import *
from pybullet_env.utils_pybullet import *
from pybullet_env.env_semantic_grasp import *
from pointnet2 import *
from aligning import *
from dataset_nunocs import NunocsIsolatedDataset
from dataset_grasp import GraspDataset
from predicter import *
import my_cpp
from functools import partial
from multiprocessing import Pool
import multiprocessing
from functools import partial
from itertools import repeat
try:
  multiprocessing.set_start_method('spawn')
except:
  pass
from collections import Counter


EXECUTE_TIME_STEP = 0.01
num_objects = 0 # 盒子里零件的数量
num_task_grasp_succ = 0
num_stable_grasp = 0
num_attempt_grasp = 0
num_grasp_null = 0
num_fail_move_to_place_top = 0
ob_file = None
cal_trasform = True


def compute_grasp_affordance_worker(grasp,finger_mesh_in_grasp,canonical_pts_in_cam,canonical_normals_in_cam,canonical_affordance,kdtree_canonical_in_cam,grip_dirs,finger_meshes,finger_ids):
  grasp_in_cam = grasp.get_grasp_pose_matrix()
  cam_in_finger = np.linalg.inv(finger_mesh_in_grasp)@np.linalg.inv(grasp_in_cam)
  p_T_given_G = []
  grasp.contacts = {}
  for i_finger in range(len(finger_ids)):
    '''
      canonical_pts_in_cam : 模板点云（相机系）
      grip_dirs : 手指方向（沿 y 轴）
    '''
    surface_pts, dist_to_finger_surface = get_finger_contact_area(finger_meshes[i_finger],ob_in_finger=cam_in_finger,
                                                                  ob_pts=canonical_pts_in_cam,
                                                                  ob_normals=canonical_normals_in_cam,
                                                                  grip_dir=grip_dirs[i_finger],surface_tol=0.005)
    if surface_pts is None:
      continue

    grasp.contacts[i_finger] = surface_pts
    dists,indices = kdtree_canonical_in_cam.query(surface_pts) # 检索接触点的索引
    p_T_given_G.append(canonical_affordance[indices].mean()) # 根据索引迁移模板热图中的任务质量


  p_T_given_G = np.array(p_T_given_G).mean()
  if not np.isfinite(p_T_given_G):
    return None

  grasp.p_T_given_G = p_T_given_G
  return grasp


def compute_grasp_affordance(grasps,env,nocs_pose,nocs_cloud,canonical,debug_dir,i_pick,nocs_predicter,gripper,ob_pts,ob_normals):
  print("Computing affordance")
  finger_mesh_in_world = PU.get_link_mesh_pose_matrix(env.robot_id,env.finger_ids[0])
  gripper_base_in_world = get_link_pose_in_world(env.robot_id,env.gripper_id)
  grasp_in_gripper_base = gripper.get_grasp_pose_in_gripper_base()
  finger_mesh_in_gripper_base = np.linalg.inv(gripper_base_in_world)@finger_mesh_in_world
  finger_mesh_in_grasp = np.linalg.inv(grasp_in_gripper_base)@finger_mesh_in_gripper_base
  # 模板点云（nunocs 系）
  canonical_pcd = toOpen3dCloud(canonical['canonical_cloud'],normals=canonical['canonical_normals']) 
  
  canonical_in_cam_pcd = copy.deepcopy(canonical_pcd)
  canonical_in_cam_pcd.transform(nocs_pose) # nocs_pose --- nunocs 参考系到相机（观察未知实例）系的变换

  kdtree_canonical_in_cam = cKDTree(np.asarray(canonical_in_cam_pcd.points).copy())

  colors = array_to_heatmap_rgb(canonical['canonical_affordance'])
  pcd = toOpen3dCloud(canonical['canonical_cloud'],colors)
  pcd.transform(nocs_pose)
  o3d.io.write_point_cloud(f'{debug_dir}/{i_pick}_canonical_affordance_in_cam.ply',pcd)

  nocs_input_cloud = nocs_predicter.data_transformed['cloud_xyz_original'] # 未知实例点云（相机系）
  pcd = toOpen3dCloud(nocs_input_cloud)
  o3d.io.write_point_cloud(f'{debug_dir}/{i_pick}_nocs_input_cloud.ply',pcd)
  assert nocs_cloud.shape==nocs_input_cloud.shape, f"nocs_cloud {nocs_cloud.shape}, nocs_input_cloud {nocs_input_cloud.shape}"
  kdtree_nocs_input_cloud = cKDTree(nocs_input_cloud)

  pcd = canonical_in_cam_pcd.voxel_down_sample(voxel_size=0.002)
  canonical_pts_in_cam = np.asarray(pcd.points).copy()
  canonical_normals_in_cam = np.asarray(pcd.normals).copy()

  new_grasps = []
  for i,grasp in enumerate(grasps):
    grasp = compute_grasp_affordance_worker(grasp,finger_mesh_in_grasp,canonical_pts_in_cam,canonical_normals_in_cam,
                                            canonical['canonical_affordance'],kdtree_canonical_in_cam,
                                            env.env_grasp.grip_dirs,env.env_grasp.finger_meshes,env.env_grasp.finger_ids)
    if grasp is not None:
      new_grasps.append(grasp)
  grasps = new_grasps
  return grasps


def compute_candidate_grasp_one_ob(ags,ob_pts,ob_normals,open_gripper_collision_pts,scene_pts,i_pick,env,
                                  symmetry_tfs,debug_dir,nocs_predicter,gripper,canonical,ik_func=None,
                                  compute_affordance=True,center_ob_between_gripper=True, ob_id_in_env=None):
  global ob_file, cal_trasform
  # 零件下采样
  pcd = toOpen3dCloud(ob_pts)
  pcd = pcd.voxel_down_sample(voxel_size=0.0005)
  ob_pts_down = np.array(pcd.points).copy()
  if len(ob_pts_down)<100:
    print(f"ob_pts_down {ob_pts_down.shape} must be noise")
    return [],None
  kdtree = cKDTree(ob_pts)
  dists,indices = kdtree.query(ob_pts_down)
  ob_pts_down = ob_pts[indices].reshape(-1,3)
  ob_normals_down = ob_normals[indices].reshape(-1,3)
  data = {
        'cloud_xyz': ob_pts_down,
        'cloud_normal': ob_normals_down,
      }
  pcd = toOpen3dCloud(ob_pts_down,normals=ob_normals_down)
  o3d.io.write_point_cloud(f'{debug_dir}/{i_pick}_ob.ply',pcd)

  # 提取背景和 gripper 碰撞的部分点云
  kdtree = cKDTree(ob_pts)
  dists,indices = kdtree.query(scene_pts)
  gripper_diameter = np.linalg.norm(ags.gripper.trimesh.vertices.max(axis=0)-ags.gripper.trimesh.vertices.min(axis=0))
  keep_ids = np.where(dists<=gripper_diameter/2)[0]
  background_pts = scene_pts[keep_ids]
  background_pts,ids = cloudA_minus_cloudB(background_pts,ob_pts,thres=0.005)
  pcd = toOpen3dCloud(background_pts)
  pcd = pcd.voxel_down_sample(voxel_size=0.001)
  octomap_resolution = 0.001
  background_pts = my_cpp.makeOccupancyGridFromCloudScan(np.asarray(pcd.points),env.K,octomap_resolution)
  pcd = toOpen3dCloud(background_pts)
  o3d.io.write_point_cloud(f'{debug_dir}/{i_pick}_grasp_collision_background_pts.ply',pcd)

  pcd = toOpen3dCloud(data['cloud_xyz'],normals=data['cloud_normal'])
  o3d.io.write_point_cloud(f'{debug_dir}/{i_pick}_nocs_input.ply',pcd)

  
  nocs_cloud, nocs_pose = nocs_predicter.predict(copy.deepcopy(data), cal_transform=cal_trasform, obj_file=ob_file, 
                                                obj_id_in_env=ob_id_in_env, debug_dir=debug_dir, i_pick=i_pick)
  
  if nocs_pose is None:
    print("Cur nocs_pose is None")
    return [],None

  np.savetxt(f'{debug_dir}/{i_pick}_nocs_pose.txt',nocs_pose)

  pcd = toOpen3dCloud(nocs_cloud)
  o3d.io.write_point_cloud(f'{debug_dir}/{i_pick}_nocs_cloud.ply',pcd)
  pcd.transform(nocs_pose) # 相机系点云
  data['nocs_transformed'] = np.asarray(pcd.points).copy()
  o3d.io.write_point_cloud(f'{debug_dir}/{i_pick}_nocs_transformed.ply',pcd)
  print('nocs_pose:\n',nocs_pose)

  # observed_center = (ob_pts.max(axis=0)+ob_pts.min(axis=0))/2 # 部分点云计算得到的形心
  # nocs_in_world = env.cam_in_world@nocs_pose
  # if nocs_predicter.best_ratio<0.5:
  #   print(f'nocs_predicter.best_ratio={nocs_predicter.best_ratio}, might be wrong')

  # if np.linalg.norm(observed_center-nocs_pose[:3,3])>=0.05 or nocs_in_world[2,3]<env.bin_in_world[2,3]:
  #   print('observed_center',observed_center)
  #   print("[WARN] nocs pose seems wrong")
  #   return [],None

  ee_in_grasp = np.linalg.inv(env.grasp_in_ee)
  pcd = toOpen3dCloud(ob_pts_down,normals=ob_normals_down)
  voxel_size = np.linalg.norm(ob_pts_down.max(axis=0)-ob_pts_down.min(axis=0))/10.0
  pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
  ob_pts_for_sample = np.asarray(pcd.points).copy()
  ob_normals_for_sample = np.asarray(pcd.normals).copy()
  grasps = ags.sample_grasps(background_pts=background_pts,points_for_sample=ob_pts_for_sample,
                            normals_for_sample=ob_normals_for_sample,num_grasps=np.inf,max_num_samples=np.inf,
                            n_sphere_dir=cfg_run['cone_grasp_smapler_n_sphere_dir'],
                            approach_step=cfg_run['cone_grasp_smapler_approach_step'],
                            open_gripper_collision_pts=open_gripper_collision_pts,ob_normals_down=ob_normals_down,
                            nocs_pts=nocs_cloud,nocs_pose=nocs_pose,cam_in_world=env.cam_in_world, 
                            ee_in_grasp=ee_in_grasp, upper=env.upper_limits, lower=env.lower_limits, 
                            filter_approach_dir_face_camera=True,ik_func=ik_func,symmetry_tfs=symmetry_tfs,
                            center_ob_between_gripper=center_ob_between_gripper)

  print(f'#grasps={len(grasps)}')

  if compute_affordance:
    grasps = compute_grasp_affordance(grasps=grasps,env=env,nocs_pose=nocs_pose,nocs_cloud=nocs_cloud,
                                      canonical=canonical,debug_dir=debug_dir,i_pick=i_pick,
                                      nocs_predicter=nocs_predicter,gripper=gripper,
                                      ob_pts=ob_pts,ob_normals=ob_normals)

  return grasps,data


def compute_candidate_grasp(rgb,depth,seg,i_pick,env,ags,symmetry_tfs,ik_func=None):
  '''
  @seg: is the gt 2-D seg in simulation. We only use it to remove background objects.
  '''
  ########## Remove background
  bg_mask = depth<0.1
  for id in env.env_body_ids:
    bg_mask[seg==id] = 1

  K = np.array(cfg['K']).reshape(3,3)
  xyz_map = depth2xyzmap(depth,K) # 深度图转点云，相机系
  scene_pts = xyz_map[xyz_map[:,:,2]>=0.1].reshape(-1,3)
  scene_colors = rgb[xyz_map[:,:,2]>=0.1].reshape(-1,3)

  scene_pts_no_bg = xyz_map[bg_mask==0].reshape(-1,3)
  scene_pts_seg_id_no_bg = seg[bg_mask==0].reshape(-1, 1)
  scene_colors_no_bg = rgb[bg_mask==0].reshape(-1,3)
  if len(scene_pts_no_bg)==0:
    print("scene_pts_no_bg is empty")
    return None

  pcd = toOpen3dCloud(scene_pts_no_bg)
  pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.002, max_nn=30))
  pcd = correct_pcd_normal_direction(pcd)
  scene_normals_no_bg = np.asarray(pcd.normals).copy()
  seg_input_data = {'cloud_xyz':scene_pts_no_bg, 'cloud_rgb':scene_colors_no_bg, 'cloud_normal':scene_normals_no_bg}
  scene_seg = seg_predicter.predict(copy.deepcopy(seg_input_data))
  pcd = toOpen3dCloud(seg_predicter.xyz_shifted)
  o3d.io.write_point_cloud(f'{debug_dir}/{i_pick}_xyz_shifted.ply',pcd)

  seg_id_to_obj_id = {}
  seg_colors = np.zeros(seg_input_data['cloud_xyz'].shape)
  seg_ids = np.unique(scene_seg)
  for seg_id in seg_ids:
    seg_mask = scene_seg==seg_id
    n_pt = np.sum(seg_mask)
    is_inlier = True
    if n_pt<500:
      print(f"segment too small, n_pt={n_pt}")
      is_inlier = False
    seg_pts = seg_input_data['cloud_xyz'][seg_mask]
    seg_pts_obj_ids = scene_pts_seg_id_no_bg[seg_mask].reshape(-1).tolist()
    seg_pts_obj_ids_statistic = Counter(seg_pts_obj_ids).most_common()
    seg_pts_obj_id = seg_pts_obj_ids_statistic[0][0]
    seg_id_to_obj_id[seg_id] = seg_pts_obj_id
    if is_inlier:
      max_xyz = seg_pts.max(axis=0)
      min_xyz = seg_pts.min(axis=0)
      diameter = np.linalg.norm(max_xyz-min_xyz)
      density = n_pt / np.prod((max_xyz-min_xyz)/0.001)
      print('segment density',density)
      if density<0.01:
        is_inlier = False

    if is_inlier:
      seg_colors[seg_mask] = np.random.randint(0,255,size=(3))
    else:
      seg_colors[seg_mask] = np.zeros((3))
      scene_seg[seg_mask] = -1

  pcd = toOpen3dCloud(seg_input_data['cloud_xyz'],seg_colors)
  o3d.io.write_point_cloud(f'{debug_dir}/{i_pick}_seg_pred.ply',pcd)

  pcd = toOpen3dCloud(scene_pts,colors=scene_colors)
  pcd = pcd.voxel_down_sample(voxel_size=0.001)
  pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.003, max_nn=30))
  pcd = correct_pcd_normal_direction(pcd)
  o3d.io.write_point_cloud('{}/{}scene.ply'.format(debug_dir,i_pick),pcd)
  scene_pts = np.asarray(pcd.points).copy()
  scene_colors = np.asarray(pcd.colors).copy()

  ob_n_pix = {}
  for seg_id in np.unique(scene_seg):
    if seg_id<0:
      continue
    n_pix = (scene_seg==seg_id).sum()
    ob_n_pix[seg_id] = n_pix
  ob_n_pix = {k: v for k, v in reversed(sorted(ob_n_pix.items(), key=lambda item: item[1]))}
  print("ob_n_pix", ob_n_pix)
  if len(ob_n_pix.keys())==0:
    print("Clutter finished")
    return None

  cnt = 0
  for seg_id in ob_n_pix.keys():
    ob_data = {}
    if ik_func is None:
      tmp_ik_func = None
    else:
      tmp_ik_func = partial(ik_func,obstacles=[])

    ob_pts = seg_input_data['cloud_xyz'][scene_seg==seg_id] # 零件部分点云（实例分割得到)
    ob_normals = seg_input_data['cloud_normal'][scene_seg==seg_id]
    open_gripper_collision_pts = copy.deepcopy(ob_pts)
    grasps,data = compute_candidate_grasp_one_ob(ags,ob_pts,ob_normals,
                                                open_gripper_collision_pts=open_gripper_collision_pts,
                                                scene_pts=scene_pts,i_pick=i_pick,env=env,symmetry_tfs=symmetry_tfs,
                                                debug_dir=debug_dir,nocs_predicter=nocs_predicter,gripper=gripper,
                                                canonical=canonical,ik_func=tmp_ik_func, ob_id_in_env=seg_id_to_obj_id[seg_id])
    if len(grasps)==0:
      continue

    ########### Find the body_id in simulator that current predicted seg correspondes to
    body_ids = np.unique(seg)
    best_body_id = None
    best_dist = np.inf
    for body_id in body_ids:
      if body_id<0:
        continue
      if body_id in env.env_body_ids:
        continue
      ob_pts_sim = xyz_map[seg==body_id].reshape(-1,3)
      ob_pts_sim = ob_pts_sim[ob_pts_sim[:,2]>=0.1]
      dist = np.linalg.norm(ob_pts_sim.mean(axis=0)-ob_pts.mean(axis=0))
      if dist<best_dist:
        best_dist = dist
        best_body_id = body_id
    body_id = copy.deepcopy(best_body_id)

    ob_data[body_id] = data

    cnt += 1

    print("Evaluating grasps...")

    grasp_poses = []
    for i_grasp in range(len(grasps)):
      if i_grasp%max(1,len(grasps)//10)==0:
        print(f"eval grasps {i_grasp}/{len(grasps)}")
      grasp_pose = grasps[i_grasp].get_grasp_pose_matrix()
      grasp_poses.append(grasp_pose)

    ret = grasp_predicter.predict_batch(data,grasp_poses)
    for i_grasp in range(len(grasp_poses)):
      pred_label,confidence,pred = ret[i_grasp]
      p_G = (pred*np.arange(len(pred))).sum()/(len(cfg_grasp['classes'])-1)
      grasps[i_grasp].p_T_G = grasps[i_grasp].p_T_given_G*p_G
      grasps[i_grasp].p_G = p_G
      grasps[i_grasp].seg_id = seg_id
      grasps[i_grasp].pred_label = pred_label
      grasps[i_grasp].pred = pred
      grasps[i_grasp].body_id = body_id

    print('#grasps={}'.format(len(grasps)))
    if len(grasps)==0:
      return None

    def candidate_compare_key(grasp):
      return -grasp.p_T_G
      # return -grasp.p_G

    grasps = sorted(grasps, key=candidate_compare_key)
    yield grasps


def pick_action(grasp,env,gripper_vis_id,i_pick,debug_dir):
  global num_attempt_grasp, num_grasp_null
  PU.set_joint_positions(env.robot_id,env.arm_ids,np.zeros((len(env.arm_ids))))
  env.env_grasp.open_gripper()
  grasp_in_cam = grasp.grasp_pose.copy()
  tmp_id = p.saveState()
  grasp_in_world = env.cam_in_world@grasp_in_cam
  gripper_in_world = grasp_in_world@np.linalg.inv(env.env_grasp.grasp_pose_in_gripper_base)
  obstacles = PU.get_bodies()
  obstacles.remove(env.robot_id)
  set_body_pose_in_world(gripper_vis_id,gripper_in_world)
  command = env.move_arm(link_id=env.gripper_id,link_pose=gripper_in_world,obstacles=[],timeout=5,use_ikfast=True)
  if command is None:
    p.restoreState(tmp_id)
    p.removeState(tmp_id)
    return 0

  p.restoreState(tmp_id)

  command.execute(time_step=EXECUTE_TIME_STEP)

  grasp_in_cam = np.linalg.inv(env.cam_in_world)@grasp_in_world
  gripper = env.env_grasp.gripper
  save_grasp_pose_mesh(gripper,grasp_in_cam,out_dir=f'{debug_dir}/{i_pick}_grasp.obj')
  save_grasp_pose_mesh(gripper,grasp_in_cam,out_dir=f'{debug_dir}/{i_pick}_grasp_enclosed.obj',enclosed=True)
  with gzip.open(f'{debug_dir}/{i_pick}_grasp.pkl','wb') as ff:
    pickle.dump(grasp,ff)

  env.env_grasp.close_gripper()
  num_attempt_grasp += 1

  lower,upper = PU.get_joint_limits(env.robot_id,env.finger_ids[0])
  finger_joint_pos = PU.get_joint_positions(env.robot_id,env.finger_ids)
  if np.all(np.abs(finger_joint_pos-upper)<0.001):
    print("Grasp nothing")
    num_grasp_null += 1
    # input('press ENTER to continue')
    p.restoreState(tmp_id)
    p.removeState(tmp_id)
    return 0

  p.removeState(tmp_id)

  return 1


def place_action(body_id,symmetry_tfs,place_id,env,gripper_vis_id,ob_concave_urdf_dir,class_name):
  global num_task_grasp_succ, num_stable_grasp, num_attempt_grasp, num_fail_move_to_place_top
  attachment = PU.create_attachment(env.robot_id,env.gripper_id,body_id)
  p.resetBaseVelocity(env.robot_id,linearVelocity=[0,0,0],angularVelocity=[0,0,0])
  p.resetBaseVelocity(body_id,linearVelocity=[0,0,0],angularVelocity=[0,0,0])
  p.changeVisualShape(body_id,-1,rgbaColor=[1,0,0,1])

  place_success_func = get_place_success_func(class_name)

  tmp_id = p.saveState()
  for symmetry_tf in symmetry_tfs:
    commands = []
    p.restoreState(tmp_id)

    ob_in_world = get_ob_pose_in_world(body_id)
    ob_in_world = ob_in_world@symmetry_tf
    set_body_pose_in_world(body_id,ob_in_world)

    ############ Move to top of place
    ob_in_gripper = get_pose_A_in_B(body_id,-1,env.robot_id,env.gripper_id)
    place_in_world = get_ob_pose_in_world(place_id)
    target_ob_in_world = place_in_world@place_pose_dict[class_name][0]
    target_gripper_in_world = target_ob_in_world@np.linalg.inv(ob_in_gripper)
    obstacles = PU.get_bodies()
    obstacles.remove(env.robot_id)
    obstacles.remove(body_id)
    set_body_pose_in_world(gripper_vis_id,target_gripper_in_world)
    command = env.move_arm(link_id=env.gripper_id,link_pose=target_gripper_in_world,obstacles=[],attachments=[attachment],
                          timeout=5,use_ikfast=True)
    if command is None:
      continue
    commands.append(command)

    ############ Move down for place
    PU.set_joint_positions(env.robot_id,env.arm_ids,command.body_paths[-1].path[-1])
    attachment.assign()
    gripper_in_wrold = get_link_pose_in_world(env.robot_id,env.gripper_id)
    set_body_pose_in_world(gripper_vis_id,gripper_in_wrold)

    place_in_world = get_ob_pose_in_world(place_id)
    target_ob_in_world = place_in_world@place_pose_dict[class_name][1]
    target_gripper_in_world = target_ob_in_world@np.linalg.inv(ob_in_gripper)

    set_body_pose_in_world(gripper_vis_id,target_gripper_in_world)
    command = env.move_arm_catesian(env.gripper_id,end_pose=target_gripper_in_world,timeout=0.1,attachments=[attachment],obstacles=[])
    if command is None:
      print("move_arm_catesian failed")
      continue

    commands.append(command)
    break

  # 没有规划出运动轨迹 ret 0 over
  if len(commands)<2:
    print("Move to top of place fail")
    p.restoreState(tmp_id)
    p.removeState(tmp_id)
    num_fail_move_to_place_top += 1
    return 0

  p.restoreState(tmp_id)
  p.removeState(tmp_id)
  
  # 运动轨迹估计错误，发生碰撞
  for i_com,command in enumerate(commands):
    obstacles = []
    if i_com==len(commands)-1:
      obstacles = [place_id]

    ret = command.execute(time_step=EXECUTE_TIME_STEP,obstacles=obstacles)
    if not ret:
      print('during place, touches')
      # input('press ENTER to continue')
      break

  env.env_grasp.open_gripper()
  # input('press ENTER to continue')

  PU.set_joint_positions(env.robot_id,env.arm_ids,np.zeros((len(env.arm_ids))))

  p.setGravity(0,0,-10)
  for _ in range(50):
    p.stepSimulation()

  ob_pose = get_ob_pose_in_world(body_id)
  place_pose = get_ob_pose_in_world(place_id)
  
  # 零件最终落位是否正确？正确 ret 1 over；错误 ret -1 over
  if place_success_func(ob_pose,place_pose):
    num_task_grasp_succ += 1
    num_stable_grasp += 1
    return 1

  num_stable_grasp += 1
  return -1


def is_done():
  '''
    零件是否抓完
  '''
  bodies = PU.get_bodies()
  has = False
  for ob_id in env.ob_ids:
    if ob_id in bodies:
      has = True
      break
  is_done = has==False
  if is_done:
    print("\n\nScene done")
  return is_done


def simulate_grasp_with_arm(ob_name):
  global num_objects, ob_file
  code_dir = os.path.dirname(os.path.realpath(__file__))
  ob_file = f"{code_dir}/data/object_models/{ob_name}.obj"
  set_body_pose_in_world(env.camera.cam_id,env.cam_in_world)
  env.add_bin()
  place_dir = ob_file.replace('.obj','_place.obj')
  place_id,_ = create_object(place_dir,scale=np.array([1,1,1]),ob_in_world=np.eye(4),mass=0.1,useFixedBase=True,
                              concave=True)
  p.changeDynamics(place_id,-1,lateralFriction=0.1,spinningFriction=0.1,collisionMargin=0.0001)
  p.changeVisualShape(place_id,-1,rgbaColor=[1,1,1,1])
  tmp = get_ob_pose_in_world(env.bin_id)
  tmp[1,3] = -tmp[1,3]
  if class_name in ['hnm','screw']:
    tmp[:3,:3] = euler_matrix(np.pi,0,0,axes='sxyz')[:3,:3]
  set_body_pose_in_world(place_id,tmp) # 放置接收器（下游任务目标）
  env.env_body_ids = PU.get_bodies()

  def remove_condition(ob_id):
    if class_name in ['hnm','screw']:   # For hnm, check object is not facing up and exists valid semantic grasp
      pose = get_ob_pose_in_world(ob_id)
      if np.dot(pose[:3,2],np.array([0,0,1]))>0:
          return True
    return False

  num_objects = np.random.randint(4,7) # 随机场景中的零件数，4~6个
  env.make_pile(obj_file=ob_file,scale_range=[1,1],n_ob_range=[num_objects,num_objects+1],
                remove_condition=remove_condition)

  #########!NOTE Replace with concave model. Dont do this when making pile, it's too slow
  ob_concave_urdf_dir = f'/tmp/{os.path.basename(ob_file)}_{uuid4()}.urdf'
  create_urdf_for_mesh(ob_file,out_dir=ob_concave_urdf_dir,concave=True,scale=np.ones((3)))

  body_ids = PU.get_bodies()
  for body_id in body_ids:
    if body_id in env.env_body_ids:
      continue
    ob_in_world = get_ob_pose_in_world(body_id)
    p.removeBody(body_id)
    ob_id = p.loadURDF(ob_concave_urdf_dir, [0, 0, 0], useFixedBase=False, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
    set_body_pose_in_world(ob_id,ob_in_world)
    p.changeDynamics(ob_id,-1,linearDamping=0.9,angularDamping=0.9,lateralFriction=0.9,spinningFriction=0.9,collisionMargin=0.0001,activationState=p.ACTIVATION_STATE_ENABLE_SLEEPING)

  PU.set_joint_positions(env.robot_id,env.arm_ids,np.zeros((7)))

  symmetry_tfs = get_symmetry_tfs(class_name,allow_reflection=False)
  print('symmetry_tfs:',len(symmetry_tfs))

  gripper_vis_id = create_gripper_visual_shape(env.env_grasp.gripper)
  tmp = np.eye(4)
  tmp[:3,3] = [0,0,2]
  set_body_pose_in_world(gripper_vis_id,tmp)

  init_gripper_in_world = get_link_pose_in_world(env.robot_id,env.gripper_id)
  p.setGravity(0,0,-10)

  grasp_seq = []
    


  body_ids = PU.get_bodies()
  for body_id in body_ids:
    p.changeDynamics(body_id,-1,collisionMargin=0.0001)

  grasp_in_gripper = ags.gripper.get_grasp_pose_in_gripper_base() #todo 抓取位姿（采样+codebook），参考系是机械手？？？
  ee_in_gripper = get_pose_A_in_B(env.robot_id,env.ee_id,env.robot_id,env.gripper_id)
  ee_in_grasp = np.linalg.inv(grasp_in_gripper)@ee_in_gripper

  def ik_func(grasp_in_cam,obstacles):
    ee_in_world = env.cam_in_world@grasp_in_cam@ee_in_grasp
    sols = env.ik_fast_feasible_solutions(ee_in_world,obstacles=obstacles)  #!NOTE Will check collision later
    if len(sols)>0:
      return True
    return False

  body_ids = PU.get_bodies()
  for body_id in body_ids:
    if body_id in env.env_body_ids:
      continue
    p.changeDynamics(body_id,-1,activationState=p.ACTIVATION_STATE_ENABLE_SLEEPING)

  i_pick = 0
  while 1:
    PU.set_joint_positions(env.robot_id,env.arm_ids,np.zeros((len(env.arm_ids))))
    # i_pick = len(grasp_seq)
    i_pick += 1
    tmp = np.eye(4)
    tmp[:3,3] = [9,0,0]
    set_body_pose_in_world(gripper_vis_id,tmp)
    env.env_grasp.open_gripper()
    env.simulation_until_stable()
    p.saveBullet('{}/{}_state.bullet'.format(debug_dir,i_pick))
    rgb,depth,seg = env.camera.render(env.cam_in_world)
    Image.fromarray(rgb).save(f'{debug_dir}/{i_pick}_rgb.png') # 观察值：rgb 图像
    cv2.imwrite(f'{debug_dir}/{i_pick}_gt_seg.png',seg) # 观察值：实例分割图像

    skipped_grasps = []
    visited_seg_ids = set()
    place_ret = -1
    pick_ret = -1
    # 零件大循环
    for grasps in compute_candidate_grasp(rgb,depth,seg,i_pick=i_pick,env=env,ags=ags,symmetry_tfs=symmetry_tfs,ik_func=ik_func):
      if grasps is None or len(grasps)==0:
        continue

      state_id = p.saveState()
      body_ids = PU.get_bodies()
      # 抓取大循环（单个零件）
      for i_grasp in range(len(grasps)):
        p.restoreState(state_id)
        grasp = grasps[i_grasp]
        body_id = grasp.body_id # 该抓取位姿对应的零件
        visited_seg_ids.add(body_id)
        if body_id not in body_ids: # 跳过非零件抓取位姿
          continue
        p.changeVisualShape(body_id,-1,rgbaColor=[0,0,1,1]) # 可视化：重点标识（颜色）正在抓取的零件

        region 按照配置文件的阈值，过滤抓取
        if grasp.p_T_G<p_T_G_thres:
          skipped_grasps.append(grasp)
          break

        if class_name in ['hnm','screw']:  #####!NOTE grasp cannot be from bottom of ob
          grasp_pose = grasp.grasp_pose
          grasp_in_ob = np.linalg.inv(nocs_predicter.nocs_pose)@grasp_pose
          grasp_in_ob = normalizeRotation(grasp_in_ob)
          approach_dir = grasp_in_ob[:3,0]
          dot = np.dot(approach_dir,np.array([0,0,1]))
          if dot<0:
            continue

        if grasp.p_T_given_G<p_T_given_G_thres:
          skipped_grasps.append(grasp)
          continue

        if grasp.p_G<p_G_thres:
          skipped_grasps.append(grasp)
          continue
        #endregion

        PU.set_joint_positions(env.robot_id,env.arm_ids,np.zeros((len(env.arm_ids))))
        pick_ret = pick_action(grasp,env,gripper_vis_id,i_pick,debug_dir=debug_dir)
        if pick_ret<=0: 
          print(f"grasp {i_grasp} pick_action failed")
          continue

        place_in_world = get_ob_pose_in_world(place_id)
        place_ret = place_action(body_id,symmetry_tfs,place_id,env,gripper_vis_id,ob_concave_urdf_dir,class_name)
        if place_ret==0: # 没有规划出运动轨迹
          continue 

        p.removeBody(body_id) # 移除零件，该零件有两种情况：落位成功或失败
        if is_done():
          return

        if place_ret>0: # 零件落位成功
          break

        print(f'body {body_id} place failed, ret=-1') # 零件落位失败

        grasp_seq.append(grasp) # 稳定抓取，但非任务相关
        print('i_pick:',i_pick)

        break

      p.removeState(state_id)

      if place_ret>0:
        break

      print(f'body {body_id} all failed')

      if len(visited_seg_ids)>3:
        break

    if place_ret>0: # 继续抓取剩余的零件
      continue

    print("No suitable pick. Checking skipped grasps for anything runnable...")
    skipped_grasps = sorted(list(skipped_grasps), key=lambda x:-x.p_T_G) # 按照联合概率密度排序
    # skipped_grasps = sorted(list(skipped_grasps), key=lambda x:-x.p_G)
    with gzip.open(f'{debug_dir}/skipped_grasps.pkl','wb') as ff:
      pickle.dump(skipped_grasps,ff)
    for i_grasp,grasp in enumerate(skipped_grasps):
      pick_ret = pick_action(grasp,env,gripper_vis_id,i_pick,debug_dir=debug_dir)
      if pick_ret<=0:
        print(f"skipped grasp {i_grasp} pick_action failed")
        continue
      else:
        place_ret = place_action(grasp.body_id,symmetry_tfs,place_id,env,gripper_vis_id,ob_concave_urdf_dir,class_name)
        if place_ret>0:
          break

    if place_ret<=0:
      if is_done():
        return
      print("\nFAIL\n")
      # input('press ENTER to continue')
      p.removeBody(body_id)
      if is_done():
        return


if __name__=="__main__":
  seed = 0
  np.random.seed(seed)
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

  debug_dir = '/tmp/catgrasp'
  os.system(f'mkdir -p {debug_dir}')

  code_dir = os.path.dirname(os.path.realpath(__file__))
  with open('{}/config.yml'.format(code_dir),'r') as ff:
    cfg = yaml.safe_load(ff)
  with open(f'{code_dir}/config_run.yml','r') as ff:
    cfg_run = yaml.safe_load(ff)
    class_name = cfg_run['class_name']
    p_G_thres = cfg_run['p_G_thres']
    p_T_given_G_thres = cfg_run['p_T_given_G_thres']
    p_T_G_thres = cfg_run['p_T_G_thres']
    object_names = str(cfg_run['ob_name']).split(',')
    model_path = cfg_run['model_path']
    cal_trans = str(cfg_run['cal_transform']).split(',')
    ite_per_obj = int(cfg_run['ite_per_obj'])
  cfg_grasp = YamlConfig("{}/config_grasp.yml".format(code_dir))
  gripper = RobotGripper.load(gripper_dir=cfg_grasp['gripper_dir'][class_name])
  
  env = Env(cfg,gripper,gui=False)
  
  grasp_predicter = GraspPredicter(class_name)
  if model_path != 'None':
    nocs_predicter = NunocsPredicter(class_name, art_dir=f'{code_dir}/logs/{class_name}_nunocs/{model_path}', env=env)
  else:
    nocs_predicter = NunocsPredicter(class_name, env=env)
  seg_predicter = PointGroupPredictor(class_name)

  code_dir = os.path.dirname(os.path.realpath(__file__))
  with gzip.open(f'{code_dir}/data/object_models/{class_name}_canonical.pkl','rb') as ff:
    canonical = pickle.load(ff)

  # samplers = [
  #   PointConeGraspSampler(gripper,cfg_grasp),
  #   NocsTransferGraspSampler(gripper,cfg_grasp, canonical, class_name=class_name, 
  #                             score_larger_than=cfg_run['nocs_grasp_sampler_score_larger_than'],
  #                             max_n_grasp=cfg_run['nocs_grasp_sampler_max_n_grasp']),
  #   ]
  samplers = [
    PointConeGraspSampler(gripper,cfg_grasp)
    ]

  ags = CombinedGraspSampler(gripper,cfg_grasp,samplers=samplers)
  
  class_name_to_artifact_id = {
      'nut': 78,
      'hnm': 73,
      'screw': 76
    }
  artifact_id = class_name_to_artifact_id[class_name]
  
  
  
  if model_path != 'None':
    save_dir = f'{code_dir}/logs/{class_name}_nunocs/{model_path}/grasp.out'
  else:
    save_dir = f'{code_dir}/artifacts/artifacts-{artifact_id}/log.out'
    # save_dir = f'{code_dir}/artifacts/artifacts-{artifact_id}/log_test_canonical.out'
  
  for obj_name in object_names:
    for cal_trasform in cal_trans:
      # with gzip.open(f'{code_dir}/data/object_models/{obj_name}_canonical.pkl','rb') as ff:
      #   canonical_temp = pickle.load(ff)
      # canonical['canonical_cloud'] = canonical_temp['canonical_cloud']
      # canonical['canonical_normals'] = canonical_temp['canonical_normals']
      # canonical['canonical_affordance'] = canonical_temp['canonical_affordance']
      
      if cal_trasform == 'True':
        cal_trasform = True
        log("network predict:", save_dir)
      else:
        cal_trasform = False
        log("computed:", save_dir)

      task_total = 0
      attem_total = 0
      stable_total = 0
      fail_total = 0
      
      for i in range(ite_per_obj):
        num_task_grasp_succ = 0
        num_stable_grasp = 0
        num_attempt_grasp = 0
        num_grasp_null = 0
        num_fail_move_to_place_top = 0
        simulate_grasp_with_arm(obj_name)
        env.reset()
        num_attempt_grasp -= num_fail_move_to_place_top
        task_perc = '{:d}' .format(int(num_task_grasp_succ / num_attempt_grasp * 100))
        stable_perc = '{:d}' .format(int(num_stable_grasp / num_attempt_grasp * 100))
        fail_perc = '{:d}' .format(int(num_grasp_null / num_attempt_grasp * 100))
        nocs_inlier_perc = '{:d}' .format(int(nocs_predicter.best_ratio * 100))
        attem_total += num_attempt_grasp
        task_total += num_task_grasp_succ
        stable_total += num_stable_grasp
        fail_total += num_grasp_null
        log_str = '{}, num_objects={}, num_task_grasp_succ={} {}%, num_stable_grasp={} {}%, num_grasp_fail={} {}%, num_fail_move_arm={}, num_attempt_grasp={}, nucs_inlier_ratio={}%'\
          .format(obj_name, num_objects, num_task_grasp_succ, task_perc, num_stable_grasp, stable_perc, num_grasp_null, fail_perc, num_fail_move_to_place_top, num_attempt_grasp, nocs_inlier_perc)
        log(log_str, save_dir)

      task_total_perc = '{:d}' .format(int(task_total / attem_total * 100))
      stable_total_perc = '{:d}' .format(int(stable_total / attem_total * 100))
      fail_total_perc = '{:d}' .format(int(fail_total / attem_total * 100))
      log_str = 'statistic: task={}% stable={}% fail={}%\n' .format(task_total_perc, stable_total_perc, fail_total_perc)
      log(log_str, save_dir)

    log('\n', save_dir)