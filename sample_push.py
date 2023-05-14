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
import json


code_dir = os.path.dirname(os.path.realpath(__file__))
bin_no_ground_dir = 'bin_no_ground.ply'
# debug_dir = '/tmp/catgrasp_vis'
# obj_name = 'screw_carr_97435A401_WEAR'
debug_dir = '/tmp/catgrasp_vis_A329_NYLON'
obj_name = 'screw_carr_94323A329_NYLON'
run_id = 1
state_id = 1
target_body_id = 6
atp = 0
scene_dir = '{}/{}_run{:0>2d}_state{:0>2d}_scene.ply' .format(debug_dir, obj_name, 1, state_id)
ob_dir = '{}/{}_run{:0>2d}_state{:0>2d}_body{:0>2d}_atp{}_ob.ply' .format(debug_dir, obj_name, run_id, state_id, 
                                                                          target_body_id, atp)

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

# rgb,depth,seg = env.camera.render(env.cam_in_world)
# Image.fromarray(rgb).save(f'{code_dir}/screw_carr_94323A329_NYLON_run{bullet_run_id:0>2d}_state{bullet_state_id:0>2d}_rgb_befor_push.png')
#endregion

#region 推移检测
def sample_push(pts, norms):
    # sample_ids = farthest_point_sample(ob_pts, 1000)
    # pts = ob_pts[sample_ids]
    # norms = ob_norms[sample_ids]

    r_ball = compute_cloud_resolution(pts) * 3
    ob_kdtree = cKDTree(pts)
    
    gripper_dir = f'{code_dir}/urdf/robotiq_hande'
    params = json.load(open(os.path.join(gripper_dir, 'params.json'), 'r'))
    
    hands = []

    pt_id = 0
    pts_homo = to_homo(pts)
    while pt_id < len(pts):
        pt = pts[pt_id]
        norm = norms[pt_id]

        #region local frame
        kd_indices = ob_kdtree.query_ball_point(pt.reshape(1,3),r=r_ball)
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
        
        # 推移系 --> 相机系
        R = np.concatenate((approach_normal.reshape(3,1), major_pc.reshape(3,1), minor_pc.reshape(3,1)), axis=1)
        #endregion

        #region grid search
        hands_local = []

        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(pts)
        posi_t = copy.deepcopy(world_frame).translate(pt)
        nega_t = copy.deepcopy(world_frame).translate(-pt) # error
        o3d.visualization.draw_geometries([world_frame, obj_pcd, posi_t, nega_t])

        
        # rot by minor axis
        for minor_rot in np.arange(-90, 90, 30):
            R_inplane = euler_matrix(0,0,minor_rot*np.pi/180,axes='sxyz')[:3,:3]
            rot_origin = R.transpose() @ R_inplane
            
            # 点云转换到 hand 系
            frame_origin = np.eye(4)
            frame_origin[:3,:3] = rot_origin
            frame_origin[:3,3] = -pt
            cloud_in_hand = frame_origin @ pts_homo.transpose() # (4, N)
            cloud_in_hand = cloud_in_hand.transpose()
            
            # 截取指背内点云
            keep_height = []
            for i in np.arange(len(cloud_in_hand)):
                point = cloud_in_hand[i]
                z_val = point[2]
                if z_val > -params['hand_height']/2 and z_val < params['hand_height']/2:
                    keep_height.append(i)
            pts_croped_h = cloud_in_hand[keep_height]

            # translation along major axis
            for major_off in np.linspace(-0.05, 0.05, 10):
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
                    tip_x = last_depth + depth_off * 0.001
                    for p in pts_croped:
                        x_val = p[0]
                        if x_val < tip_x:
                            is_coll = True
                            break
                    if is_coll == True:
                        break
                    last_depth = tip_x
                off_vec = [-last_depth, -major_off, 0]
                frame_hand = frame_origin.copy()
                frame_hand[:3,3] += off_vec
                hands_local.append(frame_hand)
        
        #endregion
        
        hands += hands_local
        pt_id += 1
    
    return hands


def EvalPush(push_candidates, ob_pts, background_pts):
    # 计算零件和环境的距离
    dist_origin = np.sum(chamfer_distance(ob_pts, background_pts))

    # TODO: 过滤越界动作
    bin_no_ground = o3d.io.read_point_cloud(bin_no_ground_dir)
    aabb_bin = bin_no_ground.get_axis_aligned_bounding_box() # 计算 bin 边界
    max_boun = aabb_bin.get_max_bound() 
    min_boun = aabb_bin.get_min_bound()
    pushs_in_bin = []
    pushs_translation = [] # 过滤碰撞动作用
    for push_cand in push_candidates:
        move_actu = push_cand[1] # 推移向量
        # pratice movement
        if move_actu[2] != 0:
            move_actu[2] = 0
        ob_moved = ob_pts + move_actu
        if (ob_moved[:,0] < min_boun[0]).any() or (ob_moved[:,0] > max_boun[0]).any() \
            or (ob_moved[:,1] < min_boun[1]).any() or (ob_moved[:,1] > max_boun[1]).any():
            continue
        pushs_in_bin.append((push_cand[0],move_actu))
        pushs_translation.append(move_actu)
    pushs_in_bin = np.array(pushs_in_bin)
    pushs_translation = np.array(pushs_translation)

    # 过滤碰撞
    # TODO: oct_resolution
    colli_keep_ids = my_cpp.detectCollisionMove(pushs_translation, ob_pts, 
                                                background_pts, 0.001)
    push_colli_free = pushs_in_bin[colli_keep_ids]
    # translation_keeped = pushs_translation[colli_keep_ids]
  
    # 打分
    # id_dist = my_cpp.distObj2Env(translation_keeped, ob_pts, background_pts)
    # score_keep_ids = id_dist[0,:]
    # push_assed = push_colli_free[score_keep_ids]
    # scores = id_dist[1,:]
    # push_filted = []
    # for i in range(len(push_assed)):
    #     p = push_assed[i]
    #     score_p = scores[i]
    #     push_filted.append((p[0], p[1], score_p))
    # print('score time: {:.0f} minute\n' .format((time.time() - tic) / 60))
    push_filted = []
    for p in push_colli_free:
        ob_moved = ob_pts + p[1]
        dist_p = np.sum(chamfer_distance(ob_moved, background_pts))
        if dist_p < dist_origin:
            continue
        push_filted.append((p[0], p[1], dist_p))
        if len(push_filted) > 10:
            break
    
    dtype = [('p_head', np.ndarray), ('move_vec', np.ndarray), ('score', float)]
    push_filted = np.array(push_filted, dtype=dtype)
    return sorted(push_filted, key=lambda x:x['score'])[::-1]
#endregion

#region 推移操作定义
def move(pose_in_cam):
  pose_in_world = env.cam_in_world @ pose_in_cam
  gripper_in_world = pose_in_world @ np.linalg.inv(env.env_grasp.grasp_pose_in_gripper_base)
  # set_body_pose_in_world(gripper_vis_id, gripper_in_world)
  command = env.move_arm(link_id=env.gripper_id, link_pose=gripper_in_world, obstacles=[], timeout=5, 
                          use_ikfast=True)
  obstacles = [env.bin_id]
  for body_id in env.ob_ids:
    if body_id != target_body_id:
        obstacles.append(body_id)
      
  if command != None:
    return command.execute(time_step=0.05, obstacles=obstacles)
  
  return False


def save_push_pose_mesh(mesh, pose, out_dir):
  grasp_in_gripper = gripper.get_grasp_pose_in_gripper_base()
  gripper_in_cam = pose@np.linalg.inv(grasp_in_gripper)
  mesh.apply_transform(gripper_in_cam)
  mesh.export(out_dir)


def push(push_start, move_vec):
    push_end = push_start.copy()
    push_end[:3,3] += move_vec
    over0 = push_start.copy()
    over1 = push_end.copy()
    over0[2, 3] = 0.61
    over1[2, 3] = 0.61
    save_push_pose_mesh(gripper_trimesh, push_start, out_dir=f'{code_dir}/push_start.obj')
    # save_grasp_pose_mesh(gripper, push_end, out_dir=f'{code_dir}/push_end.obj')
    # save_grasp_pose_mesh(gripper,push_end,out_dir=f'{code_dir}/push_end_enclosed.obj',enclosed=True)

    # Execute push.
    num_arms = len(env.arm_ids)
    PU.set_joint_positions(env.robot_id, env.arm_ids, np.zeros(num_arms))
    move(over0)
    env.env_grasp.close_gripper()
    move(push_start)

    dist = np.linalg.norm(move_vec)
    push_dir = move_vec / dist # 移动方向
    steps = np.int32(np.floor(dist / 0.001))
    target = push_start.copy()

    for _ in range(steps): # 单位毫米为移动单位，完成整个移动过程
        target[:3,3] += push_dir * 0.001
        move(target)
    
    move(push_end)
    move(over1)
    env.env_grasp.open_gripper()
#endregion

#region 位姿采样
scene_pcd = o3d.io.read_point_cloud(scene_dir) # 场景点云
scene_pts = scene_pcd.points
ob_pcd = o3d.io.read_point_cloud(ob_dir) # 目标点云
ob_pts = np.array(ob_pcd.points)
ob_norms = np.array(ob_pcd.normals)
fps_ids = farthest_point_sample(ob_pts, 1000)
ob_pts = ob_pts[fps_ids]
ob_norms = ob_norms[fps_ids]

# obstacle for collision detect
with open('{}/config.yml'.format(code_dir),'r') as ff:
    cfg = yaml.safe_load(ff)
K = np.array(cfg['K']).reshape(3,3)
gripper_in_grasp = np.linalg.inv(gripper.get_grasp_pose_in_gripper_base())
kdtree = cKDTree(ob_pts)
dists,indices = kdtree.query(scene_pts)
gripper_diameter = np.linalg.norm(gripper.trimesh.vertices.max(axis=0)-gripper.trimesh.vertices.min(axis=0))
keep_ids = np.where(dists<=gripper_diameter/2)[0]
background_pts = np.array(scene_pts)[keep_ids]
# background_pts,ids = cloudA_minus_cloudB(background_pts,ob_pts,thres=0.005)
pcd = toOpen3dCloud(background_pts)
# pcd = pcd.voxel_down_sample(voxel_size=0.001)
# background_pts = my_cpp.makeOccupancyGridFromCloudScan(np.asarray(pcd.points), K, 0.001)
pcd = pcd.voxel_down_sample(voxel_size=0.01)
background_pts = my_cpp.makeOccupancyGridFromCloudScan(np.asarray(pcd.points), K, 0.01)

# 推移采样
push_heads = np.array(sample_push(ob_pts, ob_norms))

# 碰撞检测（闭合夹具和环境之间）
gripper_dir = f"{code_dir}/urdf/robotiq_hande"
mesh_filename = os.path.join(gripper_dir, 'gripper_finger_closed.obj')
gripper_trimesh = trimesh.load(mesh_filename)
gripper_in_grasp = np.linalg.inv(gripper.get_grasp_pose_in_gripper_base())
keep_coll_free_ids = my_cpp.filterPushPose(push_heads, gripper_in_grasp, gripper_trimesh.vertices, 
                      gripper_trimesh.faces, background_pts, 0.0005)
push_heads = push_heads[keep_coll_free_ids]

# 推移评估
ob_center = ob_pcd.get_center() # 零件形心
push_dis = 0.03 # 推移距离
push_candidates = []
for head in push_heads: # push frame in camera frame
    appro_dir = head[:3,0]
    appro_dir /= np.linalg.norm(appro_dir)
    cos_val = appro_dir.dot([0, 0, 1])
    if cos_val > np.cos(30): # 垂直抵近物体上表面
        for angle_horiz in np.linspace(0, 360, 8): # 沿 xy 面，采样 8 个移动方向
            rot_z = euler_matrix(0,0,angle_horiz*np.pi/180,axes='sxyz')[:3,:3]
            push_vec = rot_z @ [0, 1, 0]
            push_candidates.append((head, push_vec * push_dis))
    else:
        head_pos = head[:3,3]
        push_vec = ob_center - head_pos
        push_vec = push_vec / np.linalg.norm(push_vec)
        push_candidates.append((head, push_vec * push_dis))
push_candidates = np.array(push_candidates)
pushs_valid = EvalPush(push_candidates, ob_pts, background_pts)
#endregion

push(pushs_valid[0][0], pushs_valid[0][1])