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

# def save_push_pose_mesh(push_in_cam, out_dir):
#     gripper_dir = f"{code_dir}/urdf/robotiq_hande"
#     mesh_filename = os.path.join(gripper_dir, 'gripper_finger_closed.obj')
#     mesh = trimesh.load(mesh_filename)
#     push_in_gripper = gripper.get_grasp_pose_in_gripper_base()
#     gripper_in_push = np.linalg.inv(push_in_gripper)
#     gripper_in_cam = push_in_cam @ gripper_in_push
#     mesh.apply_transform(gripper_in_cam)
#     mesh.export(out_dir)

#region 算法 & 数据结构定义
class Push3D:
    '''
        初始化的时候需要传入的量：
            起始位姿 pose_start
            位姿探索基于的采样点 sample_pt
            网格搜索参数：
                绕小主轴的旋转 angle
                沿大主轴的偏移 major_off
            物体点集 obj_pts
    '''
    def __init__(self, pose_origin=None, pose_start=None, sample_pt=None, angle=None, major_off=None):
        self.pose_origin_ = pose_origin.copy()
        self.pose_start_ = pose_start.copy()
        self.pose_end_ = None
        self.push_vec_ = None
        self.push_vector_expec_ = None
        self.score_ = None
        self.gripper_in_push = np.linalg.inv(gripper.get_grasp_pose_in_gripper_base())
        self.gripper_mesh_path = f'{code_dir}/urdf/robotiq_hande/gripper_finger_closed.obj'
        self.pcd_obj_moved = None
        self.sample_pt_ = sample_pt
        self.angle_ = angle
        self.major_offset_ = major_off

    @property
    def pose_origin(self):
        return self.pose_origin_

    @property
    def pose_start(self):
        return self.pose_start_
    
    @property
    def pose_end(self):
        return self.pose_end_
        
    @property
    def push_vec_origin(self):
        return self.push_vec_

    @property
    def push_vec(self):
        return self.push_vector_expec_
    
    @property
    def score(self):
        return self.score_
    
    @property
    def sample_pt(self):
        return self.sample_pt_
    
    @property
    def angle(self):
        return self.angle_
    
    @property
    def major_off(self):
        return self.major_offset_
    
    @score.setter
    def score(self, s):
        self.score_ = s

    @push_vec.setter
    def push_vec(self, push_vec):
        self.push_vec_ = push_vec
        if push_vec[2] != 0:
            push_vec[2] = 0
        self.push_vector_expec_ = push_vec
        pose_end = self.pose_start_.copy()
        pose_end[:3,3] += self.push_vec_
        self.pose_end_ = pose_end

    def log_push(self, rank, obj_pts, end=False):
        if end:
            push_in_cam = self.pose_start_
        else:
            push_in_cam = self.pose_end_
        gripper_in_cam = push_in_cam @ self.gripper_in_push
        gripper_mesh = trimesh.load(self.gripper_mesh_path)
        gripper_mesh.apply_transform(gripper_in_cam)
        gripper_mesh.export(f'/tmp/catgrasp/push_top/pose/{rank:0>2d}th_push.obj')
        gripper_origin_in_cam = self.pose_origin_ @ self.gripper_in_push
        gripper_mesh_origin = trimesh.load(self.gripper_mesh_path)
        gripper_mesh_origin.apply_transform(gripper_origin_in_cam)
        gripper_mesh.export(f'/tmp/catgrasp/push_top/pose/{rank:0>2d}th_push_origin.obj')
        moved_pts = obj_pts + self.push_vector_expec_
        pcd_obj_moved = toOpen3dCloud(moved_pts)
        o3d.io.write_point_cloud(f'/tmp/catgrasp/push_top/obj/{rank:0>2d}th_obj_after_push.ply', pcd_obj_moved)
        log(f'{rank} th:\norigin: {self.sample_pt_}, angle: {self.angle_}, major_off: {self.major_offset_}, score: {self.score_}', 
            f'/tmp/catgrasp/push_top/pose/log.out')

#region push 检测
def sample_push(pts, norms):
    # sample_ids = farthest_point_sample(pts_origin, 1000)
    # pts = pts_origin[sample_ids]
    # norms = norms_origin[sample_ids]

    r_ball = compute_cloud_resolution(pts) * 3
    ob_kdtree = cKDTree(pts)
    
    gripper_dir = f'{code_dir}/urdf/robotiq_hande'
    params = json.load(open(os.path.join(gripper_dir, 'params.json'), 'r'))
    
    hands = []
    heads = []

    pt_id = 0
    while pt_id < len(pts):
        pt = pts[pt_id]
        nm = norms[pt_id]

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

        approach_normal = -nm.reshape(3)
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

        T_ch = np.eye(4) # 夹具位姿
        T_ch[:3,:3] = R
        T_ch[:3,3] = pt
        
        # save_push_pose_mesh(gripper_trimesh, T_ch, out_dir=f'/tmp/catgrasp/push_head_sample/{pt_id:0>3d}th_origin.obj')

        # T_hc = np.linalg.inv(T_ch)
        # pts_in_hand = (T_hc @ to_homo(pts).T).T[:,:3]
        # obj_pcd = o3d.geometry.PointCloud()
        # world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
        # obj_pcd.points = o3d.utility.Vector3dVector(pts_in_hand) # (N, 3)
        # o3d.visualization.draw_geometries([world_frame, obj_pcd])

        #region grid search
        hands_local = []
        heads_local = []
        # rot by minor axis
        for minor_rot in np.arange(-90, 120, 30):
            R_inplane = euler_matrix(0,0,minor_rot*np.pi/180,axes='sxyz')[:3,:3]
            minor_rot_transformtion = np.eye(4)
            minor_rot_transformtion[:3,:3] = R_inplane
            T_ch_roted = T_ch @ minor_rot_transformtion

            # save_push_pose_mesh(gripper_trimesh, T_ch_roted, out_dir=f'/tmp/catgrasp/push_head_sample/{pt_id:0>3d}th_{minor_rot:0>2d}angle.obj')
            
            T_hc_roted = np.linalg.inv(T_ch_roted)
            pts_in_hand = (T_hc_roted @ to_homo(pts).T).T[:,:3]

            
            # 截取指侧内点云
            keep_height = []
            for i in np.arange(len(pts_in_hand)):
                point = pts_in_hand[i]
                z_val = point[2]
                if z_val > -params['hand_height']/2 and z_val < params['hand_height']/2:
                    keep_height.append(i)
            pts_croped_h = pts_in_hand[keep_height]

            # translation along major axis
            for major_off in np.linspace(-0.05, 0.05, 10):
                is_coll = False
                lfinger_y = -params['finger_width'] + major_off
                rfinger_y = params['finger_width'] + major_off
                
                pts_croped = [] # 既在指背内又在指侧内的点云
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
                for depth_off in np.arange(0, 55, 5) * 0.001:
                    for p in pts_croped:
                        x_val = p[0]
                        if x_val < depth_off:
                            is_coll = True
                            break
                    if is_coll == True:
                        break
                    last_depth = depth_off
                major_transl = np.eye(4)
                major_transl[:3,3] = [last_depth, major_off, 0]
                T_transl_major_ch = T_ch_roted @ major_transl
                hand = Push3D(pose_origin=T_ch, pose_start=T_transl_major_ch, sample_pt=pt, 
                              angle=minor_rot, major_off=major_off)
                hands_local.append(hand)
                heads_local.append(T_transl_major_ch)
        #endregion
        
        hands += hands_local
        heads += heads_local
        pt_id += 1
    hands = np.array(hands)
    heads = np.array(heads)
    return hands, heads

def EvalPush(hand_candidates, ob_pts, background_pts):
    # 过滤越界动作
    bin_no_ground = o3d.io.read_point_cloud(bin_no_ground_dir)
    aabb_bin = bin_no_ground.get_axis_aligned_bounding_box() # 计算 bin 边界
    max_boun = aabb_bin.get_max_bound() 
    min_boun = aabb_bin.get_min_bound()
    hands_in_bin = []
    hands_translation = [] # 过滤碰撞动作用
    for hand_candi in hand_candidates:
        ob_moved = ob_pts + hand_candi.push_vec
        if (ob_moved[:,0] < min_boun[0]).any() or (ob_moved[:,0] > max_boun[0]).any() \
            or (ob_moved[:,1] < min_boun[1]).any() or (ob_moved[:,1] > max_boun[1]).any():
            continue
        hands_in_bin.append(hand_candi)
        hands_translation.append(hand_candi.push_vec)
    hands_in_bin = np.array(hands_in_bin)
    hands_translation = np.array(hands_translation)

    # 过滤目标移动碰撞 TODO：check
    colli_keep_ids = my_cpp.detectCollisionMove(hands_translation, ob_pts, 
                                                background_pts, 0.001)
    hands_colli_free = hands_in_bin[colli_keep_ids]

    # 稀疏化物体点云和场景点云
    sample_ids = farthest_point_sample(ob_pts, 1000)
    ob_pts_down = ob_pts[sample_ids]
    pcd_scene = toOpen3dCloud(background_pts)
    pcd_scene_sparse = pcd_scene.voxel_down_sample(voxel_size=0.01)
    pcd_scene_sparse_pts = my_cpp.makeOccupancyGridFromCloudScan(np.asarray(pcd_scene_sparse.points), K, 0.01)

    # 计算零件和环境的距离
    dist_origin = np.sum(chamfer_distance(ob_pts_down, pcd_scene_sparse_pts))

    # 打分
    hands_filted = []
    tic = time.time()
    for hand in hands_colli_free:
        ob_moved = ob_pts_down + hand.push_vec
        dist = np.sum(chamfer_distance(ob_moved, pcd_scene_sparse_pts))
        if dist < dist_origin:
            continue
        hand.score = dist
        hands_filted.append(hand)
    toc = time.time()
    
    log(f'cost of eval push({time.asctime(time.localtime(time.time()))}): {toc - tic} s.', f'{code_dir}/logs/push/log.out')

    hands = sorted(hands_filted, key=lambda h: h.score, reverse=True)
    # dtype = [('p_head', np.ndarray), ('move_vec', np.ndarray), ('score', float)]
    # push_filted = np.array(push_filted, dtype=dtype)
    # return sorted(push_filted, key=lambda x:x['score'])[::-1]
    return hands
#endregion

#region 推移定义
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

def push(hand):
    over0 = hand.pose_start.copy()
    over1 = hand.pose_end.copy()
    over0[2, 3] = 0.61
    over1[2, 3] = 0.61
    # save_push_pose_mesh(push_start, out_dir=f'/tmp/catgrasp/push_exec/push_start.obj')
    # save_push_pose_mesh(push_end, out_dir=f'/tmp/catgrasp/push_exec/push_end.obj')

    # Execute push.
    # tic = time.time()
    num_arms = len(env.arm_ids)
    PU.set_joint_positions(env.robot_id, env.arm_ids, np.zeros(num_arms))
    move(over0)
    env.env_grasp.close_gripper()
    move(hand.pose_start)

    dist = np.linalg.norm(hand.push_vec_origin)
    push_dir = hand.push_vec_origin / dist # 移动方向
    steps = np.int32(np.floor(dist / 0.001))
    target = hand.pose_start.copy()

    for _ in range(steps): # 单位毫米为移动单位，完成整个移动过程
        target[:3,3] += push_dir * 0.001
        move(target)
    
    move(hand.pose_end)
    move(over1)
    # toc = time.time()
    env.env_grasp.open_gripper()
    # log(f'cost of exec push: {toc-tic} s.', f'{code_dir}/logs/push/log.out')
#endregion
#endregion


#region 启动仿真环境
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

#region 环境信息
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
scene_pcd = o3d.io.read_point_cloud(scene_dir) # 场景点云
scene_pts = scene_pcd.points
ob_pcd = o3d.io.read_point_cloud(ob_dir) # 目标点云
ob_pts = np.array(ob_pcd.points)
ob_norms = np.array(ob_pcd.normals)

# obstacle for collision detect
with open('{}/config.yml'.format(code_dir),'r') as ff:
    cfg = yaml.safe_load(ff)
K = np.array(cfg['K']).reshape(3,3)
ob_ktree = cKDTree(ob_pts)
dists,indices = ob_ktree.query(scene_pts)
gripper_diameter = np.linalg.norm(gripper.trimesh.vertices.max(axis=0)-gripper.trimesh.vertices.min(axis=0))
keep_ids = np.where(dists<=gripper_diameter/2)[0]
scene_pts = np.array(scene_pts)[keep_ids]
pcd_scene = toOpen3dCloud(scene_pts)
pcd_scene_down = pcd_scene.voxel_down_sample(voxel_size=0.001)
pcd_scene_down_pts = my_cpp.makeOccupancyGridFromCloudScan(np.asarray(pcd_scene_down.points), K, 0.001)
#endregion

#region 启动检测过程
gripper_dir = f"{code_dir}/urdf/robotiq_hande"
mesh_filename = os.path.join(gripper_dir, 'gripper_finger_closed.obj')
gripper_trimesh = trimesh.load(mesh_filename)

# 采样推移位姿
push_hands, push_heads = sample_push(ob_pts, ob_norms)

# 碰撞检测（闭合夹具和环境之间）
gripper_in_grasp = np.linalg.inv(gripper.get_grasp_pose_in_gripper_base())
keep_colli_free_ids = my_cpp.filterPushPose(push_heads, gripper_in_grasp, 
                                        gripper_trimesh.vertices, gripper_trimesh.faces, 
                                        pcd_scene_down_pts, 0.0005)
push_hands = push_hands[keep_colli_free_ids]
push_heads = push_heads[keep_colli_free_ids]

# 设置推移向量
ob_center = ob_pcd.get_center() # 零件形心
push_dis = 0.03 # 推移距离
hand_candidates = []
for hand in push_hands:
    appro_dir = hand.pose_start[:3,0]
    appro_dir /= np.linalg.norm(appro_dir)
    cos_val = appro_dir.dot([0, 0, 1])
    if cos_val > np.cos(30): # 垂直抵近物体上表面
        for angle_horiz in np.linspace(0, 360, 8): # 沿 xy 面，采样 8 个移动方向
            rot_z = euler_matrix(0,0,angle_horiz*np.pi/180,axes='sxyz')[:3,:3]
            push_vec = rot_z @ [0, 1, 0]
            hand_temp = copy.deepcopy(hand)
            hand_temp.push_vec = push_vec * push_dis
            hand_candidates.append(hand_temp)
    else:
        head_pos = hand.pose_start[:3,3]
        push_vec = ob_center - head_pos
        push_vec = push_vec / np.linalg.norm(push_vec)
        hand_temp = copy.deepcopy(hand)
        hand_temp.push_vec = push_vec * push_dis
        hand_candidates.append(hand_temp)
hand_candidates = np.array(hand_candidates)

# 评估推移动作
scene_excp_target_pts, ids = cloudA_minus_cloudB(scene_pts, ob_pts, thres=0.005)
push_hands = EvalPush(hand_candidates, ob_pts, scene_excp_target_pts)

# debug top 动作
for i in np.arange(3):
    push_hands[i].log_push(i, ob_pts)

push(push_hands[0])
#endregion