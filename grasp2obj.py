import gzip, pickle
import os, sys
from dexnet.grasping.gripper import *
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)

# 可视化失败场景的夹具分布
i_round = 2
i_pick = 2
file = f'/tmp/catgrasp/screw_carr_94323A329_NYLON_{i_round}_{i_pick}_sample_grasps_origin.pkl'

with gzip.open(file,'rb') as ff:
    grasps = pickle.load(ff)

gripper = RobotGripper.load('urdf/robotiq_hande')

# p_T_given_G, p_T_G, p_G
#region 排序抓取，选择五个，并输出对应的三个分数

vis_dir = '/tmp/catgrasp_vis_grasp'
if not os.path.exists(vis_dir):
    os.system(f'mkdir -p {vis_dir}')
    # os.system(f'rm -rf {vis_dir}')


num_vis_grasp = len(grasps)
num_vis_grasp = 5

#region 可视化根据 p（G
def candidate_compare_key(grasp):
    return -grasp.p_G
grasps = sorted(grasps, key=candidate_compare_key)
print('sorted by p(G):\n')
for i in range(num_vis_grasp):
    grasp = grasps[i]
    grasp_in_cam = grasp.grasp_pose.copy()
    save_grasp_pose_mesh(gripper,grasp_in_cam,out_dir=f'{vis_dir}/{i_round}_{i_pick}_G_{i+1}.obj')
    save_grasp_pose_mesh(gripper,grasp_in_cam,out_dir=f'{vis_dir}/{i_round}_{i_pick}_G_enclosed_{i+1}.obj', enclosed=True)
    print(f'P(G)={grasp.p_G}, P(T|G)={grasp.p_T_given_G}, P(T,G)={grasp.p_T_G}')
#endregion

#region 可视化根据条件概率
# def candidate_compare_key(grasp):
#     return -grasp.p_T_given_G
# grasps = sorted(grasps, key=candidate_compare_key)
# print('sorted by p(T|G):\n')
# for i in range(num_vis_grasp):
#     grasp = grasps[i]
#     grasp_in_cam = grasp.grasp_pose.copy()
#     save_grasp_pose_mesh(gripper,grasp_in_cam,out_dir=f'{vis_dir}/{i_round}_{i_pick}_T_Given_G_{i+1}.obj')
    # save_grasp_pose_mesh(gripper,grasp_in_cam,out_dir=f'{vis_dir}/{i_round}_{i_pick}_T_Given_G_enclosed_{i+1}.obj', enclosed=True)
    # print(f'P(G)={grasp.p_G}, P(T|G)={grasp.p_T_given_G}, P(T,G)={grasp.p_T_G}')
#endregion

#region 根据联合概率，输出前五分数
# def candidate_compare_key(grasp):
#     return -grasp.p_T_G
# grasps = sorted(grasps, key=candidate_compare_key)
# print('sorted by p(T,G):\n')
# for i in range(num_vis_grasp):
#     grasp = grasps[i]
#     grasp_in_cam = grasp.grasp_pose.copy()
#     path = f'{vis_dir}/{i_round}_{i_pick}_T_G_{i+1}.obj'
#     save_grasp_pose_mesh(gripper,grasp_in_cam,out_dir=f'{vis_dir}/{i_round}_{i_pick}_T_G_{i+1}.obj')
#     print(f'P(G)={grasp.p_G}, P(T|G)={grasp.p_T_given_G}, P(T,G)={grasp.p_T_G}')
#endregion

#endregion
print('end')