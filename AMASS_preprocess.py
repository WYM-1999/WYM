r"""
    Synthesize AMASS dataset.
    Please refer to the `paths` in `config.py` and set the path of each dataset correctly.
"""

import articulate as art
import torch
import os
from config import Paths, amass_data
import numpy as np
from tqdm import tqdm
import glob

vi_mask = torch.tensor([1961, 5424, 1120, 4606, 335, 3021])
vi = torch.tensor(
    [1920, 1962, 1969, 1960, 5381, 5423, 5421, 5430, 1119, 3321, 1374, 1121, 4607, 6721, 4605, 4848, 3163, 336, 3771,
     259, 3022, 1784, 1782, 5245])
ji_mask = torch.tensor([18, 19, 4, 5, 15, 0])

body_model = art.ParametricModel(Paths.smpl_file, device=torch.device('cuda'))


def _syn_acc(v, smooth_n=2):
    r"""
    Synthesize accelerations from vertex positions.
    """
    mid = smooth_n // 2
    acc = torch.stack([(v[i - 1] + v[i + 1] - 2 * v[i]) * 3600 for i in range(1, v.shape[0] - 1)])
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
    if mid != 0:
        acc[smooth_n:-smooth_n] = torch.stack(
            [(v[i - smooth_n] + v[i + smooth_n] - 2 * v[i]) * 3600 / smooth_n ** 2
             for i in range(smooth_n, v.shape[0] - smooth_n)])
    return acc


# def _syn_acc2(v, smooth_n=2):
#     r"""
#     Synthesize accelerations from 4 vertex positions.
#     """
#     v = v.view(v.shape[0], 4, 6, 3)
#     v1 = v[:, 0] - v[:, 1]
#     v2 = v[:, 2] - v[:, 3]
#     v = torch.cross(v1, v2, dim=-1)
#     mid = smooth_n // 2
#     acc = torch.stack([(v[i - 1] + v[i + 1] - 2 * v[i]) * 3600 for i in range(1, v.shape[0] - 1)])
#     acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
#     if mid != 0:
#         acc[smooth_n:-smooth_n] = torch.stack(
#             [(v[i - smooth_n] + v[i + smooth_n] - 2 * v[i]) * 3600 / smooth_n ** 2
#              for i in range(smooth_n, v.shape[0] - smooth_n)])
#     return acc


# def compute_transform_matrix(r):
#     # N, J, _, _ = r.shape
#     tran_r = torch.matmul(r[1:], torch.inverse(r[:-1]))
#     # identity_matrix = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(1, J, 1, 1)
#     # tran_r = torch.cat((identity_matrix, tran_r), dim=0)
#     return tran_r[:-1]


# def compute_angular_velocity(tr):
#     tr = tr.clone().detach()
#     tr_2 = tr.view(-1, 3, 3)
#     axis_angle = torch.zeros((tr_2.shape[0], 3))
#     for i in range(tr_2.shape[0]):
#         R = tr_2[i]
#         # print(R.shape)
#         #     # Compute rotation angle
#         trace_val = (torch.trace(R) - 1) / 2
#         # trace_val = torch.clamp(trace_val, -1.0, 1.0)  # Clamp value to avoid numerical issues
#         angle = torch.acos(trace_val)
#         #     # Compute rotation axis
#         sin_angle = torch.sin(angle)
#         axis = torch.tensor([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (
#                 2 * sin_angle)
#         axis_angle[i] = axis * angle
#     # # Compute angular velocity
#     angular_velocity = axis_angle * 60
#     angular_velocity = angular_velocity.view(-1, 24, 3)
#     return angular_velocity


# def compute_velocity(j):
#     position_diff = j[1:] - j[:-1]
#     # initial_diff = torch.zeros((1, 24, 3))
#     # position_diff = torch.cat((initial_diff, position_diff), dim=0)
#     # # Compute the velocity by multiplying the difference by the sampling frequency
#     velocity = position_diff * 60
#     return velocity[:-1]


def process_amass():
    data_pose, data_trans, data_beta, length = [], [], [], []
    for ds_name in amass_data:
        print('\rReading', ds_name)
        for npz_fname in tqdm(glob.glob(os.path.join(Paths.raw_amass_dir, ds_name, '*/*_poses.npz'))):
            try:
                cdata = np.load(npz_fname)
            except:
                continue

            framerate = int(cdata['mocap_framerate'])
            if framerate == 120:
                step = 2
            elif framerate == 60 or framerate == 59:
                step = 1
            else:
                continue

            data_pose.extend(cdata['poses'][::step].astype(np.float32))
            data_trans.extend(cdata['trans'][::step].astype(np.float32))
            data_beta.append(cdata['betas'][:10])
            length.append(cdata['poses'][::step].shape[0])

    assert len(data_pose) != 0, 'AMASS dataset not found. Check config.py or comment the function process_amass()'
    length = torch.tensor(length, dtype=torch.int)
    shape = torch.tensor(np.asarray(data_beta, np.float32))
    tran = torch.tensor(np.asarray(data_trans, np.float32))
    pose = torch.tensor(np.asarray(data_pose, np.float32)).view(-1, 52, 3)
    pose[:, 23] = pose[:, 37]  # right hand
    pose = pose[:, :24].clone()  # only use body
    # print(pose.shape)

    # align AMASS global fame with DIP
    amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]])
    tran = amass_rot.matmul(tran.unsqueeze(-1)).view_as(tran)
    pose[:, 0] = art.math.rotation_matrix_to_axis_angle(
        amass_rot.matmul(art.math.axis_angle_to_rotation_matrix(pose[:, 0])))
    print('Synthesizing IMU accelerations and orientations')

    b = 0
    out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc, out_vacc2, = [], [], [], [], [], [], []
    for i, l in tqdm(list(enumerate(length))):
        if l <= 60: b += l; print('\tdiscard one sequence with length', l); continue
        p = art.math.axis_angle_to_rotation_matrix(pose[b:b + l]).view(-1, 24, 3, 3)
        grot, joint, vert = body_model.forward_kinematics(p, shape[i], tran[b:b + l], calc_mesh=True)
        out_pose.append(pose[b:b + l].clone())  # N, 24, 3
        out_tran.append(tran[b:b + l].clone())  # N, 3
        out_shape.append(shape[i].unsqueeze(0).repeat(l, 1).clone())  # N， 10
        out_joint.append(joint[:, :24].contiguous().clone())  # N, 24, 3
        out_vacc.append(_syn_acc(vert[:, vi_mask]))  # N, 6, 3
        # out_vacc2.append(_syn_acc2(vert[:, vi]))  # N, 6, 3
        out_vrot.append(grot)  # N, 24, 3, 3
        b += l

    print('Saving')
    os.makedirs(Paths.amass_dir, exist_ok=True)
    torch.save(out_pose, os.path.join(Paths.amass_dir, 'pose.pt'))
    torch.save(out_shape, os.path.join(Paths.amass_dir, 'shape.pt'))
    torch.save(out_tran, os.path.join(Paths.amass_dir, 'tran.pt'))
    torch.save(out_joint, os.path.join(Paths.amass_dir, 'joint.pt'))
    torch.save(out_vrot, os.path.join(Paths.amass_dir, 'vrot.pt'))
    torch.save(out_vacc, os.path.join(Paths.amass_dir, 'vacc.pt'))
    # torch.save(out_vert, os.path.join(Paths.amass_dir, 'vert.pt'))
    torch.save(out_vacc2, os.path.join(Paths.amass_dir, 'vacc2.pt'))
    # torch.save(out_transform_rot, os.path.join(Paths.amass_dir, 'transform_rot.pt'))
    # torch.save(out_angular_velocity, os.path.join(Paths.amass_dir, 'ang_vel.pt'))
    # torch.save(out_vel, os.path.join(Paths.amass_dir, 'vel.pt'))
    print('Synthetic AMASS dataset is saved at', Paths.amass_dir)


# def process_totalcapture():
#     data_pose, data_trans, data_beta, length = [], [], [], []
#     ds_name = ['TotalCapture']
#     print('\rReading', ds_name)
#     for npz_fname in tqdm(glob.glob(os.path.join(Paths.raw_totalcapture_dir, '*/*_poses.npz'))):
#         try:
#             cdata = np.load(npz_fname)
#         except:
#             continue
#         framerate = int(cdata['mocap_framerate'])
#         if framerate == 120:
#             step = 2
#         elif framerate == 60 or framerate == 59:
#             step = 1
#         else:
#             continue
#
#         data_pose.extend(cdata['poses'][::step].astype(np.float32))
#         data_trans.extend(cdata['trans'][::step].astype(np.float32))
#         data_beta.append(cdata['betas'][:10])
#         length.append(cdata['poses'][::step].shape[0])
#
#     assert len(data_pose) != 0, 'totalcapture dataset not found. Check config.py or comment the function process()'
#     length = torch.tensor(length, dtype=torch.int)
#     shape = torch.tensor(np.asarray(data_beta, np.float32))
#     tran = torch.tensor(np.asarray(data_trans, np.float32))
#     pose = torch.tensor(np.asarray(data_pose, np.float32)).view(-1, 52, 3)
#     pose[:, 23] = pose[:, 37]  # right hand
#     pose = pose[:, :24].clone()  # only use body
#
#     # align AMASS global fame with DIP
#     amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]])
#     tran = amass_rot.matmul(tran.unsqueeze(-1)).view_as(tran)
#     pose[:, 0] = art.math.rotation_matrix_to_axis_angle(
#     amass_rot.matmul(art.math.axis_angle_to_rotation_matrix(pose[:, 0])))
#
#     print('Synthesizing IMU accelerations and orientations')
#     b = 0
#     out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc = [], [], [], [], [], []
#     for i, l in tqdm(list(enumerate(length))):
#         if l <= 12: b += l; print('\tdiscard one sequence with length', l); continue
#         p = art.math.axis_angle_to_rotation_matrix(pose[b:b + l]).view(-1, 24, 3, 3)
#         grot, joint, vert = body_model.forward_kinematics(p, shape[i], tran[b:b + l], calc_mesh=True)
#         out_pose.append(pose[b:b + l].clone())  # N, 24, 3
#         out_tran.append(tran[b:b + l].clone())  # N, 3
#         out_shape.append(shape[i].clone())  # 10
#         out_joint.append(joint[:, :24].contiguous().clone())  # N, 24, 3
#         out_vacc.append(_syn_acc(vert[:, vi_mask]))  # N, 6, 3
#         out_vrot.append(grot[:, ji_mask])  # N, 6, 3, 3
#         b += l
#
#     print('Saving')
#     os.makedirs(Paths.totalcapture_dir, exist_ok=True)
#     torch.save(out_pose, os.path.join(Paths.totalcapture_dir, 'pose.pt'))
#     torch.save(out_shape, os.path.join(Paths.totalcapture_dir, 'shape.pt'))
#     torch.save(out_tran, os.path.join(Paths.totalcapture_dir, 'tran.pt'))
#     torch.save(out_joint, os.path.join(Paths.totalcapture_dir, 'joint.pt'))
#     torch.save(out_vrot, os.path.join(Paths.totalcapture_dir, 'vrot.pt'))
#     torch.save(out_vacc, os.path.join(Paths.totalcapture_dir, 'vacc.pt'))
#     print('Synthetic totalcapture dataset is saved at', Paths.totalcapture_dir)
#
#
# def process_DanceDB():
#     data_pose, data_trans, data_beta, length = [], [], [], []
#     ds_name = ['DanceDB']
#     print('\rReading', ds_name)
#     for npz_fname in tqdm(glob.glob(os.path.join(Paths.raw_DanceDB_dir, '*/*_poses.npz'))):
#         try:
#             cdata = np.load(npz_fname)
#         except:
#             continue
#         framerate = int(cdata['mocap_framerate'])
#         if framerate == 120:
#             step = 2
#         elif framerate == 60 or framerate == 59:
#             step = 1
#         else:
#             continue
#
#         data_pose.extend(cdata['poses'][::step].astype(np.float32))
#         data_trans.extend(cdata['trans'][::step].astype(np.float32))
#         data_beta.append(cdata['betas'][:10])
#         length.append(cdata['poses'][::step].shape[0])
#
#     assert len(data_pose) != 0, 'DanceDB dataset not found. Check config.py or comment the function process()'
#     length = torch.tensor(length, dtype=torch.int)
#     shape = torch.tensor(np.asarray(data_beta, np.float32))
#     tran = torch.tensor(np.asarray(data_trans, np.float32))
#     pose = torch.tensor(np.asarray(data_pose, np.float32)).view(-1, 52, 3)
#     pose[:, 23] = pose[:, 37]  # right hand
#     pose = pose[:, :24].clone()  # only use body
#
#     # align AMASS global fame with DIP
#     amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]])
#     tran = amass_rot.matmul(tran.unsqueeze(-1)).view_as(tran)
#     pose[:, 0] = art.math.rotation_matrix_to_axis_angle(
#     amass_rot.matmul(art.math.axis_angle_to_rotation_matrix(pose[:, 0])))
#
#     print('Synthesizing IMU accelerations and orientations')
#     b = 0
#     out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc = [], [], [], [], [], []
#     for i, l in tqdm(list(enumerate(length))):
#         if l <= 12: b += l; print('\tdiscard one sequence with length', l); continue
#         p = art.math.axis_angle_to_rotation_matrix(pose[b:b + l]).view(-1, 24, 3, 3)
#         grot, joint, vert = body_model.forward_kinematics(p, shape[i], tran[b:b + l], calc_mesh=True)
#         out_pose.append(pose[b:b + l].clone())  # N, 24, 3
#         out_tran.append(tran[b:b + l].clone())  # N, 3
#         out_shape.append(shape[i].clone())  # 10
#         out_joint.append(joint[:, :24].contiguous().clone())  # N, 24, 3
#         out_vacc.append(_syn_acc(vert[:, vi_mask]))  # N, 6, 3
#         out_vrot.append(grot[:, ji_mask])  # N, 6, 3, 3
#         b += l
#
#     print('Saving')
#     os.makedirs(Paths.DanceDB_dir, exist_ok=True)
#     torch.save(out_pose, os.path.join(Paths.DanceDB_dir, 'pose.pt'))
#     torch.save(out_shape, os.path.join(Paths.DanceDB_dir, 'shape.pt'))
#     torch.save(out_tran, os.path.join(Paths.DanceDB_dir, 'tran.pt'))
#     torch.save(out_joint, os.path.join(Paths.DanceDB_dir, 'joint.pt'))
#     torch.save(out_vrot, os.path.join(Paths.DanceDB_dir, 'vrot.pt'))
#     torch.save(out_vacc, os.path.join(Paths.DanceDB_dir, 'vacc.pt'))
#     print('Synthetic DanceDB dataset is saved at', Paths.DanceDB_dir)
#
#
# def process_dipimu():
#     imu_mask = [7, 8, 11, 12, 0, 2]
#     # test_split = ['s_09', 's_10']
#     train_split = ['s_01', 's_02', 's_03', 's_04', 's_05', 's_06', 's_07', 's_08']
#     accs, oris, poses, trans = [], [], [], []
#     # for subject_name in test_split:
#     for subject_name in train_split:
#         for motion_name in os.listdir(os.path.join(Paths.raw_dipimu_dir, subject_name)):
#             path = os.path.join(Paths.raw_dipimu_dir, subject_name, motion_name)
#             data = pickle.load(open(path, 'rb'), encoding='latin1')
#             acc = torch.from_numpy(data['imu_acc'][:, imu_mask]).float()
#             ori = torch.from_numpy(data['imu_ori'][:, imu_mask]).float()
#             pose = torch.from_numpy(data['gt']).float()
#
#             # fill nan with nearest neighbors
#             for _ in range(4):
#                 acc[1:].masked_scatter_(torch.isnan(acc[1:]), acc[:-1][torch.isnan(acc[1:])])
#                 ori[1:].masked_scatter_(torch.isnan(ori[1:]), ori[:-1][torch.isnan(ori[1:])])
#                 acc[:-1].masked_scatter_(torch.isnan(acc[:-1]), acc[1:][torch.isnan(acc[:-1])])
#                 ori[:-1].masked_scatter_(torch.isnan(ori[:-1]), ori[1:][torch.isnan(ori[:-1])])
#
#             acc, ori, pose = acc[6:-6], ori[6:-6], pose[6:-6]
#             if torch.isnan(acc).sum() == 0 and torch.isnan(ori).sum() == 0 and torch.isnan(pose).sum() == 0:
#                 accs.append(acc.clone())
#                 oris.append(ori.clone())
#                 poses.append(pose.clone())
#                 trans.append(torch.zeros(pose.shape[0], 3))
#                 # dip-imu does not contain translations ⭐⭐⭐⭐⭐⭐⭐⭐⭐
#             else:
#                 print('DIP-IMU: %s/%s has too much nan! Discard!' % (subject_name, motion_name))
#
#     os.makedirs(Paths.dipimu_dir, exist_ok=True)
#     # torch.save({'acc': accs, 'ori': oris, 'pose': poses, 'tran': trans}, os.path.join(Paths.dipimu_dir, 'test.pt'))
#     torch.save({'acc': accs, 'ori': oris, 'pose': poses, 'tran': trans}, os.path.join(Paths.dipimu_dir, 'train.pt'))
#     print('Preprocessed DIP-IMU dataset is saved at', Paths.dipimu_dir)


# def process_totalcapture():
#     inches_to_meters = 0.0254
#     file_name = 'gt_skel_gbl_pos.txt'
#
#     accs, oris, poses, trans = [], [], [], []
#     for file in sorted(os.listdir(Paths.raw_totalcapture_dip_dir)):
#         data = pickle.load(open(os.path.join(Paths.raw_totalcapture_dip_dir, file), 'rb'), encoding='latin1')
#         ori = torch.from_numpy(data['ori']).float()[:, torch.tensor([2, 3, 0, 1, 4, 5])]
#         acc = torch.from_numpy(data['acc']).float()[:, torch.tensor([2, 3, 0, 1, 4, 5])]
#         pose = torch.from_numpy(data['gt']).float().view(-1, 24, 3)
#
#         # acc/ori and gt pose do not match in the dataset
#         if acc.shape[0] < pose.shape[0]:
#             pose = pose[:acc.shape[0]]
#         elif acc.shape[0] > pose.shape[0]:
#             acc = acc[:pose.shape[0]]
#             ori = ori[:pose.shape[0]]
#
#         assert acc.shape[0] == ori.shape[0] and ori.shape[0] == pose.shape[0]
#         accs.append(acc)  # N, 6, 3
#         oris.append(ori)  # N, 6, 3, 3
#         poses.append(pose)  # N, 24, 3
#
#     for subject_name in ['S1', 'S2', 'S3', 'S4', 'S5']:
#         for motion_name in sorted(os.listdir(os.path.join(Paths.raw_totalcapture_official_dir, subject_name))):
#             if subject_name == 'S5' and motion_name == 'acting3':
#                 continue  # no SMPL poses
#             f = open(os.path.join(Paths.raw_totalcapture_official_dir, subject_name, motion_name, file_name))
#             line = f.readline().split('\t')
#             index = torch.tensor([line.index(_) for _ in ['LeftFoot', 'RightFoot', 'Spine']])
#             pos = []
#             while line:
#                 line = f.readline()
#                 pos.append(torch.tensor([[float(_) for _ in p.split(' ')] for p in line.split('\t')[:-1]]))
#             pos = torch.stack(pos[:-1])[:, index] * inches_to_meters
#             pos[:, :, 0].neg_()
#             pos[:, :, 2].neg_()
#             trans.append(pos[:, 2] - pos[:1, 2])  # N, 3
#
#     # match trans with poses
#     for i in range(len(accs)):
#         if accs[i].shape[0] < trans[i].shape[0]:
#             trans[i] = trans[i][:accs[i].shape[0]]
#         assert trans[i].shape[0] == accs[i].shape[0]
#
#     # remove acceleration bias
#     for iacc, pose, tran in zip(accs, poses, trans):
#         pose = art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
#         _, _, vert = body_model.forward_kinematics(pose, tran=tran, calc_mesh=True)
#         vacc = _syn_acc(vert[:, vi_mask])
#         for imu_id in range(6):
#             for i in range(3):
#                 d = -iacc[:, imu_id, i].mean() + vacc[:, imu_id, i].mean()
#                 iacc[:, imu_id, i] += d
#
#     os.makedirs(Paths.totalcapture_dir, exist_ok=True)
#     torch.save({'acc': accs, 'ori': oris, 'pose': poses, 'tran': trans},
#                os.path.join(Paths.totalcapture_dir, 'test.pt'))
#     print('Preprocessed TotalCapture dataset is saved at', Paths.totalcapture_dir)


if __name__ == '__main__':
    process_amass()
    # process_dipimu()
    # process_totalcapture()
    # process_DanceDB()
