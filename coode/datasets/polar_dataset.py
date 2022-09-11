import os
import torch
import numpy as np

import utils.general as utils
from utils import rend_util

class PolarDataset(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self,
                 train_cameras,
                 data_dir,
                 img_res,
                 num_views,
                 cam_file=None
                 ):

        self.instance_dir = data_dir

        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res

        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None
        self.train_cameras = train_cameras

        image_dir = '{0}/I-sum'.format(self.instance_dir)
        image_paths_all = sorted(utils.glob_imgs(image_dir))
        mask_dir = '{0}/masks'.format(self.instance_dir)
        mask_paths_all = sorted(utils.glob_imgs(mask_dir))
        normal_dir ='{0}/normals-png'.format(self.instance_dir)
        normal_paths_all = sorted(utils.glob_imgs(normal_dir))
        aolp_dir = '{0}/params/AoLP'.format(self.instance_dir)
        aolp_paths_all = sorted(utils.glob_imgs(aolp_dir))
        dolp_dir = '{0}/params/DoLP'.format(self.instance_dir)
        dolp_paths_all = sorted(utils.glob_imgs(dolp_dir))

        # num_views sampler
        if(num_views>0):
            interval = int(len(image_paths_all) / num_views)
        else:
            interval = 1
            num_views = len(image_paths_all)

        image_paths = []
        mask_paths = []
        normal_paths = []
        aolp_paths = []
        dolp_paths = []
        for i in range(0,num_views):
            if(interval * i > len(image_paths_all)):
                image_paths.append(image_paths_all[-1])
                mask_paths.append(mask_paths_all[-1])
                normal_paths.append(normal_paths_all[-1])
                aolp_paths.append(aolp_paths_all[-1])
                dolp_paths.append(dolp_paths_all[-1])
            else:
                image_paths.append(image_paths_all[interval * i])
                mask_paths.append(mask_paths_all[interval * i])
                normal_paths.append(normal_paths_all[interval * i])
                aolp_paths.append(aolp_paths_all[interval * i])
                dolp_paths.append(dolp_paths_all[interval * i])




        self.n_images = len(image_paths)

        self.cam_file = '{0}/cameras_new.npz'.format(self.instance_dir)
        if cam_file is not None:
            self.cam_file = '{0}/{1}'.format(self.instance_dir, cam_file)

        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(len(image_paths_all))]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(len(image_paths_all))]

        intrinsics_all_all = []
        pose_all_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            intrinsics_all_all.append(torch.from_numpy(intrinsics).float())
            pose_all_all.append(torch.from_numpy(pose).float())

        self.intrinsics_all = []
        self.pose_all = []
        for i in range(0,num_views):
            if(interval * i > len(image_paths_all)):
                self.intrinsics_all.append(intrinsics_all_all[-1])
                self.pose_all.append(pose_all_all[-1])

            else:
                self.intrinsics_all.append(intrinsics_all_all[interval * i])
                self.pose_all.append(pose_all_all[interval * i])


        self.rgb_images = []
        for path in image_paths:
            rgb = rend_util.load_gray_as_rgb(path)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())

        self.object_masks = []
        for path in mask_paths:
            object_mask = rend_util.load_mask(path)
            object_mask = object_mask.reshape(-1)
            self.object_masks.append(torch.from_numpy(object_mask).bool())

        self.normal_images = []
        for path in normal_paths:
            normal = rend_util.load_rgb(path)
            normal = normal.reshape(3, -1).transpose(1, 0)
            self.normal_images.append(torch.from_numpy(normal).float())

        self.aolp_images = []
        for path in aolp_paths:
            aolp = rend_util.load_aolp_as_rgb(path)
            aolp = aolp.reshape(3, -1).transpose(1, 0) # (0,1)
            aolp = aolp * np.pi # (0,pi)
            self.aolp_images.append(torch.from_numpy(aolp).float())

        self.dolp_images = []
        for path in dolp_paths:
            dolp = rend_util.load_aolp_as_rgb(path)
            dolp = dolp.reshape(3, -1).transpose(1, 0) # (0,1)
            self.dolp_images.append(torch.from_numpy(dolp).float())

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)
        sample = {
            "object_mask": self.object_masks[idx],
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
        }

        ground_truth = {
            "rgb": self.rgb_images[idx],
            "normal": self.normal_images[idx], # (-1,1)
            "aolp":self.aolp_images[idx],     # (-1,1)
            "dolp":self.dolp_images[idx]
        }
        # import matplotlib.pyplot as plt
        # plt.imshow(sample["object_mask"].reshape(self.img_res))
        # plt.figure()
        # plt.imshow(ground_truth["rgb"].reshape([self.img_res[0],self.img_res[1],3]))
        # plt.figure()
        # plt.imshow(ground_truth["normal"].reshape([self.img_res[0],self.img_res[1],3]))
        # plt.figure()
        # plt.imshow(ground_truth["aolp"].reshape([self.img_res[0],self.img_res[1],3]))
        # plt.show()

        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            ground_truth["normal"] = self.normal_images[idx][self.sampling_idx, :]
            ground_truth["aolp"] = self.aolp_images[idx][self.sampling_idx, :]
            ground_truth["dolp"] = self.dolp_images[idx][self.sampling_idx, :]

            sample["object_mask"] = self.object_masks[idx][self.sampling_idx]
            sample["uv"] = uv[self.sampling_idx, :]

        if not self.train_cameras:
            sample["pose"] = self.pose_all[idx]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']

    def get_gt_pose(self, scaled=False):
        # Load gt pose without normalization to unit sphere
        camera_dict = np.load(self.cam_file)
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat
            if scaled:
                P = world_mat @ scale_mat
            P = P[:3, :4]
            _, pose = rend_util.load_K_Rt_from_P(None, P)
            pose_all.append(torch.from_numpy(pose).float())

        return torch.cat([p.float().unsqueeze(0) for p in pose_all], 0)

    def get_pose_init(self):
        # get noisy initializations obtained with the linear method
        cam_file = '{0}/cameras_new.npz'.format(self.instance_dir)
        camera_dict = np.load(cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        init_pose = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            _, pose = rend_util.load_K_Rt_from_P(None, P)
            init_pose.append(pose)

        init_pose = torch.cat([torch.Tensor(pose).float().unsqueeze(0) for pose in init_pose], 0).cuda()
        init_quat = rend_util.rot_to_quat(init_pose[:, :3, :3])
        init_quat = torch.cat([init_quat, init_pose[:, :3, 3]], 1)

        return init_quat

