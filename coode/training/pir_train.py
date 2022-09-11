# -*-coding:gbk-*-
import os
from datetime import datetime
from pyhocon import ConfigFactory
import sys
import torch

import utils.general as utils
import utils.plots as plt
def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    # for group in optimizer.param_groups:
    #     for param in group["params"]:
    #         if param.grad is not None:
    #             param.grad.data.clamp_(-grad_clip, grad_clip)
    for group in optimizer.param_groups:
        for param in group["params"]:
            a = param.sum()
            if(torch.isnan(a)):
                print('param error')

            if(param.grad is not None):
                b = param.grad.sum()
                if(torch.isnan(b)):
                    print('grad error')

class PIRTrainRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.batch_size = kwargs['batch_size']
        self.nepochs = kwargs['nepochs']
        self.exps_folder_name = kwargs['exps_folder_name']
        self.GPU_INDEX = kwargs['gpu_index']
        self.train_cameras = kwargs['train_cameras']

        self.expname = self.conf.get_string('train.expname') + kwargs['expname']


        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join('../',kwargs['exps_folder_name'],self.expname)):
                timestamps = os.listdir(os.path.join('../',kwargs['exps_folder_name'],self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']

        utils.mkdir_ifnotexists(os.path.join('../',self.exps_folder_name))
        self.expdir = os.path.join('../', self.exps_folder_name, self.expname)
        utils.mkdir_ifnotexists(self.expdir)
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))

        self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)

        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)
        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"
        self.scheduler_params_subdir = "SchedulerParameters"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))

        if self.train_cameras:
            self.optimizer_cam_params_subdir = "OptimizerCamParameters"
            self.cam_params_subdir = "CamParameters"

            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_cam_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.cam_params_subdir))

        os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))

        if (not self.GPU_INDEX == 'ignore'):
            os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        dataset_conf = self.conf.get_config('dataset')


        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'),name='datasets')(self.train_cameras,
                                                                                          **dataset_conf)

        print('Finish loading data ...')

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            )
        self.plot_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=True,
                                                           )
        print('Creating model ...')
        self.model = utils.get_class(self.conf.get_string('train.model_class'),name='model')(conf=self.conf.get_config('model'))
        if torch.cuda.is_available():
            self.model.cuda()
        print('Creating settings ...')
        self.loss = utils.get_class(self.conf.get_string('train.loss_class'),name='model')(**self.conf.get_config('loss'))

        self.lr = self.conf.get_float('train.learning_rate')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.sched_milestones = self.conf.get_list('train.sched_milestones', default=[])
        self.sched_factor = self.conf.get_float('train.sched_factor', default=0.0)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.sched_milestones, gamma=self.sched_factor)

        print('settings for camera optimization...')
        # settings for camera optimization
        if self.train_cameras:
            num_images = len(self.train_dataset)
            self.pose_vecs = torch.nn.Embedding(num_images, 7, sparse=True).cuda()
            self.pose_vecs.weight.data.copy_(self.train_dataset.get_pose_init())

            self.optimizer_cam = torch.optim.SparseAdam(self.pose_vecs.parameters(), self.conf.get_float('train.learning_rate_cam'))

        self.start_epoch = 0
        if is_continue:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')

            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            self.model.load_state_dict(saved_model_state["model_state_dict"])
            self.start_epoch = saved_model_state['epoch']

            data = torch.load(
                os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.scheduler.load_state_dict(data["scheduler_state_dict"])

            if self.train_cameras:
                data = torch.load(
                    os.path.join(old_checkpnts_dir, self.optimizer_cam_params_subdir, str(kwargs['checkpoint']) + ".pth"))
                self.optimizer_cam.load_state_dict(data["optimizer_cam_state_dict"])

                data = torch.load(
                    os.path.join(old_checkpnts_dir, self.cam_params_subdir, str(kwargs['checkpoint']) + ".pth"))
                self.pose_vecs.load_state_dict(data["pose_vecs_state_dict"])

        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.plot_conf = self.conf.get_config('plot')

        self.aolp_render_weight_milestones = self.conf.get_list('train.aolp_render_weight_milestones', default=[])

        self.alpha_milestones = self.conf.get_list('train.alpha_milestones', default=[])
        self.alpha_factor = self.conf.get_float('train.alpha_factor', default=0.0)
        for acc in self.alpha_milestones:
            if self.start_epoch > acc:
                self.loss.alpha = self.loss.alpha * self.alpha_factor

    def save_checkpoints(self, epoch):
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))

        if self.train_cameras:
            torch.save(
                {"epoch": epoch, "optimizer_cam_state_dict": self.optimizer_cam.state_dict()},
                os.path.join(self.checkpoints_path, self.optimizer_cam_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "optimizer_cam_state_dict": self.optimizer_cam.state_dict()},
                os.path.join(self.checkpoints_path, self.optimizer_cam_params_subdir, "latest.pth"))

            torch.save(
                {"epoch": epoch, "pose_vecs_state_dict": self.pose_vecs.state_dict()},
                os.path.join(self.checkpoints_path, self.cam_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "pose_vecs_state_dict": self.pose_vecs.state_dict()},
                os.path.join(self.checkpoints_path, self.cam_params_subdir, "latest.pth"))

    def run(self):
        print("training...")

        for epoch in range(self.start_epoch, self.nepochs + 1):

            if epoch in self.alpha_milestones:
                self.loss.alpha = self.loss.alpha * self.alpha_factor
            if(epoch > self.aolp_render_weight_milestones[0]):
                self.loss.aolp_render_weight = self.loss.aolp_render_weight_init


            if epoch % 10 == 0:
                self.save_checkpoints(epoch)

            if epoch % self.plot_freq == 0:
                self.model.eval()
                if self.train_cameras:
                    self.pose_vecs.eval()
                self.train_dataset.change_sampling_idx(-1)
                indices, model_input, ground_truth = next(iter(self.plot_dataloader))

                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input["object_mask"] = model_input["object_mask"].cuda()

                if self.train_cameras:
                    pose_input = self.pose_vecs(indices.cuda())
                    model_input['pose'] = pose_input
                else:
                    model_input['pose'] = model_input['pose'].cuda()

                split = utils.split_input(model_input, self.total_pixels)
                res = []
                i = 0
                for s in split:
                    out = self.model(s)

                    i +=1
                    res.append({
                        'points': out['points'].detach(),
                        'network_object_mask': out['network_object_mask'].detach(),
                        'object_mask': out['object_mask'].detach(),
                        'normals': out['normals'].detach(),
                        'sdf_output': out['sdf_output'].detach(),
                        'reflection_intensity': out['reflection_intensity'].detach(),
                        'transmittance_intensity': out['transmittance_intensity'].detach(),
                        'dolp_render': out['dolp_render'].detach(),
                        'aolp_render': out['aolp_render'].detach(),
                        's0': out['s0'].detach(),
                        's1': out['s1'].detach(),
                        's2': out['s2'].detach(),
                        'bottom_normals':out['bottom_normals'].detach(),
                        'bottom_mask':out['bottom_mask'].detach()
                    })

                batch_size = ground_truth['rgb'].shape[0]
                model_outputs = utils.merge_output(res, self.total_pixels, batch_size)

                plt.plot(self.model,
                         indices,
                         model_outputs,
                         model_input['pose'],
                         ground_truth['dolp'],
                         ground_truth['aolp'],
                         ground_truth['rgb'],
                         self.plots_dir,
                         epoch,
                         self.img_res,
                         **self.plot_conf
                         )
                loss_output = self.loss(model_input,model_outputs, ground_truth)
                loss = loss_output['loss']
                print('TEST LOSS: {0} [{1}]: loss = {2}, eikonal_loss = {3}, mask_loss = {4}, alpha = {5}, lr = {6}'
                        .format(self.expname, epoch, loss.item(),
                                loss_output['eikonal_loss'].item(),
                                loss_output['mask_loss'].item(),
                                self.loss.alpha,
                                self.scheduler.get_lr()[0]))

                self.model.train()
                if self.train_cameras:
                    self.pose_vecs.train()

            self.train_dataset.change_sampling_idx(self.num_pixels)

            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):

                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input["object_mask"] = model_input["object_mask"].cuda()

                if self.train_cameras:
                    pose_input = self.pose_vecs(indices.cuda())
                    model_input['pose'] = pose_input
                else:
                    model_input['pose'] = model_input['pose'].cuda()
                torch.autograd.set_detect_anomaly(True)
                with torch.autograd.detect_anomaly():
                    model_outputs = self.model(model_input)

                with torch.autograd.detect_anomaly():
                    loss_output = self.loss(model_input,model_outputs, ground_truth,self.train_dataset.sampling_idx)


                loss = loss_output['loss']

                self.optimizer.zero_grad()
                if self.train_cameras:
                    self.optimizer_cam.zero_grad()

                loss.backward()
                clip_gradient(optimizer=self.optimizer,grad_clip=3)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=2)

                self.optimizer.step()
                clip_gradient(optimizer=self.optimizer,grad_clip=3)

                if self.train_cameras:
                    self.optimizer_cam.step()

                print(
                    '{0} [{1}] ({2}/{3}): loss = {4}, eikonal_loss = {5}, mask_loss = {6}, aolp_render_loss = {7}, alpha = {8}, lr = {9}'
                        .format(self.expname, epoch, data_index, self.n_batches, loss.item(),
                                loss_output['eikonal_loss'].item(),
                                loss_output['mask_loss'].item(),
                                loss_output['aolp_render_loss'].item(),
                                self.loss.alpha,
                                self.scheduler.get_lr()[0]))



            self.scheduler.step()
