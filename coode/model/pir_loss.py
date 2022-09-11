import torch
from torch import nn
from torch.nn import functional as F



class PIRLoss(nn.Module):
    def __init__(self, eikonal_weight, mask_weight,aolp_render_weight_init, alpha):
        super().__init__()

        self.eikonal_weight = eikonal_weight
        self.mask_weight = mask_weight
        self.aolp_render_weight = 0.0
        self.aolp_render_weight_init = aolp_render_weight_init
        self.alpha = alpha
        self.l1_loss = nn.L1Loss(reduction='sum')


    def get_aolp_render_loss(self,aolp_values, aolp_gt, network_object_mask, object_mask,valid_mask,percentage):
        if (network_object_mask & object_mask & valid_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()
        aolp_predict = aolp_values.reshape(-1) # (num_pixels,) # (0,pi)
        aolp_predict = aolp_predict[network_object_mask & object_mask & valid_mask]

        aolp_gt = aolp_gt.reshape(-1)[network_object_mask & object_mask & valid_mask] # (0,pi)

        loss = torch.abs(aolp_predict - aolp_gt)
        loss_invalid_mask = (loss > torch.pi / 6.0)
        percentage = percentage[network_object_mask & object_mask & valid_mask].reshape(-1)
        loss_weighted= loss * percentage
        loss_weighted[loss_invalid_mask] = 0.0
        loss_weighted = loss_weighted.sum()
        loss_weighted = loss_weighted / float(object_mask.shape[0])
        return loss_weighted

    def get_eikonal_loss(self, grad_theta):
        if grad_theta.shape[0] == 0:
            return torch.tensor(0.0).cuda().float()

        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def get_mask_loss(self, sdf_output, network_object_mask, object_mask):
        mask = ~(network_object_mask & object_mask)
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()
        # 不满足mask的sdf值都应大于0
        sdf_pred = -self.alpha * sdf_output[mask]
        gt = object_mask[mask].float()
        mask_loss = (1 / self.alpha) * F.binary_cross_entropy_with_logits(sdf_pred.squeeze(), gt, reduction='sum') / float(object_mask.shape[0])
        return mask_loss

    def get_bottom_normal_loss(self,bottom_normals_predict,bottom_normal_target,bottom_mask):
        n_rays,_ = bottom_normals_predict.shape
        target_bottom_normals = bottom_normal_target.repeat(n_rays,1)

        loss_orientation = torch.min(torch.sqrt(((target_bottom_normals - bottom_normals_predict) ** 2).sum(dim=1)),
                                     torch.sqrt(((target_bottom_normals + target_bottom_normals) ** 2).sum(dim=1)))
        loss_orientation[~bottom_mask] = 0.0
        loss = loss_orientation.sum() / float(bottom_mask.shape[0])
        return loss

    def forward(self, model_inputs,model_outputs,ground_truth,sampling_idxs=None):
        normal_gt = ground_truth['normal'].cuda()  # [bs,num_pixels,3]
        aolp_gt = ground_truth['aolp'][:,:,0].cuda()
        dolp_gt = ground_truth['dolp'][:,:,0].cuda()
        s0_gt = (ground_truth['rgb'][:,:,0].cuda() + 1.0) / 2.0
        network_object_mask = model_outputs['network_object_mask']
        object_mask = model_outputs['object_mask']
        network_normals = model_outputs['normals'] # [num_pixels,3]
        reflection_intensity  = model_outputs['reflection_intensity'].cuda()
        transmittance_intensity  = model_outputs['transmittance_intensity'].cuda()
        aolp_render = model_outputs['aolp_render'].cuda()
        s0_render = model_outputs['s0'].cuda()
        # bottom_normals = model_outputs['bottom_normals'].cuda()
        # bottom_normal_ground_truth = torch.tensor([0.0,0.0,1.0]).cuda()
        # bottom_mask = model_outputs['bottom_mask'].cuda()

        valid_mask = (aolp_gt > 0.0).reshape(-1)
        I_sum = reflection_intensity + transmittance_intensity
        percentage = reflection_intensity / I_sum
        percentage[torch.isnan(percentage)] = 0.0
        percentage = torch.clamp(percentage, 0, 1)
        percentage.requires_grad= False



        pose = model_inputs['pose']

        mask_loss = self.get_mask_loss(model_outputs['sdf_output'], network_object_mask, object_mask)
        if(aolp_gt.shape[1]>20480):
            eikonal_loss = mask_loss
        else:
            eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        aolp_render_loss = self.get_aolp_render_loss(aolp_render,aolp_gt,network_object_mask,object_mask,valid_mask,percentage)
        # bottom_loss = self.get_bottom_normal_loss(bottom_normals,bottom_normal_ground_truth,bottom_mask)




        loss = self.mask_weight * mask_loss + self.eikonal_weight * eikonal_loss +\
               self.aolp_render_weight * aolp_render_loss #+ bottom_loss * 0.01


        return {
            'loss': loss,
            'aolp_render_loss': self.aolp_render_weight * aolp_render_loss,
            'eikonal_loss': self.eikonal_weight * eikonal_loss,
            'mask_loss': self.mask_weight * mask_loss,
        }
