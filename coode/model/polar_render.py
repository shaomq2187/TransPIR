import torch
import torch.nn as nn
import numpy as np

from utils import rend_util
from model.embedder import *
from model.ray_tracing import RayTracing
from polarTracing.mueller import Mueller
class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0
    ):
        super().__init__()

        dims = [d_in] + dims + [d_out ]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)
            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

    def forward(self, input, compute_grad=False):
        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)
            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:, :1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)

        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)

class PolarInverseRenderingNetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.implicit_network = ImplicitNetwork(**conf.get_config('implicit_network'))
        self.ray_tracer = RayTracing(**conf.get_config('ray_tracer'))
        self.mueller = Mueller()
    def forward(self, input):

        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]
        object_mask = input["object_mask"].reshape(-1)
        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)
        batch_size, num_pixels, _ = ray_dirs.shape

        first_pts, first_net_mask, first_dists,\
        second_pts, second_net_mask, second_dists,\
        first_refract_dirs, first_reflect_dirs, first_attenuate, first_refract_total_reflect_mask,\
        second_refract_dirs, second_reflect_dirs, second_attenuate, second_refract_total_reflect_mask,stokes,dict,valid_mask,\
        bottom_start_pts,bottom_pts,bottom_mask = self.ray_tracer(sdf=lambda x: self.implicit_network(x)[:, 0],
                                                                    normal_fn = lambda x: self.implicit_network.gradient(x)[:,0,:],
                                                                    cam_loc=cam_loc,
                                                                    object_mask=object_mask,
                                                                    ray_directions=ray_dirs
                                                                    )




        n_rays = first_net_mask.shape[0]
        n_valid_rays = torch.sum(first_net_mask)


        dolp_render = torch.zeros((num_pixels,1)).reshape(-1).cuda()
        aolp_render = torch.zeros((num_pixels,1)).reshape(-1).cuda()
        s0 = torch.zeros((num_pixels,1)).reshape(-1).cuda()
        s1 = torch.zeros((num_pixels,1)).reshape(-1).cuda()
        s2 = torch.zeros((num_pixels,1)).reshape(-1).cuda()


        if n_valid_rays > 0:
            L_t2 = torch.zeros(n_valid_rays,1).cuda().float()
            L_r1 = torch.zeros(n_valid_rays,1).cuda().float()
            F1 = first_attenuate
            F2 = second_attenuate
            L_t2[~second_refract_total_reflect_mask] = self.ray_tracer.sampleEnvLight(second_refract_dirs[~second_refract_total_reflect_mask])

            L_t1 = L_t2 * (1 - F2)

            L_t0 = L_t1 * (1 - F1)
            L_t0[second_refract_total_reflect_mask] = 0.5


            L_r1[~first_refract_total_reflect_mask] = self.ray_tracer.sampleEnvLight(first_reflect_dirs[~first_refract_total_reflect_mask])
            L_r0 = L_r1 * F1

            L_r0_output = torch.zeros(n_rays, 1).cuda()
            L_t0_output = torch.zeros(n_rays, 1).cuda()

            L_r0_output[first_net_mask] = L_r0
            L_t0_output[first_net_mask] = L_t0


            # polar redering
            incident_dirs = ray_dirs.squeeze(0)[first_net_mask,:]
            reflected_dirs = first_reflect_dirs
            normals_demo = self.implicit_network.gradient(first_pts[first_net_mask,:])[:,0,:]
            reflection_light = torch.ones((n_valid_rays,1)).reshape(-1)
            stokes = self.reflection_polarized_demo(pose,incident_dirs,reflected_dirs,normals_demo,reflection_light)  # (n_valid_rays,)
            stokes_cam = self.stokes_to_camera(pose,stokes,incident_dirs)
            s0_valid = stokes_cam[:,0].reshape(-1)
            s1_valid = stokes_cam[:,1].reshape(-1)
            s2_valid = stokes_cam[:,2].reshape(-1)

            dolp = torch.sqrt(s1_valid**2 + s2_valid**2) / s0_valid
            aolp = 0.5 * torch.atan2(s2_valid, s1_valid + 1e-6) # 加个eps试试
            dolp = dolp.reshape(-1)
            aolp = aolp.reshape(-1)
            aolp = torch.remainder(aolp,torch.pi)

            dolp_render[first_net_mask] = dolp
            aolp_render[first_net_mask] = aolp
            s0[first_net_mask] = s0_valid
            s1[first_net_mask] = s1_valid
            s2[first_net_mask] = s2_valid

        else:
            L_r0_output = torch.zeros_like(first_net_mask).cuda().float()
            L_t0_output = torch.zeros_like(first_net_mask).cuda().float()








        self.implicit_network.train()
        points = first_pts # (n_rays,3)
        surface_points = first_pts[first_net_mask]
        sdf_output = self.implicit_network(points)[:,0]
        network_object_mask = first_net_mask
        grad_theta = self.implicit_network.gradient(surface_points)[:,0,:]
        normals = self.implicit_network.gradient(points)[:,0,:]
        points_refractive = second_pts
        mask_refractive = second_net_mask
        normals_refractive = self.implicit_network.gradient(points_refractive)[:,0,:]
        out_mask = ~second_refract_total_reflect_mask
        out_points = second_pts + 1 * second_refract_dirs

        attenuate = second_attenuate
        if(torch.sum(first_net_mask)>0):
            reflected_points = first_pts[first_net_mask] + 1 * first_reflect_dirs
        else:
            reflected_points = first_pts + 1 * first_reflect_dirs

        reflected_light = self.ray_tracer.sampleEnvLight(first_reflect_dirs)
        out_light = self.ray_tracer.sampleEnvLight(second_refract_dirs)

        output = {
            'points': points, # (num_pixels,3)
            'first_net_mask':first_net_mask,
            'differentiable_surface_points': surface_points, # 仅有效点 (n_valid_rays,3)
            'sdf_output': sdf_output, # (num_pixels,3)
            'network_object_mask': network_object_mask, # (num_pixels,) bool
            'object_mask': object_mask, # (num_pixels,) bool
            'grad_theta': grad_theta,
            'normals': normals,  # (num_pixels,3)
            'points_refractive': points_refractive, # (n_valid_rays,3)
            'sdf_refractive': self.implicit_network(points_refractive)[:,0],
            'mask_refractive': mask_refractive, # (n_valid_rays,) bool
            'normals_refractive': normals_refractive, # (n_valid_rays,3)
            'out_mask': out_mask, # (n_valid_rays,)
            'out_points': out_points, # (n_valid_rays,3)
            'out_attenuate': attenuate, # (n_valid_days,)
            'reflected_points': reflected_points, #(n_valid_rays,3)
            'reflected_light': reflected_light, #(n_valid_rays,)
            'reflection_intensity': L_r0_output, #(n_rays,) (0,1)
            'transmittance_intensity': L_t0_output, #(n_rays,)(0,1)
            'out_light': out_light, # (n_valid_rays,)
            'first_attenuate': first_attenuate, # (n_valid_rays,)
            'dolp_render': dolp_render, #(num_pixels,)
            'aolp_render': aolp_render, #(num_pixels,)
            's0': s0,  # (num_pixels,)
            's1': s1,  # (num_pixels,)
            's2': s2,  # (num_pixels,)
            'rayTracing_dict': dict,
            'valid_mask': valid_mask, # (num_pixels,)
            'bottom_normals':self.implicit_network.gradient(bottom_pts)[:,0,:],
            'bottom_mask':bottom_mask,
            'bottom_pts': bottom_pts,
            'bottom_start_pts':bottom_start_pts

        }


        return output
    def reflection_polarized_demo(self,pose,incident_dirs,reflection_dirs,normals,reflection_light):
        normals = self.mueller.normalize(normals)
        # pose: (bs,4,4)
        if pose.shape[1] == 7:  # In case of quaternion vector representation
            R = rend_util.quat_to_rot(pose[:, :4]).squeeze(dim=0)

        else:  # In case of pose matrix representation
            R = pose[0,0:3,0:3]


        n_rays = incident_dirs.shape[0]
        eta = (torch.ones(n_rays,1) * 1.52).cuda().reshape(-1)

        stokes_init = torch.zeros((n_rays,4,1)).cuda().float()
        stokes_init[:,0,0] = reflection_light.reshape(-1)


        w_o_hat = reflection_dirs
        w_i_hat = -incident_dirs

        s_axis_in = self.mueller.normalize(torch.cross(normals,-w_o_hat))
        s_axis_out = self.mueller.normalize(torch.cross(normals,w_i_hat))


        cos_theta_i = self.mueller.dot(normals,w_o_hat)

        weight = self.mueller.specular_reflection(cos_theta_i,eta) # (n_rays,4,4)

        weight = self.mueller.rotate_mueller_basis(weight,
                                                   -w_o_hat, s_axis_in, self.mueller.stokes_basis(-w_o_hat),
                                                   w_i_hat, s_axis_out, self.mueller.stokes_basis(w_i_hat)
                                                   )
        stokes = torch.bmm(weight,stokes_init)

        return stokes

    def stokes_to_camera(self,pose,stokes,incident_dirs):
        if pose.shape[1] == 7:  # In case of quaternion vector representation
            R = rend_util.quat_to_rot(pose[:, :4]).squeeze(dim=0)

        else:  # In case of pose matrix representation
            R = pose[0,0:3,0:3]
        n_rays,_,_ = stokes.shape

        vertical_cam = torch.tensor([0.0,-1.0,0.0]).cuda().reshape(-1,1) #(3,1)
        vertical = torch.mm(R,vertical_cam).reshape(-1).unsqueeze(0)
        vertical = vertical.repeat(n_rays,1) # (n_rays,3)
        current_basis = self.mueller.stokes_basis(-incident_dirs)
        target_basis = torch.cross(incident_dirs,vertical)
        M_cam = self.mueller.rotate_stokes_basis(-incident_dirs,current_basis,target_basis)
        stokes_cam = torch.bmm(M_cam, stokes)
        return stokes_cam


