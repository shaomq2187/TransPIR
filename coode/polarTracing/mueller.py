# torch version for mitsuba2: mitsuba2/include/render/mueller.h
import torch
import torch.nn as nn
class Mueller(nn.Module):
    def __init__(self):
        super().__init__()

    def stokes_basis(self,forward):
        # forward: (n_rays,3)
        b1,b2 = self.coordinate_system(forward)
        return b1



    def specular_reflection(self,cos_theta_i,eta):
        # checked with C++
        # return: (n_rays,4,4)
        n_rays = cos_theta_i.shape[0]
        cos_theta_i = cos_theta_i.reshape(-1)
        eta = eta.reshape(-1)
        a_s,a_p,_,_,_ = self.frenel_polarized(cos_theta_i,eta)

        sin_delta,cos_delta = self.sincos_arg_diff(a_p,a_s)

        r_s = torch.abs(a_s * a_s)
        r_p = torch.abs(a_p * a_p)
        a = 0.5 * (r_s + r_p)
        b = 0.5 * (r_s - r_p)
        c = torch.sqrt(r_s * r_p + 1e-6)

        sin_delta[torch.where(c == 0)] = 0.0
        cos_delta[torch.where(c == 0)] = 0.0

        res = torch.zeros([n_rays,4,4]).cuda().float()
        res[:,0,0] = a
        res[:,1,1] = a
        res[:,0,1] = b
        res[:,1,0] = b
        res[:,2,2] = c * cos_delta
        res[:,2,3] = -c * sin_delta
        res[:,3,2] = c * sin_delta
        res[:,3,3] = c * cos_delta
        return res



    def specular_transmission(self,cos_theta_i,eta):
        # checked
        n_rays = cos_theta_i.shape[0]

        a_s, a_p, cos_theta_t, eta_it, eta_ti = self.frenel_polarized(cos_theta_i, eta)

        factor = -eta_it * self.select(torch.abs(cos_theta_i) > 1e-8,cos_theta_t / cos_theta_i,torch.zeros_like(cos_theta_i))

        a_s_r = 1.0 + torch.real(a_s)
        a_p_r = (1.0 + torch.real(a_p)) * eta_ti

        t_s = a_s_r * a_s_r
        t_p = a_p_r * a_p_r
        a = 0.5 * factor * (t_s + t_p)
        b = 0.5 * factor * (t_s - t_p)
        c = factor * torch.sqrt(t_p * t_s)

        res = torch.zeros([n_rays,4,4]).cuda().float()
        res[:,0,0] = a
        res[:,1,1] = a
        res[:,0,1] = b
        res[:,1,0] = b
        res[:,2,2] = c
        res[:,3,3] = c


        return res



    def rotate_mueller_basis(self,M,in_forward,in_basis_current,in_basis_target,out_forward,out_basis_current,out_basis_target):
        # M:(n_rays,4,4)
        # basis:(n_rays,3)
        R_in  = self.rotate_stokes_basis(in_forward,in_basis_current,in_basis_target) # (n_rays,4,4)
        R_out = self.rotate_stokes_basis(out_forward,out_basis_current,out_basis_target)# (n_rays,4,4)

        # return R_out * M * transpose(R_in)
        temp = torch.bmm(R_out,M)
        temp = torch.bmm(temp,torch.transpose(R_in,dim0=1,dim1=2))

        return temp

    def rotate_stokes_basis(self,forward,basis_current,target_basis):
        theta = self.unit_angle(self.normalize(basis_current),
                                self.normalize(target_basis))
        mask = (self.dot(forward,self.cross(basis_current,target_basis))) < 0
        theta[mask] = -1.0 * theta[mask]
        return self.rotator(theta)

    def unit_angle(self,a,b):
        # checked
        # a,b: (n_rays,3)
        dot_uv = self.dot(a,b) # (n_rays,)
        temp = 2.0 * torch.asin(0.5 * torch.norm(b - self.mulsign_vec(a,dot_uv),dim=1))
        return self.select(dot_uv>=0,temp,torch.pi - temp)


    def rotator(self,theta):
        # theta(n_rays,)
        # return: M(n_rays,4,4)
        theta = theta.reshape(-1)
        n_rays = theta.shape[0]
        c = torch.cos(2.0 * theta).reshape(-1) # (n_rays,)
        s = torch.sin(2.0 * theta).reshape(-1)
        res = torch.zeros([n_rays,4,4]).cuda().float()
        res[:,0,0] = 1.0
        res[:,3,3] = 1.0
        res[:,1,1] = c
        res[:,1,2] = s
        res[:,2,1] = -s
        res[:,2,2] = c
        return res
    ##################################  菲涅尔方程  ################################

    def frenel_polarized(self,cos_theta_i, eta):
        # checked with C++
        # eta: n2/n1
        # cos_theta_i: (n_rays,) float
        # eta: (n_rays) float
        # return: a_s: complex, ap: complex, cos_theta_t_signed: float,eta_it: float,eta_ti: float
        cos_theta_i = cos_theta_i.reshape(-1)
        eta = eta.reshape(-1)
        rcp_eta = 1.0 / eta
        outside_mask = cos_theta_i >= 0.0
        eta_it = self.select(outside_mask,eta,rcp_eta)
        eta_ti = self.select(outside_mask,rcp_eta,eta)

        cos_theta_t_sqr = 1.0 - eta_ti * eta_ti * (1 - cos_theta_i * cos_theta_i) # 1- n^2 * sin_theta_i^2
        cos_theta_i_abs = torch.abs(cos_theta_i)
        cos_theta_t = self.sqrtz(cos_theta_t_sqr)

        cos_theta_t = self.mulsign(cos_theta_t,cos_theta_t_sqr)

        a_s = (cos_theta_i_abs - eta_it * cos_theta_t) / \
              (cos_theta_i_abs + eta_it * cos_theta_t)
        a_p = (eta_it * cos_theta_i_abs - cos_theta_t) / \
              (eta_it * cos_theta_i_abs + cos_theta_t)

        index_matched = torch.eq(eta,1.0)
        invalid = torch.eq(eta,0.0)
        a_s[index_matched | invalid] = 0.0
        a_p[index_matched | invalid] = 0.0


        cos_theta_t_signed = self.select(cos_theta_t_sqr >= 0.0,self.mulsign_neg(torch.real(cos_theta_t),cos_theta_i),torch.zeros_like(cos_theta_t_sqr))

        return a_s,a_p,cos_theta_t_signed,eta_it,eta_ti

    def sincos_arg_diff(self,a,b):
        # sin(arg(a) - arg(n))和cos(arg(a) - arg(b))
        # a,b: (n_rays,)
        a = a.reshape(-1)
        b = b.reshape(-1)





        normalization = 1.0 / (torch.sqrt(torch.abs(a)**2 * torch.abs(b)**2 + 1e-6) )  # 计算模长

        norm_nan_mask = (torch.sqrt(torch.abs(a)**2 * torch.abs(b)**2) == 0.0)
        normalization[norm_nan_mask] = 0.0
        value = a * torch.conj(b) * normalization
        sin_value = torch.imag(value).reshape(-1)
        cos_value = torch.real(value).reshape(-1)
        return sin_value,cos_value




    ##################################  API  ################################
    def coordinate_system(self,n):
        # checked
        # same as <Building an Orthonormal Basis, Revisited> and mitsuba2
        # n: (n_rays,3)
        # return b1,b2
        x = n[:,0]
        y = n[:,1]
        z = n[:,2]
        sign = torch.sign(z)
        sign[torch.where(sign==0)] = 1.0
        a = -1.0 / (sign + z)
        b = x * y * a
        b1 = torch.zeros_like(n).cuda()
        b1[:,0] = 1.0 + sign * x * x * a
        b1[:,1] = sign * b
        b1[:,2] = -sign * x

        b2 = torch.zeros_like(n).cuda()
        b2[:,0] = b
        b2[:,1] = sign + y * y * a
        b2[:,2] = -y
        return b1,b2
    def cross(self,a,b):
        # checked
        # a,b: (n_rays,3)
        return torch.cross(a,b,dim=1)
    def normalize(self,a):
        # a: (n_rays,3)
        return a / torch.linalg.norm(a,ord = 2,dim=1).reshape(-1,1)
    def dot(self,a,b):
        # checked
        return torch.sum(a * b, dim=1).reshape(-1)
    def sign(self,a):
        # checked
        sign = torch.sign(a)
        sign[torch.where(sign==0)] = 1.0
        return sign
    def sign_neg(self,a):
        sign = torch.sign(-a)
        sign[torch.where(sign==0)] = -1.0
        return sign
    def mulsign_vec(self,a,b):
        # a: (n_rays,3)
        # b: (n_rays,)
        assert a.shape[1] == 3,'error shape'
        sign = self.sign(b).unsqueeze(1).repeat(1,3) # (n_rays,3)
        return a * sign
    def mulsign(self,a,b):
        # checked
        # a: (n_rays,)
        # b: (n_rays,)

        assert len(a.shape) == 1,'error shape'
        sign = self.sign(b)
        return a * sign
    def mulsign_neg(self,a,b):
        sign = self.sign_neg(b)
        return a * sign
    def select(self,condition,true_values,false_values):
        # checked
        # condition: mask: (n_rays,)
        # true_values: (n_rays)
        condition = condition.reshape(-1)
        true_values = true_values.reshape(-1)
        false_values = false_values.reshape(-1)

        result = false_values.clone()
        result[condition] = true_values[condition]
        return result.reshape(-1)
    def sqrtz(self,a):
        # checked
        # a: float, (n_rays,)
        a_complex = torch.complex(a,torch.zeros_like(a))
        return torch.sqrt(a_complex + 1e-6)


if __name__ == '__main__' :

    ##########----test stoke_basis()----#########
    ml = Mueller()
    forward = torch.tensor([[1,0,0],[0,1,0],[0,0,1],[0.707,0.707,0]])
    a = ml.stokes_basis(forward)
    print('sokes_basis',a)

    ##########----test rotator()----#########
    a = torch.tensor([0.25*torch.pi,1*torch.pi,2*torch.pi],requires_grad=True)
    res = ml.rotator(a)
    print('rotator grad_fn:',res.grad_fn)
    print('rotator:',res)

    ##########----test select()----#########
    mask = torch.tensor([True,False,False])
    a = torch.tensor([1,2,3])
    b = torch.tensor([0.1,0.2,0.3])
    c = ml.select(mask,a,b)
    print('select:',c)

    ##########----test unit_angle()----#########
    a = torch.tensor([[1,0,0],[0,1,0],[0,0,1]])
    b = torch.tensor([[-0.707,0.707,0],[1,0,0],[0,0,-1]])
    c = ml.unit_angle(a,b)
    print('unit_angle:',c)

    ##########----test sign()----#########
    a = torch.tensor([1,1,1,1,1])
    b = torch.tensor([-0.00000001,0.0000001,0,-0.5,3])
    c = ml.mulsign(a,b)
    print('sign:',c)

    ##########----test normalize()----#########
    a = torch.tensor([[1.0,-1.0,1.0],[0.0,1.0,1.0]])
    print('normalize:',ml.normalize(a))

    ##########----test cross()----#########
    a = torch.tensor([[1.0,-1.0,1.0],[0.0,1.0,0.0]])
    b = torch.tensor([[1.0,1.0,0.0],[1.0,0.0,0.0]])
    print('cross:',ml.cross(a,b))

    ##########----test sqrtz()----#########
    a = torch.tensor([-0.5,-0.4,0.2,0.3])
    b = ml.sqrtz(a)
    print('sqrtz:',ml.sqrtz(a))
    print('sqrtz mulsign:',ml.mulsign(b,a))

    ##########----test fernel_polarized()----#########
    cos_theta_i = torch.tensor([1,1])
    eta = torch.tensor([1.52,1.52])
    a_s, a_p,cos_theta_t_signed,eta_it,eta_ti = ml.frenel_polarized(cos_theta_i,eta)
    print('fernel_polarized a_s:',a_s)
    print('fernel_polarized a_p:',a_p)
    print('fernel_polarized cos_theta_t_signed:',cos_theta_t_signed)
    print('fernel_polarized eta_it:',eta_it)
    print('fernel_polarized eta_ti:',eta_ti)

    ##########----test sincos_arg_diff()----#########
    real = torch.tensor([[0,-0.4,0.2,0.3]])
    imag = torch.tensor([[0,0.2,-0.2,0.3]])
    a = torch.complex(real,imag)
    b = torch.complex(imag,real)
    sin,cos = ml.sincos_arg_diff(a,b)
    print('sincos_arg_diff sin:',sin)
    print('sincos_arg_diff cos:',cos)


    ##########----test specular_reflection()----#########

    cos_theta_i = torch.tensor([0,1,-1,0.5,-0.5])
    eta = torch.tensor([1.52,1.52,1.52,1.52,1.52])
    res = ml.specular_reflection(cos_theta_i,eta)
    print('specular_reflection:',res)

    ##########----test specular_transmission()----#########

    cos_theta_i = torch.tensor([-0.5108,1,-1,0.5,-0.5])
    eta = torch.tensor([1.52,1.52,1.52,1.52,1.52])
    res = ml.specular_transmission(cos_theta_i,eta)
    print('specular_transmission:',res)

    ##########----test rotate_mueller_basis()----#########

    cos_theta_i = torch.tensor([0.5,0.0,-1.0])
    eta = torch.tensor([1.52,1.52,1.52])
    res = ml.specular_transmission(cos_theta_i,eta)

    # in_forward, in_basis_current, in_basis_target, out_forward, out_basis_current, out_basis_target
    in_forward = torch.tensor([[0.707,0.707,0],[-0.707,0,0.707],[0,0,1]])
    in_basis_current = torch.tensor([[-0.707,0.707,0],[-0.707,0,-0.707],[1,0,0]])
    in_basis_target = torch.tensor([[0.57735,0.57735,0.57735],[0,0,1],[0,0.707,-0.707]])
    out_forward = torch.tensor([[0,1,0],[-1,0,0],[-0.57735,0.57735,-0.57735]])
    out_basis_current = torch.tensor([[0,0,-1],[0.707,-0.707,0],[0.57735,0.57735,0.57735]])
    out_basis_target = torch.tensor([[1.0,0,0],[0,1,0],[0,0,1]])

    M = ml.rotate_mueller_basis(res,in_forward,in_basis_current,in_basis_target, out_forward, out_basis_current, out_basis_target)

    print('rotate_mueller_basis:',M)

