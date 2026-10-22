import matplotlib.pyplot as plt
import psgd_ref
import psgd_pro
import torch

def loss(x : torch.Tensor):
    return torch.sum(x)

#dtype d double
torch.manual_seed(0)
xdref = torch.randn(100, dtype=torch.double, requires_grad=True)
xsref = xdref.to(torch.float)

torch.manual_seed(0)
xdpro = torch.randn(100, dtype=torch.double, requires_grad=True)
xspro = xdpro.to(torch.float)

x_list = [xdref, xdpro, xsref, xspro]

# create the optimizer with seed for damping noise rng
opt_dref = psgd_ref.KronWhiten(xdref)
opt_dpro = psgd_pro.KronWhiten(xdpro)
opt_sref = psgd_ref.KronWhiten(xsref)
opt_spro = psgd_pro.KronWhiten(xspro)

opt_list = [opt_dref, opt_dpro, opt_sref, opt_spro]

# step
for x, opt in zip(x_list, opt_list):
    torch.manual_seed(0)
    opt.step(lambda : loss(x))

# unit test type integrity
assert (xdref.dtype == xdpro.dtype == torch.float64)
assert (xsref.dtype == xspro.dtype == torch.float32)

# bitwise equality
print(torch.equal(xdref, xdpro)) # double ref & double pro equal?
print(torch.equal(xsref, xspro)) # simple ref & simple pro equal?

# floor
print((torch.sum(xdref) - torch.sum(xspro)).item())
print((torch.sum(xdref) - torch.sum(xsref)).item())

# average diff
print((torch.sum(xdref) - torch.sum(xdpro)).item())