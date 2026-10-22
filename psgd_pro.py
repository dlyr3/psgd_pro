"""
The new PSGD-Kron Newton/Whitening preconditioners support five kinds of local coordinates for updating Q:
    Q0.5EQ1.5/Q0p5EQ1p5): dQ = Q^0.5 * mathcal{E} * Q^1.5
    The default and recommended choice for fitting Q.
    An online orthogonal Procrustes problem solver is used to keep Q approximately SPD (no need to be exactly SPD).

Please refer to
    https://github.com/lixilinx/psgd_torch/blob/master/wrapped_as_torch_optimizer_for_ddp.py
    https://github.com/lixilinx/psgd_torch/blob/master/wrapped_as_torch_optimizer_for_dtensor.py
for torch.optim optimizer wrappings for DDP, FSDP, FP, etc. trainings and typical settings.

Xi-Lin Li, lixilinx@gmail.com; last updated in Oct., 2025.
Main refs: https://arxiv.org/abs/1512.04202; https://arxiv.org/abs/2402.11858.
"""

import opt_einsum
import torch

def norm_lower_bound_spd(A, k=32, half_iters=2):
    """
    Returns a cheap lower bound for the spectral norm of a symmetric positive definite matrix A, where,
        k: the dim of subspace, suggesting 128 for bfloat16 and 32 for float32 (tested on my laptop 4070 GPU);
        half_iters: half of the number of subspace iterations, suggesting 2.
    A rough norm estimation with bfloat16 is good enough, and we don't orthonormalize the subspace vectors.

    The initial noise space V is rotated such that its centroid aligns with the largest row of A.
    Hence, each row of V and the largest row of A has an angle about acos(1/sqrt(k)) when k << dim(A).
    This feature makes the subspace iteration more robust for large matrices with very low rank.
    A simplified branchless approximate implementation is provided here.
    """
    smallest_normal = torch.finfo(A.dtype).smallest_normal
    normalizing_factor = A.diagonal().real.amax() + smallest_normal
    A = A / normalizing_factor # (complex tensor) / (subnormal number) could produce inf or nan unexpectedly
    j = torch.argmax(torch.linalg.vector_norm(A, dim=1))
    V = torch.randn(k, A.shape[1], dtype=A.dtype, device=A.device)
    V = A[j] + torch.sgn(torch.sum(A[j] * V.conj(), dim=1, keepdim=True)) * V # torch.sign for real
    for _ in range(half_iters):
        V = V @ A
        V /= torch.linalg.vector_norm(V, dim=1, keepdim=True) + smallest_normal
        V = V @ A
    return normalizing_factor * torch.amax(torch.linalg.vector_norm(V, dim=1))

def norm_lower_bound_skh(A, k=32, half_iters=2):
    """
    Returns a cheap lower bound for the spectral norm of a skew-Hermitian matrix A,
        k: the dim of subspace, suggesting 128 for bfloat16 and 32 for float32 (tested on my laptop 4070 GPU);
        half_iters: half of the number of subspace iterations, suggesting 2.
    A rough norm estimation with bfloat16 is good enough, and we don't orthonormalize the subspace vectors.

    The initial noise space V is rotated such that its centroid aligns with the largest row of A.
    Hence, each row of V and the largest row of A has an angle about acos(1/sqrt(k)) when k << dim(A).
    This feature makes the subspace iteration more robust for large matrices with very low rank.
    A simplified branchless approximate implementation is provided here.
    """
    smallest_normal = torch.finfo(A.dtype).smallest_normal
    normalizing_factor = A.abs().amax() + smallest_normal
    A = A / normalizing_factor # (complex tensor) / (subnormal number) could produce inf or nan unexpectedly
    j = torch.argmax(torch.linalg.vector_norm(A, dim=1))
    V = torch.randn(k, A.shape[1], dtype=A.dtype, device=A.device)
    V = A[j] + torch.sgn(torch.sum(A[j] * V.conj(), dim=1, keepdim=True)) * V # torch.sign for real
    for _ in range(half_iters):
        V = V @ A
        V /= torch.linalg.vector_norm(V, dim=1, keepdim=True) + smallest_normal
        V = V @ A
    return normalizing_factor * torch.amax(torch.linalg.vector_norm(V, dim=1))

def lift2single(x):
    # lift half or lower precision to single precision; leave single precision unchanged
    return x.to(torch.float32) if torch.finfo(x.dtype).eps > 1e-6 else x

def procrustes_step2(Q, max_step_size=1/8):
    """
    A in-place (update Q directly) online solver for the orthogonal Procrustes problem,
        min_U || U Q - I ||_F,   s.t. U^H U = I
    by rotating Q as exp(a R) Q, where R = Q^H - Q is the generator and ||a R|| < 1.

    We expand U = exp(a R) to its 2nd term as
        U ~ I + aR + (aR)^2/2
    and the truncation error ||U^H U - I|| is upper bounded as ||a R||^4/4.
    Set max_step_size <= 1/4 such that the truncation error <= (1/4)^4/4 < 1e-3.

    Note that U(n) is connected and such rotations can make almost any complex Q SPD except for convergence to saddle points.
    However, O(n) is not connected. Hence, such SO(n) rotations can only make real Q with det(Q) > 0 SPD.

    We have simplified the original implementation. The one branch here is necessary for line search.
    """
    R = Q.H - Q
    R /= norm_lower_bound_skh(R) + torch.finfo(R.dtype).smallest_normal # normalize R as typically it's too small
    RQ = R @ Q
    RRQ = R @ RQ
    tr_RQ = RQ.diagonal().real.sum() # tr_RQ >=0 by theory; torch.trace not implemented for CPU bfloat16, so using sum(diag(.)) here
    tr_RRQ = RRQ.diagonal().real.sum() # line search is needed if tr_RRQ < 0
    a = torch.where(tr_RRQ < 0, torch.clamp(-tr_RQ / tr_RRQ, max=max_step_size), max_step_size)
    Q.add_(a * (RQ + 0.5 * a * RRQ))

def balance_kron_precond(Q):
    """
    In place balancing the dynamic ranges of the factors of Q to avoid over/under-flow.
    """
    order = len(Q)  # order of tensor or the number of factors in Q
    if order>1:
        norms = [torch.max(torch.abs(q)) for q in Q]
        gmean = torch.prod(torch.stack(norms))**(1/order) # geometric mean
        for i, q in enumerate(Q):
            q.mul_(gmean/norms[i])

def precond_grad_kron(QL, exprs, G):
    """
    Precondition gradient G with Kron preconditioner Q.
    """
    Q, exprP = QL[0], exprs[0]
    return exprP(*[q.conj() for q in Q], *Q, G)

def update_precond_kron_whiten_q0p5eq1p5(QL, exprs, G, lr=0.1, betaL=0.9, damping=1e-9):
    """
    Update the Kron preconditioner Q as dQ = Q^0.5 * E * Q^1.5.
    """
    Q, L = QL
    exprP, exprGs = exprs

    total_numel = G.numel()
    damping = damping + torch.finfo(G.dtype).eps * G.abs()
    Pg = exprP(*[q.conj() for q in Q], *Q, G + damping*torch.randn_like(G))
    for i, q in enumerate(Q):
        term1 = exprGs[i](Pg, Pg.conj())
        if q.dim() < 2: # diagonal or scalar Q
            term2 = total_numel/q.numel() # times I
            ell = torch.max(torch.real(term1)) + term2
            L[i].copy_(torch.max(betaL*L[i] + (1 - betaL)*ell, ell))
            q.mul_(1 - lr/L[i] * (term1 - term2))
        else: # matrix Q
            term2 = total_numel/q.shape[0] # times I
            ell = norm_lower_bound_spd(term1) + term2
            L[i].copy_(torch.max(betaL*L[i] + (1 - betaL)*ell, ell))
            q.sub_(lr/L[i] * (term1 @ q - term2 * q))
            procrustes_step2(q)

    if torch.rand([]) < 0.01: # balance factors of Q
        balance_kron_precond(Q)

def init_kron(t, Scale=1.0, max_size=float("inf"), max_skew=1.0, dQ="Q0.5EQ1.5"):
    """
    For a scalar or tensor t, we initialize its states (preconditioner Q and Lipschitz smoothness constant L),
    and reusable contraction expressions for updating Q and preconditioning gradient.

    1, The preconditioner Q is initialized to
        Q = Scale * I = Scale * kron(eye(t.shape[0]), eye(t.shape[1]), ...)
       where the eye(.) may be replaced with diag(ones(.)) if that dim is too large, determined by max_size and max_skew.

       The Lipschitz smoothness constant L for Q is initialized to zero.

    2, A series of einsum contract expressions. The following subscript examples are for a 5th order tensor.
        2.1, exprP is the expression for applying the Preconditioner on the gradient, e.g.,
                'aA,bB,cC,dD,eE,aα,bβ,cγ,dδ,eε,αβγδε->ABCDE'
        2.2, the i-th expression of exprGs is for the contraction of two tensors that only keeps the i-th dim, e.g.,
                'abCde,abγde->Cγ'
            for i=2. It's useful for Gradient calculation.
        2.3, exprA is the expression for applying All the factors of Q on a tensor, e.g.,
                'aA,bB,cC,dD,eE,ABCDE->abcde'
        2.4, the i-th expression of exprQs is the expression for applying the i-th factor of Q on a tensor, e.g.,
                'Cγ,abγde->abCde'
            for i=2.

        Please check https://drive.google.com/file/d/1CEEq7A3_l8EcPEDa_sYtqr5aMLVeZWL7/view?usp=drive_link for notations and derivations.
    """
    if dQ in {"QUAD4P", "PRO4P"}:  # the only two cases that we fit P directly; so square Scale
        Scale = Scale ** 2
    shape = t.shape
    if len(shape) == 0:  # scalar
        Q = [Scale * torch.ones_like(t), ]
        L = [lift2single(torch.zeros_like(t.real)), ]
        exprA = opt_einsum.contract_expression(",->", Q[0].shape, t.shape)
        exprP = opt_einsum.contract_expression(",,->", Q[0].shape, Q[0].shape, t.shape)
        exprGs = [opt_einsum.contract_expression(",->", t.shape, t.shape), ]
        exprQs = [opt_einsum.contract_expression(",->", Q[0].shape, t.shape), ]
    else:  # tensor
        if len(shape) > 26:
            raise ValueError(
                f"Got tensor with dim {len(t.shape)}; einsum runs out of letters; replace 26 with larger numbers.")

        scale = Scale ** (1 / len(shape))

        Q, L = [], []
        exprGs, exprQs = [], []
        piece1A, piece2A, piece3A = [], "", ""  # used for getting the subscripts for exprA
        piece1P, piece2P, piece3P, piece4P = [], [], "", ""  # used for getting the subscripts for exprP
        for i, size in enumerate(shape):
            L.append(lift2single(torch.zeros([], dtype=t.real.dtype, device=t.device)))
            if size <= 1 or size > max_size or size ** 2 > max_skew * t.numel():
                # use diagonal matrix as preconditioner for this dim
                Q.append(scale * torch.ones(size, dtype=t.dtype, device=t.device))

                piece1A.append(opt_einsum.get_symbol(i))
                piece2A = piece2A + opt_einsum.get_symbol(i)
                piece3A = piece3A + opt_einsum.get_symbol(i)

                piece1P.append(opt_einsum.get_symbol(i + 26))
                piece2P.append(opt_einsum.get_symbol(i + 26))
                piece3P = piece3P + opt_einsum.get_symbol(i + 26)
                piece4P = piece4P + opt_einsum.get_symbol(i + 26)

                piece1 = "".join(
                    [opt_einsum.get_symbol(i + 26) if j == i else opt_einsum.get_symbol(j) for j in range(len(shape))])
                subscripts = piece1 + "," + piece1 + "->" + opt_einsum.get_symbol(i + 26)
                exprGs.append(opt_einsum.contract_expression(subscripts, t.shape, t.shape))

                subscripts = opt_einsum.get_symbol(i + 26) + "," + piece1 + "->" + piece1
                exprQs.append(opt_einsum.contract_expression(subscripts, Q[-1].shape, t.shape))
            else:  # use matrix preconditioner for this dim
                Q.append(scale * torch.eye(size, dtype=t.dtype, device=t.device))

                piece1A.append(opt_einsum.get_symbol(i) + opt_einsum.get_symbol(i + 26))
                piece2A = piece2A + opt_einsum.get_symbol(i + 26)
                piece3A = piece3A + opt_einsum.get_symbol(i)

                a, b, c = opt_einsum.get_symbol(i), opt_einsum.get_symbol(i + 26), opt_einsum.get_symbol(i + 805)
                piece1P.append(a + b)
                piece2P.append(a + c)
                piece3P = piece3P + c
                piece4P = piece4P + b

                piece1 = "".join(
                    [opt_einsum.get_symbol(i + 26) if j == i else opt_einsum.get_symbol(j) for j in range(len(shape))])
                piece2 = "".join(
                    [opt_einsum.get_symbol(i + 805) if j == i else opt_einsum.get_symbol(j) for j in range(len(shape))])
                subscripts = piece1 + "," + piece2 + "->" + opt_einsum.get_symbol(i + 26) + opt_einsum.get_symbol(
                    i + 805)
                exprGs.append(opt_einsum.contract_expression(subscripts, t.shape, t.shape))

                subscripts = opt_einsum.get_symbol(i + 26) + opt_einsum.get_symbol(
                    i + 805) + "," + piece2 + "->" + piece1
                exprQs.append(opt_einsum.contract_expression(subscripts, Q[-1].shape, t.shape))

        subscripts = ",".join(piece1A) + "," + piece2A + "->" + piece3A
        exprA = opt_einsum.contract_expression(subscripts, *[q.shape for q in Q], t.shape)

        subscripts = ",".join(piece1P) + "," + ",".join(piece2P) + "," + piece3P + "->" + piece4P
        exprP = opt_einsum.contract_expression(subscripts, *[q.shape for q in Q], *[q.shape for q in Q], t.shape)

    exprGs, exprQs = tuple(exprGs), tuple(exprQs)
    if dQ == "QEP":
        return [[Q, L], (exprP, exprGs, exprQs)]
    elif dQ == "EQ":
        return [[Q, L], (exprP, exprGs, exprA)]
    elif dQ in {"QEQ", "QUAD", "Q0p5EQ1p5", "Q0.5EQ1.5"}:
        return [[Q, L], (exprP, exprGs)]
    else:  # the only two cases that we fit P directly; dQ actually is dP
        assert dQ in {"QUAD4P", "PRO4P"}, "Invalid choice for dQ"
        return [[Q, L], (exprA, exprGs)]

class KronWhiten:
    """
    Implements the PSGD optimizer with the Kronecker product gradient/momentum whitening preconditioner.
    Most of the time, the hyperparameter name says it all. Here are some comments on a few key hyperparameters.

    1, preconditioner_max_size and preconditioner_max_skew. These two together control the complexity of the preconditioners.
    For example, we are to precondition a 2D gradient with shape 10 x 50.
    With preconditioner_max_size 20, we use a dense preconditioner for the first dim since 10 <= 20 and diagonal preconditioner for the second dim since 50 > 20.
    With preconditioner_max_skew 1.5, we use a dense preconditioner for the first dim since 10/50 <= 1.5 and diagonal preconditioner for the second dim since 50/10 > 1.5.

    2, grad_clip_max_amps, betaL and damping. These three together help to stabilize the training.
    PSGD here tries to normalize the gradients to unit amplitude. This can be problematic when gradients approach zeros.
    The most effective way is to clip the preconditioned gradients if their average/element-wise amplitudes exceed grad_clip_max_amps[0]/[1], respectively.
    Another way is to damp and upper bound the fitted preconditioner such that P < eye/damping.
    For extremely sparse gradients, increasing betaL (say to 0.999) helps a lot, where betaL is the EMA factor for the L-smoothness constant (wrt Q) estimation.

    3, Lastly, dQ is for the selection of geometry for preconditioner update.
    The two recommended choices are dQ = Q0.5EQ1.5 and dP = P0.5EP (online Newton-Schulz iterations).
    Q is initialized to preconditioner_init_scale * eye. Boolean setting whiten_grad decides to whiten whether the gradient or momentum.
    Always good to check https://arxiv.org/abs/2402.11858 for math details.
    """
    def __init__(self,  params_with_grad,
                 preconditioner_max_size=float("inf"),
                 preconditioner_max_skew=1.0,
                 preconditioner_init_scale:float|None=None,
                 lr_params=0.001,
                 lr_preconditioner=0.1,
                 betaL=0.9,
                 damping=1e-9,
                 momentum=0.0,
                 grad_clip_max_amps=(2.0, 10.0),
                 preconditioner_update_probability=1.0,
                 update_preconditioner_first=True,
                 whiten_grad=True):

        # mutable members
        self.lr_params = lr_params
        self.lr_preconditioner = lr_preconditioner
        self.betaL = betaL # beta for the Lipschitz smoothness constant estimation; set to a large value for sparse gradients
        self.damping = damping # to damp and upper bound the preconditioner such that P < eye/damping
        self.momentum = momentum if (0<momentum<1) else 0.0
        self.grad_clip_max_amps = grad_clip_max_amps # clip grad with thresholds (max average amplitude, max element-wise amplitude)
        self.preconditioner_update_probability = preconditioner_update_probability
        self.update_preconditioner_first = update_preconditioner_first # True for biased update; False for unbiased update.

        #region Protected members
        self._preconditioner_max_size = preconditioner_max_size
        self._preconditioner_max_skew = preconditioner_max_skew
        params_with_grad = [params_with_grad,] if isinstance(params_with_grad, torch.Tensor) else params_with_grad
        self._params_with_grad = [param for param in params_with_grad if param.requires_grad] # double check requires_grad flag

        if preconditioner_init_scale is None:
            self._QLs_exprs = None # initialize on the fly
            print("FYI: Will set the preconditioner initial scale on the fly. Recommend to set it manually.")
        else:
            self._QLs_exprs = [init_kron(p.squeeze(), preconditioner_init_scale, preconditioner_max_size, preconditioner_max_skew) for p in self._params_with_grad]
        self._ms, self._counter_m = None, 0 # momentum buffers and counter

        self._whiten_grad = whiten_grad # set to False to whiten momentum.
        if not whiten_grad:
            assert self.momentum > 0, "Cannot whiten momentum if the momentum setting is invalid."
            print(f"Recommend reducing the lr_params for gradient whitening by a factor of {((1 + self.momentum)/(1 - self.momentum))**0.5} for this momentum whitening setting.")

        # default to dQ="Q0.5EQ1.5"
        self._update_precond = update_precond_kron_whiten_q0p5eq1p5
        self._precond_grad = precond_grad_kron
        #endregion

    @torch.no_grad()
    def step(self, closure):
        """
        Performs one step of PSGD with the Kronecker product gradient/momentum whitening preconditioner.
        """
        with torch.enable_grad():
            closure_returns = closure()
            loss = closure_returns if isinstance(closure_returns, torch.Tensor) else closure_returns[0]
            grads = [g.squeeze() for g in torch.autograd.grad(loss, self._params_with_grad)]

        if self._QLs_exprs is None:
            scale = max([torch.mean((torch.abs(g))**4) for g in grads])
            scale = (scale + self.damping**4)**(-1/8)
            self._QLs_exprs = [init_kron(g, scale, self._preconditioner_max_size, self._preconditioner_max_skew) for g in grads]

        if self.momentum > 0:
            beta = min(self._counter_m/(1 + self._counter_m), self.momentum)
            self._counter_m += 1
            if self._ms is None:
                self._ms = [torch.zeros_like(g) for g in grads]

            for (m, g) in zip(self._ms, grads):
                m.mul_(beta).add_(g, alpha=1 - beta)
        else:
            self._ms, self._counter_m = None, 0

        if torch.rand([]) < self.preconditioner_update_probability:
            update_preconditioner_first, update_preconditioner_last = self.update_preconditioner_first, not self.update_preconditioner_first
        else:
            update_preconditioner_first, update_preconditioner_last = False, False

        if update_preconditioner_first: # update Q
            if self._whiten_grad: # Q whitens gradient
                for (QL_exprs, g) in zip(self._QLs_exprs, grads):
                    self._update_precond(*QL_exprs, g, lr=self.lr_preconditioner, betaL=self.betaL, damping=self.damping)
            else: # Q whitens momentum
                for (QL_exprs, m) in zip(self._QLs_exprs, self._ms):
                    self._update_precond(*QL_exprs, m, lr=self.lr_preconditioner, betaL=self.betaL, damping=self.damping)

        if self.momentum > 0: # precondition momentum
            pre_grads = [self._precond_grad(*QL_exprs, m) for (QL_exprs, m) in zip(self._QLs_exprs, self._ms)]
        else: # precondition gradient
            pre_grads = [self._precond_grad(*QL_exprs, g) for (QL_exprs, g) in zip(self._QLs_exprs, grads)]

        if update_preconditioner_last: # update Q
            if self._whiten_grad: # Q whitens gradient
                for (QL_exprs, g) in zip(self._QLs_exprs, grads):
                    self._update_precond(*QL_exprs, g, lr=self.lr_preconditioner, betaL=self.betaL, damping=self.damping)
            else: # Q whitens momentum
                for (QL_exprs, m) in zip(self._QLs_exprs, self._ms):
                    self._update_precond(*QL_exprs, m, lr=self.lr_preconditioner, betaL=self.betaL, damping=self.damping)

        # Update the parameters after clipping the preconditioned gradient per tensor
        max_avg_amp, max_element_amp = self.grad_clip_max_amps
        for param, g in zip(self._params_with_grad, pre_grads):
            avg_amp = torch.sqrt(torch.real(torch.mean(g*g.conj())))
            if avg_amp > max_avg_amp:
                g *= max_avg_amp/avg_amp
            if torch.is_complex(g):
                g /= torch.clamp(torch.abs(g)/max_element_amp, min=1.0)
            else:
                g.clamp_(min=-max_element_amp, max=max_element_amp)
            param.subtract_(g.view_as(param), alpha=self.lr_params)

        # return whatever closure returns
        return closure_returns