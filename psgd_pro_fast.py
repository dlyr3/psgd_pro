"""
PSGD Pro (KronWhiten with dQ = "Q0.5EQ1.5", real parameters) restructured so that the optimizer math compiles with
torch.compile(fullgraph=True). In eager mode it is bitwise identical to psgd_pro.py (tests/test_fast.py):
  1. The opt_einsum expressions are executed by `Path`, which replays opt_einsum's precomputed contraction list with
     the torch.tensordot / permute / torch.einsum calls that opt_einsum's torch backend makes. Dynamo cannot trace
     opt_einsum itself (its backend lookup goes through a functools.lru_cache, which Dynamo skips).
  2. The coin flips (update probability, factor balancing) are drawn on the CPU before the math, in the original order.
     Balancing runs eagerly after the update, only for the parameters whose coin fired; parameters are independent,
     so the order change does not alter any value.
  3. The Gaussian draws (damping noise, the two subspace-iteration starts per dense factor) are made eagerly before
     the update, with the same torch calls in the same order, and passed in. Inside a compiled graph Inductor does not
     keep random ops in program order, so drawing them there gives different values from the same seed.
  4. The row gather A[j] in the norm bounds is index_select (Dynamo turns a 0-d tensor index into a data-dependent int).
  5. The clipping branch `if avg_amp > max_avg_amp` (a GPU-to-CPU sync per parameter) is a multiply by
     clamp(max_avg_amp / avg_amp, max=1), which is exactly 1.0 whenever the branch would not fire.
Compiled mode passes the momentum factor and lr_params as 0-d tensors, so annealing them does not recompile.
`ops` lets custom kernels replace the dense-factor arithmetic (Gram matrix, norm bounds, term1 @ q; see kernels/).
batch=True (not bitwise, same math) stacks the state of same-shape parameters and runs them under torch.vmap;
batch="update" batches only the preconditioner update and keeps the preconditioning per parameter.
cache_p=True (eager, bitwise) reuses P_i = Q_i^T Q_i between calls while Q_i is unchanged.
graph_rng=True replays the Gaussian draws from a captured CUDA graph (same values, one launch).
p2d=True (same math, not bitwise) runs 2-D parameters through update_2d/apply_2d with P_i = Q_i^T Q_i kept as explicit
state (works under compile and CUDA graphs); r2=True uses procrustes_step_r2 there.
"""
import torch
from opt_einsum.parser import convert_to_valid_einsum_chars
from psgd_pro import balance_kron_precond, init_kron

K = 32  # subspace dimension of the norm lower bounds (psgd_pro default)


def _norm_lower_bound(A, normalizing_factor, V, half_iters=2):
    """Shared body of psgd_pro.norm_lower_bound_spd/_skh with the random start V passed in."""
    smallest_normal = torch.finfo(A.dtype).smallest_normal
    A = A / normalizing_factor
    j = torch.argmax(torch.linalg.vector_norm(A, dim=1))
    Aj = A.index_select(0, j.view(1))
    V = Aj + torch.sgn(torch.sum(Aj * V.conj(), dim=1, keepdim=True)) * V
    for _ in range(half_iters):
        V = V @ A
        V /= torch.linalg.vector_norm(V, dim=1, keepdim=True) + smallest_normal
        V = V @ A
    return normalizing_factor * torch.amax(torch.linalg.vector_norm(V, dim=1))


def norm_lower_bound_spd(A, V):
    return _norm_lower_bound(A, A.diagonal().real.amax() + torch.finfo(A.dtype).smallest_normal, V)


def norm_lower_bound_skh(A, V):
    return _norm_lower_bound(A, A.abs().amax() + torch.finfo(A.dtype).smallest_normal, V)


class Path:
    """An opt_einsum ContractExpression replayed with the same torch calls (see contract.py: _core_contract).
    With nq (exprP of nq factors), the steps that form a Gram matrix Q_i^H Q_i are recorded, so that a caller can
    supply them (gram=...) instead of the tensordot."""

    def __init__(self, expr, nq=None):
        self.steps, self.gram_steps = [], {}
        names = [("c", i) for i in range(nq)] + [("q", i) for i in range(nq)] + [("g", 0)] if nq else None
        for k, (inds, idx_rm, einsum_str, _, blas) in enumerate(expr.contraction_list):
            inp, res = einsum_str.split("->")
            if blas and "EINSUM" not in blas:
                left, right = inp.split(",")
                axes = tuple(zip(*sorted((left.find(s), right.find(s)) for s in idx_rm))) if idx_rm else ((), ())
                tres = "".join(s for s in left + right if s not in idx_rm)
                self.steps.append((inds, axes, tuple(map(tres.index, res)) if tres != res else None, None))
            else:
                self.steps.append((inds, None, None, convert_to_valid_einsum_chars(einsum_str)))
            if names is not None:
                popped = [names.pop(x) for x in inds]
                ops_str = inp.split(",")
                if (len(popped) == 2 and {popped[0][0], popped[1][0]} == {"c", "q"} and popped[0][1] == popped[1][1]
                        and all(len(o) == 2 for o in ops_str) and set(idx_rm) == {ops_str[0][0]} == {ops_str[1][0]}):
                    self.gram_steps[k] = [x[0] for x in popped].index("q")  # position of the plain Q operand
                names.append(("x", k))

    def __call__(self, *operands, gram=None, cache=None):
        """cache (eager only): dict reused across calls; a Gram step is skipped when its Q is the same tensor at the
        same in-place version (Q unchanged since), so the result is bitwise the one the step would compute."""
        ops = list(operands)
        for k, (inds, axes, perm, eq) in enumerate(self.steps):
            tmp = [ops.pop(x) for x in inds]
            if cache is not None and k in self.gram_steps:
                q = tmp[self.gram_steps[k]]
                hit = cache.get((id(self), k, id(q)))
                if hit is not None and hit[0] is q and hit[1] == q._version:
                    ops.append(hit[2])
                    continue
            if gram is not None and k in self.gram_steps:
                v = gram(tmp[0])  # Q_i^T Q_i, exactly symmetric, so the step's permute does not matter
            elif eq is None:
                v = torch.tensordot(*tmp, dims=axes)
                if perm is not None:
                    v = v.permute(*perm)
            else:
                v = torch.einsum(eq, tmp)
            if cache is not None and k in self.gram_steps:
                q = tmp[self.gram_steps[k]]
                cache[(id(self), k, id(q))] = (q, q._version, v)
            ops.append(v)
        return ops[0]


class EagerOps:
    """The original arithmetic. Kernel-backed replacements implement the same methods."""
    gram = None    # None: term1 from the opt_einsum path; else gram(Pg, i, expr) -> term1 of factor i
    gram_q = None  # None: P_i = Q_i^T Q_i inside exprP by tensordot; else gram_q(q) -> Q_i^T Q_i
    norm_lower_bound_spd = staticmethod(norm_lower_bound_spd)
    norm_lower_bound_skh = staticmethod(norm_lower_bound_skh)

    @staticmethod
    def symm(A, q):  # S @ X, S symmetric in storage form (term1 @ q; P_1 @ X in the 2-D path)
        return A @ q

    @staticmethod
    def syrk(X, trans):  # full symmetric X^T X (trans) or X X^T
        return X.T @ X if trans else X @ X.T

    # storage form of term1 and P in the 2-D path (B12: kernels may keep one triangle or a packed triangle)
    sym = syrk

    @staticmethod
    def symm_r(X, S):  # X @ S, S symmetric in storage form
        return X @ S


def procrustes_step2(Q, V, max_step_size=1/8, ops=EagerOps):
    """psgd_pro.procrustes_step2 with the norm bound's random start V passed in."""
    R = Q.H - Q
    R /= ops.norm_lower_bound_skh(R, V) + torch.finfo(R.dtype).smallest_normal
    RQ = R @ Q
    RRQ = R @ RQ
    tr_RQ = RQ.diagonal().real.sum()
    tr_RRQ = RRQ.diagonal().real.sum()
    a = torch.where(tr_RRQ < 0, torch.clamp(-tr_RQ / tr_RRQ, max=max_step_size), max_step_size)
    Q.add_(a * (RQ + 0.5 * a * RRQ))


def procrustes_step_r2(Q, V, max_step_size=1/8, ops=EagerOps):
    """procrustes_step2 with R R formed once by a symmetric rank-n product: 3 n^3 FLOPs instead of 4 n^3, the same
    algebra with a different rounding order. Works on Rn = Q - Q^T = -R (row-major like Q; its norm bound equals R's
    bitwise, the sign cancels through the random start), S = Rn^T Rn = -R R:
      tr(R Q) = sum(Rn * Q) (R skew), tr(R R Q) = -sum(S * Q) (S symmetric),
      Q + a R Q + a^2/2 R R Q = Q - (a Rn + a^2/2 S) Q."""
    Rn = Q - Q.T
    Rn /= ops.norm_lower_bound_skh(Rn, V) + torch.finfo(Rn.dtype).smallest_normal
    S = ops.syrk(Rn, True)
    tr_RQ = (Rn * Q).sum()
    tr_RRQ = -(S * Q).sum()
    a = torch.where(tr_RRQ < 0, torch.clamp(-tr_RQ / tr_RRQ, max=max_step_size), max_step_size)
    Q.sub_(S.mul_(0.5 * a).add_(Rn).mul_(a) @ Q)


def apply_2d(P, X, Q, ops=EagerOps):
    """P_1 X P_2 for a 2-D tensor; P_i is the Gram matrix of a dense factor (in ops' storage form: full, upper triangle
    or packed) or the squared diagonal factor."""
    (P1, P2), (q1, q2) = P, Q
    X = ops.symm(P1, X) if q1.dim() == 2 else P1[:, None] * X
    return ops.symm_r(X, P2) if q2.dim() == 2 else X * P2


def gram_2d(q, ops):
    return ops.sym(q, True) if q.dim() == 2 else q * q


def update_2d(QL, P, G, rnd, lr=0.1, betaL=0.9, damping=1e-9, ops=EagerOps, r2=False):
    """update_precond_kron_whiten_q0p5eq1p5 for a 2-D gradient with P = [Q_1^T Q_1, Q_2^T Q_2] kept as state: the
    preconditioned gradient uses the cached P (Q is unchanged since P was formed), term1 is a symmetric rank-k product,
    and P is refreshed after the update. Same draws, same order; r2 selects procrustes_step_r2."""
    Q, L = QL
    rnd = iter(rnd)
    total_numel = G.numel()
    damping = damping + torch.finfo(G.dtype).eps * G.abs()
    Pg = apply_2d(P, G + damping*next(rnd), Q, ops)
    for i, q in enumerate(Q):
        if q.dim() < 2:
            term1 = (Pg * Pg).sum(1 - i)
            term2 = total_numel/q.numel()
            ell = torch.max(term1) + term2
            L[i].copy_(torch.max(betaL*L[i] + (1 - betaL)*ell, ell))
            q.mul_(1 - lr/L[i] * (term1 - term2))
        else:
            term1 = ops.sym(Pg, i == 1)
            term2 = total_numel/q.shape[0]
            ell = ops.norm_lower_bound_spd(term1, next(rnd)) + term2
            L[i].copy_(torch.max(betaL*L[i] + (1 - betaL)*ell, ell))
            q.sub_(lr/L[i] * (ops.symm(term1, q) - term2 * q))
            (procrustes_step_r2 if r2 else procrustes_step2)(q, next(rnd), ops=ops)
    for p, q in zip(P, Q):
        p.copy_(gram_2d(q, ops))


def precond_grad_kron(QL, exprs, G, ops=EagerOps, cache=None):
    Q, exprP = QL[0], exprs[0]
    return exprP(*[q.conj() for q in Q], *Q, G, gram=ops.gram_q, cache=cache)


def draw_normals(Q, G):
    """The Gaussian draws of one update_precond_kron_whiten_q0p5eq1p5 call, in the original order and shapes."""
    out = [torch.randn_like(G)]
    for q in Q:
        if q.dim() == 2:
            out += [torch.randn(K, q.shape[0], dtype=q.dtype, device=q.device) for _ in range(2)]  # spd, then skh
    return out


def update_precond_kron_whiten_q0p5eq1p5(QL, exprs, G, rnd, lr=0.1, betaL=0.9, damping=1e-9, ops=EagerOps, cache=None):
    """psgd_pro.update_precond_kron_whiten_q0p5eq1p5 with the draws rnd = draw_normals(Q, G) passed in and
    without the balancing coin (drawn by the caller)."""
    Q, L = QL
    exprP, exprGs = exprs
    rnd = iter(rnd)

    total_numel = G.numel()
    damping = damping + torch.finfo(G.dtype).eps * G.abs()
    Pg = exprP(*[q.conj() for q in Q], *Q, G + damping*next(rnd), gram=ops.gram_q, cache=cache)
    for i, q in enumerate(Q):
        if q.dim() < 2:  # diagonal or scalar Q
            term1 = exprGs[i](Pg, Pg.conj())
            term2 = total_numel/q.numel()
            ell = torch.max(torch.real(term1)) + term2
            L[i].copy_(torch.max(betaL*L[i] + (1 - betaL)*ell, ell))
            q.mul_(1 - lr/L[i] * (term1 - term2))
        else:  # matrix Q
            term1 = exprGs[i](Pg, Pg.conj()) if ops.gram is None else ops.gram(Pg, i, exprGs[i])
            term2 = total_numel/q.shape[0]
            ell = ops.norm_lower_bound_spd(term1, next(rnd)) + term2
            L[i].copy_(torch.max(betaL*L[i] + (1 - betaL)*ell, ell))
            q.sub_(lr/L[i] * (ops.symm(term1, q) - term2 * q))
            if getattr(ops, "qsym", False):  # Q-sym (fact 6): exact symmetrization instead of the Procrustes step
                next(rnd)
                q.copy_((q + q.H) / 2)
            else:
                procrustes_step2(q, next(rnd), ops=ops)


class KronWhiten:
    """psgd_pro.KronWhiten with the same arguments, plus
         compile: None for eager, a torch.compile mode ("default", "reduce-overhead", ...) or a dict of Inductor options;
         ops: arithmetic for the dense-factor update (EagerOps or a kernel-backed class).
       step(closure) behaves as the original; step_math(grads) is the part after the gradients exist."""

    def __init__(self, params_with_grad,
                 preconditioner_max_size=float("inf"),
                 preconditioner_max_skew=1.0,
                 preconditioner_init_scale: float | None = None,
                 lr_params=0.001,
                 lr_preconditioner=0.1,
                 betaL=0.9,
                 damping=1e-9,
                 momentum=0.0,
                 grad_clip_max_amps=(2.0, 10.0),
                 preconditioner_update_probability=1.0,
                 update_preconditioner_first=True,
                 whiten_grad=True,
                 compile=None,
                 ops=EagerOps,
                 batch=False,
                 cache_p=False,
                 graph_rng=False,
                 p2d=False,
                 r2=False):
        self.lr_params = lr_params
        self.lr_preconditioner = lr_preconditioner
        self.betaL = betaL
        self.damping = damping
        self.momentum = momentum if (0 < momentum < 1) else 0.0
        self.grad_clip_max_amps = grad_clip_max_amps
        self.preconditioner_update_probability = preconditioner_update_probability
        self.update_preconditioner_first = update_preconditioner_first

        self._preconditioner_max_size = preconditioner_max_size
        self._preconditioner_max_skew = preconditioner_max_skew
        params_with_grad = [params_with_grad, ] if isinstance(params_with_grad, torch.Tensor) else params_with_grad
        self._params_with_grad = [param for param in params_with_grad if param.requires_grad]
        self._QLs_exprs = None
        if preconditioner_init_scale is not None:
            self._init([p.squeeze() for p in self._params_with_grad], preconditioner_init_scale)
        self._ms, self._counter_m = None, 0
        self._whiten_grad = whiten_grad
        if not whiten_grad:
            assert self.momentum > 0, "Cannot whiten momentum if the momentum setting is invalid."
        self.ops, self.capture, self.pre = ops, False, []
        self._scalars, self._static, self._batch, self._groups = {}, False, batch, None
        assert not (cache_p and compile is not None), "cache_p relies on tensor version counters: eager only"
        self._pcache = {} if cache_p else None
        self._graph_rng, self._rng_graph, self._rng_out = graph_rng, None, None
        self._gbuf = None
        assert not (p2d and batch), "p2d keeps per-parameter P state; batch stacks parameters"
        self._p2d, self._r2, self._P = p2d, r2, None
        self._compiled = compile is not None
        if self._compiled:
            opts = torch._inductor.list_mode_options(compile) if isinstance(compile, str) else compile
            c = lambda f: torch.compile(f, fullgraph=True, dynamic=False, options=opts)
            self._momentum_fn, self._update_fn, self._apply_fn = c(self._momentum), c(self._update_all), c(self._precond_apply)
        else:
            self._momentum_fn, self._update_fn, self._apply_fn = self._momentum, self._update_all, self._precond_apply

    def _init(self, grads, scale):
        self._QLs_exprs = []
        for g in grads:
            (Q, L), (exprP, exprGs) = init_kron(g, scale, self._preconditioner_max_size, self._preconditioner_max_skew)
            self._QLs_exprs.append([[Q, L], (Path(exprP, len(Q)), tuple(Path(e) for e in exprGs))])

    def _scalar(self, name, value, like):
        """0-d device tensor holding value, refilled only when value changes (compiled mode)."""
        t = self._scalars.get(name)
        if t is None:
            t = self._scalars[name] = [torch.empty([], dtype=torch.float32, device=like.device), None]
            torch._dynamo.mark_static_address(t[0])
        if t[1] != value:
            t[0].fill_(value)
            t[1] = value
        return t[0]

    @staticmethod
    def _momentum(ms, grads, beta):
        if isinstance(beta, torch.Tensor):
            for m, g in zip(ms, grads):
                m.mul_(beta).add_(g * (1 - beta))
        else:
            for m, g in zip(ms, grads):
                m.mul_(beta).add_(g, alpha=1 - beta)

    def _build_groups(self, grads):
        """batch=True: parameters with the same shape and factor structure share stacked Q, L and momentum storage
        (the per-parameter tensors become views), so their update and preconditioning run once under torch.vmap."""
        sig = {}
        for i, (QL_exprs, g) in enumerate(zip(self._QLs_exprs, grads)):
            key = (tuple(g.shape), tuple(q.dim() for q in QL_exprs[0][0])) if self._batch else i
            sig.setdefault(key, []).append(i)
        self._groups = []
        for idx in sig.values():
            if len(idx) == 1:
                self._groups.append((idx, None))
                continue
            Q0, L0 = self._QLs_exprs[idx[0]][0]
            st = dict(Q=[torch.stack([self._QLs_exprs[i][0][0][f] for i in idx]) for f in range(len(Q0))],
                      L=[torch.stack([self._QLs_exprs[i][0][1][f] for i in idx]) for f in range(len(L0))],
                      m=torch.stack([self._ms[i] for i in idx]) if self._ms is not None else None)
            for b, i in enumerate(idx):
                for f in range(len(Q0)):
                    self._QLs_exprs[i][0][0][f] = st["Q"][f][b]
                    self._QLs_exprs[i][0][1][f] = st["L"][f][b]
                if st["m"] is not None:
                    self._ms[i] = st["m"][b]
            self._groups.append((idx, st))

    def _update_all(self, Gs, rnds):
        kw = dict(lr=self.lr_preconditioner, betaL=self.betaL, damping=self.damping, ops=self.ops, cache=self._pcache)
        for idx, st in self._groups:
            if st is None and self._P[idx[0]] is not None:
                update_2d(self._QLs_exprs[idx[0]][0], self._P[idx[0]], Gs[idx[0]], rnds[idx[0]], lr=kw["lr"],
                          betaL=kw["betaL"], damping=kw["damping"], ops=self.ops, r2=self._r2)
                continue
            if st is None:
                update_precond_kron_whiten_q0p5eq1p5(*self._QLs_exprs[idx[0]], Gs[idx[0]], rnds[idx[0]], **kw)
                continue
            exprs = self._QLs_exprs[idx[0]][1]
            G = torch.stack([Gs[i] for i in idx])
            R = [torch.stack([rnds[i][k] for i in idx]) for k in range(len(rnds[idx[0]]))]

            def one(Q, L, g, r):
                update_precond_kron_whiten_q0p5eq1p5([Q, L], exprs, g, r, **kw)
                return g
            torch.vmap(one)(st["Q"], st["L"], G, R)

    def _draws(self, Gs):
        """The update's Gaussian draws, in the original order. graph_rng=True captures them once in a CUDA graph and
        replays it: one launch instead of one per draw, same values (tests/test_fast.py), fixed output addresses."""
        if not self._graph_rng:
            return [draw_normals(QL_exprs[0][0], g) for QL_exprs, g in zip(self._QLs_exprs, Gs)]
        if self._rng_graph is None:
            state = torch.cuda.get_rng_state(Gs[0].device)
            self._rng_graph = torch.cuda.CUDAGraph()
            side = torch.cuda.Stream(Gs[0].device)
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side), torch.cuda.graph(self._rng_graph):
                self._rng_out = [draw_normals(QL_exprs[0][0], g) for QL_exprs, g in zip(self._QLs_exprs, Gs)]
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.set_rng_state(state, Gs[0].device)
            if self._compiled:
                for t in [t for r in self._rng_out for t in r]:
                    torch._dynamo.mark_static_address(t)
        self._rng_graph.replay()
        return self._rng_out

    def _update(self, Gs, coins):
        self._update_fn(Gs, self._draws(Gs))
        for k, (QL_exprs, coin) in enumerate(zip(self._QLs_exprs, coins)):
            if coin:
                balance_kron_precond(QL_exprs[0][0])
                if self._P[k] is not None:
                    for p, q in zip(self._P[k], QL_exprs[0][0]):
                        p.copy_(gram_2d(q, self.ops))

    def _precond_apply(self, Gs, lr):
        max_avg_amp, max_element_amp = self.grad_clip_max_amps
        pre_grads, done = [None] * len(Gs), set()
        for idx, st in self._groups:
            if st is None or self._batch == "update":  # batch="update": batched update, per-parameter preconditioning
                for i in idx:
                    pre_grads[i] = (apply_2d(self._P[i], Gs[i], self._QLs_exprs[i][0][0], self.ops) if self._P[i] is not None else
                                    precond_grad_kron(*self._QLs_exprs[i], Gs[i], ops=self.ops, cache=self._pcache))
                continue
            exprs = self._QLs_exprs[idx[0]][1]
            G = st["m"] if st["m"] is not None and Gs is self._ms else torch.stack([Gs[i] for i in idx])
            P = torch.vmap(lambda Q, g: precond_grad_kron([Q, None], exprs, g, ops=self.ops))(st["Q"], G)
            for b, i in enumerate(idx):
                pre_grads[i] = P[b]
            if not self.capture:  # clip the whole group at once
                avg_amp = torch.sqrt(torch.mean(P * P, dim=tuple(range(1, P.dim())), keepdim=True))
                P *= torch.clamp(max_avg_amp/avg_amp, max=1.0)
                P.clamp_(min=-max_element_amp, max=max_element_amp)
                for i in idx:
                    param = self._params_with_grad[i]
                    if isinstance(lr, torch.Tensor):
                        param.sub_(lr * pre_grads[i].view_as(param))
                    else:
                        param.subtract_(pre_grads[i].view_as(param), alpha=lr)
                done.update(idx)
        if self.capture:
            self.pre = [g.clone() for g in pre_grads]
        for i, (param, g) in enumerate(zip(self._params_with_grad, pre_grads)):
            if i in done:
                continue
            avg_amp = torch.sqrt(torch.real(torch.mean(g*g.conj())))
            g *= torch.clamp(max_avg_amp/avg_amp, max=1.0)
            if torch.is_complex(g):
                g /= torch.clamp(torch.abs(g)/max_element_amp, min=1.0)
            else:
                g.clamp_(min=-max_element_amp, max=max_element_amp)
            if isinstance(lr, torch.Tensor):
                param.sub_(lr * g.view_as(param))
            else:
                param.subtract_(g.view_as(param), alpha=lr)

    @torch.no_grad()
    def step_math(self, grads):
        grads = [g.squeeze() for g in grads]
        if self._QLs_exprs is None:
            scale = max([torch.mean((torch.abs(g))**4) for g in grads])
            scale = (scale + self.damping**4)**(-1/8)
            self._init(grads, scale)
        if self._P is None:  # p2d: each 2-D parameter keeps P_i = Q_i^T Q_i (q * q if diagonal), refreshed after updates
            self._P = [[gram_2d(q, self.ops) for q in QL[0][0]] if self._p2d and g.dim() == 2 else None
                       for QL, g in zip(self._QLs_exprs, grads)]
        if self._groups is None:
            if self.momentum > 0 and self._ms is None:
                self._ms = [torch.zeros_like(g) for g in grads]
            self._build_groups(grads)
        if self._compiled and not self._static:  # state lives at fixed addresses: lets CUDA graphs mutate it
            for t in ([t for QL_exprs in self._QLs_exprs for part in QL_exprs[0] for t in part] + self._params_with_grad
                      + [t for P in self._P if P is not None for t in P]
                      + (self._ms or []) + [t for _, st in self._groups if st for k in ("Q", "L") for t in st[k]]
                      + [st["m"] for _, st in self._groups if st and st["m"] is not None]):
                torch._dynamo.mark_static_address(t)
            self._static = True

        if self.momentum > 0:
            beta = min(self._counter_m/(1 + self._counter_m), self.momentum)
            self._counter_m += 1
            if self._ms is None:
                self._ms = [torch.zeros_like(g) for g in grads]
                if self._compiled:
                    for m in self._ms:
                        torch._dynamo.mark_static_address(m)
            self._momentum_fn(self._ms, grads, self._scalar("beta", beta, grads[0]) if self._compiled else beta)
        else:
            self._ms, self._counter_m = None, 0

        if self._compiled:  # one copy into fixed buffers instead of one CUDA-graph input copy per gradient and graph
            if self._gbuf is None:
                self._gbuf = [torch.empty_like(g) for g in grads]
                for t in self._gbuf:
                    torch._dynamo.mark_static_address(t)
            torch._foreach_copy_(self._gbuf, grads)
            grads = self._gbuf

        update = bool(torch.rand([]) < self.preconditioner_update_probability)
        coins = (torch.rand(len(self._QLs_exprs)) < 0.01).tolist() if update else []  # = one torch.rand([]) each
        src = grads if self._whiten_grad else self._ms
        lr = self._scalar("lr", self.lr_params, grads[0]) if self._compiled else self.lr_params
        if update and self.update_preconditioner_first:
            self._update(src, coins)
        self._apply_fn(self._ms if self.momentum > 0 else grads, lr)
        if update and not self.update_preconditioner_first:
            self._update(src, coins)

    @torch.no_grad()
    def step(self, closure):
        with torch.enable_grad():
            closure_returns = closure()
            loss = closure_returns if isinstance(closure_returns, torch.Tensor) else closure_returns[0]
            grads = torch.autograd.grad(loss, self._params_with_grad)
        self.step_math(grads)
        return closure_returns
