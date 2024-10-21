from .dr2d import DiscreteRod2D

import torch

from torch.profiler import record_function


class DiscreteElasticRod2D(DiscreteRod2D):

    def __init__(self, l, r_0, E, ms):
        super().__init__(l, ms)
        self.EI = E * torch.pi * r_0 ** 4 / 4   # stretching stiffness
        self.EA = E * torch.pi * r_0 ** 2  # bending stiffness

    def forward(self, t, x):
        with record_function('linear'):
            p, v, F = super().forward(t, x)

        with record_function('stretch'):
            # Compute stretching forces
            stretch_tensor = self.get_stretch_force_tensor(
                self.__to_pair(p[..., :-2]),
                self.__to_pair(p[..., 2:])
            )
            for i in range(self.N-1):
                F[..., i*2:(i+2)*2] += stretch_tensor[..., i, :]

        with record_function('bend'):
            # Compute bending forces
            bend_tensor = self.__get_bend_force_tensor(
                self.__to_pair(p[..., :-4]),
                self.__to_pair(p[..., 2:-2]),
                self.__to_pair(p[..., 4:])
            )
            for i in range(self.N-2):
                F[..., i*2:(i+3)*2] += bend_tensor[..., i, :]

        return torch.cat([v, F / self.M], dim=-1)   # x', x''

    def get_stretch_force_tensor(self, xk_batch, xkp1_batch):
        dxy = xkp1_batch - xk_batch
        dist = torch.norm(dxy, dim=-1).clamp_min(1e-8)
        factor = (1 - dist / self.l_k) / (dist * self.l_k)
        F = torch.cat([-2 * dxy, 2 * dxy], dim=-1) * factor.unsqueeze(-1)
        F *= 0.5 * self.EA * self.l_k
        return F

    def __get_bend_force_tensor(self, nm1, n, np1, curvature=0):
        nm1_3d = torch.nn.functional.pad(nm1, (0, 1), value=0.0)
        n_3d = torch.nn.functional.pad(n, (0, 1), value=0.0)
        np1_3d = torch.nn.functional.pad(np1, (0, 1), value=0.0)

        ee = n_3d - nm1_3d
        ef = np1_3d - n_3d

        norm_e = torch.norm(ee, dim=-1, keepdim=True).clamp_min(1e-8)
        norm_f = torch.norm(ef, dim=-1, keepdim=True).clamp_min(1e-8)

        te = ee / norm_e
        tf = ef / norm_f

        dot_te_tf = (te * tf).sum(dim=-1, keepdim=True)
        kb = 2.0 * torch.cross(te, tf, dim=-1) / (1.0 + dot_te_tf)

        chi = 1.0 + dot_te_tf
        tilde_t = (te + tf) / chi
        tilde_d2 = torch.tensor(
            [0.0, 0.0, 2.0], device=te.device).view(1, 3) / chi

        kappa1 = kb[..., 2]
        Dkappa1De = (torch.cross(tf, tilde_d2, dim=-1) -
                     kappa1.unsqueeze(-1) * tilde_t) / norm_e
        Dkappa1Df = (-torch.cross(te, tilde_d2, dim=-1) -
                     kappa1.unsqueeze(-1) * tilde_t) / norm_f

        grad_kappa = torch.cat([
            -Dkappa1De[..., :2],
            Dkappa1De[..., :2] - Dkappa1Df[..., :2],
            Dkappa1Df[..., :2]
        ], dim=-1)

        dkappa = kappa1 - curvature
        return -grad_kappa * self.EI * dkappa.unsqueeze(-1) / self.l_k

    def __to_pair(self, t):
        return t.view(*t.shape[:-1], -1, 2)
