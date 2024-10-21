class DiscreteElasticRod2D(DiscreteRod2D):

    def __init__(self, l, r_0, E, ms):
        super().__init__(l, ms)
        self.EI = E * torch.pi * r_0 ** 4 / 4   # stretching stiffness
        self.EA = E * torch.pi * r_0 ** 2  # bending stiffness

    def forward(self, t, x):
        """
        x = {x_1, y_1, x_2, y_2, x_3, y_3, x'_1 ...} 12
        """
        p, v, F = super().forward(t, x)
        N = len(F)
        # F is a tensor with shape (self.N*2)
        stretch_tensor = self.get_stretch_force_tensor(
            self.__to_pair(p[..., :-2]), self.__to_pair(p[..., 2:]))
        for i in range(self.N-1):
            F[..., i*2:(i+2)*2] += stretch_tensor[..., i, :]

        # Calculate bending force between trio
        bend_tensor = self.__get_bend_force_tensor(
            self.__to_pair(p[..., :-4]), self.__to_pair(p[..., 2:-2]), self.__to_pair(p[..., 4:]))
        for i in range(self.N-2):
            F[..., i*2:(i+3)*2] += bend_tensor[..., i, :]

        return torch.cat([v, F / self.M])   # x', x''

    def get_stretch_force_tensor(self, xk_batch, xkp1_batch):
        dxy = xkp1_batch - xk_batch
        dist = torch.norm(dxy, dim=-1).clamp(min=1e-8)
        factor = (1 - dist / self.l_k) / (dist * self.l_k)
        F = torch.cat([-2 * dxy, 2 * dxy], dim=-1) * factor.unsqueeze(-1)
        F *= 0.5 * self.EA * self.l_k
        return F

    def __get_bend_force_tensor(self, nm1, n, np1, curvature=0):
        nm1_3d = torch.cat([nm1, torch.zeros(*nm1.shape[:-1], 1)], dim=-1)
        n_3d = torch.cat([n, torch.zeros(*n.shape[:-1], 1)], dim=-1)
        np1_3d = torch.cat([np1, torch.zeros(*np1.shape[:-1], 1)], dim=-1)

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

        grad_kappa = torch.cat(
            [-Dkappa1De[..., :2], Dkappa1De[..., :2] - Dkappa1Df[..., :2], Dkappa1Df[..., :2]], dim=-1)
        dkappa = kappa1 - curvature

        return -grad_kappa * self.EI * dkappa.unsqueeze(-1) / self.l_k

    def __to_pair(self, t):
        return t.view(*t.shape[:-1], -1, 2)