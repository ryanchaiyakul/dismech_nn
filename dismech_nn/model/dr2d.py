import abc
import typing
import torch


from torch.profiler import profile, record_function, ProfilerActivity


class DiscreteRod2D(torch.nn.Module, metaclass=abc.ABCMeta):
    """
    Generic 2D rod class which allows you add proportional (viscousity) and constant (gravity) forces onto a discretized rod
    """

    def __init__(self, l: float, ms: typing.List[float], dtype=torch.float64):
        super().__init__()

        self.N = len(ms)
        self.l_k = l / (self.N - 1)

        self.dtype = dtype
        self.M = torch.zeros(self.N*2, dtype=self.dtype)

        for i in range(self.N):
            self.M[i*2] = ms[i]
            self.M[i*2+1] = ms[i]

        self.a_p = torch.zeros(self.N*2, dtype=self.dtype)
        self.a_v = torch.zeros(self.N*2, dtype=self.dtype)
        self.c = torch.zeros(self.N*2, dtype=self.dtype)

    @abc.abstractmethod
    def forward(self, _, x: torch.Tensor):
        """
        Returns p, v, F after linear operations are applied
        """
        p, v = x.split(self.N * 2, dim=-1)
        F = self.a_p * p + self.a_v * v + self.c
        return p, v, F

    def add_position_proportional_force(self, f_x: typing.List[float], f_y: typing.List[float]):
        self.a_p += self.__interleave(f_x, f_y)

    def add_velocity_proportional_force(self, f_x: typing.List[float], f_y: typing.List[float]):
        self.a_v += self.__interleave(f_x, f_y)

    def add_constant_force(self, f_x: typing.List[float], f_y: typing.List[float]):
        self.c += self.__interleave(f_x, f_y)

    def __interleave(self, f_x: typing.List[float], f_y: typing.List[float]):
        f = torch.zeros(self.N*2, dtype=self.dtype)
        for i in range(self.N):
            f[i*2] = f_x[i]
            f[i*2+1] = f_y[i]
        return f

    def at_rest(self):
        """
        Return a torch.Tensor representing the rod at rest with even spacing in the x direction with node 0 at the origin.
        """
        x_positions = torch.arange(self.N, dtype=self.dtype) * self.l_k
        return torch.cat([torch.stack([x_positions, torch.zeros(self.N)], dim=1).flatten(), torch.zeros(self.N * 2, dtype=torch.float64)])

    def run_profile(self, steps=100):
        # Warm-up loop to avoid initial GPU launch latency affecting profiling
        for _ in range(10):
            self([], self.at_rest() + torch.rand(self.N*4))

        # Profile the function with torch.profiler
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            for _ in range(steps):
                self([], self.at_rest() + torch.rand(self.N*4))

        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))