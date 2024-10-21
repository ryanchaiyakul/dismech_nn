import torchdiffeq
import torch
import csv
import os
import typing

import torch.utils.data.dataset as dataset


class SegmentedDataset(dataset.Dataset):

    def __init__(self, dataset_size: int, batch_steps: int):
        if batch_steps == -1:
            self.batch_steps = dataset_size
            self.batch_count = 1
        else:
            self.batch_steps = batch_steps
            self.batch_count = dataset_size - self.batch_steps + 1

    def __len__(self):
        return self.sol.shape[0] * self.batch_count

    def __getitem__(self, index: int):
        d_idx = int(index / self.batch_count)       # floor
        t_idx = index - (d_idx * self.batch_count)  # remainder
        return (self.t[t_idx:t_idx+self.batch_steps], self.sol[d_idx][t_idx], self.sol[d_idx][t_idx:t_idx+self.batch_steps])


class ODEDataset(SegmentedDataset):
    """
    Create a dataset from an analytical ordinary differential equation with a numerical ODE solver.
    """

    def __init__(self, module: torch.nn.Module, y0s: torch.Tensor, t_span: typing.Tuple[int, int], steps: int, batch_steps: int = -1):
        """
        - module: torch.nn.Module that represents your ODE
        - y0s: torch.Tensor (N x V) of initial conditions
        - t_span: 2 item list with t_0 < t_1
        - steps: number of time steps
        - batch_steps: number of time steps for each sequence ()
        """
        self.t = torch.linspace(*t_span, steps)
        with torch.no_grad():
            self.sol = torchdiffeq.odeint_adjoint(
                module, y0s, self.t).permute(1, 0, 2)

        super().__init__(self.sol.shape[1], batch_steps)


class HomogenousTensorDataset(SegmentedDataset):

    def __init__(self, t: torch.Tensor, sol: torch.Tensor, batch_steps: int = -1):
        """
        - solution tensor in M x T x V
        - batch_steps: number of time steps for each sequence
        """
        self.t = t
        self.sol = sol

        super().__init__(self.sol.shape[1], batch_steps)

class HomogenousCSVDataset(SegmentedDataset):
    """
    Create a dataset from CSV file(s) of the same time range (but should have different initial conditions)
    """

    def __init__(self, path: str, batch_steps: int = -1):
        """
        - path: path to a csv file or folder of CSV files (all CSVs must have the same time range)
        - batch_steps: number of time steps for each sequence
        """
        if os.path.isfile(path):
            self.t, self.sol = self.__read_csv(path)
            self.sol = torch.unsqueeze(self.sol, 0)  # Needs to be N x T x V
        elif os.path.isdir(path):
            sols = []
            self.t = None   # TODO: Fix inefficient code

            for f_path in os.listdir(path):
                t, sol = self.__read_csv("{}/{}".format(path, f_path))
                if self.t is None:
                    self.t = t
                assert (t.equal(self.t))

                sols.append(sol)

            self.sol = torch.stack(sols)

        super().__init__(self.sol.shape[1], batch_steps)

    def __read_csv(self, f_path):
        """
        Read t, ys from a csv in the format t, y_0, y_1, y_2 ...
        """
        with open(f_path, 'r') as f:
            ts = []
            ys = []

            for row in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC):
                ts.append(torch.tensor(row[0]))
                ys.append(torch.tensor(row[1:]))

        return torch.stack(ts), torch.stack(ys)
