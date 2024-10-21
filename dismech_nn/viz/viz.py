import torch
import torchdiffeq
import matplotlib.pyplot as plt

def visualize_2d(fun, truth, axes, t_span, y0, x_span=[-10.0, 10.0], v_span=[-10.0, 10.0], step=1):
    """
    Visualize a 2D (position and velocity) differential equation which is agnostic to time.
    WIP as it runs slow
    """
    # Create X, Y inputs for function
    X, Y = torch.meshgrid(torch.arange(*x_span, step),
                          torch.arange(*v_span, step), indexing='ij')
    grid = torch.stack([X, Y], dim=-1).view(-1, 2)
    magnitudes = fun([], grid).detach().reshape(
        len(X), len(Y), 2).permute(2, 1, 0)

    # Create a 1D tensor for all time evaluated points
    t = torch.linspace(*t_span, 1000)

    sol = torchdiffeq.odeint_adjoint(fun, y0, t)

    truth_sol = torchdiffeq.odeint_adjoint(truth, y0, t)

    axes[0].cla()
    axes[1].cla()
    axes[2].cla()

    axes[0].quiver(X, Y, magnitudes[1], magnitudes[0])
    axes[0].set_xlabel("Position")
    axes[0].set_ylabel("Velocity")
    axes[0].set_title("Vector Field")

    axes[1].plot(t, sol, label=["x", "v"])
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Position or Velocity")
    axes[1].legend()
    axes[1].set_title("Position and Velocity vs Time of MSD")

    axes[2].plot(t, sol - truth_sol, label=["x", "v"])
    axes[2].legend()
    axes[2].set_title("Error vs Time")

    plt.draw()
    plt.pause(0.15)