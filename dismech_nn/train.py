from tqdm.autonotebook import tqdm
import torch
from torch.utils.data import DataLoader
import torchdiffeq


def train_ode(nn: torch.nn.Module,
              train_dataloader: DataLoader,
              valid_dataloader: DataLoader,
              epochs: int,
              device='cpu',
              method=None,
              options=None,
              running_mean=0.9):
    """
    Basic ODE training loop with 2D visualization. Turn off visualization if your NN is not 2D and time agnostic.
    """
    optimizer = torch.optim.Adam(nn.parameters(), lr=0.001)
    loss = torch.nn.MSELoss()

    nn.to(device)
    for epoch in tqdm(range(epochs), desc='Epochs', leave=False):

        nn.train()
        train_loss = 0.0
        for batch_t, batch_y0, batch_y in (pbar := tqdm(train_dataloader, desc='Training Batch', leave=False)):
            optimizer.zero_grad()
            pred_y = torchdiffeq.odeint_adjoint(
                nn, batch_y0.to(device), batch_t[0].to(device), method=method, options=options).permute(1, 0, 2)
            output = loss(pred_y.to(device), batch_y.to(device))
            train_loss = train_loss * (1-running_mean) + running_mean * output
            pbar.set_postfix(loss=train_loss.item())
            output.backward()
            optimizer.step()

        nn.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for (batch_t, batch_y0, batch_y) in tqdm(valid_dataloader, desc='Validation Batch', leave=False):
                pred_y = torchdiffeq.odeint_adjoint(
                    nn, batch_y0.to(device), batch_t[0].to(device), method=method, options=options).permute(1, 0, 2)
                output = loss(pred_y.to(device), batch_y.to(device))
                valid_loss += output
        print("Epoch {}/{} - Training Loss: {} Validation Loss: {}".format(epoch+1,
              epochs, train_loss, valid_loss/len(valid_dataloader)))
