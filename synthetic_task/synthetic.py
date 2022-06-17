import random
import numpy as np
import torch
import scipy.fft
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from SCRNOptimizer import SCRNOptimizer
from adahessian import Adahessian
import os

# Setting the device of the experiment
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"


# Setting the amplitudes, phases, and frequencies of the target function
amplitudes = np.array([0.25, 0.5, 0.75, 1.0])
phases = np.array([0., 0., 0., 0.])
frequencies = np.array([1., 2., 4., 8.])

# Calculate the target function which is a sum of sinusoid signals
def f(x, amplitudes, frequencies, phases):
    return (np.sin(2 * np.pi * np.outer(x, frequencies) + phases) * amplitudes).sum(axis=1)


def train_eval(args):
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 1)).to(device)

    x = np.linspace(0, 1, args.samples, endpoint=False).astype(np.float32)
    y = f(x, amplitudes, frequencies, phases).astype(np.float32)
    x = x.reshape((-1, 1))

    train_x = torch.tensor(x, device=device)
    train_y = torch.tensor(y.reshape((-1, 1)), device=device)

    # Creating datasets and dataloaders
    train_ds = torch.utils.data.TensorDataset(train_x, train_y)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batchsize, shuffle=True)

    # Defining the optimizer
    if args.optim.lower() == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim.lower() == 'scrn':
        print(args.rho, args.l)
        opt = SCRNOptimizer(model.parameters(), inner_itr=10, ro=args.rho, l=args.l, epsilon=1e-3, c_prime=0.1,
                            step_size=0.001)
    elif args.optim.lower() == 'adahessian':
        opt = Adahessian(model.parameters(), lr=args.lr)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    loss_func = torch.nn.MSELoss()
    frequenices_while_learning = []
    time_stamps = []
    losses = []
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_func(pred, yb)
            train_loss += loss
            loss.backward(create_graph=True, retain_graph=True)
            opt.step()
            opt.zero_grad()
        if epoch % max(10, 1) == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                if epoch % min(args.epochs // 20, 1) == 0 or epoch == args.epochs - 1:
                    print(f"Epoch: {epoch}, \t Training Loss: {train_loss.cpu().detach().numpy() / len(train_dl)}")
                freq = scipy.fft.fft(model(train_x).cpu().detach().numpy().ravel())
                freq = np.abs(freq)[:args.samples // 2]
                frequenices_while_learning.append(freq)
                time_stamps.append(epoch)
                losses.append(train_loss.cpu().detach().numpy() / len(train_dl))
    return time_stamps, losses, frequenices_while_learning


if __name__ == '__main__':
    parser = ArgumentParser(description="Training script for synthetic experiments of the report",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-samples', default=200, type=int, help='number of equally spaced samples in [0,1]')
    parser.add_argument('-batchsize', default=20, type=int, help='batch size')
    parser.add_argument('-seed', default=0, type=int, help='random seed', required=True)
    parser.add_argument('-epochs', default=100, type=int, help='number of epochs', required=True)
    parser.add_argument('-lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('-momentum', default=0.0, type=float, help='momentum')
    parser.add_argument('-l', default=0.1, type=float, help='l for SCRN optimizer')
    parser.add_argument('-rho', default=5.0, type=float, help='rho for SCRN optimizer')
    parser.add_argument('-optim', default='SGD', type=str, help='name of the optimizer (SGD, Adam, SCRN, or AdaHessian)',
                        required=True)
    args = parser.parse_args()

    if args.optim.lower() not in ['sgd', 'adam', 'scrn', 'adahessian']:
        print("The optimizer is not valid.")
        exit()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    for k, v in vars(args).items():
        print(f"{k}: {v}")

    x = np.linspace(0, 1, args.samples, endpoint=False)
    y = f(x, amplitudes, frequencies, phases)
    true_Fourier_coefficients = np.abs(scipy.fft.fft(y))[:args.samples // 2]
    print("TRAINING LOG")
    time, loss, freq = train_eval(args)

    os.makedirs('./results_synthetic', exist_ok=True)
    saved_data = {
        'time': time,
        'loss': loss,
        'freq': np.array(freq) / (args.samples // 2)
    }
    saved_data.update(vars(args))
    with open(f"./results_synthetic/{args.optim.lower()}_{args.seed}.npz", "wb") as f:
        np.savez(f, **saved_data)


