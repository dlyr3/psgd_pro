"""LSTM network with the classic delayed XOR problem, used as a quick profiling harness for a few training steps with the PSGD KronWhiten optimizer. It shows where the time goes per step; it does not try to prove convergence.

python test/lstm_with_xor_pb.py                       # 20 profiled steps, timing table
python test/lstm_with_xor_pb.py --plot                # also records and plots the loss
python test/lstm_with_xor_pb.py --trace trace.json    # dump for ui.perfetto.dev / chrome://tracing
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import psgd_pro as psgd

batch_size, seq_len = 2048, 50  # increasing sequence_length
dim_in, dim_hidden, dim_out = 2, 256, 1  # or decreasing dimension_hidden_layer will make learning harder

def generate_train_data():
    x = np.zeros([batch_size, seq_len, dim_in], dtype=np.float32)
    y = np.zeros([batch_size, dim_out], dtype=np.float32)
    for i in range(batch_size):
        x[i, :, 0] = np.random.choice([-1.0, 1.0], seq_len)

        i1 = int(np.floor(np.random.rand() * 0.1 * seq_len))
        i2 = int(np.floor(np.random.rand() * 0.4 * seq_len + 0.1 * seq_len))
        x[i, i1, 1] = 1.0
        x[i, i2, 1] = 1.0
        if x[i, i1, 0] == x[i, i2, 0]:  # XOR
            y[i] = -1.0  # lable 0
        else:
            y[i] = 1.0  # lable 1

    # tranpose x to format (sequence_length, batch_size, dimension_of_input)
    return torch.tensor(np.transpose(x, [1, 0, 2])), torch.tensor(y)

class LSTM_net(torch.nn.Module):
    def __init__(self):
        super(LSTM_net, self).__init__()
        W1 = 0.1 * torch.randn(dim_in + 2 * dim_hidden + 1, 4 * dim_hidden)
        W1[-1, dim_hidden:2 * dim_hidden] += 1.0  # forget gate with large bias to encourage long term memory
        W1[:, 2 * dim_hidden:3 * dim_hidden] *= 2.0  # cause tanh(x)=2*sigmoid(2*x) - 1
        self.W1 = torch.nn.Parameter(W1)
        self.W2 = torch.nn.Parameter(0.1 * torch.randn(dim_hidden + 1, dim_out))

    def forward(self, xs):
        with torch.profiler.record_function("forward_LSTM"):
            h, c = torch.zeros(batch_size, dim_hidden, device=xs.device), torch.zeros(batch_size,
                                                                    dim_hidden, device=xs.device)  # initial hidden and cell states
            for x in torch.unbind(xs):
                ifgo = torch.cat([x, h, c], dim=1) @ self.W1[:-1] + self.W1[
                    -1]  # here cell state is in the input feature as well
                i, f, g, o = torch.chunk(torch.sigmoid(ifgo), 4, dim=1)
                c = f * c + i * (2.0 * g - 1.0)  # new cell state; note that tanh(x)=2*sigmoid(2*x) - 1
                h = o * torch.tanh(c)  # new hidden state
            return h @ self.W2[:-1] + self.W2[-1]

def train_loss(net, xy_pair):  # logistic loss
    return -torch.mean(torch.log(torch.sigmoid(xy_pair[1] * net(xy_pair[0]))))

def main():
    parser = argparse.ArgumentParser(description="Profile a few PSGD training steps on the delayed XOR LSTM.")
    parser.add_argument("--steps", type=int, default=20, help="number of profiled training steps")
    parser.add_argument("--lr", type=float, default=0.02, help="parameter learning rate")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--plot", action="store_true", help="record the loss of each step and plot it at the end")
    parser.add_argument("--trace", default=None, help="write a chrome trace file, viewable at ui.perfetto.dev")
    args = parser.parse_args()

    device = torch.device(args.device)
    lstm_net = LSTM_net().to(device)
    opt = psgd.KronWhiten(lstm_net.parameters(), preconditioner_init_scale=1.0, lr_params=args.lr,
                          preconditioner_max_skew=dim_hidden+1) # full quality preconditionner, max skew nether reacher

    def training_step():
        with torch.profiler.record_function("generate_data"):
            x, y = generate_train_data()
            x, y = x.to(device), y.to(device)
        with torch.profiler.record_function("opt_step"):
            return opt.step(lambda: train_loss(lstm_net, (x, y)))

    for _ in range(5):  # warmup outside the profiler: lazy preconditioner init, thread pools, etc.
        training_step()

    losses = []
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
        for num_iter in range(args.steps):
            loss = training_step().item()
            if args.plot:
                losses.append(loss)
            print('Iteration: {}; loss: {:.4f}'.format(num_iter, loss))

    print(prof.key_averages().table(
        sort_by="self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total", row_limit=30))
    if args.trace:
        prof.export_chrome_trace(args.trace)
    if args.plot:
        plt.plot(losses)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Delayed XOR, PSGD KronWhiten')
        plt.show()

if __name__ == "__main__":
    main()
