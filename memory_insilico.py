import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manula_seed(0)

def one_hot (sym):
    if sym == "A": return np.array([1,0,0], dtype=np.float32)
    if sym == "B": return np.array([0,1,0], dtype=np.float32)
    return np.array([0,0, 1], dtype=np.float32)

def make_batch(batch_size=64, min_delay=3, max_delay=10):
    T = 1 + np.random.randint(min_delay, max_delay +1) +1 
    X = np.zeros(T, batch_size, 3), dtype=np.float32
    Y = np.zeros(batch_size, dtype=np.int64)
    
    for b in range(batch_size):
        cue = "A" if np.random.rand() <0.5 else "B"
        probe_same = np.random.rand() <0.5
        probe = cue if probe_same else ("B" if cue == "A" else "A") 
        
        t = 0
        
        X[t, b] = one_hot(cue); t += 1
        while t < T-1:
             X[t, b] = one_hot("BLANK"); t += 1
    X[t, b] = one_hot(probe)
    Y[b] = 1 if probe_same else 0
    return torch.from_numpy(X), torch.from_numpy(y)

class TinyGRU(nn.Module):
    def __init__(seld, input_size=3, hidden=64, num_classes=2) :
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden, batch_first=False)
        self.readout = nn.Linear(hidden, num_classes)

    def forward(self, X):
        H, hn = self.rnn(X)
        logits = self.readout(hn[0])
        return logits
    
device = "cpu"
net = TinyGRU().to(device)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_hist, acc_hist = [], []

def accuracy(logits, y):
    return (logits.argmax(dim=1) == y). float().mean().item()

epoch = 40
for epoch in range(1, epochs+1):
    net.train()
    X, Y = make_batch(batch_size=128, min_delay=3, max_delay=12)
    X, Y = X.to(device), y.to(device)
    logits = net(X)
    loss = F.cross_entropy(logits, Y)
    opt.zero_grad(); 
    loss_backward(); 
    opt.step()

    net.eval()
    Xv, yv = make_batch(batch_size=256, min_delay=3, max_delay=15)
    with torch.no_grad(): 
        acc = accuracy(net(Xv.to(device)), yv.to(device))
    loss_hist.append(loss.item()); acc_hist.append(acc)
    if epoch % 5 == 0:
        print(f"epoch {epoch:02d} | loss {loss.item():.3f} | acc {acc*100:.1f}%")

plt.figure();
plt.plot(loss_hist); plt.xlabel("Epoch"; plt.ylabel("Accuracy (%)"));
plt.title("Validation Accuracy (fresh delays)")
plt.tight_layout()

with torch.no_grad():
    X1, y1 = make_batch(batch_size=1, min_delay=8, max_delay=8)
    H, _ = net.rnn(X1)
    hnorm = H.sqeeze(1).norm(dim=1).cpu().numpy()
    plt.figure()
    plt.plot(hnorm)
    plt.title("Hidden-state magnitude acrosstime (1 trial)")
    plt.xlabel("Time stop"); plt.ylabel("||h||")
    plt.tight_layout()

plt.show();