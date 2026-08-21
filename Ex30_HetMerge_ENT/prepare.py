"""Підготовка чекпоінтів батьківських моделей для експериментів злиття (Ex30).

Детерміністично (SEED=42) відтворює батьківські моделі A (класи 0-4) і B
(класи 5-9) точно за протоколом e34_benchmark.py — та сама архітектура MLP
[784, 128, 64, 10], той самий оптимізатор Adam(lr=0.003), 15 епох, ті самі
підмножини даних — і зберігає їх у checkpoints/parentA.pth та
checkpoints/parentB.pth. Ваги не поширюються в репозиторії (див. .gitignore)
і відтворюються цим скриптом.

Запуск:  python3 prepare.py
"""
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

HERE = Path(__file__).resolve().parent
OUT = HERE / "checkpoints"
OUT.mkdir(exist_ok=True)

# Дані — як у e34_benchmark.py: перші 20000 зразків train MNIST
tfm = transforms.Compose([transforms.ToTensor(), lambda x: x.view(-1)])
tr = datasets.MNIST("/tmp/mnist", train=True, download=True, transform=tfm)
X_tr = torch.stack([tr[i][0] for i in range(20000)])
y_tr = torch.tensor([tr[i][1] for i in range(20000)])


class MLP(nn.Module):
    def __init__(s, a):
        super().__init__(); l = []
        for i in range(len(a) - 1):
            l.append(nn.Linear(a[i], a[i + 1]))
            if i < len(a) - 2:
                l.append(nn.ReLU())
        s.net = nn.Sequential(*l); s.arch = a

    def forward(s, x):
        return s.net(x)


def train_model(arch, X, y, cls, epochs=15):
    m = MLP(arch)
    mask = sum(y == c for c in cls).bool()
    Xs, ys = X[mask][:5000], y[mask][:5000]
    opt = torch.optim.Adam(m.parameters(), lr=0.003); m.train()
    for _ in range(epochs):
        l = nn.CrossEntropyLoss()(m(Xs), ys)
        opt.zero_grad(); l.backward(); opt.step()
    m.eval(); return m


if __name__ == "__main__":
    arch = [784, 128, 64, 10]
    clA, clB = list(range(5)), list(range(5, 10))

    torch.manual_seed(SEED)
    print("Тренування моделі A (класи 0-4), 15 епох...")
    modelA = train_model(arch, X_tr, y_tr, clA)
    print("Тренування моделі B (класи 5-9), 15 епох...")
    modelB = train_model(arch, X_tr, y_tr, clB)

    torch.save(modelA.state_dict(), OUT / "parentA.pth")
    torch.save(modelB.state_dict(), OUT / "parentB.pth")
    print(f"Збережено: {OUT / 'parentA.pth'}")
    print(f"Збережено: {OUT / 'parentB.pth'}")
    print("Примітка: бенчмарки (e34_benchmark.py та ін.) тренують ці самі моделі "
          "в пам'яті з тим самим сідом; чекпоінти призначені для інспекції ваг "
          "та зовнішнього використання.")
