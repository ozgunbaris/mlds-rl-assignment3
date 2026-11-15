"""Federated EMNIST data: loading, the per-client 80/20 split and the Laplace perturbation.

The files are pickled object arrays (README.txt of the data): train_data[client]['images'|'labels']
and test_data[0]['images'|'labels']. Pixels are float32 in [0, 1]; the page background is 1.
"""
import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import torch

N_CLASSES = 62
N_PIXELS = 784
# LEAF label order: 0-9 digits, 10-35 upper-case letters, 36-61 lower-case letters
CLASS_NAMES = [str(d) for d in range(10)] + [chr(ord("A") + i) for i in range(26)] + [chr(ord("a") + i) for i in range(26)]


def default_data_dir():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.environ.get("FEMNIST_DIR", os.path.join(here, "data"))


def _to_arrays(entry):
    x = np.stack([np.asarray(a, dtype=np.float32) for a in entry["images"]]).reshape(-1, N_PIXELS)
    y = np.asarray(entry["labels"], dtype=np.int64).reshape(-1)
    if len(x) != len(y):
        raise ValueError("images and labels differ in length")
    return x, y


def load_raw(data_dir=None):
    """Return (clients, test): clients is a list of (X [n, 784] float32 in [0, 1], y [n] int64)."""
    data_dir = data_dir or default_data_dir()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train = np.load(os.path.join(data_dir, "train_data.npy"), allow_pickle=True)
        test = np.load(os.path.join(data_dir, "test_data.npy"), allow_pickle=True)
    clients = [_to_arrays(c) for c in train]
    return clients, _to_arrays(test[0])


def split_indices(n, client_id, seed, train_frac=0.8):
    """Random split of one client's n samples. The validation part gets round(0.2 n), at least 1
    and at most n - 1 samples, so that both parts are non-empty for every n >= 2."""
    rng = np.random.default_rng([seed, client_id, 1])
    perm = rng.permutation(n)
    n_val = int(round((1.0 - train_frac) * n))
    n_val = min(max(n_val, 1), n - 1) if n >= 2 else 0
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def laplace_noise(shape, b, rng):
    """Zero-centred Laplace noise with scale b (density exp(-|x|/b) / (2b)). b = 0 gives zeros."""
    if b < 0:
        raise ValueError("the Laplace scale b must be non-negative")
    if b == 0:
        return np.zeros(shape, dtype=np.float32)
    return rng.laplace(0.0, b, size=shape).astype(np.float32)


def preprocess(x, mode):
    """Map raw pixels (background 1, ink 0) to network inputs. 'invert' gives ink 1, background 0."""
    if mode == "invert":
        return 1.0 - x
    if mode == "raw":
        return x
    if mode == "center":
        return 1.0 - 2.0 * x
    raise ValueError(mode)


@dataclass
class Federation:
    """All tensors of one experiment. Client k owns rows off_tr[k]:off_tr[k+1] of the pooled
    training tensors (and the same for validation); nothing else of client k exists anywhere."""
    x_tr: torch.Tensor
    y_tr: torch.Tensor
    x_va: torch.Tensor
    y_va: torch.Tensor
    off_tr: np.ndarray
    off_va: np.ndarray
    x_te: torch.Tensor
    y_te: torch.Tensor
    x_tr_clean: torch.Tensor = None  # training images before the noise (diagnostic only)
    meta: dict = field(default_factory=dict)

    @property
    def n_clients(self):
        return len(self.off_tr) - 1

    @property
    def n_tr(self):
        return np.diff(self.off_tr)

    @property
    def n_va(self):
        return np.diff(self.off_va)

    def client_train(self, k):
        s = slice(self.off_tr[k], self.off_tr[k + 1])
        return self.x_tr[s], self.y_tr[s]

    def client_val(self, k):
        s = slice(self.off_va[k], self.off_va[k + 1])
        return self.x_va[s], self.y_va[s]

    def client_index(self, part):
        off = self.off_tr if part == "train" else self.off_va
        return torch.from_numpy(np.repeat(np.arange(self.n_clients), np.diff(off)))


def build_federation(data_dir=None, split_seed=0, noise_b=0.0, noise_seed=0, noise_val=True,
                     clip=False, prep="invert", partition="writer", shuffle_labels=False,
                     dtype=torch.float32, raw=None, max_clients=None):
    """Build the tensors of one experiment.

    noise_b     Laplace scale on the raw [0, 1] pixel scale. The noise is drawn once per image.
    noise_val   also perturb the validation split (the test data is never perturbed).
    clip        clip perturbed pixels back to [0, 1].
    partition   'writer' keeps the real clients; 'iid' deals the pooled training+validation images
                to the clients at random, keeping every client's sample count.
    shuffle_labels  permute the training labels inside each client (validation labels stay).
    """
    if not np.isfinite(noise_b) or noise_b < 0:
        raise ValueError("the Laplace scale b must be finite and non-negative")
    clients, test = raw if raw is not None else load_raw(data_dir)
    if max_clients:
        clients = clients[:max_clients]
    if partition == "iid":
        sizes = [len(y) for _, y in clients]
        xs = np.concatenate([x for x, _ in clients])
        ys = np.concatenate([y for _, y in clients])
        perm = np.random.default_rng([split_seed, 7]).permutation(len(ys))
        cuts = np.cumsum(sizes)[:-1]
        clients = [(xs[i], ys[i]) for i in np.split(perm, cuts)]
    elif partition != "writer":
        raise ValueError(partition)

    xtr, ytr, xva, yva, xclean = [], [], [], [], []
    for k, (x, y) in enumerate(clients):
        tr, va = split_indices(len(y), k, split_seed)
        x_noisy = x
        if noise_b > 0:
            # one noise draw per image of the client, made before the split is applied
            eps = laplace_noise(x.shape, noise_b, np.random.default_rng([noise_seed, k, 2]))
            x_noisy = x + eps
            if clip:
                x_noisy = np.clip(x_noisy, 0.0, 1.0)
        y_train = y[tr]
        if shuffle_labels:
            y_train = np.random.default_rng([split_seed, k, 3]).permutation(y_train)
        xtr.append(x_noisy[tr]); ytr.append(y_train); xclean.append(x[tr])
        xva.append(x_noisy[va] if noise_val else x[va]); yva.append(y[va])

    t = lambda parts: torch.from_numpy(preprocess(np.concatenate(parts), prep)).to(dtype)
    lab = lambda parts: torch.from_numpy(np.concatenate(parts))
    off = lambda parts: np.concatenate([[0], np.cumsum([len(p) for p in parts])])
    return Federation(
        x_tr=t(xtr), y_tr=lab(ytr), x_va=t(xva), y_va=lab(yva), off_tr=off(ytr), off_va=off(yva),
        x_te=torch.from_numpy(preprocess(test[0], prep)).to(dtype), y_te=torch.from_numpy(test[1]),
        x_tr_clean=t(xclean) if noise_b > 0 else None,
        meta=dict(split_seed=split_seed, noise_b=noise_b, noise_seed=noise_seed, noise_val=noise_val,
                  clip=clip, prep=prep, partition=partition, shuffle_labels=shuffle_labels),
    )
