"""Parallel client updates with Ray Core (Part 1, question 2).

One actor is created for every client selected in a round and is killed as soon as its
update has been collected, so at most `clients_per_round` actors exist at any time. An actor
receives the data of its own client only. It reserves 1 CPU and, when CUDA devices exist,
a share of one GPU (see gpu_share): a whole GPU when there are at least as many devices as
clients per round, otherwise an equal share of one device.
"""
import math
import os
import time

import numpy as np
import ray
import torch

from .core import client_update, update_seed


def gpu_share(n_gpus, clients_per_round):
    """GPU fraction one actor reserves. Ray places a fractional request on a single device, so the
    share must let ceil(m / G) actors fit on one GPU: with G = 3 and m = 4 the share is 0.5, because
    a share of G / m = 0.75 would leave room for only 3 actors at a time."""
    n_gpus, m = int(n_gpus), int(clients_per_round)
    if n_gpus <= 0:
        return 0.0
    if n_gpus >= m:
        return 1.0
    return math.floor(1e4 / math.ceil(m / n_gpus)) / 1e4  # Ray resolves resources to 0.0001


@ray.remote
class ClientActor:
    def __init__(self, cid, x, y, use_gpu=False):
        self.t_init0 = time.time()
        torch.set_num_threads(1)  # the actor owns 1 CPU
        self.cid = int(cid)
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.x = torch.tensor(x).to(self.device)
        self.y = torch.tensor(y).to(self.device)
        self.gpu_ids = [str(g) for g in ray.get_gpu_ids()]  # devices Ray assigned to this actor
        self.t_created = time.time()

    def train(self, global_vec, epochs, batch_size, lr, seed, fault=None):
        t0 = time.time()
        if fault == "crash":       # test hook: the client process dies during the round
            os._exit(1)
        if fault and fault.startswith("sleep:"):  # test hook: a slow client
            time.sleep(float(fault.split(":")[1]))
        if fault == "nan":         # test hook: the client returns non-finite weights
            lr = float("inf")
        w0 = torch.tensor(global_vec).to(self.device)
        vec, stats = client_update(w0, self.x, self.y, epochs, batch_size, lr, seed)
        vec = vec.cpu()
        stats.update(t_created=self.t_created, t_init0=self.t_init0, t_start=t0, t_end=time.time(), pid=os.getpid(),
                     device=self.device.type, gpu_ids=self.gpu_ids)
        if self.device.type == "cuda":
            free, total = torch.cuda.mem_get_info()
            stats.update(gpu_name=torch.cuda.get_device_name(0), gpu_tensor_bytes=int(torch.cuda.max_memory_allocated()),
                         gpu_used_bytes=int(total - free))
        return self.cid, vec.numpy(), stats


class RayRunner:
    """Drop-in replacement for core.SequentialRunner."""

    def __init__(self, fed, cfg, num_cpus=None, timeout=None, faults=None, num_gpus=None, use_gpu=True,
                 object_store_memory=None, gpu_per_actor=None):
        self.fed, self.cfg = fed, cfg
        self.timeout = timeout        # seconds a round waits for its clients (None = no limit)
        self.faults = faults or {}    # {(round, client): 'crash' | 'sleep:<s>' | 'nan'} for the tests
        m = cfg.clients_per_round
        cpus = num_cpus or min(os.cpu_count() or 1, m)
        if not ray.is_initialized():
            kw = dict(num_cpus=cpus, include_dashboard=False, log_to_driver=False, logging_level="ERROR")
            if num_gpus is not None:  # None lets Ray count the CUDA devices itself
                kw["num_gpus"] = num_gpus
            if object_store_memory:
                kw["object_store_memory"] = int(object_store_memory)
            ray.init(**kw)
        self.num_cpus = int(ray.cluster_resources().get("CPU", cpus))
        self.num_gpus = int(ray.cluster_resources().get("GPU", 0))
        self.gpu_per_actor = gpu_share(self.num_gpus, m) if use_gpu else 0.0
        if gpu_per_actor is not None:  # test hook: a share other than the rule of gpu_share
            self.gpu_per_actor = float(gpu_per_actor)
        self.data_refs = {}
        self.log = []                 # one dictionary per returned update (device, GPU ids, times)

    def _data(self, k):
        if k not in self.data_refs:  # the client's data enters the object store once
            x, y = self.fed.client_train(k)
            self.data_refs[k] = (ray.put(x.numpy()), ray.put(y.numpy()))
        return self.data_refs[k]

    def __call__(self, rnd, selected, global_vec, lr):
        w_ref = ray.put(global_vec.numpy())
        pending = {}
        for k in selected:
            k = int(k)
            x_ref, y_ref = self._data(k)
            actor = ClientActor.options(num_cpus=1, num_gpus=self.gpu_per_actor).remote(
                k, x_ref, y_ref, self.gpu_per_actor > 0)
            fut = actor.train.remote(w_ref, self.cfg.epochs, self.cfg.batch_size, lr,
                                     update_seed(self.cfg.seed, rnd, k), self.faults.get((rnd, k)))
            pending[fut] = (k, actor)
        results = {}
        deadline = None if self.timeout is None else time.time() + self.timeout
        while pending:
            wait = None if deadline is None else max(0.0, deadline - time.time())
            done, _ = ray.wait(list(pending), num_returns=1, timeout=wait)
            if not done:              # the round's time limit passed: drop the clients still running
                break
            k, actor = pending.pop(done[0])
            try:
                cid, vec, stats = ray.get(done[0])
                results[k] = dict(cid=cid, vec=torch.from_numpy(vec.copy()), stats=stats)
                self.log.append(dict(round=int(rnd), cid=int(cid), **{a: b for a, b in stats.items() if a != 'n'}))
            except (ray.exceptions.RayActorError, ray.exceptions.RayTaskError):
                results[k] = None     # the client died; the server continues without it
            ray.kill(actor)           # frees the actor's CPU and memory
        for k, actor in pending.values():
            ray.kill(actor)
            results[k] = None
        return [results[int(k)] for k in selected]  # client order, so the sum order is fixed

    def close(self):
        ray.shutdown()
