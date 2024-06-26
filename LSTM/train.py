#!/usr/bin/env python3

import datetime
import os
import re
import time

import numpy as np
import torch
import json
from contextlib import contextmanager
from io import StringIO

# import gradient_reducers
import tasks
from mean_accumulator import MeanAccumulator
# from timer import Timer
import argparse
from config import config
"""
When you run this script, it uses the default parameters below.
To change them, you can make another script, say `experiment.py`
and write, e.g.
```
import train
train.config["num_epochs"] = 200
train.config["n_workers"] = 4
train.config["rank"] = 0
train.main()
```

The configuration overrides we used for all our experiments can be found in the folder schedule/neurips19.
"""

# config = dict(
#     average_reset_epoch_interval=30,
#     distributed_backend="nccl",
#     fix_conv_weight_norm=False,
#     num_epochs=300,
#     checkpoints=[],
#     num_train_tracking_batches=1,
#     optimizer_batch_size=128,  # per worker
#     optimizer_conv_learning_rate=0.1,  # tuned for batch size 128
#     optimizer_decay_at_epochs=[150, 250],
#     optimizer_decay_with_factor=10.0,
#     optimizer_learning_rate=0.1,  # Tuned for batch size 128 (single worker)
#     optimizer_memory=False,
#     optimizer_momentum_type="nesterov",
#     optimizer_momentum=0.9,
#     optimizer_reducer="ExactReducer",
#     # optimizer_reducer_compression=0.01,
#     # optimizer_reducer_rank=4,
#     # optimizer_reducer_reuse_query=True,
#     # optimizer_reducer_n_power_iterations=0,
#     optimizer_scale_lr_with_factor=None,  # set to override world_size as a factor
#     optimizer_scale_lr_with_warmup_epochs=5,  # scale lr by world size
#     optimizer_mom_before_reduce=False,
#     optimizer_wd_before_reduce=False,
#     optimizer_weight_decay_conv=0.0001,
#     optimizer_weight_decay_other=0.0001,
#     optimizer_weight_decay_bn=0.0,
#     task="LanguageModeling",
#     dataset_name='wikitext103',
#     task_architecture="ResNet18",
#     seed=42,
#     local_rank=0,
#     rank=0,
#     n_workers=1,
#     distributed_init_file=None,
#     log_verbosity=2,
# )

output_dir = "/home/mist/output.tmp"  # will be overwritten by run.py


class Timer:
    """
    Timer for PyTorch code
    Comes in the form of a contextmanager:

    Example:
    >>> timer = Timer()
    ... for i in range(10):
    ...     with timer("expensive operation"):
    ...         x = torch.randn(100)
    ... print(timer.summary())
    """

    def __init__(self, verbosity_level=1, log_fn=None, skip_first=True):
        self.verbosity_level = verbosity_level
        self.log_fn = log_fn if log_fn is not None else self._default_log_fn
        self.skip_first = skip_first

        self.reset()

    def reset(self):
        """Reset the timer"""
        self.totals = {}  # Total time per label
        self.first_time = {}  # First occurrence of a label (start time)
        self.last_time = {}  # Last occurence of a label (end time)
        self.call_counts = {}  # Number of times a label occurred

    @contextmanager
    def __call__(self, label, epoch=-1.0, verbosity=1):
        # Don't measure this if the verbosity level is too high
        if verbosity > self.verbosity_level:
            yield
            return
        # 这段代码使用了Python的contextmanager装饰器和yield关键字来定义一个上下文管理器，用于测量代码块的执行时间。当进入with语句块时，代码首先同步CUDA（如果使用），记录开始时间，然后执行yield之前的代码。yield之后的代码会在with语句块结束时执行，此时再次同步CUDA并记录结束时间，从而计算出代码块的运行时间。
        # Measure the time
        self._cuda_sync()
        # change time_ns() to time()
        start = time.time() 
        yield
        self._cuda_sync()
        end = time.time()

        # Update first and last occurrence of this label
        if not label in self.first_time:
            self.first_time[label] = start
        self.last_time[label] = end

        # Update the totals and call counts
        if not label in self.totals and self.skip_first:
            self.totals[label] = 0.0
            del self.first_time[label]
            self.call_counts[label] = 0
        elif not label in self.totals and not self.skip_first:
            self.totals[label] = end - start
            self.call_counts[label] = 1
        else:
            self.totals[label] += end - start
            self.call_counts[label] += 1

        if self.call_counts[label] > 0:
            # We will reduce the probability of logging a timing linearly with the number of times
            # we have seen it.
            # It will always be recorded in the totals, though
            if np.random.rand() < 1 / self.call_counts[label]:
                self.log_fn(
                    "timer", {"epoch": float(epoch), "value": end - start}, {"event": label}
                )

    def summary(self):
        """
        Return a summary in string-form of all the timings recorded so far
        """
        with StringIO() as buffer:
            print('current_time '+ time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            print("--- Timer summary -----------------------------------------------", file=buffer)
            print("  Event                          |  Count | Average time |  Frac.", file=buffer)
            for event_label in sorted(self.totals):
                total = self.totals[event_label]
                count = self.call_counts[event_label]
                if count == 0:
                    continue
                avg_duration = total / count
                total_runtime = self.last_time[event_label] - self.first_time[event_label]
                runtime_percentage = 100 * total / total_runtime
                print(
                    f"- {event_label:30s} | {count:6d} | {avg_duration:11.5f}s | {runtime_percentage:5.1f}%",
                    file=buffer,
                )
            print("-----------------------------------------------------------------", file=buffer)
            return buffer.getvalue()

    def save_summary(self, json_file_path):
        data = {}
        for event_label in sorted(self.totals):
            total = self.totals[event_label]
            count = self.call_counts[event_label]
            if count == 0:
                continue
            avg_duration = total / count
            data[event_label] = {
                "label": event_label,
                "average_duration": avg_duration,
                "n_events": count,
                "total_time": total,
            }

        with open(json_file_path, "w") as fp:
            json.dump(data, fp)

    def _cuda_sync(self):
        """Finish all asynchronous GPU computations to get correct timings"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _default_log_fn(self, _, values, tags):
        label = tags["label"]
        epoch = values["epoch"]
        duration = values["value"]
        print(f"Timer: {label:30s} @ {epoch:4.1f} - {duration:8.5f}s")


def main():
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    if torch.distributed.is_available():
        # if config["distributed_init_file"] is None:
        #     config["distributed_init_file"] = os.path.join(output_dir, "dist_init")
        
        torch.distributed.init_process_group(
            backend=config["distributed_backend"],
            # init_method="file://" + os.path.abspath(config["distributed_init_file"]),
            timeout=datetime.timedelta(seconds=120),
            # world_size=config["n_workers"],
            # rank=config["rank"],
        )
        config['n_workers'] = torch.distributed.get_world_size()
        config['rank'] = torch.distributed.get_rank()
        print(
                "Distributed init: rank {}/{}".format(
                    config["rank"], config["n_workers"]
                )
            )
    if config['rank'] == 0:
        print(config)
        
    torch.manual_seed(config["seed"] + config["rank"])
    np.random.seed(config["seed"] + config["rank"])

    device = torch.device("cuda:"+str(config['local_rank']) if torch.cuda.is_available() else "cpu")

    timer = Timer(verbosity_level=config["log_verbosity"], log_fn=metric)

    task = tasks.build(task_name=config["task"], device=device, timer=timer, **config)
    print('Rank {} build task {} on {} '.format(config['rank'], config['task'], config['dataset_name']))
    reducer = get_reducer(device, timer)
    warm_start_reducer = getattr(gradient_reducers, 'ExactReducer')(
            config["seed"], device, timer
        )
    print('Rank {} get reducer'.format(config['rank']))
    bits_communicated = 0
    runavg_model = MeanAccumulator()

    memories = [torch.zeros_like(param) for param in task.state]
    momenta = [torch.empty_like(param) for param in task.state]  # need initialization
    send_buffers = [torch.zeros_like(param) for param in task.state]
    for epoch in range(config["num_epochs"]):
        epoch_metrics = MeanAccumulator()
        info({"state.progress": float(epoch) / config["num_epochs"], "state.current_epoch": epoch})

        # This seems fine ...
        # check_model_consistency_across_workers(task._model, epoch)

        # Determine per-parameter optimization parameters
        wds = [get_weight_decay(epoch, name) for name in task.parameter_names]

        # Reset running average of the model
        if epoch % config["average_reset_epoch_interval"] == 0:
            runavg_model.reset()

        train_loader = task.train_iterator(config["optimizer_batch_size"])
        for i, batch in enumerate(train_loader):
            epoch_frac = epoch + i / len(train_loader)
            lrs = [get_learning_rate(epoch_frac, name) for name in task.parameter_names]

            with timer("batch", epoch_frac):
                _, grads, metrics = task.batch_loss_and_gradient(batch)
                epoch_metrics.add(metrics)

                # Compute some derived metrics from the raw gradients
                with timer("batch.reporting.lr", epoch_frac, verbosity=2):
                    for name, param, grad, lr in zip(task.parameter_names, task.state, grads, lrs):
                        # if is_embed_param(name) and config['rank'] == 0:
                        #     print(tensor_nnz(grad))
                        #     np.save(os.path.join(os.path.dirname(__file__), str(i)+'_'+name+'.npy'), grad.cpu().numpy())
                        if np.random.rand() < 0.001:  # with a small probability
                            tags = {"weight": name.replace("module.", "")}
                            metric(
                                "effective_lr",
                                {
                                    "epoch": epoch_frac,
                                    "value": lr / max(l2norm(param).item() ** 2, 1e-8),
                                },
                                tags,
                            )
                            metric(
                                "grad_norm",
                                {"epoch": epoch_frac, "value": l2norm(grad).item()},
                                tags,
                            )

                if config["optimizer_wd_before_reduce"]:
                    with timer("batch.weight_decay", epoch_frac, verbosity=2):
                        for grad, param, wd in zip(grads, task.state, wds):
                            if wd > 0:
                                grad.add_(wd, param.detach())

                if config["optimizer_mom_before_reduce"]:
                    with timer("batch.momentum", epoch_frac, verbosity=2):
                        for grad, momentum in zip(grads, momenta):
                            if epoch == 0 and i == 0:
                                momentum.data = grad.clone().detach()
                            else:
                                if (
                                    config["optimizer_momentum_type"]
                                    == "exponential_moving_average"
                                ):
                                    momentum.mul_(config["optimizer_momentum"]).add_(
                                        1 - config["optimizer_momentum"], grad
                                    )
                                else:
                                    momentum.mul_(config["optimizer_momentum"]).add_(grad)
                            replace_grad_by_momentum(grad, momentum)

                with timer("batch.accumulate", epoch_frac, verbosity=2):
                    for grad, memory, send_bfr in zip(grads, memories, send_buffers):
                        if config["optimizer_memory"]:
                            send_bfr.data[:] = grad + memory
                        else:
                            send_bfr.data[:] = grad
                    if np.random.rand() < 0.01:
                        grad_norm = l2norm(torch.cat([send_bfr.view(-1) for send_bfr in send_buffers]))
                        metric("grad norm l2", {"epoch": epoch_frac, "value": grad_norm.item() ** 2})

                with timer("batch.reduce", epoch_frac):
                    # Set 'grads' to the averaged value from the workers
                    if config['warm_reducer'] == True:
                        if epoch == 0:
                            bits_communicated += warm_start_reducer.reduce(send_buffers, grads, memories)
                        else:
                            bits_communicated += reducer.reduce(send_buffers, grads, memories, task.parameter_names)
                    else:
                        bits_communicated += reducer.reduce(send_buffers, grads, memories, task.parameter_names)
                if config["optimizer_memory"]:
                    with timer("batch.reporting.compr_err", verbosity=1):
                        if np.random.rand() < 0.001:
                            tags = {"weight": 'whole weight'}
                            rel_compression_error = l2norm(torch.cat([memory.view(-1) for memory in memories])) / l2norm(torch.cat([grad.view(-1) for grad in send_buffers]))
                            metric(
                                "rel_compression_error",
                                {"epoch": epoch_frac, "value": rel_compression_error.item() ** 2},
                                tags,
                            )

                if not config["optimizer_wd_before_reduce"]:
                    with timer("batch.wd", epoch_frac, verbosity=2):
                        for grad, param, wd in zip(grads, task.state, wds):
                            if wd > 0:
                                grad.add_(wd, param.detach())

                if not config["optimizer_mom_before_reduce"]:
                    with timer("batch.mom", epoch_frac, verbosity=2):
                        for grad, momentum in zip(grads, momenta):
                            if epoch == 0 and i == 0:
                                momentum.data = grad.clone().detach()
                            else:
                                if (
                                    config["optimizer_momentum_type"]
                                    == "exponential_moving_average"
                                ):
                                    momentum.mul_(config["optimizer_momentum"]).add_(
                                        1 - config["optimizer_momentum"], grad
                                    )
                                else:
                                    momentum.mul_(config["optimizer_momentum"]).add_(grad)
                            replace_grad_by_momentum(grad, momentum)

                with timer("batch.step", epoch_frac, verbosity=2):
                    for param, grad, lr in zip(task.state, grads, lrs):
                        param.data.add_(-lr, grad)

                if config["fix_conv_weight_norm"]:
                    with timer("batch.normfix", epoch_frac, verbosity=2):
                        for param_name, param in zip(task.parameter_names, task.state):
                            if is_conv_param(param_name):
                                param.data[:] /= l2norm(param)

                with timer("batch.update_runavg", epoch_frac, verbosity=2):
                    runavg_model.add(task.state_dict())

                if config["optimizer_memory"]:
                    with timer("batch.reporting.memory_norm", epoch_frac, verbosity=2):
                        if np.random.rand() < 0.001:
                            sum_of_sq = 0.0
                            for parameter_name, memory in zip(task.parameter_names, memories):
                                tags = {"weight": parameter_name.replace("module.", "")}
                                sq_norm = torch.sum(memory ** 2)
                                sum_of_sq += sq_norm
                                # metric(
                                #     "memory_norm",
                                #     {"epoch": epoch_frac, "value": sq_norm.item()},
                                #     tags,
                                # )
                            metric(
                                "compression_error",
                                {"epoch": epoch_frac, "value": sum_of_sq.item()},
                            )
                            
        with timer("epoch_metrics.collect", epoch + 1.0, verbosity=2):
            epoch_metrics.reduce()
            for key, value in epoch_metrics.value().items():
                metric(
                    key,
                    {"value": value, "epoch": epoch + 1.0, "bits": bits_communicated},
                    tags={"split": "train"},
                )
                metric(
                    f"last_{key}",
                    {"value": value, "epoch": epoch + 1.0, "bits": bits_communicated},
                    tags={"split": "train"},
                )

        with timer("test.last", epoch):
            test_stats = task.test()
            for key, value in test_stats.items():
                metric(
                    f"last_{key}",
                    {"value": value, "epoch": epoch + 1.0, "bits": bits_communicated},
                    tags={"split": "test"},
                )
        # mute average model test to save cuda memory
        # with timer("test.runavg", epoch):
        #     test_stats = task.test(state_dict=runavg_model.value())
        #     for key, value in test_stats.items():
        #         metric(
        #             f"runavg_{key}",
        #             {"value": value, "epoch": epoch + 1.0, "bits": bits_communicated},
        #             tags={"split": "test"},
        #         )

        if epoch in config["checkpoints"] and torch.distributed.get_rank() == 0:
            with timer("checkpointing"):
                save(
                    os.path.join(output_dir, "epoch_{:03d}".format(epoch)),
                    task.state_dict(),
                    epoch + 1.0,
                    test_stats,
                )
                # Save running average model @TODO

        print(timer.summary())
        if config["rank"] == 0:
            timer.save_summary(os.path.join(output_dir, "timer_summary.json"))

    info({"state.progress": 1.0})

    torch.distributed.destroy_process_group()

def save(destination_path, model_state, epoch, test_stats):
    """Save a checkpoint to disk"""
    # Workaround for RuntimeError('Unknown Error -1')
    # https://github.com/pytorch/pytorch/issues/10577
    time.sleep(1)

    torch.save(
        {"epoch": epoch, "test_stats": test_stats, "model_state_dict": model_state},
        destination_path,
    )


def get_weight_decay(epoch, parameter_name):
    """Take care of differences between weight decay for parameters"""
    if is_conv_param(parameter_name):
        return config["optimizer_weight_decay_conv"]
    elif is_batchnorm_param(parameter_name):
        return config["optimizer_weight_decay_bn"]
    else:
        return config["optimizer_weight_decay_other"]


def get_learning_rate(epoch, parameter_name):
    """Apply any learning rate schedule"""
    if is_conv_param(parameter_name):
        lr = config["optimizer_conv_learning_rate"]
    else:
        lr = config["optimizer_learning_rate"]

    if config["optimizer_scale_lr_with_warmup_epochs"]:
        warmup_epochs = config["optimizer_scale_lr_with_warmup_epochs"]
        max_factor = config.get("optimizer_scale_lr_with_factor", None)
        if max_factor is None:
            max_factor = (
                torch.distributed.get_world_size() if torch.distributed.is_available() else 1.0
            )
        factor = 1.0 + (max_factor - 1.0) * min(epoch / warmup_epochs, 1.0)
        lr *= factor

    for decay_epoch in config["optimizer_decay_at_epochs"]:
        if epoch >= decay_epoch:
            lr /= config["optimizer_decay_with_factor"]
        else:
            return lr
    return lr


def is_conv_param(parameter_name):
    """
    Says whether this parameter is a conv linear layer that 
    needs a different treatment from the other weights
    """
    return "conv" in parameter_name and "weight" in parameter_name

def is_batchnorm_param(parameter_name):
    """
    Is this parameter part of a batchnorm parameter?
    """
    return re.match(r""".*\.bn\d+\.(weight|bias)""", parameter_name)

def is_embed_param(parameter_name):
    """
    Says whether this parameter is a embedding layer that 
    needs a different treatment from the other weights
    """
    return "encoder.weight" in parameter_name

def replace_grad_by_momentum(grad, momentum):
    """
    Inplace operation that applies momentum to a gradient.
    This distinguishes between types of momentum (heavy-ball vs nesterov)
    """
    if config["optimizer_momentum_type"] == "heavy-ball":
        grad.data[:] = momentum
    if config["optimizer_momentum_type"] == "exponential_moving_average":
        grad.data[:] = momentum
    elif config["optimizer_momentum_type"] == "nesterov":
        grad.data[:] += momentum
    else:
        raise ValueError("Unknown momentum type")


def get_reducer(device, timer):
    """Configure the reducer from the config"""
    if config["optimizer_reducer"] in ["RankKReducer"]:
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            random_seed=config["seed"],
            device=device,
            timer=timer,
            n_power_iterations=config["optimizer_reducer_n_power_iterations"],
            reuse_query=config["optimizer_reducer_reuse_query"],
            rank=config["optimizer_reducer_rank"],
        )
    elif config["optimizer_reducer"] == "AtomoReducer":
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            random_seed=config["seed"],
            device=device,
            timer=timer,
            rank=config["optimizer_reducer_rank"],
        )
    elif config["optimizer_reducer"] == "RandomSparseReducer":
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            random_seed=config["seed"],
            device=device,
            timer=timer,
            rank=config["optimizer_reducer_rank"],
        )
    elif config["optimizer_reducer"] == "RandomSparseBlockReducer":
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            random_seed=config["seed"],
            device=device,
            timer=timer,
            rank=config["optimizer_reducer_rank"],
        )
    elif (
        config["optimizer_reducer"] == "GlobalTopKReducer"
        or config["optimizer_reducer"] == "TopKReducer"
        or config["optimizer_reducer"] == "UniformRandomSparseBlockReducer"
        or config["optimizer_reducer"] == "UniformRandomSparseReducer"
    ):
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            random_seed=config["seed"],
            device=device,
            timer=timer,
            compression=config["optimizer_reducer_compression"],
        )
    elif config["optimizer_reducer"] == "HalfRankKReducer":
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            random_seed=config["seed"],
            device=device,
            timer=timer,
            rank=config["optimizer_reducer_rank"],
        )
    elif config["optimizer_reducer"] == "SVDReducer":
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            config["seed"], device, timer, config["optimizer_reducer_rank"]
        )
    elif config["optimizer_reducer"] == "CASReducer":
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            config["seed"], device, timer, config["optimizer_reducer_rank"]
        )
    elif config["optimizer_reducer"] == "CAS_Layer_ResNet_Reducer":
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            config["seed"], device, timer, config["optimizer_reducer_rank"]
        )
    elif config["optimizer_reducer"] == "CAS_Layer_LM_Reducer":
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            config["seed"], device, timer, config["optimizer_reducer_rank"]
        )    
    elif config["optimizer_reducer"] == "CAS_Merge_Reducer":
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            config["seed"], device, timer, config["optimizer_reducer_rank"]
        )
    elif config["optimizer_reducer"] == "CAS_Fast_Reducer":
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            config["seed"], device, timer, config["optimizer_reducer_rank"]
        )
    elif config["optimizer_reducer"] == "Sketch_Embed_Reducer":
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            random_seed=config["seed"],
            device=device,
            timer=timer,
            reuse_sketch=config["optimizer_reducer_reuse_sketch"],
            rank=config["optimizer_reducer_rank"],
            sparsity = config["optimizer_sparsity"],
        )

    else:
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            config["seed"], device, timer
        )


@torch.jit.script
def l2norm(tensor):
    """Compute the L2 Norm of a tensor in a fast and correct way"""
    # tensor.norm(p=2) is buggy in Torch 1.0.0
    # tensor.norm(p=2) is really slow in Torch 1.0.1
    return torch.sqrt(torch.sum(tensor ** 2))


def log_info(info_dict):
    """Add any information to MongoDB
       This function will be overwritten when called through run.py"""
    pass


def log_metric(name, values, tags={}):
    """Log timeseries data
       This function will be overwritten when called through run.py"""
    value_list = []
    for key in sorted(values.keys()):
        value = values[key]
        value_list.append(f"{key}:{value:7.3f}")
    values = ", ".join(value_list)
    tag_list = []
    for key, tag in tags.items():
        tag_list.append(f"{key}:{tag}")
    tags = ", ".join(tag_list)
    print("{name:30s} - {values} ({tags})".format(name=name, values=values, tags=tags))


def info(*args, **kwargs):
    if config["rank"] == 0:
        log_info(*args, **kwargs)


def metric(*args, **kwargs):
    if config["rank"] == 0:
        log_metric(*args, **kwargs)


def check_model_consistency_across_workers(model, epoch):
    signature = []
    for name, param in model.named_parameters():
        signature.append(param.view(-1)[0].item())

    rank = config["rank"]
    signature = ",".join(f"{x:.4f}" for x in signature)
    print(f"Model signature for epoch {epoch:04d} / worker {rank:03d}:\n{signature}")

def tensor_nnz(grad):
    return grad.to_sparse(), torch.numel(grad.to_sparse()._indices()[0].unique())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--local_world_size", type=int, default=1)
    args = parser.parse_args()

    config['local_rank'] = args.local_rank
    os.environ['CUDA_DEVICE'] = str(config['local_rank'])
    import gradient_reducers

    main()
