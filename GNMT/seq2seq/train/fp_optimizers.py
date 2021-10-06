import logging
import math
from time import time, strftime

import torch
from torch.autograd import grad
from torch.nn.utils import clip_grad_norm_

import seq2seq.train.gradient_reducers as gradient_reducers

from seq2seq.train.compression_config import config
import numpy as np
def get_reducer(device, timer, compression_method):
    """Configure the reducer from the config"""
    if config["optimizer_reducer"] == "RankKReducer":
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            random_seed=config["seed"],
            device=device,
            timer=timer,
            n_power_iterations=config["optimizer_reducer_n_power_iterations"],
            reuse_query=config["optimizer_reducer_reuse_query"],
            rank=config["optimizer_reducer_rank"],
        )
    elif config["optimizer_reducer"] == "Multi_RankKReducer":
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            random_seed=config["seed"],
            device=device,
            timer=timer,
            n_power_iterations=config["optimizer_reducer_n_power_iterations"],
            reuse_query=config["optimizer_reducer_reuse_query"],
            rank=config["optimizer_reducer_rank"],
        )
    elif config["optimizer_reducer"] == "RankKSparseReducer":
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            random_seed=config["seed"],
            device=device,
            timer=timer,
            n_power_iterations=config["optimizer_reducer_n_power_iterations"],
            reuse_query=config["optimizer_reducer_reuse_query"],
            rank=config["optimizer_reducer_rank"],
            sparsity = config["optimizer_sparsity"],
        )
    elif config["optimizer_reducer"] == "Sketch_Embed_Reducer":
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            random_seed=config["seed"],
            device=device,
            timer=timer,
            reuse_sketch=config["optimizer_reducer_reuse_sketch"],
            rank=config["optimizer_reducer_rank"],
            sparsity = config["embedding_sparsity"],
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
    
class Fp16Optimizer:
    """
    Mixed precision optimizer with dynamic loss scaling and backoff.
    https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#scalefactor
    """
    @staticmethod
    def set_grads(params, params_with_grad):
        """
        Copies gradients from param_with_grad to params

        :param params: dst parameters
        :param params_with_grad: src parameters
        """
        for param, param_w_grad in zip(params, params_with_grad):
            if param.grad is None:
                param.grad = torch.nn.Parameter(torch.empty_like(param))
            param.grad.data.copy_(param_w_grad.grad.data)

    @staticmethod
    def set_weights(params, new_params):
        """
        Copies parameters from new_params to params

        :param params: dst parameters
        :param new_params: src parameters
        """
        for param, new_param in zip(params, new_params):
            param.data.copy_(new_param.data)

    def __init__(self, fp16_model, grad_clip=float('inf'), loss_scale=8192,
                 dls_downscale=2, dls_upscale=2, dls_upscale_interval=128):
        """
        Constructor for the Fp16Optimizer.

        :param fp16_model: model (previously casted to half)
        :param grad_clip: coefficient for gradient clipping, max L2 norm of the
            gradients
        :param loss_scale: initial loss scale
        :param dls_downscale: loss downscale factor, loss scale is divided by
            this factor when NaN/INF occurs in the gradients
        :param dls_upscale: loss upscale factor, loss scale is multiplied by
            this factor if previous dls_upscale_interval batches finished
            successfully
        :param dls_upscale_interval: interval for loss scale upscaling
        """
        logging.info('Initializing fp16 optimizer')
        self.initialize_model(fp16_model)

        self.since_last_invalid = 0
        self.loss_scale = loss_scale
        self.dls_downscale = dls_downscale
        self.dls_upscale = dls_upscale
        self.dls_upscale_interval = dls_upscale_interval
        self.grad_clip = grad_clip

    def initialize_model(self, model):
        """
        Initializes internal state and build fp32 master copy of weights.

        :param model: fp16 model
        """
        logging.info('Initializing fp32 clone weights')
        self.fp16_model = model
        self.fp16_model.zero_grad()
        self.fp32_params = [param.to(torch.float32).detach()
                            for param in model.parameters()]

        for param in self.fp32_params:
            param.requires_grad = True

    def step(self, loss, optimizer, scheduler, update=True):
        """
        Performs one step of the optimizer.
        Applies loss scaling, computes gradients in fp16, converts gradients to
        fp32, inverts scaling and applies optional gradient norm clipping.
        If gradients are finite, it applies update to fp32 master weights and
        copies updated parameters to fp16 model for the next iteration. If
        gradients are not finite, it skips the batch and adjusts scaling factor
        for the next iteration.

        :param loss: value of loss function
        :param optimizer: optimizer
        :param update: if True executes weight update
        """
        loss *= self.loss_scale
        loss.backward()

        if update:
            self.set_grads(self.fp32_params, self.fp16_model.parameters())
            if self.loss_scale != 1.0:
                for param in self.fp32_params:
                    param.grad.data /= self.loss_scale

            norm = clip_grad_norm_(self.fp32_params, self.grad_clip)

            if math.isfinite(norm):
                scheduler.step()
                optimizer.step()
                self.set_weights(self.fp16_model.parameters(),
                                 self.fp32_params)
                self.since_last_invalid += 1
            else:
                self.loss_scale /= self.dls_downscale
                self.since_last_invalid = 0
                logging.info(f'Gradient norm: {norm}')
                logging.info(f'Skipped batch, new scale: {self.loss_scale}')

            if self.since_last_invalid >= self.dls_upscale_interval:
                self.loss_scale *= self.dls_upscale
                self.loss_scale = min(self.loss_scale, 8192.0)
                logging.info(f'Upscaling, new scale: {self.loss_scale}')
                self.since_last_invalid = 0

            self.fp16_model.zero_grad()


class Fp32Optimizer:
    """
    Standard optimizer, computes backward and applies weight update.
    """
    def __init__(self, model, grad_clip=None, local_rank=0, timer=None, optimizer_memory=True):
        """
        Constructor for the Fp32Optimizer

        :param model: model
        :param grad_clip: coefficient for gradient clipping, max L2 norm of the
            gradients
        """
        logging.info('Initializing fp32 optimizer')
        self.initialize_model(model)
        self.parameter_names = [name for (name, _) in model.named_parameters()]
        # if torch.distributed.get_rank()==0:
        #     for name, param in model.named_parameters():
        #         print(f'{name}: {param.size()}')
        self.grad_clip = grad_clip
        self.memories = [torch.zeros_like(param) for param in model.parameters()]
        # self.momenta = [torch.empty_like(param) for param in task.state]  # need initialization
        self.send_buffers = [torch.zeros_like(param) for param in model.parameters()]
        self.bits_communicated = 0
        self.optimizer_memory = optimizer_memory
        self.timer = timer
        self.device = torch.device(local_rank)
        self.reducer = get_reducer(self.device, self.timer, config["optimizer_reducer"])
        
        

    def initialize_model(self, model):
        """
        Initializes state of the model.

        :param model: model
        """
        self.model = model
        self.model.zero_grad()

    def step(self, loss, optimizer, scheduler, update=True, epoch=0):
        """
        Performs one step of the optimizer.

        :param loss: value of loss function
        :param optimizer: optimizer
        :param update: if True executes weight update
        """
        with self.timer("batch.backward", epoch):
            loss.backward()
        if update:
            if self.grad_clip != float('inf'):
                clip_grad_norm_(self.model.parameters(), self.grad_clip)
            # get grad
            grads = [parameter.grad for parameter in self.model.parameters()]
            # error compensation
            with self.timer("batch.reduce", epoch):
                for grad, memory, send_bfr in zip(grads, self.memories, self.send_buffers):
                    if self.optimizer_memory == True:
                        send_bfr.data[:] = grad + memory
                    else:
                        send_bfr.data[:] = grad

                self.bits_communicated += self.reducer.reduce(self.send_buffers, grads, self.memories)

                                        
                # if torch.distributed.get_rank()==0:
                #     for (name, param), memory, send_bfr in zip(self.model.named_parameters(), self.memories, self.send_buffers):
                #             # np.save(str(time())+ '_grad.npy', named[1].grad.to('cpu').numpy())
                #             # np.save(str(time())+ '_memory.npy', memory.to('cpu').numpy())
                #             # np.save(str(time())+ '_sendbuffer.npy', send_bfr.to('cpu').numpy())
                            
                #             np.save(str(time()) + name + '.npy', param.grad.to('cpu').numpy())
                
                # for name, memory, send_bfr in zip(
                #                 self.parameter_names, self.memories, self.send_buffers
                #             ):
                #             if 'weight' in name and torch.distributed.get_rank() == 0:
                #                 tags = {"weight": name.replace("module.", "")}
                #                 rel_compression_error = l2norm(memory) / l2norm(send_bfr)
                #                 logging.info(f'[rel_compression_error] epoch: {epoch} value: {rel_compression_error.item()} {tags}')

                # param updating is included in the synchronization time
                
                optimizer.step()
                scheduler.step()
            self.model.zero_grad()

class Sketch_Embed_Optimizer:
    """
    get the dense gradient, convert embedding gradient to sparse, and compress it with sketch wo communicate with allreduce, allreduce the other gradient normally
    """
    def __init__(self, model, grad_clip=None, local_rank=0, timer=None, optimizer_memory=False):
        """
        Constructor for the Optimizer

        :param model: model
        :param grad_clip: coefficient for gradient clipping, max L2 norm of the
            gradients
        """
        logging.info('Initializing fp32 optimizer')
        self.initialize_model(model)
        self.parameter_names = [name for (name, _) in model.named_parameters()]
        # if torch.distributed.get_rank()==0:
        #     for name, param in model.named_parameters():
        #         print(f'{name}: {param.size()}')
        self.grad_clip = grad_clip
        # self.memories = [torch.zeros_like(param) for param in model.parameters()]
        # self.momenta = [torch.empty_like(param) for param in task.state]  # need initialization
        # self.send_buffers = [torch.zeros_like(param) for param in model.parameters()]
        self.bits_communicated = 0
        self.optimizer_memory = optimizer_memory
        self.timer = timer
        self.device = torch.device(local_rank)
        self.reducer = get_reducer(self.device, self.timer, config["optimizer_reducer"])
        
        

    def initialize_model(self, model):
        """
        Initializes state of the model.

        :param model: model
        """
        self.model = model
        self.model.zero_grad()

    def step(self, loss, optimizer, scheduler, update=True, epoch=0):
        """
        Performs one step of the optimizer.

        :param loss: value of loss function
        :param optimizer: optimizer
        :param update: if True executes weight update
        """
        with self.timer("batch.backward", epoch):
            loss.backward()
        if update:
            if self.grad_clip != float('inf'):
                clip_grad_norm_(self.model.parameters(), self.grad_clip)
            # get grad
            named_grads = [(name, parameter.grad) for (name, parameter) in self.model.named_parameters()]
            # error compensation
            with self.timer("batch.reduce", epoch):
                # for grad, send_bfr in zip(grads, self.send_buffers):
                #     send_bfr.data[:] = grad

                self.bits_communicated += self.reducer.reduce(named_grads)

                                        
                # if torch.distributed.get_rank()==0:
                #     for (name, param), memory, send_bfr in zip(self.model.named_parameters(), self.memories, self.send_buffers):
                #             # np.save(str(time())+ '_grad.npy', named[1].grad.to('cpu').numpy())
                #             # np.save(str(time())+ '_memory.npy', memory.to('cpu').numpy())
                #             # np.save(str(time())+ '_sendbuffer.npy', send_bfr.to('cpu').numpy())
                            
                #             np.save(str(time()) + name + '.npy', param.grad.to('cpu').numpy())
                
                # for name, memory, send_bfr in zip(
                #                 self.parameter_names, self.memories, self.send_buffers
                #             ):
                #             if 'weight' in name and torch.distributed.get_rank() == 0:
                #                 tags = {"weight": name.replace("module.", "")}
                #                 rel_compression_error = l2norm(memory) / l2norm(send_bfr)
                #                 logging.info(f'[rel_compression_error] epoch: {epoch} value: {rel_compression_error.item()} {tags}')

                # param updating is included in the synchronization time
                
                optimizer.step()
                scheduler.step()
            self.model.zero_grad()