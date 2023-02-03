# 
import re
import inspect
from torch import optim

# 
import torch
import math

from .optimizers.sag import SimpleSAG, SAGBase, SAGWithd
from .optimizers.sgd import CustomSGD, SAGSGDBase, SAGSGDWithd
from .optimizers.adam import CustomAdam, \
    SAGAdamSimpleBase, SAGAdamBase, \
    SAGAdamSimpleWithd, SAGAdamWithd, \
    AdamCosineWithWarmup, AdamInverseSqrtWithWarmup


class CustomAdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        noise_factor=0.0,
        weight_decay_form="to_zero",
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not weight_decay_form in ["to_zero", "to_init", "jiggle", "honest"]:
            raise ValueError(
                f"Invalid weight decay form: {weight_decay_form}, should be one of ['to_zero', 'to_init', 'jiggle']"
            )
        # if not 0.0 <= weight_decay:
        #     raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            noise_factor=noise_factor,
            weight_decay_form=weight_decay_form,
        )
        super(CustomAdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(CustomAdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perform optimization step
                grad = p.grad

                if group["weight_decay"] > 0:
                    if group["weight_decay_form"] == "honest":
                        grad = grad + group["weight_decay"] * p.detach()

                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                amsgrad = group["amsgrad"]

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if group["weight_decay_form"] == "to_init":
                        state["init"] = p.detach().clone()
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                if group["weight_decay"] > 0:
                    if group["weight_decay_form"] == "to_zero":
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    elif group["weight_decay_form"] == "to_init":
                        p.add_(
                            (state["init"] - p) * (group["lr"] * group["weight_decay"])
                        )
                    elif group["weight_decay_form"] == "jiggle":
                        p.mul_(
                            torch.exp(
                                torch.randn(1).cuda()
                                * (group["lr"] * group["weight_decay"])
                            )
                        )
                    elif group["weight_decay_form"] == "honest":
                        pass
                    else:
                        raise ValueError(
                            f"Invalid weight decay form: {group['weight_decay_form']}"
                        )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                        group["eps"]
                    )
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                        group["eps"]
                    )

                step_size = group["lr"] / bias_correction1

                upd = exp_avg / denom
                # add uniform gaussian noise to the update
                if group["noise_factor"] > 0:
                    upd += torch.randn_like(upd) * group["noise_factor"]
                # if group['noise_factor'] > 0:
                #     upd *= torch.exp(torch.randn_like(upd) * group['noise_factor'])
                p.add_(-step_size * upd)

        return loss


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert (
            closure is not None
        ), "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(
            closure
        )  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0
        ].device  # put everything on the same device, in case of model parallelism
        grad_norms = [
            p.grad.norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        print("grad norms is ", grad_norms, "!" * 1000)
        norm = torch.norm(
            torch.stack(grad_norms),
            p=2,
        )
        return norm

class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

def get_params_from_string(s : str, separator = ",", have_method=True):
    re_float="^[+-]?(\d+(\.\d*)?|\.\d+)$"
    # https://stackoverflow.com/a/41668652/11814682
    re_scient="[+\-]?[^A-Za-z]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)"
    if "," in s:
        if have_method :
            method = s[:s.find(separator)]
            s = s[s.find(separator) + 1:]
        else :
            method = ""
        optim_params = {}
        for x in s.split(separator):
            split = x.split('=')
            assert len(split) == 2
            try:
                float(split[1])
                assert (re.match(re_float, split[1]) is not None) or (re.match(re_scient, split[1]) is not None)
                optim_params[split[0]] = float(split[1])
            except ValueError:
                optim_params[split[0]] = split[1]
    else:
        method = s
        optim_params = {}
        
    return method, optim_params

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {'off', 'false', '0'}
    TRUTHY_STRINGS = {'on', 'true', '1'}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise TypeError("Invalid value for a boolean flag!")

def get_optimizer(parameters, s, noamopt=""):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.001"
        - "adagrad,lr=0.001,lr_decay=0.05"
    noamopt : Optim wrapper that implements rate.
        -"factor_ae=1,warmup_ae=200"
    """
    method, optim_params = get_params_from_string(s, separator = ",", have_method=True)
    #al = "sgd","asgd","rmsprop","rprop","adadelta","adagrad","adam","adamax","custom_adam","adam_inverse_sqrt","adam_cosine","sag"
    #al = "sgd","asgd","rmsprop","rprop","adadelta","adagrad","adam","adamax","sag"
    if method == 'sgd':
        # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
        #"lr=,momentum=0,dampening=0,weight_decay=0,nesterov=False"
        optim_params["nesterov"] = bool_flag(optim_params.get("nesterov", 'False'))
        optim_fn = optim.SGD
        #optim_fn = CustomSGD
        assert 'lr' in optim_params
    elif method == 'asgd':
        # https://pytorch.org/docs/stable/generated/torch.optim.ASGD.html
        # https://sci-hub.se/10.1137/0330046
        # https://www.quora.com/How-does-Averaged-Stochastic-Gradient-Decent-ASGD-work
        # lr=,lambd=0.0001,alpha=0.75,t0=1000000.0,weight_decay=0
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        # https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html
        #lr=,alpha=0.99,weight_decay=0,momentum=0,centered=False
        optim_params["centered"] = bool_flag(optim_params.get("centered", 'False'))
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        # https://pytorch.org/docs/stable/generated/torch.optim.Rprop.html
        # lr=,etas=(0.5, 1.2), step_sizes=(1e-06, 50)
        # lr=,etaplus=0.5,etaminus=1.2,step_min=1e-06,step_max=50
        optim_params['etas'] = (optim_params.pop('etaplus', 0.5), optim_params.pop('etaminus', 1.2))
        optim_params['step_sizes'] = (optim_params.pop('step_min', 1e-06), optim_params.pop('step_max', 50))
        optim_fn = optim.Rprop
    elif method == 'adadelta':
        # https://pytorch.org/docs/stable/generated/torch.optim.Adadelta.html
        # lr=,rho=0.9,weight_decay=0
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        # https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html
        # lr=,lr_decay=0,weight_decay=0,initial_accumulator_value=0
        optim_fn = optim.Adagrad
    elif method in ['adam', 'adamax', 'custom_adam', 'adam_inverse_sqrt', 'adam_cosine']:
        # lr, betas,weight decay,amsgrad
        optim_params['betas'] = (optim_params.pop('beta1', 0.9), optim_params.pop('beta2', 0.999))
        if method == 'adam' :
            # lr=0.001, betas=(0.9,0.999),eps=1e-08,weight_decay=0,amsgrad=False 
            optim_params["amsgrad"] = bool_flag(optim_params.get("amsgrad", 'False'))
            optim_fn = optim.Adam
        elif method == 'adamax':
            # https://pytorch.org/docs/stable/generated/torch.optim.Adamax.html
            # lr=0.002, betas=(0.9,0.999),eps=1e-08,weight_decay=0
            optim_fn = optim.Adamax
        elif method == 'custom_adam' :
            # lr=,betas=(0.9,0.999),eps=1e-8,weight_decay=0
            optim_fn = CustomAdam 
        elif method == 'adam_inverse_sqrt':
            # lr=,betas=(0.9,0.999),eps=1e-8,weight_decay=0,warmup_updates=4000,warmup_init_lr=1e-7,exp_factor=0.5
            optim_fn = AdamInverseSqrtWithWarmup
        elif method == 'adam_cosine':
            # lr=,betas=(0.9, 0.999),eps=1e-8,weight_decay=0,warmup_updates=4000,warmup_init_lr=1e-7,min_lr=1e-9,init_period=1000000,period_mult=1,lr_shrink=0.75
            optim_fn = AdamCosineWithWarmup
    elif "sag" in method :
        if method == 'sagbase':
            optim_fn = SAGBase
        else :
            assert 'batch_mode' in optim_params
            assert 'init_y_i' in optim_params
            optim_params["batch_mode"] = bool_flag(optim_params["batch_mode"])
            optim_params["init_y_i"] = bool_flag(optim_params["init_y_i"])
            optim_params["sum_all"] = bool_flag(optim_params.get("sum_all", 'False'))
            with_d = bool_flag(optim_params.pop("with_d", 'True'))
            if method == 'sag' : 
                # good without decay
                # SAGWithd is slightly better than SAGBase
                optim_fn = SAGWithd if with_d else SAGBase 
            elif method == 'sag_sgd' :
                optim_params["nesterov"] = bool_flag(optim_params.get("nesterov", 'False'))
                # good without decay
                # SAGSGDWithd is slightly better than SAGSGDBase
                optim_fn =  SAGSGDWithd if with_d else SAGSGDBase
            elif method == 'sag_adam' :
                optim_params['betas'] = (optim_params.pop('beta1', 0.9), optim_params.pop('beta2', 0.999))
                #optim_params['weight_decay'] = 1.0 
                if with_d :
                    optim_fn = SAGAdamSimpleWithd # good, but unstable
                    #optim_fn = SAGAdamWithd # good, but unstable
                else :
                    optim_fn = SAGAdamSimpleBase # good
                    #optim_fn = SAGAdamBase # bad
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    try : 
        expected_args = inspect.getargspec(optim_fn.__init__)[0]
    except ValueError: #Function has keyword-only parameters or annotations, use getfullargspec() API which can support them
        expected_args = inspect.getfullargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    if noamopt != "" :
        _, params = get_params_from_string(s=noamopt, have_method=False)
        return NoamOpt(params["d_model"], params["factor_ae"], params["warmup_ae"], optim_fn(parameters, **optim_params))
    else :
        return optim_fn(parameters, **optim_params)


def get_lr_scheduler(optimizer, s) :
    """
    Parse lr_scheduler parameters.
    Input should be of the form:
        - "reduce_lr_on_plateau,factor=0.2,patience=20,min_lr=0.00005,mode=min,monitor=val_loss"

    """
    method, scheduler_params = get_params_from_string(s, separator = ",", have_method=True)
    monitor = scheduler_params.pop("monitor", "val_loss")
    if method == "reduce_lr_on_plateau" :
        # Reduce learning rate when a metric has stopped improving.
        scheduler_fn =  optim.lr_scheduler.ReduceLROnPlateau
    elif method == "constant_lr" :
        # Decays the learning rate of each parameter group by a small constant factor until the number of epoch reaches a pre-defined milestone: total_iters.
        scheduler_fn =  optim.lr_scheduler.ConstantLR
    elif method == "linear_lr" :
        # Decays the learning rate of each parameter group by linearly changing small multiplicative factor until the number of epoch reaches a pre-defined milestone: total_iters.
        scheduler_fn =  optim.lr_scheduler.LinearLR
    elif method == "cosine_annealing_lr" :
        # Set the learning rate of each parameter group using a cosine annealing schedule, where η_max is set to the initial lr and T_{cur} is the number of epochs since the last restart in SGDR
        scheduler_fn =  optim.lr_scheduler.CosineAnnealingLR
    elif method == "exponential_lr" :
        # Decays the learning rate of each parameter group by gamma every epoch.
        scheduler_fn =  optim.lr_scheduler.ExponentialLR
    elif method == "lambda_lr" :
        # Sets the learning rate of each parameter group to the initial lr times a given function.
        scheduler_fn = optim.lr_scheduler.LambdaLR
    elif method == "multiplicative_lr" :
        # Multiply the learning rate of each parameter group by the factor given in the specified function.
        scheduler_fn = optim.lr_scheduler.MultiplicativeLR
    elif method == "step_lr" :
        # Decays the learning rate of each parameter group by gamma every step_size epochs.
        scheduler_fn = optim.lr_scheduler.StepLR
    elif method == "multi_step_lr" :
        # Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones.
        scheduler_fn = optim.lr_scheduler.MultiStepLR
    elif method == "cyclic_lr" :
        # Sets the learning rate of each parameter group according to cyclical learning rate policy (CLR).
        scheduler_fn = optim.lr_scheduler.CyclicLR
    elif method == "one_cycle_lr" :
        # Sets the learning rate of each parameter group according to the 1cycle learning rate policy.
        scheduler_fn = optim.lr_scheduler.OneCycleLR
    elif method == "cosine_annealing_warm_restarts" :
        # Set the learning rate of each parameter group using a cosine annealing schedule, where η_max is set to the initial lr, T_cur is the number of epochs since the last restart and 
        # T_i is the number of epochs between two warm restarts in SGDR
        scheduler_fn = optim.lr_scheduler.CosineAnnealingWarmRestarts
    # elif method == "chained_scheduler" :
    #     # Chains list of learning rate schedulers.
    #     scheduler_fn = optim.lr_scheduler.ChainedScheduler
    # elif method == "sequential_lr" :
    #     # Receives the list of schedulers that is expected to be called sequentially during optimization process and milestone points that provides exact intervals to reflect which scheduler is supposed to be called at a given epoch.
    #     scheduler_fn = optim.lr_scheduler.SequentialLR
    else:
        raise Exception('Unknown lr scheduler method: "%s"' % method)

    # check that we give good parameters to the optimizer
    try : 
        expected_args = inspect.getargspec(scheduler_fn.__init__)[0]
    except ValueError: #Function has keyword-only parameters or annotations, use getfullargspec() API which can support them
        expected_args = inspect.getfullargspec(scheduler_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'optimizer']
    if not all(k in expected_args[2:] for k in scheduler_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(scheduler_params.keys())))

    return scheduler_fn(optimizer, **scheduler_params), monitor

def configure_optimizers(parameters, s_optim, s_lr_scheduler):
    optimizer = get_optimizer(parameters, s_optim)
    if s_lr_scheduler :
        scheduler, monitor = get_lr_scheduler(optimizer, s_lr_scheduler)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": monitor}
    else :
        return {"optimizer": optimizer}


