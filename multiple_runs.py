import itertools
import submitit
from multiprocessing import Process
from functools import partial

from train import train, get_default_params, AttrDict

def run_task_list(function, task_list, id=None) :
    for command in task_list : function(command)

def run_task_dict(function, task_list_dict) :
    all_process = []
    for id, task_list in task_list_dict.items() :
        task = partial(run_task_list, function, task_list, id)
        p = Process(target=task)
        p.start()
        all_process.append(p)
    for p in all_process : p.join()

if __name__ == "__main__":

    params = get_default_params()

    #################################################
    ############## run experiment ###################
    #################################################
    # print()
    # for k, v in vars(params).items() : print(k, " --> ", v)
    # print()
    # logdir = train(params)
    # hparams = torch.load(logdir + "/hparams.pt")
    # data_module = torch.load(logdir+"/data.pt")
    # states = torch.load(logdir+"/states.pt")
    
    #################################################
    ############## phase diagram ###################
    #################################################

    params.train_data_pct=40
    params.math_operator="+"
    params.dropout=0.0
    params.opt="adamw"
    max_lr=0.001
    
    dump_path=".."
    params.max_epochs=20000
    params.every_n_epochs=5000 # save every x epochs
    params.accelerator = "gpu" #"auto"

    #params.devices = "auto"
    # list of device id, or the number of devices to use
    params.devices = [0] # "auto"
    #params.devices = 1 # "auto"

    params.use_wandb=False#True
    #params.group_name=f"wd={params.weight_decay}-lr={params.max_lr}"
    params.wandb_entity="grokking_ppsp"
    params.wandb_project=f"grokking_wd_lr={params.math_operator}-{params.train_data_pct}"
    
    #lrs = [0.001]
    lrs = [0.001, 0.002, 0.003, 0.004, 0.005] 
    #lrs = np.linspace(start=1e-1, stop=1e-5, num=10)

    #wds = [0.0]
    wds = [0.0, 0.2, 0.4, 0.7, 0.9, 1.1, 1.3, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    #wds = list(range(20))
    #wds = np.linspace(start=0, stop=20, num=21)

    print(lrs, wds)

    commands = []
    for max_lr, weight_decay in itertools.product(lrs, wds) :

        params["max_lr"] = max_lr 
        params["weight_decay"] = weight_decay
        params["group_name"] = f"wd={weight_decay}-lr={max_lr}"

        params.logdir=f"{dump_path}/logs/{params.train_data_pct}/{params.group_name}"
        params.datadir=f"{dump_path}/data/{params.train_data_pct}/{params.group_name}"

        for random_seed in [0, 100] :
            params.random_seed=random_seed
            
            #commands.append(params)
            command = AttrDict()
            for att_name in params.keys() : setattr(command, att_name, getattr(params, att_name))
            commands.append(command)

    """
    The current implementation does not support multi-devices training. 
    So when we pass a list of gpus (tpu, ...) as devices, we distribute them in 
    such a way that each run is only done on one gpu
    """
    ids_for_task = []
    if type(params.devices) == list :
        L = len(params.devices)
        i = 0
        for j in range(len(commands)):
            commands[j].devices = [params.devices[i]]
            ids_for_task.append(i)
            i%=L

    ## run experiment

    slurm_partition = None # TODO
    if slurm_partition is not None:
        # slurm
        # https://github.com/facebookincubator/submitit/blob/main/docs/examples.md
        slurm_output_dir = f"{dump_path}/logs" # TODO
        max_time = None # TODO
        executor = submitit.SlurmExecutor(folder=slurm_partition)
        executor.update_parameters(
            time=max_time,
            gpus_per_node=1,
            array_parallelism=512,
            cpus_per_task=4,
            partition=slurm_partition)
        executor.map_array(train, commands)
    else:
        # When we have multiple devices as described above, we launch in parallel a sequence of runs on each device
        if type(params.devices) == list :
            task_list_dict = {}
            for i, command in enumerate(commands) :
                task_list_dict[ids_for_task[i]] = task_list_dict.get(ids_for_task[i], []) + [command]
            run_task_dict(train, task_list_dict)
        else :
            for command in commands: train(command)