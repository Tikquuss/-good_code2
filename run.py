import itertools
import submitit
from multiprocessing import Process
from functools import partial
import os

from train import train, get_default_params, create_data_module, AttrDict

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

def distribute_devices(commands, p_devices, n_tasks_per_device=1, lunch=False, function=None) :
    is_list = type(p_devices) == list
    
    ids_for_task = []

    if is_list :
        assert n_tasks_per_device >= 1
        ##
        if n_tasks_per_device == 1 :
            i, L = 0, len(p_devices)
            for j in range(len(commands)):
                commands[j].devices = [p_devices[i]]
                ids_for_task.append(i)
                i=(i+1)%L
        else :
            devices = []
            for id_task in range(n_tasks_per_device) :
                for id_device in p_devices : devices.append((id_device, id_task))
            i, L = 0, len(devices)
            for j in range(len(commands)):
                id_device, id_task = devices[i]
                commands[j].devices = [id_device]
                ids_for_task.append(f"{id_device}-{id_task}")
                i=(i+1)%L
    ##
    if lunch :
        if is_list :
            task_list_dict = {}
            for i, command in enumerate(commands) :
                task_list_dict[ids_for_task[i]] = task_list_dict.get(ids_for_task[i], []) + [command]

            for k, v in task_list_dict.items() : print(k, len(v), v)
            run_task_dict(function, task_list_dict)
        else :
            for command in commands: train(command)

    return commands, ids_for_task

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
    params.max_epochs=25000
    params.every_n_epochs=5000 # save every x epochs
    params.accelerator="gpu" #"auto"

    ############################## important #####
    """
    The current implementation does not support multi-devices training 
    (I just don't trust the results produced in that case). 
    So when we pass a list of gpus (tpu, ...) as devices, we distribute them in 
    such a way that each run is only done on one gpu (see below).
    When the slurm partition is set to none, everything is launched as standard 

    That said, it is better to pass the list of devices id than the number of devices to use.
    """
    #params.devices = "auto"
    # list of device id, or the number of devices to use
    params.devices = [0] # "auto"
    #params.devices = 1 # "auto"

    slurm_partition = None # TODO

    if slurm_partition is not None :
        data_module, data_flag = create_data_module(params)

    """
    If you don't intend to use slurm, but still want multiple runs in parallel on each device, 
    specify the number of runs per device.
    In fact, each run of grokking is cheap, so parallelizing can speed things up
    This only works if slurm is not used, since I implemented this feature myself (see the functions above)
    """
    n_tasks_per_device = 1
    ###################### end important ###

    params.use_wandb=True#False
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

        ####
        checkpoint_path = params.logdir + "/checkpoints"
        os.makedirs(checkpoint_path, exist_ok=True)
        setattr(params, "checkpoint_path", checkpoint_path)
        #params.save_top_k = -1
        setattr(params, "save_top_k", -1)
        setattr(params, "external_call", True)
        ###

        for random_seed in [0, 100] :
            params.random_seed=random_seed
            
            #commands.append(params)
            command = AttrDict()
            for att_name in params.keys() : setattr(command, att_name, getattr(params, att_name))
            commands.append(command)

    # run experiments
    """
    The current implementation does not support multi-devices training 
    (In fact, I don't trust the results produced in that case). 
    So when we pass a list of gpus (tpu, ...) as devices, we distribute them in 
    such a way that each run is only done on one gpu

    When the slurm partition is set to none, everything is launched as standard 
    """
    commands, ids_for_task = distribute_devices(
        commands, params.devices, 
        n_tasks_per_device=n_tasks_per_device, 
        lunch = slurm_partition is None, 
        function=train
    )
    # slurm
    # https://github.com/facebookincubator/submitit/blob/main/docs/examples.md
    slurm_output_dir = f"{dump_path}/logs" # TODO
    max_time = None # TODO
    if slurm_partition is not None:
        executor = submitit.SlurmExecutor(folder=slurm_partition)
        executor.update_parameters(
            time=max_time,
            gpus_per_node=1,
            array_parallelism=512,
            cpus_per_task=4,
            partition=slurm_partition)
        executor.map_array(train, commands)