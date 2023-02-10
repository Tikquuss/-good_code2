# Usage : python train_parallel.py --parallel True/False

from multiprocessing import Process
from functools import partial
import subprocess

from argparse import ArgumentParser

from src.utils import bool_flag

SCRIPT_NAME="train.sh"
SCRIPT_PATH=f"./{SCRIPT_NAME}"

result = subprocess.run(f'chmod +x {SCRIPT_NAME}', shell=True, capture_output=True, text=True)
print(result)

def run_train(train_data_pct, math_operator, weight_decay, dropout, opt, max_lr, random_seed):
    
    group_name=f"tdp={train_data_pct}-wd={weight_decay}-d={dropout}-opt={opt}-mlr={max_lr}-mo{math_operator}"
    print("Start Group name %s"%group_name)
    print(f"Random seed : {random_seed}")

    command=f"{SCRIPT_PATH} {train_data_pct} {math_operator} {weight_decay} {dropout} {opt} {max_lr} {random_seed}"
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    stdoutdata, _ = process.communicate()

    if process.returncode != 0 :
        print("Error %s"%group_name)
    else :
        print("Success %s"%group_name)

    print("Finish Group name %s"%group_name)

    output = stdoutdata.decode("utf-8")
    print("*"*10)
    print(output)
    print("*"*10,"\n")

    #return stdoutdata

if __name__ == '__main__':
    parser = ArgumentParser(description="Grokking")
    parser.add_argument("--parallel", type=bool_flag, default=False)
    parallel = parser.parse_args().parallel

    math_operator = "+"

    all_process = []
    for train_data_pct in [80] :
        for weight_decay in [0.0] :
            for dropout in [0.0] : 
                for opt in ["adamw"] :
                    for max_lr in [0.001] :
                        for random_seed in [0, 100] :
                            if not parallel : 
                                run_train(
                                    train_data_pct, math_operator, weight_decay, dropout, opt, max_lr, random_seed
                                )
                            else :
                                task = partial(
                                    run_train, 
                                    train_data_pct, math_operator, weight_decay, dropout, opt, max_lr, random_seed
                                )
                                p = Process(target=task)
                                p.start()
                                all_process.append(p)
            
    for p in all_process : p.join()