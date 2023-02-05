#!/bin/bash

none="_None_"

### usage ###
#filename=train.sh 
#cat $filename | tr -d '\r' > $filename.new && rm $filename && mv $filename.new $filename
# . train.sh $train_data_pct $math_operator $weight_decay $dropout $opt $max_lr $random_seed $use_wandb $group_name
# all this parameters are optional (see the default values below)

### params ###
train_data_pct=${1-5}
math_operator=${2-+}
weight_decay=${3-1}
dropout=${4-0.0}
opt=${5-adamw}
max_lr=${6-0.001}
random_seed=${7-0}

max_steps=100000
max_epochs=100000
every_n_epochs=1
save_weights_only=True

lr_scheduler=$none
lr_scheduler="default"
#lr_scheduler=reduce_lr_on_plateau,factor=0.2,patience=20,min_lr=0.00005,mode=min,monitor=val_loss

clip_grad=$none
#clip_grad="gradient_clip_val=float(0.5),gradient_clip_algorithm=str(norm)"
#clip_grad="gradient_clip_val=float(0.5),gradient_clip_algorithm=str(value)"

### wandb ###
# wandb_entity is the name of the team on wandb and is optional
# wandb_project is the name of the project
use_wandb=False
#group_name="tdp=${train_data_pct}-wd=${weight_decay}-d=${dropout}-opt=${opt}-mlr=${max_lr}-rs=${random_seed}-mo${math_operator}"
# remove random_seed in group_name
group_name="tdp=${train_data_pct}-wd=${weight_decay}-d=${dropout}-opt=${opt}-mlr=${max_lr}-mo${math_operator}"
wandb_entity="grokking_ppsp"
wandb_project="grokking_operator=${math_operator}"

watch=$none
#watch="log=str(all),log_freq=int(1)"

### Experiment dump path ###
dump_path=..
logdir=${dump_path}/logs/$group_name
datadir=${dump_path}/data/$group_name

### Early_stopping (for grokking) : Stop the training `patience` epochs after the `metric` has reached the value `metric_threshold` ###
#early_stopping_grokking=$none
early_stopping_grokking="patience=int(1000),metric=str(val_accuracy),metric_threshold=float(90.0)"

# sgd
momentum=0.9
# adam
beta1=0.9
beta2=0.99
# sag
with_d=True
batch_mode=False
init_y_i=False
#
if [[ $opt == "adamw" ]]; then
	opttmptmp="${opt}"
elif [[ $opt == "sgd" ]]; then
	opttmptmp="${opt},momentum=0,dampening=0,weight_decay=${weight_decay},nesterov=False"
elif [[ $opt == "momentum" ]]; then
	opttmptmp="sgd,momentum=${momentum},dampening=0.9,weight_decay=${weight_decay},nesterov=False"
elif [[ $opt == "nesterov" ]]; then
	opttmptmp="sgd,momentum=${momentum},dampening=0,weight_decay=${weight_decay},nesterov=True"
elif [[ $opt == "asgd" ]]; then
	opttmptmp="${opt},lambd=0.0001,alpha=0.75,t0=1000000.0,weight_decay=${weight_decay}"
elif [[ $opt == "rmsprop" ]]; then
	opttmptmp="${opt},alpha=0.99,weight_decay=${weight_decay},momentum=0,centered=False"
elif [[ $opt == "rmsprop_mom" ]]; then
	opttmptmp="rmsprop,alpha=0.99,weight_decay=${weight_decay},momentum=${momentum},centered=False"
elif [[ $opt == "rprop" ]]; then
	opttmptmp="${opt},etaplus=0.5,etaminus=1.2,step_min=1e-06,step_max=50"
elif [[ $opt == "adadelta" ]]; then
	opttmptmp="${opt},rho=0.9,weight_decay=${weight_decay}"
elif [[ $opt == "adagrad" ]]; then
	opttmptmp="${opt},lr_decay=0,weight_decay=${weight_decay},initial_accumulator_value=0"
elif [[ $opt == "adam" ]]; then
	opttmptmp="${opt},weight_decay=${weight_decay},beta1=${beta1},beta2=${beta2},amsgrad=False"
elif [[ $opt == "amsgrad" ]]; then
	opttmptmp="adam,weight_decay=${weight_decay},beta1=${beta1},beta2=${beta2},amsgrad=True"
elif [[ $opt == "adamax" ]]; then
	opttmptmp="${opt},weight_decay=${weight_decay},beta1=${beta1},beta2=${beta2}"
elif [[ $opt == "custom_adam" ]]; then
	opttmptmp="${opt},weight_decay=${weight_decay},beta1=${beta1},beta2=${beta2}"
elif [[ $opt == "adam_inverse_sqrt" ]]; then
	opttmptmp="${opt},weight_decay=${weight_decay},beta1=${beta1},beta2=${beta2},warmup_updates=4000,warmup_init_lr=1e-7,exp_factor=0.5"
elif [[ $opt == "adam_cosine" ]]; then
	opttmptmp="${opt},weight_decay=${weight_decay},beta1=${beta1},beta2=${beta2},warmup_updates=4000,warmup_init_lr=1e-7,min_lr=1e-9"
	opttmptmp="${opttmptmp},init_period=1000000,period_mult=1,lr_shrink=0.75"
elif [[ $opt == "sag" ]]; then
	opttmptmp="${opt},weight_decay=${weight_decay},batch_mode=${batch_mode},init_y_i=${init_y_i},with_d=${with_d}"
elif [[ $opt == "sag_sgd" ]]; then
	opttmptmp="${opt},weight_decay=${weight_decay},batch_mode=${batch_mode},init_y_i=${init_y_i},with_d=${with_d}"
	opttmptmp="${opttmptmp},momentum=${momentum},dampening=0.9,weight_decay=${weight_decay},nesterov=False"
elif [[ $opt == "sag_adam" ]]; then
	opttmptmp="${opt},weight_decay=${weight_decay},batch_mode=${batch_mode},init_y_i=${init_y_i},with_d=${with_d}"
	opttmptmp="${opttmptmp},beta1=${beta1},beta2=${beta2}"
else 
	echo "Error $opt"
	exit
fi

###
python train.py \
		--batchsize -1 \
		--n_layers 2 \
		--n_heads 4 \
		--d_model 128 \
		--dropout $dropout \
		--weight_noise 0.0 \
		--non_linearity relu \
		--max_context_len 50 \
		--math_operator $math_operator \
		--train_data_pct $train_data_pct \
		--warmup_steps 10 \
		--anneal_lr_steps 100000 \
		--anneal_lr False \
		--max_lr $max_lr \
		--lr_scheduler $lr_scheduler \
		--weight_decay $weight_decay \
		--weight_decay_kind to_zero \
		--noise_factor 0 \
		--clip_grad $clip_grad \
		--save_activations False \
		--save_outputs False \
		--logdir $logdir \
		--datadir $datadir \
		--save_checkpoint True \
		--use_wandb $use_wandb \
		--group_name $group_name \
		--wandb_entity $wandb_entity \
		--wandb_project $wandb_project \
		--watch $watch \
		--opt $opttmptmp \
		--momentum $momentum \
		--random_seed $random_seed \
		--max_steps $max_steps \
		--max_epochs $max_epochs \
		--accelerator auto \
		--devices auto \
		--early_stopping_grokking $early_stopping_grokking \
		--eval_only False \
		--every_n_epochs $every_n_epochs \
		--save_weights_only $save_weights_only \
#		--load_from_ckpt $logdir/checkpoints/last.ckpt \
#		--operand_length \
