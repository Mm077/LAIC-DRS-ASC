import argparse
import torch
import os
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom
import datasets.eurosat_x_u

import datasets.cifar100_x_u
import datasets.dtd_x_u
import datasets.flowers_x_u
import datasets.cifar10_x_u
import datasets.stl10_x_u
import datasets.ucf101_x_u
import datasets.imagenet10_x_u
import datasets.imagenetdogs_x_u

import trainers.clustering


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):

    ###
    if args.RUN_TYPE_ == "GEN_EMB" or args.RUN_TYPE_ == "EVAL_PRED":
        cfg.DATALOADER.TRAIN_U.SAMPLER = "SequentialSampler"
        cfg.DATALOADER.TRAIN_X.SAMPLER = "SequentialSampler"

    if args.root:
        cfg.DATASET.ROOT = args.root

    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = 32
    
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer
    # cfg.TEST.FINAL_MODEL = "best_val"
    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    # （自行添加）改变epoch,bs的值：
    cfg.RUN_TYPE = args.RUN_TYPE_
    cfg.OPTIM.MAX_EPOCH = int(args.MAX_EPOCH_)
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = int(args.BS_)
    # set NAME
    cfg.USE_LOSS = args.LOSS_
    cfg.DATASET.NUM_SHOTS = int(args.SHOTS_)
    cfg.PROMPT_SAVEPATH = args.output_dir
    cfg.NAME_THISTRAIN = f"./prompt-{cfg.DATASET.NAME}-seed{args.seed}-{cfg.DATASET.NUM_SHOTS}shots-{cfg.PROMPT_INIT}-{args.CFG_}-{args.LOSS_}-{args.PL_}-bs={args.BS_}-ep={args.MAX_EPOCH_}.pt"
    cfg.PICNAME = f"seed{args.seed}-{cfg.DATASET.NUM_SHOTS}shots-{cfg.PROMPT_INIT}-{args.CFG_}-{args.LOSS_}-{args.PL_}-bs={args.BS_}-ep={args.MAX_EPOCH_}"

def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.PROMPT_INIT = "ONLY_NAME_X"
    cfg.NAME_THISTRAIN = "none"
    

    cfg.TEST.FINAL_MODEL == "best_val"
    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.N_CTX_Y = 12  # number of context vectors for new classes
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

  
    cfg.TRAINER.COOP.FIRST_STAGE_EP = 3

def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    args.dataset_config_file = f"./configs/datasets/{args.dataset_}.yaml"
    dataset_config_file = f"./configs/datasets/{args.dataset_}.yaml"
    if dataset_config_file:
        cfg.merge_from_file(dataset_config_file)
 
   
    # 2. From the method config file
    args.config_file = f"./configs/trainers/{args.trainer}/{args.CFG_}.yaml"
    config_file = f"./configs/trainers/{args.trainer}/{args.CFG_}.yaml"
    if config_file:
        cfg.merge_from_file(config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg

import json
from types import SimpleNamespace

def load_args_from_json(json_path):
    with open(json_path, 'r') as f:
        args_dict = json.load(f)
    return SimpleNamespace(**args_dict)

def train_prompt(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print(f"Setting fixed seed: {cfg.SEED}")
        set_random_seed(cfg.SEED)

    setup_logger(cfg.OUTPUT_DIR)
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    output_dir = cfg.OUTPUT_DIR

    cfg.defrost()

    cfg.PROMPT_STAGE = "FIRST"
    trainer = build_trainer(cfg)

    trainer.train()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )

    parser.add_argument(
        "--RUN_TYPE_", type=str, default="", help="用于分辨此次运行是为了提取特征还是训练coop"
    )
    parser.add_argument(
        "--MAX_EPOCH_", type=str, default="", help=" "
    )
    parser.add_argument(
        "--BS_", type=str, default="", help=" "
    )
    parser.add_argument(
        "--CFG_", type=str, default="", help=" "
    )
    parser.add_argument(
        "--N_CTX_", type=str, default="", help=""
    )
    parser.add_argument(
        "--CSC_", type=str, default="", help=""
    )
    parser.add_argument(
        "--CTP_", type=str, default="", help=""
    )
    parser.add_argument(
        "--LOSS_", type=str, default="", help=""
    )

    parser.add_argument(
        "--PL_", type=str, default="", help=""
    )
    # parser.add_argument(
    #     "--SHOTS_TEST_SELF", type=str, default="", help=""
    # )

    parser.add_argument(
        "--SHOTS_", type=str, default="", help=""
    )
    parser.add_argument(
        "--dataset_", type=str, default="", help=""
    )


    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    # args.output_dir = f"./output/{args.dataset_}/{args.trainer}/{args.CFG_}_{args.SHOTS_}shots/nctx{args.N_CTX_}_csc{args.CSC_}_ctp{args.CTP_}/seed{args.seed}"
    args.output_dir = f"./output/{args.dataset_}/{args.trainer}/{args.CFG_}_{args.SHOTS_}shots/{args.LOSS_}_{args.PL_}_BS{args.BS_}/seed{args.seed}/"

    train_prompt(args)
    print("finished!")
    a=1

   
