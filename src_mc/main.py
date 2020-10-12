from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import glob
import apex
from prettytable import PrettyTable
import numpy as np
import torch
from train_eval import train, set_seed, evaluate

from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForMultipleChoice, BertTokenizer,
                                  XLNetConfig, XLNetForMultipleChoice,
                                  XLNetTokenizer, RobertaConfig,
                                  RobertaForMultipleChoice, RobertaTokenizer)
from utils import load_and_cache_examples, MULTIPLE_CHOICE_TASKS_NUM_LABELS, load_and_cache_several_examples
from train_eval import train, evaluate
from model import BertForMultipleChoice_CVC
ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, RobertaConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForMultipleChoice_CVC, BertTokenizer),
}
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", default='bert', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default='bert-base-uncased', type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--output_dir", default='../output_mc', type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--raw_data_dir", default='../data_mc', type=str)
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--task_name", default='DREAM')
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help='Whether to run test on the test set')
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=2.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--time_stamp", default='', type=str)
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        logger.info("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    # Set seed
    set_seed(args)
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    config.output_hidden_states = True
    config.num_options = int(MULTIPLE_CHOICE_TASKS_NUM_LABELS[args.task_name.lower()])
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config)
    model.MLP.copy_from_bert(model.bert)
    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, 'einsum')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    if args.do_train:
        logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)  # Reduce logging
        train_dataset = load_and_cache_examples(args, task=args.task_name, tokenizer=tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    time_stamp = args.time_stamp
    # Evaluation
    # We do not use dev set
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)  # Reduce logging
            logging.getLogger("transformers.configuration_utils").setLevel(logging.WARN)  # Reduce logging
        checkpoints = [checkpoint for checkpoint in checkpoints if time_stamp in checkpoint]
        logger.info("Evaluate the following checkpoints for validation: %s", checkpoints)
        best_ckpt = 0
        best_acc = 0
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            logger.info("Load the model: %s", checkpoint)

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, args.task_name, model, tokenizer, prefix=prefix)
            if result[0]['eval_acc'] > best_acc:
                best_ckpt = checkpoint
                best_acc = result[0]['eval_acc']
    if args.do_test and args.local_rank in [-1, 0]:
        try:
            checkpoints = [best_ckpt]
        except:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
        checkpoints = [checkpoint for checkpoint in checkpoints if time_stamp in checkpoint]
        logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logging.getLogger("transformers.configuration_utils").setLevel(logging.WARN)  # Reduce logging
        logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)  # Reduce logging

        logger.info("Evaluate the following checkpoints for final testing: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            logger.info("Load the model: %s", checkpoint)
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            task_string = ['', '-Add1OtherTruth2Opt', '-Add2OtherTruth2Opt', '-Add1PasSent2Opt', '-Add1NER2Pass']
            task_string = [args.task_name + item for item in task_string]
            result = evaluate(args, task_string, model, tokenizer, prefix=prefix, test=True)

if __name__ == "__main__":
    main()
