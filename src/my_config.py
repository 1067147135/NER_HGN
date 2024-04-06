import argparse
from datasets import config


task = "ner" # Should be one of "ner", "pos" or "chunk"
model_checkpoint = "roberta-base"
batch_size = 32
config.HF_DATASETS_CACHE = "/research/d2/msc/wlshi23/.cache/huggingface/datasets"

parser = argparse.ArgumentParser()
## Required parameters
parser.add_argument("--use_crf",
                    action='store_true',
                    help="Whether use crf")
parser.add_argument("--bert_model", default=model_checkpoint, type=str,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                    "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                    "bert-base-multilingual-cased, bert-base-chinese.")
parser.add_argument("--task_name",
                    default=task,
                    type=str,
                    help="The name of the task to train.")
parser.add_argument("--output_dir",
                    default='./output',
                    type=str,
                    help="The output directory where the model predictions and checkpoints will be written.")
## Other parameters
parser.add_argument("--cache_dir",
                    default="/research/d2/msc/wlshi23/.cache",
                    type=str,
                    help="Where do you want to store the pre-trained models downloaded from s3")
parser.add_argument("--label_list",
                    default=["O"],
                    type=str,
                    nargs='+',
                    help="Where do you want to store the pre-trained models downloaded from s3")
parser.add_argument("--max_seq_length",
                    default=128,
                    type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                          "Sequences longer than this will be truncated, and sequences shorter \n"
                          "than this will be padded.")
parser.add_argument("--eval_on",
                    default="dev",
                    help="Whether to run eval on the dev set or test set.")
parser.add_argument("--do_lower_case",
                    action='store_true',
                    help="Set this flag if you are using an uncased model.")
parser.add_argument("--train_batch_size",
                    default=batch_size,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--eval_batch_size",
                    default=batch_size,
                    type=int,
                    help="Total batch size for eval.")
parser.add_argument("--learning_rate",
                    default=2e-5,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--num_train_epochs",
                    default=3.0,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                          "E.g., 0.1 = 10%% of training.")
parser.add_argument("--weight_decay", default=0.01, type=float,
                    help="Weight deay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
parser.add_argument("--no_cuda",
                    action='store_true',
                    help="Whether not to use CUDA when available")
parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for initialization")
parser.add_argument('--gradient_accumulation_steps',
                    type=int,
                    default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument('--fp16',
                    action='store_true',
                    help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--fp16_opt_level', type=str, default='O1',
                    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                          "See details at https://nvidia.github.io/apex/amp.html")
parser.add_argument('--loss_scale',
                    type=float, default=0,
                    help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                          "0 (default value): dynamic loss scaling.\n"
                          "Positive power of 2: static loss scaling value.\n")
parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
parser.add_argument("--hidden_dropout_prob",
                    default=0.1,
                    type=float,
                    help="hidden_dropout_prob")
parser.add_argument("--window_size",
                    default=-1,
                    type=int,
                    help="window_size")
parser.add_argument("--d_model",
                    default=768,
                    type=int,
                    help="pre-trained model size")
#####
parser.add_argument("--use_bilstm",
                    default=True,
                    action='store_true',
                    help="Set this flag if you are using an uncased model.")
parser.add_argument("--use_single_window",
                    action='store_true',
                    help="Set this flag if you are using an uncased model.")
parser.add_argument("--use_multiple_window",
                    default=True,
                    action='store_true',
                    help="Set this flag if you are using an multiple.")
parser.add_argument("--use_global_lstm",
                    action='store_true',
                    help="Set this flag if you are using an uncased model.")
parser.add_argument("--use_n_gram",
                    action='store_true',
                    help="Set this flag if you are using an uncased model.")
parser.add_argument('--windows_list', type=str, default='1qq3qq5qq7', help="window list")
parser.add_argument('--connect_type', type=str, default='dot-att', help="window list")
parser.add_argument('-f', help="compatible with notebook.")
args = parser.parse_args()
