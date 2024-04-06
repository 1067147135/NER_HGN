
import os
import torch
import numpy as np
import random
import logging
import logging.handlers
from datasets import load_dataset

import my_config as cfg

args = cfg.args

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None, domain_label=None, seq_len=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        self.domain_label = domain_label
        self.seq_len = seq_len

def setup_seed(seed):
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
	random.seed(seed)
	np.random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmard = False
	torch.random.manual_seed(seed)

def remove_dir(path):
    if os.path.isdir(path):
        for subpath in os.listdir(path):
            subpath = os.path.join(path, subpath)
            if os.path.isdir(subpath):
                remove_dir(subpath)
            else:
                os.remove(subpath)
        os.rmdir(path)

def generate_logger(name):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # initialize logging
    logging.basicConfig(format = '[%(asctime)s] %(levelname)s - %(name)s: %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.DEBUG)
    # initialize logger
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    else:
        # write in file, maximum 1MB, back up 5 files
        handler = logging.handlers.RotatingFileHandler(f"{args.output_dir}/log_{name}.log", maxBytes=1e6, backupCount=5)
        logger.addHandler(handler)
        return logger

def prepare_datasets(tokenizer, path="../englishv12/", task="ner"):
    datasets = load_dataset(path, trust_remote_code=True)
    label_list = datasets["train"].features[f"{task}_tags"].feature.names

    label_all_tokens = True

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples[f"{task}_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to 0 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(0)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or 0, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else 0)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)

    return datasets, tokenized_datasets, label_list

def convert_dataset_to_features(datasets_part:str, datasets, tokenized_datasets, max_seq_length=128):
    """Loads a data file into a list of `InputFeatures`s."""
    # datasets_part = "train" or "validation" or "test"
    dataset = tokenized_datasets[datasets_part]
    orig_dataset = datasets[datasets_part]

    features = []
    ori_sents = []
    for i in range(dataset.num_rows):
        sentence = dataset[i]
        orig_sentence = orig_dataset[i]

        ori_sents.append(orig_sentence['tokens'])

        input_ids = sentence['input_ids']
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
        label_ids = sentence['labels']
        valid = [1 if ele % 2 or ele == 0 else 0 for ele in label_ids]
        label_mask = [1] * len(label_ids)
        seq_len = []
        seq_len.append(len(orig_sentence['tokens']))

        if len(input_ids) >= max_seq_length - 1:
            input_ids = input_ids[0:(max_seq_length - 2)]
            input_mask = input_mask[0:(max_seq_length - 2)]
            segment_ids = segment_ids[0:(max_seq_length - 2)]
            label_ids = label_ids[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        features.append(
                InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_id=label_ids,
                            valid_ids=valid,
                            label_mask=label_mask,
                            seq_len=seq_len))
    return features, ori_sents

