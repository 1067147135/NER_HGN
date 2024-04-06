from __future__ import absolute_import, division, print_function

import os

import torch
import torch.nn.functional as F
from  transformers import AdamW, AutoTokenizer, get_linear_schedule_with_warmup

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm, trange

from seqeval.metrics import classification_report

from model import HGNER

import my_config as cfg
import my_util as utl

args = cfg.args
logger = utl.generate_logger(__name__)

if __name__ == "__main__":
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))
    
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    utl.setup_seed(args.seed)

    task_name = args.task_name.lower()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_checkpoint, add_prefix_space=True)
    datasets, tokenized_datasets, label_list = utl.prepare_datasets(tokenizer, "../englishv12/", task_name)
    
    num_train_optimization_steps = 0
    num_train_optimization_steps = int(
        datasets["train"].num_rows / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    num_labels = len(label_list)
    logger.info(args)

    # Prepare model
    model = HGNER(args,
                hidden_dropout_prob=args.hidden_dropout_prob,
                num_labels=num_labels,
                windows_list = [int(k) for k in args.windows_list.split('qq')] if args.windows_list else args.window_size,
                )

    n_params = sum([p.nelement() for p in model.parameters()])
    logger.info(f'n_params: {n_params}')

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias','LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    label_map = {i : label for i, label in enumerate(label_list,0)}

    logger.info("*** Label map ***")
    logger.info(label_map)
    logger.info("*******************************************")

    best_epoch = -1
    best_p = -1
    best_r = -1
    best_f = -1
    best_test_f = -1
    best_eval_f = -1


    #load train data
    train_features,_ = utl.convert_dataset_to_features(
        "train", datasets, tokenized_datasets, args.max_seq_length)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", datasets["train"].num_rows)
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    all_valid_ids = torch.tensor([f.valid_ids for f in train_features], dtype=torch.long)
    all_lmask_ids = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)
    all_seq_lens = torch.tensor([f.seq_len for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_valid_ids,all_lmask_ids, all_seq_lens)

    #load valid data

    eval_features,_ = utl.convert_dataset_to_features("validation", datasets, tokenized_datasets, args.max_seq_length)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    all_valid_ids = torch.tensor([f.valid_ids for f in eval_features], dtype=torch.long)
    all_lmask_ids = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
    all_seq_lens = torch.tensor([f.seq_len for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_valid_ids,all_lmask_ids, all_seq_lens)

    #load test data
    test_features,_ = utl.convert_dataset_to_features("test", datasets, tokenized_datasets, args.max_seq_length)
    all_input_ids_dev = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask_dev = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids_dev = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    all_label_ids_dev = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    all_valid_ids_dev = torch.tensor([f.valid_ids for f in test_features], dtype=torch.long)
    all_lmask_ids_dev = torch.tensor([f.label_mask for f in test_features], dtype=torch.long)
    all_seq_lens_dev = torch.tensor([f.seq_len for f in test_features], dtype=torch.long)
    test_data = TensorDataset(all_input_ids_dev, all_input_mask_dev, all_segment_ids_dev, all_label_ids_dev, all_valid_ids_dev, all_lmask_ids_dev, all_seq_lens_dev)

    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        test_sampler = SequentialSampler(test_data)
        # test_sampler = RandomSampler(test_data, replacement=True, num_samples=100)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        eval_sampler = SequentialSampler(eval_data)
        # eval_sampler = RandomSampler(eval_data, replacement=True, num_samples=100)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
        # train_sampler = RandomSampler(train_data, replacement=True, num_samples=100)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    # Fine-tuning Training

    test_f1 = []
    dev_f1 = []

    for epoch_ in trange(int(args.num_train_epochs), desc="Epoch"):
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            # begin_time = time.time()
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, valid_ids,l_mask, seq_len = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids,valid_ids,l_mask)#, seq_len=seq_len)

            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
        try:
            model_path = f'{args.output_dir}/model_epoch_{epoch_}.pt'
            torch.save(model.state_dict(), model_path)
        except Exception as e:
            logger.info(f"Failed to store model: {e}")

        # eval in each epoch.
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        y_true = []
        y_pred = []
        for input_ids, input_mask, segment_ids, label_ids,valid_ids,l_mask, seq_len in eval_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            valid_ids = valid_ids.to(device)
            label_ids = label_ids.to(device)
            l_mask = l_mask.to(device)
            seq_len = seq_len.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask,valid_ids=valid_ids,attention_mask_label=l_mask)


            if not args.use_crf:
                logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            input_mask = input_mask.to('cpu').numpy()


            for i, label in enumerate(label_ids):
                temp_1 = []
                temp_2 = []
                for j,m in enumerate(label):
                    if j == 0:
                        continue
                    elif j == len(label)-1:
                        y_true.append(temp_1)
                        y_pred.append(temp_2)
                        break
                    else:

                        temp_1.append(label_map[label_ids[i][j]])
                        try:
                            temp_2.append(label_map[logits[i][j]])
                        except:
                            temp_2.append('O')
                    

        report = classification_report(y_true, y_pred,digits=4)
        logger.info("\n******evaluate on the dev data*******")
        logger.info("\n%s", report)
        temp = report.split('\n')[-3]
        f_eval = eval(temp.split()[-2])
        dev_f1.append(f_eval)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")


        #if os.path.exists(output_eval_file):
        with open(output_eval_file, "a") as writer:
            writer.write('*******************epoch*******'+str(epoch_)+'\n')
            writer.write(report+'\n')


        y_true = []
        y_pred = []
        for input_ids, input_mask, segment_ids, label_ids,valid_ids,l_mask, seq_len in test_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            valid_ids = valid_ids.to(device)
            label_ids = label_ids.to(device)
            l_mask = l_mask.to(device)
            seq_len = seq_len

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask,valid_ids=valid_ids,attention_mask_label=l_mask)
                shape = logits.shape
                if len(shape) < 3:
                    logits = logits.unsqueeze(dim=0)

            try:
                if not args.use_crf:
                    logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                input_mask = input_mask.to('cpu').numpy()
            except:
                import pdb
                pdb.set_trace()

            for i, label in enumerate(label_ids):
                temp_1 = []
                temp_2 = []
                for j,m in enumerate(label):
                    if j == 0:
                        continue
                    elif j == len(label)-1:
                        y_true.append(temp_1)
                        y_pred.append(temp_2)
                        break
                    else:
                        temp_1.append(label_map[label_ids[i][j]])
                        try:
                            temp_2.append(label_map[logits[i][j]])
                        except:
                            temp_2.append('O')

        report = classification_report(y_true, y_pred,digits=4)

        logger.info("\n******evaluate on the test data*******")
        logger.info("\n%s", report)
        temp = report.split('\n')[-3]
        f_test = eval(temp.split()[-2])
        test_f1.append(f_test)



        output_eval_file_t = os.path.join(args.output_dir, "test_results.txt")

        with open(output_eval_file_t, "a") as writer2:
            writer2.write('*******************epoch*******'+str(epoch_)+'\n')
            writer2.write(report+'\n')


