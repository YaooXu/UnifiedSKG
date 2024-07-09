import copy
import os
from copy import deepcopy

import sys
sys.path.append('./')

import torch
from datasets import DatasetDict, load_dataset
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from transformers import AutoTokenizer
from tqdm import tqdm
from utils.configue import Configure
from utils.processor import get_custom_processor, get_default_processor

import numpy as np

class Constructor(object):
    def __init__(self, args):
        self.args = args

    def to_seq2seq(self, raw_datasets: DatasetDict, cache_root: str):
        if not len(raw_datasets) == 3:
            raise AssertionError("Train, Dev, Test sections of dataset expected.")
        train_dataset = TrainDataset(self.args, raw_datasets['train'], cache_root)
        dev_dataset = DevDataset(self.args, raw_datasets['validation'], cache_root)
        test_dataset = TestDataset(self.args, raw_datasets['test'], cache_root)

        return train_dataset, dev_dataset, test_dataset


"""
    Raw data are formatted as:
    {
        "id": datasets.Value("string"),
        "question": datasets.Value("string"),
        "table_id": datasets.Value("string"),
        "table": {"header": datasets.features.Sequence(datasets.Value("string")),
                  "rows": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("string")))},
        "answer_text": datasets.features.Sequence(datasets.Value("string")),
    }
    """


class TrainDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        # This tab processor is for table truncation and linearize.
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(cache_root, 'wikitq_train.cache')

        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:            
            self.tab_processor = get_custom_processor(max_cell_length=15,
                                                       tokenizer=AutoTokenizer.from_pretrained(args.bert.location, use_fast=False),
                                                       max_input_length=args.seq2seq.table_truncation_max_length)

            self.extended_data = []
            expansion = args.seq2seq.expansion if args.seq2seq.expansion else 1
            for expand_id in range(expansion):
                for raw_data in tqdm(self.raw_datasets):
                    extend_data = deepcopy(raw_data)
                    question = extend_data["question"]
                    table = extend_data['table']
                    gold_result = extend_data['answer_text']

                    table_context = copy.deepcopy(table)
                    # modify a table internally
                    for truncate_func in self.tab_processor.table_truncate_funcs:
                        truncate_func.truncate_table(table_context, question, gold_result)
                    # linearize a table into a string
                    linear_table = self.tab_processor.table_linearize_func.process_table(
                        table_context, question)
                    seq_out = self.tab_processor.process_output(gold_result)
                    
                    extend_data.update({"struct_in": linear_table.lower(),
                                        "text_in": question.lower(),
                                        "seq_out": seq_out.lower()})
                    # # added
                    # dist_mat, attn_mask = self.tab_processor.table_linearize_func.get_dist_mat_and_attn_mask(
                    #     linear_table, args.seq2seq.table_truncation_max_length
                    # )
                    # extend_data.update({"dist_mat": dist_mat,
                    #                     "attn_mask": attn_mask})
                    extend_data['edge_index'] = self.tab_processor.table_linearize_func.edge_index
                    
                    self.extended_data.append(extend_data)
                    
            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)


class DevDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        # This tab processor is for table truncation and linearize.
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(cache_root, 'wikitq_dev.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            self.tab_processor = get_custom_processor(max_cell_length=15,
                                                       tokenizer=AutoTokenizer.from_pretrained(
                                                           args.bert.location, use_fast=False),
                                                       max_input_length=args.seq2seq.table_truncation_max_length)

            self.extended_data = []
            for raw_data in tqdm(self.raw_datasets):
                extend_data = deepcopy(raw_data)
                question = extend_data["question"]
                table = extend_data['table']
                gold_result = extend_data['answer_text']

                table_context = copy.deepcopy(table)
                # modify a table internally
                for truncate_func in self.tab_processor.table_truncate_funcs:
                    truncate_func.truncate_table(table_context, question, [])
                # linearize a table into a string
                linear_table = self.tab_processor.table_linearize_func.process_table(table_context, question)
                seq_out = self.tab_processor.process_output(gold_result)

                extend_data.update({"struct_in": linear_table.lower(),
                                    "text_in": question.lower(),
                                    "seq_out": seq_out.lower()})
                # # added
                # dist_mat, attn_mask = self.tab_processor.table_linearize_func.get_dist_mat_and_attn_mask(
                #     linear_table, args.seq2seq.table_truncation_max_length
                # )
                # extend_data.update({"dist_mat": dist_mat,
                #                     "attn_mask": attn_mask})
                extend_data['edge_index'] = self.tab_processor.table_linearize_func.edge_index
                
                self.extended_data.append(extend_data)
            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)


class TestDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        # This tab processor is for table truncation and linearize.
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(cache_root, 'wikitq_test.cache')
        
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            self.tab_processor = get_custom_processor(max_cell_length=15,
                                                       tokenizer=AutoTokenizer.from_pretrained(
                                                           args.bert.location, use_fast=False),
                                                       max_input_length=args.seq2seq.table_truncation_max_length)

            self.extended_data = []
            for raw_data in tqdm(self.raw_datasets):
                extend_data = deepcopy(raw_data)
                question = extend_data["question"]
                table = extend_data['table']
                gold_result = extend_data['answer_text']

                table_context = copy.deepcopy(table)
                # modify a table internally
                for truncate_func in self.tab_processor.table_truncate_funcs:
                    truncate_func.truncate_table(table_context, question, [])
                # linearize a table into a string
                linear_table = self.tab_processor.table_linearize_func.process_table(table_context, question)
                seq_out = self.tab_processor.process_output(gold_result)

                extend_data.update({"struct_in": linear_table.lower(),
                                    "text_in": question.lower(),
                                    "seq_out": seq_out.lower()})

                # # added
                # dist_mat, attn_mask = self.tab_processor.table_linearize_func.get_dist_mat_and_attn_mask(
                #     linear_table, args.seq2seq.table_truncation_max_length
                # )
                # extend_data.update({"dist_mat": dist_mat,
                #                     "attn_mask": attn_mask})
                extend_data['edge_index'] = self.tab_processor.table_linearize_func.edge_index
                
                self.extended_data.append(extend_data)
            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)


if __name__ == "__main__":
    args = Configure.Get("Salesforce/T5_large_finetune_wikitq.cfg")
    cache_root = os.path.join('graph_output', 'cache')

    os.makedirs(cache_root, exist_ok=True)
    meta_tuning_data = {}
    for task, arg_path in args.arg_paths:
        task_args = Configure.Get(arg_path)
        task_args.bert = args.bert
        print('task_args.bert.location:', task_args.bert.location)
        task_raw_datasets_split: DatasetDict = load_dataset(
            path=task_args.dataset.loader_path,
            cache_dir=task_args.dataset.data_store_path)
        task_seq2seq_dataset_split: tuple = Constructor(task_args).\
            to_seq2seq(task_raw_datasets_split, cache_root)
