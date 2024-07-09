import json
import logging
from multiprocessing import Pool, pool
import pickle
import sys

import pandas as pd

sys.path.append("./")

import copy
import os
from copy import deepcopy

import torch
from datasets import DatasetDict, load_dataset
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from transformers import AutoTokenizer
from tqdm import tqdm
from utils.configue import Configure
from utils.processor import get_default_processor


class Constructor(object):
    def __init__(self, args):
        self.args = args

    def to_seq2seq(self, raw_datasets: DatasetDict, cache_root: str):
        if not len(raw_datasets) == 3:
            raise AssertionError("Train, Dev, Test sections of dataset expected.")
        train_dataset = TrainDataset(self.args, raw_datasets["train"], cache_root)
        dev_dataset = DevDataset(self.args, raw_datasets["validation"], cache_root)
        test_dataset = TestDataset(self.args, raw_datasets["test"], cache_root)

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


def _process_table(process_idx, raw_datasets, tab_processor, is_train=True):
    extended_data = []

    for i, raw_data in enumerate(raw_datasets):
        if (i + 1) % 10 == 0:
            print(process_idx, i / len(raw_datasets))

        extend_data = deepcopy(raw_data)
        question = extend_data["question"]
        table = extend_data["table"]
        gold_result = extend_data["answer_text"]

        table_context = copy.deepcopy(table)
        # modify a table internally
        for truncate_func in tab_processor.table_truncate_funcs:
            truncate_func.truncate_table(table_context, question, gold_result if is_train else [])
        # linearize a table into a string
        linear_table = tab_processor.table_linearize_func.process_table(table_context, question)
        seq_out = tab_processor.process_output(gold_result)

        extend_data.update({"struct_in": linear_table.lower(), "text_in": question.lower(), "seq_out": seq_out.lower()})
        extend_data["graph"] = tab_processor.table_linearize_func.get_edge_index(linear_table.lower())

        extended_data.append(extend_data)

    print(process_idx, "finish")

    return extended_data


def parallel_process_tables(raw_datasets, tab_processor, is_train=True, n_process=8):
    num_samples = len(raw_datasets)
    num_samples_in_chunk = num_samples // n_process + 1
    jobs = []
    st = 0
    for i in range(n_process):
        ed = st + num_samples_in_chunk
        ed = min(ed, num_samples)
        jobs.append([i, raw_datasets.select(range(st, ed)), tab_processor, is_train])

        st = ed

    with Pool(processes=n_process) as pool:
        try:
            results = pool.starmap(_process_table, jobs)
        except Exception as e:
            logging.error(f"An error occurred: {e}")
        finally:
            pool.close()
            pool.join()

    print("all finish")
    extended_data = []
    for res in results:
        extended_data.extend(res)

    print(len(extended_data))

    return extended_data


class TrainDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        # This tab processor is for table truncation and linearize.
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(cache_root, "wikitq_train.cache")
        if os.path.exists(cache_path) and args.dataset.use_cache:
            with open(cache_path, "rb") as f:
                self.extended_data = pickle.load(f)
        else:
            self.tab_processor = get_default_processor(
                max_cell_length=15,
                tokenizer=AutoTokenizer.from_pretrained(args.bert.location, use_fast=False),
                max_input_length=args.seq2seq.table_truncation_max_length,
            )

            self.extended_data = []
            expansion = args.seq2seq.expansion if args.seq2seq.expansion else 1
            for expand_id in range(expansion):
                self.extended_data.extend(parallel_process_tables(self.raw_datasets, self.tab_processor, True))
                # for raw_data in tqdm(self.raw_datasets):
                #     extend_data = deepcopy(raw_data)
                #     question = extend_data["question"]
                #     table = extend_data['table']
                #     gold_result = extend_data['answer_text']

                #     table_context = copy.deepcopy(table)
                #     # modify a table internally
                #     for truncate_func in self.tab_processor.table_truncate_funcs:
                #         truncate_func.truncate_table(table_context, question, gold_result)
                #     # linearize a table into a string
                #     linear_table = self.tab_processor.table_linearize_func.process_table(table_context, question)
                #     seq_out = self.tab_processor.process_output(gold_result)

                #     extend_data.update({"struct_in": linear_table.lower(),
                #                         "text_in": question.lower(),
                #                         "seq_out": seq_out.lower()})
                #     extend_data.update(**self.tab_processor.table_linearize_func. \
                #                        get_position_bias_and_attn_mask(linear_table.lower()))
                #     self.extended_data.append(extend_data)
            if args.dataset.use_cache:
                with open(cache_path, "wb") as f:
                    pickle.dump(self.extended_data, f)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)


class DevDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        # This tab processor is for table truncation and linearize.
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(cache_root, "wikitq_dev.cache")
        if os.path.exists(cache_path) and args.dataset.use_cache:
            with open(cache_path, "rb") as f:
                self.extended_data = pickle.load(f)
        else:
            self.tab_processor = get_default_processor(
                max_cell_length=15,
                tokenizer=AutoTokenizer.from_pretrained(args.bert.location, use_fast=False),
                max_input_length=args.seq2seq.table_truncation_max_length,
            )

            self.extended_data = parallel_process_tables(self.raw_datasets, self.tab_processor, False)
            # for raw_data in tqdm(self.raw_datasets):
            #     extend_data = deepcopy(raw_data)
            #     question = extend_data["question"]
            #     table = extend_data['table']
            #     gold_result = extend_data['answer_text']

            #     table_context = copy.deepcopy(table)
            #     # modify a table internally
            #     for truncate_func in self.tab_processor.table_truncate_funcs:
            #         truncate_func.truncate_table(table_context, question, [])
            #     # linearize a table into a string
            #     linear_table = self.tab_processor.table_linearize_func.process_table(table_context, question)
            #     seq_out = self.tab_processor.process_output(gold_result)

            #     extend_data.update({"struct_in": linear_table.lower(),
            #                         "text_in": question.lower(),
            #                         "seq_out": seq_out.lower()})
            #     extend_data.update(**self.tab_processor.table_linearize_func. \
            #                         get_position_bias_and_attn_mask(linear_table.lower()))
            #     self.extended_data.append(extend_data)
            if args.dataset.use_cache:
                with open(cache_path, "wb") as f:
                    pickle.dump(self.extended_data, f)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)


class TestDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        # This tab processor is for table truncation and linearize.
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(cache_root, "wikitq_test.cache")
        if os.path.exists(cache_path) and args.dataset.use_cache:
            with open(cache_path, "rb") as f:
                self.extended_data = pickle.load(f)
        else:
            self.tab_processor = get_default_processor(
                max_cell_length=15,
                tokenizer=AutoTokenizer.from_pretrained(args.bert.location, use_fast=False),
                max_input_length=args.seq2seq.table_truncation_max_length,
            )

            self.extended_data = parallel_process_tables(self.raw_datasets, self.tab_processor, False)
            # for raw_data in tqdm(self.raw_datasets):
            #     extend_data = deepcopy(raw_data)
            #     question = extend_data["question"]
            #     table = extend_data['table']
            #     gold_result = extend_data['answer_text']

            #     table_context = copy.deepcopy(table)
            #     # modify a table internally
            #     for truncate_func in self.tab_processor.table_truncate_funcs:
            #         truncate_func.truncate_table(table_context, question, [])
            #     # linearize a table into a string
            #     linear_table = self.tab_processor.table_linearize_func.process_table(table_context, question)
            #     seq_out = self.tab_processor.process_output(gold_result)

            #     extend_data.update({"struct_in": linear_table.lower(),
            #                         "text_in": question.lower(),
            #                         "seq_out": seq_out.lower()})
            #     extend_data.update(**self.tab_processor.table_linearize_func. \
            #                         get_position_bias_and_attn_mask(linear_table.lower()))
            #     self.extended_data.append(extend_data)
            if args.dataset.use_cache:
                with open(cache_path, "wb") as f:
                    pickle.dump(self.extended_data, f)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)


if __name__ == "__main__":
    args = Configure.Get("Salesforce/T5_base_finetune_wikitq.cfg")
    cache_root = os.path.join("output_graph", "cache")

    os.makedirs(cache_root, exist_ok=True)
    meta_tuning_data = {}
    for task, arg_path in args.arg_paths:
        task_args = Configure.Get(arg_path)
        task_args.bert = args.bert
        print("task_args.bert.location:", task_args.bert.location)
        task_raw_datasets_split: DatasetDict = load_dataset(
            path=task_args.dataset.loader_path, cache_dir=task_args.dataset.data_store_path
        )
        task_seq2seq_dataset_split: tuple = Constructor(task_args).to_seq2seq(task_raw_datasets_split, cache_root)
