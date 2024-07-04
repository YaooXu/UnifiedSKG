from dataclasses import dataclass
import os
from typing import Dict, Sequence
import torch
from torch.utils.data import Dataset
import numpy as np
import transformers

def generate_dist_matrix(rows, cols):
    # generate
    # [[2, 3, 4],
    #  [1, 2, 3]]
    base = np.arange(cols)  # 生成1到cols的数组
    result = np.tile(base, (rows, 1))  # 将base数组扩展成rows行的矩阵
    result += np.arange(rows, 0, -1).reshape(rows, 1)  # 加上行索引的偏移量
    return result

def generate_upper_tri_dist_matrix(rows, cols):
    # generate
    # [[0, 1, 2],
    #  [0, 0, 1],
    #  [0, 0, 0]]
    matrix = np.triu(np.ones((rows, cols), dtype=int), 1)
    matrix = matrix.cumsum(-1)
    return matrix

def generate_symmetric_matrix(matrix, rows, cols, sign=1):
    # 获取上三角部分的索引，不包括对角线
    upper_indices = np.triu_indices(rows, 1, cols)
    
    values = np.arange(1, len(upper_indices[0]) + 1)
    values = matrix[upper_indices]
    
    # 将下三角部分设置为上三角部分的负数
    lower_indices = (upper_indices[1], upper_indices[0])
    matrix[lower_indices] = sign * values
    
    return matrix

def get_dist_mat_and_attn_mask(edge_index, tokenizer, input_ids, real_length):
    node_tokens = ["[node]"]
    node_id = tokenizer.convert_tokens_to_ids(node_tokens)[0]
    node_idxes = np.where(np.array(input_ids)==node_id)[0].tolist()
    node_idxes.append(real_length)

    cell_ranges = []
    for i in range(len(node_idxes) - 1):
        cell_ranges.append((node_idxes[i], node_idxes[i + 1]))
    
    length = len(input_ids)
    dist_mat = np.zeros((length, length))
    attn_mask = np.zeros((length, length))

    for st, ed in cell_ranges:
        dist_mat[st:ed, st:ed] = generate_upper_tri_dist_matrix(ed-st,ed-st)
        attn_mask[st:ed, st:ed] = 1

    for cell1, cell2 in edge_index:
        if cell1 >= len(cell_ranges) or cell2 >= len(cell_ranges):
            # might be truncated
            continue
        range1 = cell_ranges[cell1]
        range2 = cell_ranges[cell2]
        sub_dist_mat = generate_dist_matrix(range1[1]-range1[0], range2[1]-range2[0])
        # print(sub_dist_mat.max(), range1, range2)
        dist_mat[range1[0]:range1[1], range2[0]:range2[1]] = sub_dist_mat
        attn_mask[range1[0]:range1[1], range2[0]:range2[1]] = 1
    
    dist_mat = generate_symmetric_matrix(dist_mat, length, length, sign=-1)
    attn_mask = generate_symmetric_matrix(attn_mask, length, length, sign=1)
    
    question_range = cell_ranges[0]
    question_st, question_ed = question_range
    dist_mat[question_st:question_ed, question_ed:] = attn_mask[question_st:question_ed, question_ed:] = 0

    return dist_mat, attn_mask

class TokenizedDataset(Dataset):
    # TODO: A unified structure-representation.
    def __init__(self, args, training_args, tokenizer, seq2seq_dataset, ):
        self.args = args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.seq2seq_dataset = seq2seq_dataset

        self.conv_sep = " || "

    def __getitem__(self, index):
        raw_item = self.seq2seq_dataset[index]

        if raw_item["text_in"]:
            ###################
            # With text input #
            ###################
            if self.conv_sep in raw_item["text_in"]:
                ##################
                # Conversational #
                ##################
                # TODO (commented by Chen): the context part roughly follows the implementation of CoSQL by Tianbao.
                # text_in = "[utt n] || [utt n-1] | [utt n-2] | ..."
                index = raw_item["text_in"].index(self.conv_sep)
                if self.args.model.knowledge_usage == 'concatenate' or self.args.model.knowledge_usage is None:
                    # seq_in  = "[utt n] ; structured knowledge: struct_in ; context: [utt n-1] | [utt n-2] | ..."
                    seq_in = "{} ; structured knowledge: {} ; context: {}".format(raw_item["text_in"][:index],
                                                                                  raw_item["struct_in"],
                                                                                  raw_item["text_in"][index + len(self.conv_sep):])
                elif self.args.model.knowledge_usage == 'separate':
                    # seq_in  = "[utt n] ; context: [utt n-1] | [utt n-2] | ..."
                    seq_in = "{} ; context: {}".format(raw_item["text_in"][:index],
                                                       raw_item["text_in"][index + len(self.conv_sep):])
                else:
                    raise ValueError()
            else:
                ######################
                # Non-conversational #
                ######################
                if self.args.model.knowledge_usage == 'concatenate' or self.args.model.knowledge_usage is None:
                    # seq_in  = "text_in ; structured knowledge: struct_in"
                    # seq_in = "{} ; structured knowledge: {}".format(raw_item["text_in"], raw_item["struct_in"])
                    seq_in = raw_item["struct_in"] # TODO: Xu
                elif self.args.model.knowledge_usage == 'separate':
                    # seq_in  = "text_in"
                    seq_in = raw_item["text_in"]
                else:
                    raise ValueError()
        else:
            ######################
            # Without text input #
            ######################
            if self.args.model.knowledge_usage == 'concatenate':
                # seq_in  = "structured knowledge: struct_in"
                seq_in = "structured knowledge: {}".format(raw_item["struct_in"])
            elif self.args.model.knowledge_usage == 'separate':
                # seq_in  = ""
                seq_in = ""
            else:
                raise ValueError()

        # Concatenate description.
        if self.args.model.use_description and self.args.model.concatenate_description:
            seq_in = "{} ; {}".format(raw_item["description"], seq_in)

        tokenized_question_and_schemas = self.tokenizer(
            seq_in,
            padding="max_length",
            truncation=True,
            max_length=self.training_args.input_max_length,
            # We found that set it as large as possible can boost the performance significantly
            # , meanwhile, due to the t5 uses a relative position coding, we need to manually
            # assign the max input length into some large numbers, instead of using the "max_model_length"
            # ,which the default is 512, which will hurt the performance a lot.
        )
        tokenized_inferred = self.tokenizer(
            raw_item["seq_out"],
            padding="max_length",
            truncation=True,
            max_length=self.training_args.generation_max_length,
            # We set the max_length of "seq_out" during training is the same with the one in inference.
        )

        tokenized_inferred_input_ids = torch.LongTensor(tokenized_inferred.data["input_ids"])
        # Here -100 will let the model not to compute the loss of the padding tokens.
        tokenized_inferred_input_ids[tokenized_inferred_input_ids == self.tokenizer.pad_token_id] = -100
        
        # TODO: Xu
        dist_mat, attn_mask = get_dist_mat_and_attn_mask(raw_item['edge_index'], self.tokenizer, 
            tokenized_question_and_schemas.data["input_ids"],
            sum(tokenized_question_and_schemas.data["attention_mask"]))

        item = {
            'input_ids': torch.LongTensor(tokenized_question_and_schemas.data["input_ids"]),
            # 'attention_mask': torch.LongTensor(tokenized_question_and_schemas.data["attention_mask"]),
            'attention_mask': torch.LongTensor(attn_mask),
            'dist_mat': torch.LongTensor(dist_mat),
            'labels': tokenized_inferred_input_ids,
        }
        # Add task name.
        if 'task_id' in raw_item:
            item['task_ids'] = raw_item['task_id']

        # Separate description tokenization.
        if self.args.model.use_description and self.args.model.map_description:
            tokenized_description = self.tokenizer(raw_item["description"],
                                                   padding="max_length",
                                                   truncation=True,
                                                   max_length=self.args.dataset.description_max_length,
                                                   )
            item['description_input_ids'] = torch.LongTensor(tokenized_description.data["input_ids"])
            item['description_attention_mask'] = torch.LongTensor(tokenized_description.data["attention_mask"])

        # Separate knowledge tokenization.
        if self.args.model.knowledge_usage == 'separate':
            tokenized_knowledge = self.tokenizer(raw_item["struct_in"],
                                                 padding="max_length",
                                                 truncation=True,
                                                 max_length=self.training_args.input_max_length,
                                                 )
            item['knowledge_input_ids'] = torch.LongTensor(tokenized_knowledge.data["input_ids"])
            item['knowledge_attention_mask'] = torch.LongTensor(tokenized_knowledge.data["attention_mask"])

        return item

    def __len__(self):
        return len(self.seq2seq_dataset)
  
    def select(self, idxes):
        self.seq2seq_dataset = self.seq2seq_dataset[:10]
        return self

@dataclass
class UniSKGDataCollator:
    tokenizer: transformers.PreTrainedTokenizer
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        
        
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        
    