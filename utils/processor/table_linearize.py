# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Utils for linearizing the table content into a flatten sequence
"""
import abc
from collections import defaultdict
from typing import Dict, List

import numpy as np


class TableLinearize(abc.ABC):
    PROMPT_MESSAGE = """
        Please check that your table must follow the following format:
        {"header": ["col1", "col2", "col3"], "rows": [["row11", "row12", "row13"], ["row21", "row22", "row23"]]}
    """

    def process_table(self, table_content: Dict) -> str:
        """
        Given a table, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        pass

    def process_header(self, headers: List):
        """
        Given a list of headers, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        pass

    def process_row(self, row: List, row_index: int):
        """
        Given a row, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        pass


# class IndexedRowTableLinearize(TableLinearize):
#     """
#     FORMAT: col: col1 | col2 | col 3 row 1 : val1 | val2 | val3 row 2 : ...
#     """
#     def __init__(self):
#         self.n_row = None
#         self.n_col = None
        
#     def process_table(self, table_content: Dict, question):
#         """
#         Given a table, TableLinearize aims at converting it into a flatten sequence with special symbols.
#         """
#         assert "header" in table_content and "rows" in table_content, self.PROMPT_MESSAGE
#         self.n_row = 1 + len(table_content["rows"])
#         self.n_col = len(table_content['rows'][0])

#         _table_str = f'[node] {question}' + " "
#         # process header
#         _table_str += self.process_header(table_content["header"]) + " "
#         # process rows
#         for i, row_example in enumerate(table_content["rows"]):
#             # NOTE: the row should start from row 1 instead of 0
#             _table_str += self.process_row(row_example, row_index=i + 1) + " "
#         return _table_str.strip()

#     def process_header(self, headers: List):
#         """
#         Given a list of headers, TableLinearize aims at converting it into a flatten sequence with special symbols.
#         """
#         # return "col : " + " | ".join(headers)
        
#         headers = [f'[node] {item}' for item in headers]
#         return ' '.join(headers)

#     def process_row(self, row: List, row_index: int):
#         """
#         Given a row, TableLinearize aims at converting it into a flatten sequence with special symbols.
#         """
#         # row_str = ""
#         # row_cell_values = []
#         # for cell_value in row:
#         #     if isinstance(cell_value, int):
#         #         row_cell_values.append(str(cell_value))
#         #     else:
#         #         row_cell_values.append(cell_value)
#         # row_str += " | ".join(row_cell_values)
#         # return "row " + str(row_index) + " : " + row_str

#         row = [f'[node] {item}' for item in row]
#         return ' '.join(row)
        


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
    
    values = matrix[upper_indices]
    
    # 将下三角部分设置为上三角部分
    lower_indices = (upper_indices[1], upper_indices[0])
    matrix[lower_indices] = sign * values
    
    return matrix

class IndexedRowTableLinearize(TableLinearize):
    """
    FORMAT: [node] Jens Hartel [node] club [node] Berliner AK 07
    """
    
    def __init__(self, tokenizer, max_input_length) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.n_row = None
        self.n_col = None
                
        node_tokens = ["[node]"]
        self.node_id = tokenizer.convert_tokens_to_ids(node_tokens)[0]
        self.max_input_length = max_input_length
        self.edge_index = None

    def process_table(self, table_content: Dict, question):
        """
        Given a table, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        assert "header" in table_content and "rows" in table_content, self.PROMPT_MESSAGE
        self.n_row = 1 + len(table_content["rows"])
        self.n_col = len(table_content['rows'][0])
        
        all_nodes, edge_index = [], []

        all_nodes.append(f'Question: {question} ; ')
        
        col_i_to_col_node_ids = defaultdict(list)
        
        # # process header
        # header_node_ids = []
        # for i, header in enumerate(table_content["header"]):
        #     node = f'[node] {header}'
        #     head_node_id = len(all_nodes)
            
        #     col_i_to_col_node_ids[i].append(head_node_id)
        #     header_node_ids.append(head_node_id)
        #     all_nodes.append(node)
            
        #     edge_index.append([question_node_id, head_node_id])
                   
        # for j in range(len(header_node_ids)):
        #     for i in range(0, j):
        #         edge_index.append([header_node_ids[i], header_node_ids[j]])

        # process rows
        for i, row in enumerate([table_content["header"]] + table_content["rows"]):
            
            row_node_ids = []

            for col_i, word in enumerate(row):

                node = f'[node] {word}'
                node_id = len(all_nodes)
                
                row_node_ids.append(node_id)
                all_nodes.append(node)
                
                col_i_to_col_node_ids[col_i].append(node_id)

            # row links
            for j in range(len(row_node_ids)):
                for i in range(0, j):
                   edge_index.append([row_node_ids[i], row_node_ids[j]])
        
        # col links
        for _, col_node_ids in col_i_to_col_node_ids.items():
            for j in range(len(col_node_ids)):
                for i in range(0, j):
                   edge_index.append([col_node_ids[i], col_node_ids[j]])            

        _table_str = ' '.join(all_nodes)
        
        self.edge_index = edge_index
        
        return _table_str.strip()

    def process_header(self, headers: List):
        """
        Given a list of headers, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        headers = ['[node] ' + header for header in headers]
        return " ".join(headers)

    def process_row(self, row: List, row_index: int):
        """
        Given a row, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        # row_str = ""
        # row_cell_values = []
        # for cell_value in row:
        #     if isinstance(cell_value, int):
        #         row_cell_values.append(str(cell_value))
        #     else:
        #         row_cell_values.append(cell_value)
        # row_str += " | ".join(row_cell_values)
        # return "row " + str(row_index) + " : " + row_str
        
        row = [f'[node] row {str(row_index)}'] + ['[node] ' + str(item) for item in row]
        return " ".join(row)

        
    def get_edge_index(self, linear_table):
        tokenized_question_and_table = self.tokenizer(
            linear_table.lower(), 
            # padding="max_length",
            truncation=True,
            max_length=self.max_input_length,
        )
        return {
            "edge_index": self.edge_index,
            "input_ids": tokenized_question_and_table['input_ids']
        }