# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Utils for linearizing the table content into a flatten sequence
"""
import abc
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


class IndexedRowTableLinearize(TableLinearize):
    """
    FORMAT: col: col1 | col2 | col 3 row 1 : val1 | val2 | val3 row 2 : ...
    """

    def process_table(self, table_content: Dict):
        """
        Given a table, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        assert "header" in table_content and "rows" in table_content, self.PROMPT_MESSAGE
        # process header
        _table_str = self.process_header(table_content["header"]) + " "
        # process rows
        for i, row_example in enumerate(table_content["rows"]):
            # NOTE: the row should start from row 1 instead of 0
            _table_str += self.process_row(row_example, row_index=i + 1) + " "
        return _table_str.strip()

    def process_header(self, headers: List):
        """
        Given a list of headers, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        return "col : " + " | ".join(headers)

    def process_row(self, row: List, row_index: int):
        """
        Given a row, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        row_str = ""
        row_cell_values = []
        for cell_value in row:
            if isinstance(cell_value, int):
                row_cell_values.append(str(cell_value))
            else:
                row_cell_values.append(cell_value)
        row_str += " | ".join(row_cell_values)
        return "row " + str(row_index) + " : " + row_str

def generate_dist_matrix(rows, cols):
    base = np.arange(1, cols + 1)  # 生成1到cols的数组
    result = np.tile(base, (rows, 1))  # 将base数组扩展成rows行的矩阵
    result += np.arange(rows).reshape(rows, 1)  # 加上行索引的偏移量
    return result

def generate_upper_triangular_matrix(rows, cols):
    matrix = np.zeros((rows, cols), dtype=int)  # 创建一个全零矩阵
    indices = np.triu_indices(rows, 1, cols)  # 获取上三角部分的索引
    matrix[indices] = np.arange(1, len(indices[0]) + 1)
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

class TableGraphLinearize(TableLinearize):
    """
    FORMAT: [node] Jens Hartel [node] club [node] Berliner AK 07
    """
    
    def __init__(self, tokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        
        node_tokens = ["[node]"]
        node_id = tokenizer.convert_tokens_to_ids(node_tokens)[0]
        if node_id == tokenizer.unk_token_id:
            tokenizer.add_tokens(node_tokens, special_tokens=True)
            node_id = tokenizer.convert_tokens_to_ids(node_tokens)[0]
        self.node_id = node_id
        self.edge_index = None

    def process_table(self, table_content: Dict, question):
        """
        Given a table, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        assert "header" in table_content and "rows" in table_content, self.PROMPT_MESSAGE

        all_nodes, edge_index = [], []

        all_nodes.append(f'[node] {question}')
        question_node_id = 0
        
        # process header
        for header in table_content["header"]:
            node = f'[node] {header}'
            head_node_id = len(all_nodes)
            all_nodes.append(node)
            
            edge_index.append([question_node_id, head_node_id])                
            

        # process rows
        for i, row in enumerate(table_content["rows"]):
            row_node_id = len(all_nodes)
            row_node = f'[node] row {i + 1}'
            all_nodes.append(row_node)
            
            edge_index.append([question_node_id, row_node_id])                
            
            for col_i, word in enumerate(row):
                head_node_id = col_i + 1
                
                node_id = len(all_nodes)
                node = f'[node] {word}'
                all_nodes.append(node)
                
                # edge_index.append([node_id, head_node_id])
                edge_index.append([head_node_id, node_id])
                
                # edge_index.append([node_id, row_node_id])
                edge_index.append([row_node_id, node_id])            
                
                edge_index.append([question_node_id, node_id])                

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

        
    def get_dist_mat_and_attn_mask(self, linear_table, max_length):
        input_ids = self.tokenizer(linear_table.lower())['input_ids']
        node_idxes = np.where(np.array(input_ids)==self.node_id)[0].tolist()
        node_idxes.append(len(input_ids))

        cell_ranges = []
        for i in range(len(node_idxes) - 1):
            cell_ranges.append((node_idxes[i], node_idxes[i + 1]))
        
        length = len(input_ids)
        dist_mat = np.zeros((length, length))
        attn_mask = np.zeros((length, length))

        for st, ed in cell_ranges:
            dist_mat[st:ed, st:ed] = generate_upper_triangular_matrix(ed-st,ed-st)
            attn_mask[st:ed, st:ed] = 1

        for cell1, cell2 in self.edge_index:
            range1 = cell_ranges[cell1]
            range2 = cell_ranges[cell2]
            dist_mat[range1[0]:range1[1], range2[0]:range2[1]] = generate_dist_matrix(range1[1]-range1[0], range2[1]-range2[0])
            attn_mask[range1[0]:range1[1], range2[0]:range2[1]] = 1
        
        dist_mat = generate_symmetric_matrix(dist_mat, length, length, sign=-1)
        attn_mask = generate_symmetric_matrix(attn_mask, length, length, sign=1)
                
        return dist_mat[:max_length, :max_length], attn_mask[:max_length, :max_length]