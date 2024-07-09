"""
Processor adapts from TAPEX code, see https://github.com/microsoft/Table-Pretraining
(We need to add more features in table handling)
"""
from .table_linearize import IndexedRowTableLinearize
from .table_processor import TableProcessor
from .table_truncate import CellLimitTruncate, RowDeleteTruncate


def get_default_processor(tokenizer, max_cell_length, max_input_length):
    node_tokens = ["[node]"]
    node_id = tokenizer.convert_tokens_to_ids(node_tokens)[0]
    if node_id == tokenizer.unk_token_id:
        tokenizer.add_tokens(node_tokens, special_tokens=True)
        node_id = tokenizer.convert_tokens_to_ids(node_tokens)[0]
   
    table_linearize_func = IndexedRowTableLinearize(tokenizer, max_input_length)
    table_truncate_funcs = [
        CellLimitTruncate(max_cell_length=max_cell_length,
                          tokenizer=tokenizer,
                          max_input_length=max_input_length),
        RowDeleteTruncate(table_linearize=table_linearize_func,
                          tokenizer=tokenizer,
                          max_input_length=max_input_length)
    ]
    processor = TableProcessor(table_linearize_func=table_linearize_func,
                               table_truncate_funcs=table_truncate_funcs)
    return processor
