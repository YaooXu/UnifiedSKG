import os
import time
import torch
import datasets
from transformers import (
    HfArgumentParser,
    set_seed,
    AutoTokenizer
)
from utils.configue import Configure
from utils.training_arguments import WrappedSeq2SeqTrainingArguments

from filelock import FileLock
import nltk
with FileLock(".lock") as lock:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)

# sys.argv = ['/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py',  # This is the name of your .py launcher when you run this line of code.
#             # belows are the parameters we set, take wikitq for example
#             '--cfg', 'Salesforce/T5_base_prefix_wikitq.cfg',
#             '--output_dir', './tmp']
# parser = HfArgumentParser((WrappedSeq2SeqTrainingArguments,))
# training_args, = parser.parse_args_into_dataclasses()
# set_seed(training_args.seed)
# args = Configure.Get(training_args.cfg)

# tokenizer = AutoTokenizer.from_pretrained("hkunlp/from_all_T5_base_prefix_wikitq2", use_fast=False)
# from models.unified.prefixtuning import Model
# model = Model(args)
# model.load("hkunlp/from_all_T5_3b_prefix_wikitq2")

# struct_in = "col : series # | season # | title | notes | original air date row 1 : 1 | 1 | \"the charity\" | alfie, dee dee, and melanie are supposed to be helping | october 15, 1994 row 2 : 2 | 1 | \"the practical joke war\" | alfie and goo unleash harsh practical jokes on dee dee | october 22, 1994 row 3 : 3 | 1 | \"the weekend aunt helen came\" | the boy's mother, jennifer, leaves for the weekend and she leaves | november 1, 1994 row 4 : 4 | 1 | \"robin hood play\" | alfie's school is performing the play robin hood and alfie is | november 9, 1994 row 5 : 5 | 1 | \"basketball tryouts\" | alfie tries out for the basketball team and doesn't make it | november 30, 1994 row 6 : 6 | 1 | \"where's the snake?\" | dee dee gets a snake, but he doesn't | december 6, 1994 row 7 : 7 | 1 | \"dee dee's girlfriend\" | a girl kisses dee dee in front of harry and | december 15, 1994 row 8 : 8 | 1 | \"dee dee's haircut\" | dee dee wants to get a hair cut by cool doctor money | december 20, 1994 row 9 : 9 | 1 | \"dee dee runs away\" | dee dee has been waiting to go to a monster truck show | december 28, 1994 row 10 : 10 | 1 | '\"donnell's birthday party\" | donnell is having a birthday party and brags about all the | january 5, 1995 row 11 : 11 | 1 | \"alfie's birthday party\" | goo and melanie pretend they are dating and they leave alfie out of | january 19, 1995 row 12 : 12 | 1 | \"candy sale\" | alfie and goo are selling candy to make money for some expensive jacket | january 26, 1995 row 13 : 13 | 1 | \"the big bully\" | dee dee gets beat up at school and his friends try to teach | february 2, 1995"
# text_in = "alfie's birthday party aired on january 19. what was the airdate of the next episode?"
# # seq_out = "january 26, 1995"


# def play(txt, model, tokenizer):
#     print("=====❓Request=====")
#     print(txt)
#     tokenized_txt = tokenizer([txt], max_length=1024, padding="max_length", truncation=True)
#     pred = tokenizer.batch_decode(
#         model.generate(
#             torch.LongTensor(tokenized_txt.data['input_ids']),
#             torch.LongTensor(tokenized_txt.data['attention_mask']),
#             num_beams=1,
#             max_length=256
#         ),
#         skip_special_tokens=True
#     )  # More details see utils/dataset.py and utils/trainer.py
#     print("=====💡Answer=====")
#     print(pred)


# play("{} ; structed knowledge: {}".format(text_in, struct_in), model, tokenizer)
