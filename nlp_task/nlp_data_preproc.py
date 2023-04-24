import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
import json
from tqdm import tqdm


def prepare_data(data_path, history_len=None):
    ARCHITECT_PREFIX = '<Architect> '
    BUILDER_PREFIX = '<Builder> '
    SEP_TOKEN = ' <sep1> '

    data = json.load(open(data_path))
    input_seqs, output_seqs = [], []

    for key, event in tqdm(data.items()):
        architect_lines, builder_lines = [], []
        cur_builder_line = ''
        for line in event[0].split('\n'):
            if line.startswith(ARCHITECT_PREFIX):
                architect_lines.append(line[len(ARCHITECT_PREFIX):])
                if cur_builder_line:
                    # Remove the trailing period
                    builder_lines.append(cur_builder_line[:-1])
                    cur_builder_line = ''
            else:
                # line = line.replace('cube', 'block')
                cur_builder_line += line[len(BUILDER_PREFIX):] + '. '

        # Discard the last architect line if there are no matching builder lines
        if len(architect_lines) != len(builder_lines):
            architect_lines = architect_lines[:-1]

        assert len(architect_lines) == len(builder_lines)

        for i in range(len(architect_lines)):
            context = ''
            if history_len:
                range_start = max(i - history_len, 0)
                range_end = min(i, range_start + history_len)
            else:
                range_start, range_end = 0, i

            for j in range(range_start, range_end):
                context += ARCHITECT_PREFIX + architect_lines[j] + SEP_TOKEN + BUILDER_PREFIX + builder_lines[j] + SEP_TOKEN

            input_seq = context + ARCHITECT_PREFIX + architect_lines[i]
            output_seq = builder_lines[i]
            input_seqs.append(input_seq)
            output_seqs.append(output_seq)

    return input_seqs, output_seqs


def prepare_inputs(batch, tokenizer: PreTrainedTokenizer, max_source_length: int, max_target_length: int, task_prefix: str = "implement given instructions: "):
    """
    Preprocesses a batch of input-output pairs for training or evaluation.
    """
    input_sequences = [x[0] for x in batch]
    output_sequences = [x[1] for x in batch]

    encoding = tokenizer(
        [task_prefix + sequence for sequence in input_sequences],
        padding="longest",
        max_length=max_source_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

    target_encoding = tokenizer(
        output_sequences, padding="longest", max_length=max_target_length, truncation=True
    )
    labels = target_encoding.input_ids

    # replace padding token id's of the labels by -100 so it's ignored by the loss
    labels.masked_fill_(labels == tokenizer.pad_token_id, -100)

    return input_ids, attention_mask, labels
