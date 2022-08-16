from typing import Sequence
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict, load_from_disk
import pickle
import random
import pandas as pd
import numpy as np

TRANSLATE_PARAMS_RU_EN = {
    'path': 'translation/ru-en',
    'source': 'ru',
    'target': 'en',
    'prefix': 'translate ru-en: ',
    'max_length': 128, 
    'truncation': True, 
    'drop_duplicates_column': 'id_ru', 
    'drop_duplicates_split':'test',
    'delete_original_columns': True, 
    'add_targets_column': True,
}

TRANSLATE_PARAMS_EN_RU = {
    'path': 'translation/en-ru',
    'source': 'en',
    'target': 'ru',
    'prefix': 'translate en-ru: ',
    'max_length': 128, 
    'truncation': True, 
    'drop_duplicates_column': 'id_en', 
    'drop_duplicates_split':'test',
    'delete_original_columns': True, 
    'add_targets_column': True,
}

DIALOG_PARAMS = {
    'source': 'source',
    'target': 'target',
    'prefix': 'dialog: ',
    'max_length': 250, 
    'truncation': True, 
    'delete_original_columns': True, 
    'add_targets_column': True,
}


def _get_preprocess_function(tokenizer, source, target, prefix=None, **tokenizer_kwargs):

    def preprocess_function(examples):

        if prefix is not None:
            inputs = [prefix + e for e in examples[source]]
        else:
            inputs = examples[source]

        model_inputs = tokenizer(inputs, **tokenizer_kwargs)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples[target], **tokenizer_kwargs)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return preprocess_function


def _group_by_word(dataset, column, max_examples, wish_words=None):
    numwords = {}
    dict_dataset ={ c: [] for c in dataset.column_names}
    for example in dataset:
        for word in example[column].lower().split():
            if wish_words is not None:
                if word not in wish_words:
                    continue
            if word not in numwords:
                numwords[word] = 1
                for k, v in example.items():
                    dict_dataset[k].append(v)
                break
                    
            elif numwords[word] < max_examples:
                
                numwords[word]+=1
                for k, v in example.items():
                    dict_dataset[k].append(v)
                break
    return Dataset.from_dict(dict_dataset)


def _drop_duplicates(dataset,  drop_duplicates_column, add_targets_column, target,):
    if add_targets_column:
        dataset = dataset.sort(drop_duplicates_column)
        df = dataset.to_pandas()
        targets = df.groupby(drop_duplicates_column).agg(list)[target]

        dataset = dataset.select(
                np.unique(dataset[drop_duplicates_column], return_index=True)[1])
        dataset = dataset.add_column('targets', targets)
    else:
        dataset = dataset.select(
                np.unique(dataset[drop_duplicates_column], return_index=True)[1])
 
        

    return dataset

def load_tokenized_dataset(path, tokenizer, source, target, prefix=None,
                           max_sentenses_by_words=None, wish_words = None,
                           drop_duplicates_column=None, drop_duplicates_split=None, delete_original_columns=False,
                           add_targets_column=False,
                           preprocess_function=None, **tokenizer_kwargs):

    if preprocess_function is None:
        preprocess_function = _get_preprocess_function(
            tokenizer, source, target, prefix, **tokenizer_kwargs)

    dataset = load_from_disk(path)


    if isinstance(dataset, DatasetDict):
        if delete_original_columns:
            remove_columns = dataset.column_names[list(dataset.keys())[0]]
            
        for key in dataset.keys():
            if drop_duplicates_column:
                drop_duplicates_split = dataset.keys() if drop_duplicates_split is None else drop_duplicates_split
                drop_duplicates_split = [drop_duplicates_split] if isinstance(drop_duplicates_split, str) else drop_duplicates_split

                if key in drop_duplicates_split:
                    dataset[key] =  _drop_duplicates(dataset[key],  drop_duplicates_column, add_targets_column, target)
            elif add_targets_column:
                dataset[key] = dataset[key].add_column('targets', [[_] for _ in dataset[key][target]])
            if max_sentenses_by_words:
                dataset[key] = _group_by_word(dataset[key], source, max_sentenses_by_words, wish_words=wish_words)

    elif isinstance(dataset, Dataset):
        if delete_original_columns:
            remove_columns = dataset.column_names

        if drop_duplicates_column:
            dataset =_drop_duplicates(dataset,  drop_duplicates_column, add_targets_column, target)
        elif add_targets_column:
            dataset = dataset.add_column('targets', [[_] for _ in dataset[target]])
            
        if max_sentenses_by_words:
            dataset = _group_by_word(dataset, source, max_sentenses_by_words, wish_words=wish_words)
   

    return dataset.map(preprocess_function, batched=True, remove_columns=remove_columns)
