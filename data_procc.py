from typing import List
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
import pickle
import random
import pandas as pd
import random


def get_translation_dataset(tokenizer, source_lang=None, target_lang=None, max_examples_by_words = 100, **kwargs):

    both_directions = (not source_lang and not target_lang)

    if both_directions:
        source_lang = 'en'
        target_lang = 'ru'

    assert (source_lang in ['en', 'ru'] and target_lang in [
            'en', 'ru']), "Supports languges en, ru"

    translation_dataset = DatasetDict()

    for split in ['train', 'test']:
        with open(f'data/translation_dataset_{split}.pickle', 'rb') as f:
            en_dict = pickle.load(f)

        examples = []
        for list_of_examples in en_dict.values():
            random.shuffle(list_of_examples)
            examples.extend(list_of_examples[:max_examples_by_words])

        translation_dataset[split] = Dataset.from_dict(
            {'translation': examples})

    def preprocess_function(examples):
        prefix = "translate"
        inputs, targets = [], []

        for example in examples["translation"]:

            inputs.append(
                prefix + f' {source_lang}-{target_lang}: ' + example[source_lang])
            targets.append(example[target_lang])

            if both_directions:
                inputs.append(
                    prefix + f' {target_lang}-{source_lang}: ' + example[target_lang])
                targets.append(example[source_lang])

        model_inputs = tokenizer(inputs, max_length=128, truncation=True)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=128, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    remove_columns = translation_dataset.column_names[list(translation_dataset.keys())[0]]
    return translation_dataset.map(preprocess_function, batched=True, remove_columns=remove_columns)


def get_dialog_dataset(tokenizer, **kwargs):
    dataset = load_dataset('daily_dialog')

    dataset['train'] = concatenate_datasets(
        [dataset['train'], dataset['validation']])
    dataset.pop('validation')

    prefix = 'dialog: '
    users = ['user1>>: ', 'user2>>: ']

    def preprocess_function(examples):
        inputs, targets = [], []
        for example in examples['dialog']:

            example_with_userid = [
                u+e for e, u in zip(example, users*(1 + len(example)//2))]
            for i in range(len(example)-1):
                inputs.append(prefix + ' '.join(example_with_userid[:i+1]))
                targets.append(example[i+1])

        model_inputs = tokenizer(inputs, max_length=300, truncation=True)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=128, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    remove_columns = dataset.column_names[list(dataset.keys())[0]]
    return dataset.map(preprocess_function, batched=True, remove_columns=remove_columns)


def get_dataset_by_name(tokenizer, dataset_name, **kwargs):

    if dataset_name == 'translate':
        return get_translation_dataset(tokenizer, **kwargs)
    elif dataset_name in ['translate en-ru', 'translate ru-en']:
        src = dataset_name[10:12]
        tgt = dataset_name[13:15]
        return get_translation_dataset(tokenizer, src, tgt, **kwargs)
    elif dataset_name == 'dialog':
        return get_dialog_dataset(tokenizer, **kwargs)


def get_tokenized_dataset(tokenizer, dataset_name, **kwargs):

    if isinstance(dataset_name, List):
        datasets_list = []
        for d_n in dataset_name:
            datasets_list.append(get_dataset_by_name(tokenizer, d_n, **kwargs))

        dataset = DatasetDict()
        keys = datasets_list[0].keys()
        for key in keys:
            dataset[key] = concatenate_datasets(
                [d[key] for d in datasets_list])
        return dataset
    else:
        return get_dataset_by_name(tokenizer, dataset_name, **kwargs)
