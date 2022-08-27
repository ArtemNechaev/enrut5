from typing import Sequence
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict, load_from_disk
import pickle
import random
import pandas as pd
import numpy as np

TRANSLATE_PARAMS_RU_EN = {
    'path_or_dataset': 'translation/ru-en',
    'source': 'ru',
    'target': 'en',
    'prefix': 'translate ru-en: ',
    'max_length': 128,
    'truncation': True,
    'drop_duplicates_column': 'id_ru',
    'drop_duplicates_split': 'test',
    'delete_original_columns': True,
    'add_targets_column': True,
}

TRANSLATE_PARAMS_EN_RU = {
    'path_or_dataset': 'translation/en-ru',
    'source': 'en',
    'target': 'ru',
    'prefix': 'translate en-ru: ',
    'max_length': 128,
    'truncation': True,
    'drop_duplicates_column': 'id_en',
    'drop_duplicates_split': 'test',
    'delete_original_columns': True,
    'add_targets_column': True,
}

DIALOG_PARAMS = {
    'source': 'source',
    'target': 'target',
    'prefix': 'dialog: ',
    'max_length': 480,
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
    dict_dataset = {c: [] for c in dataset.column_names}
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

                numwords[word] += 1
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


def load_tokenized_dataset(path_or_dataset, tokenizer, source, target, prefix=None,
                           max_sentenses_by_words=None, wish_words=None,
                           drop_duplicates_column=None, drop_duplicates_split=None, delete_original_columns=False,
                           add_targets_column=False,
                           preprocess_function=None, **tokenizer_kwargs):

    if preprocess_function is None:
        preprocess_function = _get_preprocess_function(
            tokenizer, source, target, prefix, **tokenizer_kwargs)

    if isinstance(path_or_dataset, (Dataset, DatasetDict)):
        dataset = path_or_dataset
    else:
        dataset = load_from_disk(path_or_dataset)

    if isinstance(dataset, DatasetDict):
        if delete_original_columns:
            remove_columns = dataset.column_names[list(dataset.keys())[0]]

        for key in dataset.keys():
            if drop_duplicates_column:
                drop_duplicates_split = dataset.keys(
                ) if drop_duplicates_split is None else drop_duplicates_split
                drop_duplicates_split = [drop_duplicates_split] if isinstance(
                    drop_duplicates_split, str) else drop_duplicates_split

                if key in drop_duplicates_split:
                    dataset[key] = _drop_duplicates(
                        dataset[key],  drop_duplicates_column, add_targets_column, target)
            elif add_targets_column:
                dataset[key] = dataset[key].add_column(
                    'targets', [[_] for _ in dataset[key][target]])
            if max_sentenses_by_words:
                dataset[key] = _group_by_word(
                    dataset[key], source, max_sentenses_by_words, wish_words=wish_words)

    elif isinstance(dataset, Dataset):
        if delete_original_columns:
            remove_columns = dataset.column_names

        if drop_duplicates_column:
            dataset = _drop_duplicates(
                dataset,  drop_duplicates_column, add_targets_column, target)
        elif add_targets_column:
            dataset = dataset.add_column(
                'targets', [[_] for _ in dataset[target]])

        if max_sentenses_by_words:
            dataset = _group_by_word(
                dataset, source, max_sentenses_by_words, wish_words=wish_words)

    return dataset.map(preprocess_function, batched=True, remove_columns=remove_columns)


def build_daily_dialogs(**kwargs):
    dataset = load_dataset('daily_dialog', **kwargs)

    users = ['user1>>: ', 'user2>>: ']

    greetings = [
        ['Hello', 'Hi', 'How are you?', "I can't complain"],
        ["Hey, What’s up?", "Hi! I’m great"],
        ["Good afternoon", "Hello", "How are you doing?", "Awesome! You?"],
        ["Hello", "Hello", "What's new?"]
    ]
    greetings = greetings * 500

    def prepare_dialogs(examples):
        model_inputs = {}
        inputs, targets = [], []
        for example in examples['dialog']:

            if greetings and random.random() > 0.7:
                example = greetings[-1] + example
                greetings.pop()

            example_with_userid = [
                u+e for e, u in zip(example, users*(1 + len(example)//2))]
            for i in range(len(example)-1):
                inputs.append(' '.join(example_with_userid[:i+1]))
                targets.append(example[i+1])

        model_inputs['source'] = inputs
        model_inputs["target"] = targets

        return model_inputs

    PARAMS = DIALOG_PARAMS.copy()
    PARAMS['path_or_dataset'] = dataset.map(
        prepare_dialogs, batched=True, remove_columns=dataset.column_names['train'])
    return PARAMS


def build_wow(path='dialogs/wow', task='dialog', **kwargs):

    wow_dataset = load_from_disk(path)
    dataset = DatasetDict()
    users = ["user1>>: ", "user2>>: "]

    for key in wow_dataset.keys():

        res = {'source': [], 'target': []}

        for example in wow_dataset[key]:

            dialog = example['dialog']

            example_with_userid = [
                u+e for e, u in zip(dialog, users*(1 + len(dialog)//2))]
            for i in range(len(dialog)-1):

                if task == 'dialog':
                    knowlege = "" if len(
                        example['checked_passage'][i+1].split()) > 500 else example['checked_passage'][i+1]
                    res['source'].append(
                        f"knowlege: {knowlege} " + ' '.join(example_with_userid[:i+1]))
                    res['target'].append(dialog[i+1])
                elif task == 'keyword':
                    res['source'].append(' '.join(example_with_userid[:i+1]))
                    res['target'].append(example['checked_passage_name'][i+1])

        dataset[key] = Dataset.from_dict(res)

    PARAMS = DIALOG_PARAMS.copy()
    PARAMS['path_or_dataset'] = dataset
    PARAMS['prefix'] = f'{task}: '
    return PARAMS


def build_cola(**kwargs):
    labels = ['acceptable', 'unacceptable']
    cola = load_dataset('glue', 'cola', **kwargs)

    def pre_procc(example):
        return {'source': example['sentence'], 'target': labels[example['label']]}

    PARAMS = DIALOG_PARAMS.copy()
    PARAMS['path_or_dataset'] = cola.map(pre_procc, remove_columns=cola.column_names['train'])
    PARAMS['prefix']  = 'cola: '
    return PARAMS


def build_stsb(**kwargs):

    stsb = load_dataset('glue', 'stsb', **kwargs)

    def pre_procc(example):
        return {
            'source': f"sentence1: {example['sentence1']} sentence2: {example['sentence2']}",
            'target': str(round(round(example['label']/0.2, 0) * 0.2, 2))
        }

    PARAMS = DIALOG_PARAMS.copy()
    PARAMS['path_or_dataset'] = stsb.map(pre_procc, remove_columns=stsb.column_names['train'])
    PARAMS['prefix'] = 'stsb '
    return PARAMS


def build_blended_skill_talk(**kwargs):
    blended = load_dataset('blended_skill_talk', **kwargs)
    dataset = DatasetDict()
    users = ["user1>>: ", "user2>>: "]

    for key in blended.keys():

        res = {'source': [], 'target': []}

        for example in blended[key]:

            dialog = example['previous_utterance']

            for i in range(len(example['free_messages'])):
                dialog.append(example['free_messages'][i])
                dialog.append(example['guided_messages'][i])

            example_with_userid = [
                u+e for e, u in zip(dialog, users*(1 + len(dialog)//2))]
            for i in range(len(dialog)-1):
                res['source'].append(' '.join(example_with_userid[:i+1]))
                res['target'].append(dialog[i+1])

        dataset[key] = Dataset.from_dict(res)

        PARAMS = DIALOG_PARAMS.copy()
        PARAMS['path_or_dataset'] = dataset
    return PARAMS


def build_dataset(name, **kwargs):
    PARAMS = globals()['build_' + name](**kwargs)
    return PARAMS
