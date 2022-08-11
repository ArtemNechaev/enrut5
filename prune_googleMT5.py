import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5Tokenizer, pipeline
import torch
import re


def main():
    parser = argparse.ArgumentParser()

    
    parser.add_argument("--new_vocab_size", default=None, required=True, type=int)
    parser.add_argument("--saving_path", default=None, required=True, type=str)
    parser.add_argument("--source_model_size", default='base', type=str)

    args= parser.parse_args()

    model_checkpoint=f'google/mt5-{args.source_model_size}'

    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast =False)


    symbols_regex = re.compile(r"[0-9!@#$%^&*()\"№;%:?*\|\[\]{}▁,<>=+-`~\']")
    ru_regex = re.compile(r"[а-яА-ЯёЁ0-9!@#$%^&*()\"№;%:?*\|\[\]{}▁,<>=+-`~\']")
    en_regex = re.compile(r"[a-zA-Z0-9!@#$%^&*()\"№;%:?*\|\[\]{}▁,<>=+-`~\']")

    keep_sym, keep_ru, keep_en = [], [], []
    for word, index in tokenizer.get_vocab().items():
        if len(symbols_regex.findall(word)) > 0.95*(len(word)) or index < 260 or index > tokenizer.vocab_size-101:
            keep_sym.append(index)
        elif len(ru_regex.findall(word)) > 0.95*(len(word)):
            keep_ru.append(index)
        elif len(en_regex.findall(word)) > 0.95*(len(word)):
            keep_en.append(index)
    keep_sym = sorted(keep_sym)
    keep_ru = sorted(keep_ru)
    keep_en = sorted(keep_en)

    num_ru_words = (args.new_vocab_size - 1000)//2
    num_en_words = (args.new_vocab_size - 1000) - num_ru_words

    keep = keep_sym[:1000] + keep_ru[:num_ru_words] + keep_en[:num_en_words] + keep_sym[-100:]
    keep = sorted(keep)

    # prune tokenizer
    from sentencepiece import sentencepiece_model_pb2 as spmp

    smp = tokenizer.sp_model.serialized_model_proto()
    m = spmp.ModelProto()
    m.ParseFromString(smp)
    print('the loaded model has pieces:', len(m.pieces))
    new_pieces = [m.pieces[idx] for idx in keep]
    print('the new pieces:', len(new_pieces))


    for i, p in enumerate(new_pieces):
        m.pieces[i].piece = p.piece
        m.pieces[i].score = p.score
        m.pieces[i].type = p.type
        
    # drop the remaining pieces
    n = len(new_pieces)
    for i in range(len(m.pieces) - n):
        m.pieces.pop()


    print(len(m.pieces))
    with open('new_sp.model', 'wb') as f:
        f.write(m.SerializeToString())

    #prune model
    embed=torch.nn.Embedding(len(keep), model.shared.embedding_dim,)
    embed.load_state_dict({'weight': model.shared.weight[keep]})

    model.shared = embed
    model.encoder.embed_tokens = embed
    model.decoder.embed_tokens = embed

    lm_head = torch.nn.Linear(model.shared.embedding_dim, len(keep), bias = False)
    lm_head.load_state_dict({'weight': model.lm_head.weight[keep]})
    model.lm_head = lm_head

    model.config.__dict__['vocab_size'] = len(keep)
    model.config.__dict__['_name_or_path'] = args.saving_path

    new_tokenizer = T5Tokenizer('new_sp.model', extra_ids=0)

    new_tokenizer.save_pretrained(args.saving_path)
    model.save_pretrained(args.saving_path)


if __name__ == "__main__":
    main()