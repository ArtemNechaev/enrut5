import torch

class Generator():
    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, text, **kwargs):
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors='pt').to(self.model.device)
        with torch.no_grad():
             hypotheses = self.model.generate(**inputs,  **kwargs, )
        return self.tokenizer.batch_decode(hypotheses, skip_special_tokens=True)

    