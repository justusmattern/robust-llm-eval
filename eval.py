from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from torch import nn


class LM(torch.nn.Module):
    def __init__(self, model_type):
        super.__init__(LM, self)
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_type)
        self.model = GPT2LMHeadModel.from_pretrained(model_type)

    def get_text_loss(self, text):
        tokenized_text = self.tokenizer(text, return_tensors='pt').input_ids
        loss = self.model(tokenized_text).loss

        return loss
    
    def make_prediction(self, pos_text, neg_text):
        pos = self.get_text_loss(pos_text)
        neg = self.get_text_loss(neg_text)

        return 1 if pos > neg else 0



def prepare_data():
    texts = []
    return texts


def main(model_type: str, all_combinations: bool):

    model = LM(model_type)
    data = prepare_data()
    all_predictions = []

    for task in data:
        pos_texts, neg_texts = task[0], task[1]
        task_predictions = []

        if all_combinations:
            for p in pos_texts:
                for n in neg_texts:
                    pred = model.make_prediction(p, n)
                    task_predictions.append(pred)

        else:
            for p, n in zip(pos_texts, neg_texts):
                pred = model.make_prediction(p, n)
                task_predictions.append(pred)

        all_predictions.append(task_predictions)


    for i, task_predictions in enumerate(all_predictions):
        ratio = task_predictions.count(0)/len(task_predictions)

        print(f'task {i} has a ratio of {ratio}')



        


