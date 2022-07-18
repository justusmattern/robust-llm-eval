import openai


def prepare_data():
    pos_texts, neg_texts = [], []
    return pos_texts, neg_texts


def paraphrase(text, num):
    response = openai.Completion.create(
                model='text-davinci-002',
                prompt=f"Paraphrase the following text:\n\"{text}\"\n\Now write the paraphrase:",
                temperature=0.9,
                max_tokens=70,
                top_p=0.8,
                frequency_penalty=0,
                presence_penalty=0,
                logprobs=0,
                )

    return paraphrases