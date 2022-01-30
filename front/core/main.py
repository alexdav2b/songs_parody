# test du modèle Bert pour fill the mask, ne fonctionne pas sur mon jupyter parcequ'il ne veut pas install torch :(
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, \
    DataCollatorForLanguageModeling, Trainer
from datasets import load_dataset
from random import randint
import os


model_checkpoint = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
block_size = 128


def load_base_model():
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    return model


def tokenize_function(examples):
    return tokenizer(examples["text"])


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def mlm_fine_tuning(dataset, name):
    # datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')
    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    # model_name = model_checkpoint.split("/")[-1]
    training_args = TrainingArguments(
        f"{name}",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        # push_to_hub=True,
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(f"{name}/model")
    return trainer.model


def fill_mask(_model, sentence):
    # unmasker = pipeline('fill-mask', model=trainer.model, tokenizer=tokenizer)
    unmasker = pipeline('fill-mask', model=_model, tokenizer=tokenizer)
    res = unmasker(sentence)[0]["sequence"]
    return res


def generate_models(models):
    for model in models:
        print(model)
        data_files = {
            "train": f"emotion-detection-from-text/tweet_emotions_{model}_train.csv",
            "test": f"emotion-detection-from-text/tweet_emotions_{model}_test.csv",
            "validation": f"emotion-detection-from-text/tweet_emotions_{model}_valid.csv"
        }
        my_dataset = load_dataset('csv', data_files=data_files)
        print(my_dataset)
        my_model = mlm_fine_tuning(my_dataset, model)
        my_res = fill_mask(my_model, "I am so <mask>.")
        print(my_res)


def load_model(name):
    print(f"{name}/model")
    trained_model = AutoModelForMaskedLM.from_pretrained(f"{name}/model")
    base_res = fill_mask(trained_model, "I am so <mask>.")
    print(base_res)
    return trained_model


def do_parody(_lyrics, _theme):
    my_model = load_model(_theme)
    sentence_changed = dict()
    word_changed = dict()
    lyrics_to_fill = []

    # sentence tokenization
    sentences = _lyrics.splitlines()

    for x in range(len(sentences)):
        if sentences[x] in sentence_changed.keys():  # selon les phrases, on récupère le format à compléter pour éviter différents refrains
            sentences[x] = sentence_changed[sentences[x]]
        else:
            words = sentences[x].split(" ")  # word tokenization
            # random de mot à remplacer par phrase
            if len(words) > 1:
                longest_id = randint(0, len(words) - 1)
                size = 1
                for word_id, word in enumerate(words):
                    if len(word) >= size:
                        size = len(word)
                        longest_id = word_id
                words[longest_id] = "<mask>"

            sentence_changed[sentences[x]] = " ".join(words)  # on garde les phrases déja rencontrés et modifiées
            sentences[x] = " ".join(words)
    lyrics_to_fill = sentences
    print(lyrics_to_fill)
    # On applique le modèle à nos phrases
    for x in range(len(lyrics_to_fill)):
        if "<mask>" in lyrics_to_fill[x]:
            lyrics_to_fill[x] = fill_mask(my_model, lyrics_to_fill[x]) # model()  ###########  modifier le call au modèle préentrainé et changement de la phrase

    _lyrics = "\n".join(lyrics_to_fill)
    return _lyrics


if __name__ == "__main__":
    list_models = [
        "love",
        "happiness",
        "sadness",
        "hate",
    ]
    # generate_models(list_models)
    # model = load_model(list_models[2])
    lyrics = "this is my text\nthis is my second line"
    theme = "love"
    res = do_parody(lyrics, theme)
    print(res)
