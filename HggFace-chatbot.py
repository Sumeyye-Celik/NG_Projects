# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 09:55:43 2023

@author: Sümeyye
"""

import requests


def query(payload, model_id, api_token): 
    headers = {"Authorization": f"Bearer {api_token}"}
    API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

model_id = "bert-base-cased"
api_token = "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" 
data = query("The goal of life is [MASK].", model_id, api_token)
    
from huggingface_hub import notebook_login

notebook_login()

from datasets import load_dataset

squad = load_dataset("squad", split="train[:1000]")
    
squad= squad.train_test_split(test_size= 0.2)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="longest_first",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []
    
    # print(answers)
    # print(type(answers))

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)
        
        # print(answer)
        # print(type(answer))

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1
        
        # print(offset)
        # print(type(offset))

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)

print("Fonksiyonun çalışması bitti!!! ")

from transformers import DefaultDataCollator

data_collator = DefaultDataCollator()

from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

model = AutoModelForQuestionAnswering.from_pretrained("bert-base-cased")

training_args = TrainingArguments(
    output_dir="my_qa_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_squad["train"],
    eval_dataset=tokenized_squad["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()


print(trainer.train())  
# from transformers import pipeline
trainer.push_to_hub()

question = "How many programming languages does BLOOM support?"
context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."

from transformers import pipeline

question_answerer = pipeline("question-answering", model="my_awesome_qa_model")
question_answerer(question=question, context=context)
print(question_answerer)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("my_qa_model")
inputs = tokenizer(question, context, return_tensors="pt")

import torch
from transformers import AutoModelForQuestionAnswering

model = AutoModelForQuestionAnswering.from_pretrained("my_qa_model")
with torch.no_grad():
    outputs = model(**inputs)
    
answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()


predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
print(tokenizer.decode(predict_answer_tokens))














    




