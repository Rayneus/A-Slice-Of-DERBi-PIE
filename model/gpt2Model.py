from transformers import GPT2LMHeadModel, Trainer, TrainingArguments

def getModel():
    #load model
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    return model

def getTrainer(model, tokenizedDatasets):
    #training arguments
    trainingArgs = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        logging_dir="./logs",
        logging_steps=100,
        eval_strategy="steps"
    )

    #initialize trainer

    trainer = Trainer(
        model=model,
        args=trainingArgs,
        train_dataset=tokenizedDatasets["train"],
        eval_dataset=tokenizedDatasets["validation"]
    )

    return trainer
