from data.preprocess import load_process
from model.gpt2Model import getModel, getTrainer


#load and preprocess data
tokenizedDatasets = load_process()

#load model
model = getModel()
trainer= getTrainer(model, tokenizedDatasets)

#evaluate model
evalResults = trainer.evaluate()

#save model
print(f"Perplexity: {evalResults['eval_loss']}")
