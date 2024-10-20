import sys
import os
# Add the root directory (where data/ and models/ exist) to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocess import load_process
from model.gpt2Model import getModel, getTrainer


#load and preprocess data
tokenizedDatasets = load_process()
print(tokenizedDatasets)

#load model
model = getModel()
trainer= getTrainer(model, tokenizedDatasets)

#evaluate model
# evalResults = trainer.evaluate()

#print results
# print(evalResults)
# print(f"Perplexity: {evalResults['eval_loss']}")
