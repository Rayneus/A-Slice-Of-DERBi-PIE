import sys
import os

# Add the root directory (where data/ and models/ exist) to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from data.preprocess import load_process
from model.gpt2Model import getModel, getTrainer

#load and preprocess data
tokenizedDatasets = load_process()

#load model
model = getModel()
trainer= getTrainer(model, tokenizedDatasets)

#train model
trainer.train()

#save model
model.save_pretrained("./results")