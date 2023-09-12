import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "MineChat"
#print("Let's chat! (type 'quit' to exit)")
sentences = {"Hello" : "greeting",
             "Hi" : "greeting",
             "What is the criteria of employement?" : "employedcriteria", 
             "What is Mine?" : "Minedef1952", 
             "What is the function of a chief inspector?" : "function_of_inspector_1952",
             "What powers does the Chief Inspector have to authorizing other Inspectors?": "chief_inspector_authorization_func_1952",
             "Who are considered public servants in The Coal act" : "chief_inspector_public_servant_1952",
             "What are Limitation of district magistrate's power in mining" : "district_magistrate_powers_sm_1952",
             "What are the rules related to Indian Standard Time" : "indian_standard_time_rules_1952",
             "What is working above ground?": "working_above_ground_def1952",
             "What is relay and shift?": "relay_and_shift_definition_def1952"}




































total = 0
right = 0
for sentence in sentences:
    # sentence = "do you use credit cards?"
    #sentence = sentences[i]
    cp = sentence
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        if tag == sentences[cp]:
            right = right + 1
    total = total + 1
percentage = float((right * 100)/total)
print(f"Model Accuracy: {percentage} %")
#print(f"Total Cases: {total}\nAccurate Cases: {right}")