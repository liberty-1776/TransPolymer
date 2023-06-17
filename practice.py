import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaModel, RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from PolymerSmilesTokenization import PolymerSmilesTokenizer
from dataset import Downstream_Dataset, DataAugmentation, LoadPretrainData
import torch
import torch.nn as nn
from torchmetrics import R2Score
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader

fingerprint = torch.empty(2, 768)
pred_output = torch.empty(2,1)



class DownstreamRegression(nn.Module):
    def __init__(self, drop_rate=0.1):
        super(DownstreamRegression, self).__init__()
        self.PretrainedModel = deepcopy(PretrainedModel)
        self.PretrainedModel.resize_token_embeddings(len(tokenizer))
        
        self.Regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(self.PretrainedModel.config.hidden_size, self.PretrainedModel.config.hidden_size),
            nn.SiLU(),
            nn.Linear(self.PretrainedModel.config.hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask,step):
        outputs = self.PretrainedModel(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.last_hidden_state[:, 0, :] #fingerprint
        fingerprint[step] = logits
        output = self.Regressor(logits)
        return output

def test(model, loss_fn, train_dataloader,device):

    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(train_dataloader):
            print(f'Smiles: {step+1}')
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prop = batch["prop"].to(device).float()
            outputs = model(input_ids, attention_mask,step).float()
            pred_output[step] = outputs
            
            
            

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data = pd.read_csv('data/Egc.csv')
train_data = train_data[0:2]


scaler = StandardScaler()
train_data.iloc[:, 1] = scaler.fit_transform(train_data.iloc[:, 1].values.reshape(-1, 1))

PretrainedModel = RobertaModel.from_pretrained('ckpt/pretrain.pt')
tokenizer = PolymerSmilesTokenizer.from_pretrained("roberta-base", max_len=411)
train_dataset = Downstream_Dataset(train_data, tokenizer, 411)

model = DownstreamRegression(drop_rate=0.1).to(device)
model = model.double()
loss_fn = nn.MSELoss()

train_dataloader = DataLoader(train_dataset, 1, shuffle=True, num_workers=8)

steps_per_epoch = train_data.shape[0] // 1
training_steps = steps_per_epoch * 1
warmup_steps = int(training_steps * 0.05)

optimizer = AdamW(
                    [
                        {"params": model.PretrainedModel.parameters(), "lr":  0.00005,
                         "weight_decay": 0.0},
                        {"params": model.Regressor.parameters(), "lr": 0.0001,
                         "weight_decay": 0.01},
                    ],
    				no_deprecation_warning=True
                )
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                        num_training_steps=training_steps)

for epoch in range(1):
    test(model, loss_fn, train_dataloader, device)

fingerprint = fingerprint.detach().cpu().numpy()
pred_output = pred_output.detach().cpu().numpy()

for i in range(2):
    p = (fingerprint[i],pred_output[i])
    print(p)
    np.savetxt('scores.txt',p, delimiter=',',fmt='%s')
   


#np.savetxt('scores.txt', [p for p in zip(list(fingerprint), list(pred_output))], delimiter=',', fmt='%s')