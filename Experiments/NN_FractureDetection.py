import torch
from torch  import nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
#from ray.tune.integration.pytorch_lightning import TuneReportCallback
from torchmetrics.classification import AUROC
#from ray import tune
# Define custom dataset
class CustomDataset(Dataset):
    def __init__(self, df):
        self.dataset = df.copy()
        self.X = df["embeddings"].apply(lambda x: x.cpu()).to_numpy()#X#torch.tensor(X,dtype=torch.float32)
        y = df['fracture_grading'].to_numpy()
        y = torch.from_numpy(y)
        self.y = torch.tensor(y,dtype=torch.long)
        self.subject = df["subject"].to_numpy()
        self.label = df["label"].to_numpy()
        self.fracture = df["fracture_flag"].to_numpy()
        self.grade = df["fracture_grading"].to_numpy()

    def sample_weights(self):#sample_weights
        class_counts = self.dataset.fracture_flag.value_counts()
        class_weights = 1 / class_counts
        sample_weights = [class_weights[i] for i in self.dataset.fracture_flag.values]
        #print(class_weights)

        return sample_weights
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        #label_name = str(v_idx2name.get(self.label[idx],-1))
        return {"emb": self.X[idx], "class": self.y[idx], "sub": self.subject[idx], "label": self.label[idx], "frac": self.fracture[idx], "grade":self.grade[idx] }
        #return self.X[idx], self.y[idx]



class SimpleClassifier(pl.LightningModule):
    ###### INIT PROCESS ######
    def __init__(self, data_path, lr, conv, batch_size, epochs, layers):

        super().__init__()
        self.epochs = epochs
        self.conv = conv
        self.batch_size = batch_size
        self.lr = lr
        self.path = data_path
        self.layer_sizes =  layers
        self.criterion = nn.CrossEntropyLoss()
        self.fc1 = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 2) 
        self.fc_one = nn.Linear(512,2)
        self.conv1 = nn.Conv1d(512, 64, 1, stride=1)
        self.conv2 = nn.Conv1d(64, 16, 1, stride=1)
        self.fc_16_2 = nn.Linear(16, 2)
        self.sig = nn.Sigmoid() # Output size is 2 for binary classification (F or not F)
        # Define convolutional layers with decreasing output channels
        input_size = 512
        output_size = 2
        self.conv1x1_1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=1)
        self.conv1x1_2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1)
        self.conv1x1_3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=1)
        self.conv1x1_4 = nn.Conv1d(in_channels=16, out_channels=output_size, kernel_size=1)
        self.layer_1 = torch.nn.Linear(512, self.layer_sizes[0])
        self.layer_2 = torch.nn.Linear(self.layer_sizes[0], self.layer_sizes[1])
        self.layer_3 = torch.nn.Linear(self.layer_sizes[1], 2)
        self.auc = AUROC(task = 'binary')
        self.save_hyperparameters()

    def prepare_data(self):
        df = self.load_data()
        train_df, val_df = train_test_split(df, test_size=0.4, random_state=42)
        self.train_data = CustomDataset(train_df)
        self.val_data = CustomDataset(val_df)
        
        print("train data:", len(self.train_data))
        print("val data:", len(self.val_data))
        #print()

    def load_data(self):
        df =torch.load(self.path)
        df = df[df['fracture_flag'] != 'U']
        df = df[df['fracture_grading'] != -1]
        df = df[df['fracture_grading'] != 4]
        df = df[df['fracture_grading'] != 1]
        class_mapping = {0: 0, 2: 1, 3: 1}
        df['fracture_grading'] = df['fracture_grading'].map(class_mapping)
        return df
       # fxclass_df["embeddings"] = fxclass_df["embeddings"]
    
    def train_dataloader(self):
        sampler = WeightedRandomSampler(self.train_data.sample_weights(), len(self.train_data))
        return DataLoader(self.train_data, batch_size=self.batch_size,sampler=sampler)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size,shuffle=False)
    
    def forward(self, x):
        #x = x.view(x.size(0), -1)  # Flatten the input
        if self.conv: 
            x = x.permute(0, 2, 1)
            x = self.conv1x1_1(x)
            x = self.relu(x)
            x = self.conv1x1_2(x)
            x = self.relu(x)
            x = self.conv1x1_3(x)
            x = self.relu(x)
            x = x.squeeze(2)
            
            # Apply the final linear layer
            x = self.fc_16_2(x)
        else:
            x = x.view(x.size(0), -1)  # Flatten the input
            #x = self.fc1(x)
            x = self.layer_1(x)
            x = self.relu(x)
            x = self.layer_2(x)
            # x = self.conv2(x)
            # x = self.relu(x)
            # x = self.fc_16_2(x)
            # x = self.fc2(x)
            # x = self.relu(x)
            # x = self.fc3(x)
            x = self.relu(x)
            #x = self.fc_out(x)
            x = self.layer_3(x)
        return x

    def training_step(self, train_batch, batch_idx):
        inputs = train_batch['emb']
        labels = train_batch['class']
        x, y = inputs, labels
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        #print('train_loss', loss)
        self.log('train_loss', loss)
        #auc = self.auc(logits, y)
        #self.log('AUC', auc)
        return loss


    def validation_step(self, val_batch, batch_idx):
        inputs = val_batch['emb']
        labels = val_batch['class']
        x, y = inputs, labels
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        #_, predicted = torch.max(logits, 1)
        #auc = roc_auc_score(predicted.cpu().numpy(), y.cpu().numpy())
        #print('val_loss', loss)
        #auc = self.auc(logits, y)
        self.log('val_loss', loss)
        #self.log('AUC', auc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    

if __name__ == '__main__':   # train
    train = False
    if train: 
        monitor_str = "val_loss"#"loss/train_loss"#
        lr = 1e-3
        conv = False
        batch_size = 32 
        max_epochs = 150
        layers = [64,16]
        checkpoint = ModelCheckpoint(
            filename="{epoch}-{step}_latest",
            monitor=monitor_str,
            mode="min",
            save_last=True,
            save_top_k=1,
            auto_insert_metric_name=True,
            #every_n_train_steps=5,#opt.save_every_samples // opt.batch_size_effective,
            save_on_train_epoch_end= True
        )
        early_stopping_patience: int = 100
        early_stopping = EarlyStopping(
            monitor=monitor_str,
            mode="min",
            verbose=False,
            patience=early_stopping_patience,
            # check_on_train_epoch_end=True,
        )
        #metrics = {"loss": "ptl/val_loss", "auc": "ptl/val_auc"}
        #callbacks_tune = TuneReportCallback(metrics, on="validation_end")
        path = "/media/DATA/martina_ma/emb_df_cleaned_epoch10_tsne.pt"#'/media/DATA/martina_ma/emb_dict_3D_cleaned_balanced_tsne.pt'#'/media/DATA/martina_ma/emb_dict_3D_corpus_tsne_epoch15.pt'
        model = SimpleClassifier(data_path=path,lr=lr,conv = conv, batch_size= batch_size,epochs=max_epochs,layers=layers)
        trainer = pl.Trainer(max_epochs=max_epochs,callbacks=[checkpoint])

        trainer.fit(model)
    else:
        checkpoint_path ="/media/DATA/martina_ma/dae/lightning_logs/version_34/checkpoints/epoch=108-step=37823_latest.ckpt"#"/media/DATA/martina_ma/dae/lightning_logs/version_33/checkpoints/epoch=92-step=32271_latest.ckpt"#"/media/DATA/martina_ma/dae/lightning_logs/version_32/checkpoints/epoch=95-step=33312_latest.ckpt"#"/media/DATA/martina_ma/dae/lightning_logs/version_31/checkpoints/epoch=98-step=34353_latest.ckpt"#"/media/DATA/martina_ma/dae/lightning_logs/version_30/checkpoints/epoch=95-step=33312_latest.ckpt"#"/media/DATA/martina_ma/dae/lightning_logs/version_29/checkpoints/epoch=79-step=27760_latest.ckpt"#"/media/DATA/martina_ma/dae/lightning_logs/version_28/checkpoints/epoch=74-step=21750_latest.ckpt"#"/media/DATA/martina_ma/dae/lightning_logs/version_27/checkpoints/epoch=81-step=23780_latest.ckpt"#"/media/DATA/martina_ma/dae/lightning_logs/version_26/checkpoints/epoch=69-step=20300_latest.ckpt"#"/media/DATA/martina_ma/dae/lightning_logs/version_25/checkpoints/epoch=51-step=15080_latest.ckpt"#"/media/DATA/martina_ma/dae/lightning_logs/version_25/checkpoints/epoch=90-step=26390_latest.ckpt"#"/media/DATA/martina_ma/dae/lightning_logs/version_24/checkpoints/epoch=78-step=22910_latest.ckpt"#"/media/DATA/martina_ma/dae/lightning_logs/version_23/checkpoints/epoch=95-step=13920_latest.ckpt"#"/media/DATA/martina_ma/dae/lightning_logs/version_22/checkpoints/epoch=88-step=25810_latest.ckpt"#"/media/DATA/martina_ma/dae/lightning_logs/version_21/checkpoints/epoch=95-step=27840_latest.ckpt"#"/media/DATA/martina_ma/dae/lightning_logs/version_20/checkpoints/epoch=90-step=26390_latest.ckpt"#"/media/DATA/martina_ma/dae/lightning_logs/version_19/checkpoints/epoch=42-step=12470_latest.ckpt"#"/media/DATA/martina_ma/dae/lightning_logs/version_17/checkpoints/epoch=93-step=27260_latest.ckpt"#"/media/DATA/martina_ma/dae/lightning_logs/version_16/checkpoints/epoch=49-step=14500_latest.ckpt"#"/media/DATA/martina_ma/dae/lightning_logs/version_15/checkpoints/epoch=32-step=9570_latest.ckpt"#"/media/DATA/martina_ma/dae/lightning_logs/version_14/checkpoints/epoch=19-step=5800_latest.ckpt"#"/media/DATA/martina_ma/dae/lightning_logs/version_13/checkpoints/epoch=19-step=2900_latest.ckpt"#"/media/DATA/martina_ma/dae/lightning_logs/version_12/checkpoints/epoch=16-step=4930_latest.ckpt"
        #assert Path(checkpoint_path).exists()
        device = "cuda:0"
        model = SimpleClassifier.load_from_checkpoint(checkpoint_path)
        model.eval()
        model.to(device)
        model.prepare_data()
        val_preds = []
        val_targets = []
        all_probs = []
        with torch.no_grad():
            for x in model.val_dataloader():
                emb = x['emb'].to(device)
                label = x['class'].to(device)
                #emb, label = emb.to(device), label.to(device)
                outputs = model.forward(emb)
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(label.cpu().numpy())
                probs = torch.softmax(outputs, dim=1)
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1 (F)

        auc = roc_auc_score(val_targets, all_probs)
        print(auc)

# '/media/DATA/martina_ma/emb_dict_3D_corpus_tsne_epoch15.pt'
# "/media/DATA/martina_ma/dae/lightning_logs/version_12/checkpoints/epoch=16-step=4930_latest.ckpt"
# conv layers
# train data: 9257
# val data: 6172
# 0.8907095319027273
        
#/media/DATA/martina_ma/dae/lightning_logs/version_13/checkpoints/epoch=19-step=2900_latest.ckpt
#0.84502861741612
        
#/media/DATA/martina_ma/dae/lightning_logs/version_14/checkpoints/epoch=19-step=5800_latest.ckpt
# 0.8922280286544786 epochs 20
        
# /media/DATA/martina_ma/dae/lightning_logs/version_15/checkpoints/epoch=32-step=9570_latest.ckpt
# 0.900274790730181
        
#/media/DATA/martina_ma/dae/lightning_logs/version_16/checkpoints/epoch=49-step=14500_latest.ckpt
        #0.887571940636731

# /media/DATA/martina_ma/dae/lightning_logs/version_17/checkpoints/epoch=93-step=27260_latest.ckpt
# 0.8896876704205515
# 
# /media/DATA/martina_ma/dae/lightning_logs/version_19/checkpoints/epoch=42-step=12470_latest.ckpt
# 0.9154449333142728
# 
# /media/DATA/martina_ma/dae/lightning_logs/version_20/checkpoints/epoch=90-step=26390_latest.ckpt
# 0.9157636693828063
# 
# /media/DATA/martina_ma/dae/lightning_logs/version_21/checkpoints/epoch=95-step=27840_latest.ckpt
# 0.9228478697100135
# 
# /media/DATA/martina_ma/dae/lightning_logs/version_22/checkpoints/epoch=88-step=25810_latest.ckpt
# 0.9235382880378671 best value
# batch_size: 32
# conv: false
# data_path: /media/DATA/martina_ma/emb_dict_3D_corpus_tsne_epoch15.pt
# epochs: 100
# layers:
# - 64
# - 16
# lr: 0.001
# 
# /media/DATA/martina_ma/dae/lightning_logs/version_23/checkpoints/epoch=95-step=13920_latest.ckpt
# 0.920933335451181
# 
# /media/DATA/martina_ma/dae/lightning_logs/version_24/checkpoints/epoch=78-step=22910_latest.ckpt
# 0.9156673073155751
# 
# 
# /media/DATA/martina_ma/emb_dict_3D_cleaned_balanced_tsne.pt
#         
# /media/DATA/martina_ma/dae/lightning_logs/version_25/checkpoints/epoch=90-step=26390_latest.ckpt
# 0.6418581995118361
# 
# /media/DATA/martina_ma/dae/lightning_logs/version_25/checkpoints/epoch=51-step=15080_latest.ckpt
# 0.6520821089526714
# 
# /media/DATA/martina_ma/dae/lightning_logs/version_26/checkpoints/epoch=69-step=20300_latest.ckpt
# 0.6608648230803034         
# 
# /media/DATA/martina_ma/dae/lightning_logs/version_27/checkpoints/epoch=81-step=23780_latest.ckpt
# 0.6353500537403836
# 
# /media/DATA/martina_ma/dae/lightning_logs/version_28/checkpoints/epoch=74-step=21750_latest.ckpt
# 0.6466773617972056
# 
# 
# /media/DATA/martina_ma/emb_df_cleaned_epoch10_tsne.pt
# /media/DATA/martina_ma/dae/lightning_logs/version_29/checkpoints/epoch=79-step=27760_latest.ckpt
# 0.9215516445051882
# 
# /media/DATA/martina_ma/dae/lightning_logs/version_30/checkpoints/epoch=95-step=33312_latest.ckpt
# 0.9188519403999098
# /media/DATA/martina_ma/dae/lightning_logs/version_31/checkpoints/epoch=98-step=34353_latest.ckpt
#  0.8800071912647984
# 
#      
# /media/DATA/martina_ma/dae/lightning_logs/version_32/checkpoints/epoch=95-step=33312_latest.ckpt
# 0.9129588776859224
#         
        #/media/DATA/martina_ma/dae/lightning_logs/version_33/checkpoints/epoch=92-step=32271_latest.ckpt
        #0.8625867495979747
# /media/DATA/martina_ma/dae/lightning_logs/version_34/checkpoints/epoch=108-step=37823_latest.ckpt
# 0.8905646434115996
# 
# 
# 
# 
#         