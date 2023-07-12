import timm
import torch
import torch.nn as nn
import pytorch_lightning as pl
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torchmetrics.classification import BinaryRecall, BinaryPrecision, BinaryF1Score

class FullModel(pl.LightningModule):
    def __init__(self, 
                 lr: float=0.001, 
                 total_steps: int=821,
                 width:int=224,
                 height:int=224,
                 normalize = True,
                 w_p: float=0.5,
                 w_n: float=0.5
                 ):
        super().__init__()
        self.lr = lr
        self.save_hyperparameters()
        self.model = timm.create_model('hf-hub:timm/convnext_small.fb_in22k_ft_in1k_384',
                             pretrained=True)
#         self.model = ConvNextForImageClassification.from_pretrained("facebook/convnext-tiny-224")
        self.model.head.fc = nn.Linear(in_features=768, out_features=2, bias=True)
        pos_weight = torch.FloatTensor([w_p, w_n])  # All weights are equal to 1
        self.criterion = nn.CrossEntropyLoss(weight=pos_weight, label_smoothing=0.1)
        self.total_steps = total_steps
        self.metric = BinaryF1Score()

        
 
    def initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
#             nn.init.xavier_normal_(m.weight.data)
            nn.init.xavier_uniform_(m.weight.data,gain=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
#             nn.init.xavier_normal_(m.weight.data)
            nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('sigmoid'))
            nn.init.constant_(m.bias.data, 0)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.99998), weight_decay=1e-5)
#         optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
#         optimizer = torch.optim.RMSprop(self.parameters(), lr=0.0004)
#         scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
#                                                         max_lr=self.lr, 
#                                                         total_steps=self.total_steps, 
#                                                         verbose=False,
#                                                         )
        scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                          first_cycle_steps=self.total_steps,
                                          cycle_mult=1.0,
                                          max_lr=1e-3,
                                          min_lr=1e-5,
                                          warmup_steps=self.total_steps*0.2,
                                          gamma=1.0)
#         scheduler = transformers.get_cosine_schedule_with_warmup(optimizer,
#                                                                  num_warmup_steps=2000,
#                                                                  num_training_steps= self.total_steps)
        scheduler = {
            "scheduler": scheduler,
            "interval": "step",  # or 'epoch'
            "frequency": 1,
        }
        return [optimizer], [scheduler]
    
    def forward(self, x):
        out = self.model(x)
        return out
    def training_step(self, batch, batch_idx):
        precision = BinaryPrecision().cuda()
        recall = BinaryRecall().cuda()
        img, labels = batch
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        img, labels = img.float().to(device), labels.to(device)
        outputs = self.model(img)
        loss = self.criterion(outputs.squeeze(0), labels)
        self.log("train/loss", loss.item())
        #Accuracy
        output = torch.argmax(outputs, dim=1)
        correct = (output == labels).float().sum() 
        
        self.log("train/loss", loss.item())
        self.log("train/acc", correct/len(labels))
        self.log("F1_score/train", self.metric(output, labels))
#         self.log("Precision/train", precision(output.cuda(), labels.cuda()))
#         self.log("Recall/train", recall(output.cuda(), labels.cuda()))

        return loss

    def validation_step(self, batch, batch_idx):
        precision = BinaryPrecision().cuda()
        recall = BinaryRecall().cuda()
        img, labels = batch
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        img, labels = img.float().to(device), labels.to(device)
        outputs = self.model(img)
        loss = self.criterion(outputs, labels)
        #Accuracy
        output = torch.argmax(outputs, dim=1)
        correct = (output == labels).float().sum()
        # lưu lại loss của validate
        self.log("val/loss", loss.item())
        self.log("val/acc", correct/len(labels))
        self.log("F1_score/val", self.metric(output, labels))
        self.log("Precision/val", precision(output.cuda(), labels.cuda()))
        self.log("Recall/val", recall(output.cuda(), labels.cuda()))
        return loss

    def test_step(self, batch, batch_idx):
        precision = BinaryPrecision().cuda()
        recall = BinaryRecall().cuda()
        img, labels = batch
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        img, labels = img.float().to(device), labels.to(device)
        outputs = self.model(img)
        loss = self.criterion(outputs, labels)
        #Accuracy
        output = torch.argmax(outputs, dim=1)
        correct = (output == labels).float().sum()
        # lưu lại loss của test
        self.log("test/loss", loss.item())
        self.log("test/acc", correct/len(labels))
        self.log("F1_score/test", self.metric(output, labels))
        self.log("Precision/test", precision(torch.tensor(output).cuda(), torch.tensor(labels).cuda()))
        self.log("Recall/test", recall(torch.tensor(output).cuda(), torch.tensor(labels).cuda()))
        return loss