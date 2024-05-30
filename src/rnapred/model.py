from torch import nn
from torch.nn.functional import cross_entropy
import torch
from tqdm import tqdm
import pandas as pd
import math
import utils
from metrics import contact_f1

def rnapred(pretrained = False, **kwargs):
    '''
    This function trains the model
    '''
    # Load the data
    '''
    The data is loaded using the Data class
    '''
    model = RNAPrediction(**kwargs)

    if pretrained:
        model.load_state_dict(torch.load("model.pt"))
    else:
        model.load_state_dict(torch.load("model.pt", map_location=torch.device(model.device)))
    
    return model


class RNAPrediction(nn.Module):
    def __init__(
        self,
        train_len=0,
        embedding_dim=4,
        device="cpu",
        negative_weight=0.1,
        lr=1e-4,
        loss_l1=0,
        loss_beta=0,
        scheduler="none",
        verbose=True,
        interaction_prior=False,
        output_th=0.5,
        **kwargs
    ):
        super().__init__()

        self.device = device
        self.class_weights = torch.tensor([1, negative_weight]).float.to(device)
        self.loss_l1 = loss_l1
        self.loss_beta = loss_beta
        self.verbose = verbose
        self.config = kwargs
        self.output_th = output_th

        self.interaction_prior = interaction_prior

        self.build_graph(embedding_dim)
    
    def build_graph(
        self,
        embedding_dim,
        kernel_size=3,
        filters=32,
        layers=2,
        resnet_bottleneck_factor=0.5,
        mid_channels=1,
        kernel_resnet2d = 5,
        bottleneck1_resnet2d = 256,
        bottleneck2_resnet2d = 128,
        filters_resnet2d = 256,
        rank=64,
        dilation_resnet1d=3,
        dilation_resnet2d=3,
        **kwargs
    ):
        pad = (kernel_size - 1) // 2

        self.use_restriction = mid_channels != 1

        self.resnet1d = [nn.Conv1d(embedding_dim, filters, kernel_size, padding="same")]

        for i in range(layers):
            self.resnet1d.append(
                ResidualLayer1D(dilation_resnet1d, resnet_bottleneck_factor, filters, kernel_size)
            )

        self.resnet1d = nn.Sequential(*self.resnet1d)

        self.convrank1 = nn.Conv1d(in_channels=filters, out_channels=rank, kernel_size=kernel_size, padding = pad, stride = 1)

        self.convrank2 = nn.Conv1d(in_channels=filters, out_channels=rank, kernel_size=kernel_size, padding = pad, stride = 1)

        self.resnet2d = [nn.Conv2d(in_channels=mid_channels, out_channels=filters_resnet2d, kernel_size=7, padding=pad, stride = 1)]

        self.resnet2d.extend([ResidualLayer2D(filters_resnet2d, bottleneck1_resnet2d, kernel_resnet2d, dilation_resnet2d), 
                              ResidualLayer2D(filters_resnet2d, bottleneck2_resnet2d, kernel_resnet2d, dilation_resnet2d)])

        self.resnet2d = nn.Sequential(*self.resnet2d)

        self.conv2Dout = nn.Conv2d(in_channels=filters_resnet2d, out_channels=1, kernel_size=kernel_size, padding="same")

    def forward(self, batch):
        x = batch["embedding"].to(self.device)
        batch_size = x.shape[0]
        L = x.shape[2]

        y = self.resnet1d(x)
        ya = self.convrank1(y)
        ya = torch.transpose(ya, -1, -2)

        yb = self.convrank2(y)

        y = ya @ yb
        yt = torch.transpose(y, -1, -2)
        y = (y + yt) / 2

        y0 = y.view(-1, L, L)

        if self.interaction_prior != None:
            prob_mat = batch["interaction_prior"].to(self.device)
            x1 = torch.zero([batch_size, 2, L, L]).to(self.device)
            x1[:, 0, :, :] = y0
            x1[:, 1, :, :] = prob_mat
        else:
            x1 = y0.unsqueeze(1)
        
        y = self.resnet2d(x1)

        y = self.conv2Dout(torch.relu(y)).squeeze(1)
        if batch["canonical_mask"] is not None:
            y = y.multiply(batch["canonical_mask"].to(self.device))
        yt = torch.transpose(y, -1, -2)
        y = (y + yt) / 2

        return y, y0

    def loss(self, yhat, y):
        y = y.view(y.shape[0], -1)
        yhat, y0 = yhat
        yhat = yhat.view(yhat.shape[0], -1)
        y0 = y0.view(y0.shape[0], -1)

        l1_loss = torch.mean(torch.relu(yhat[y != 1]))

        yhat = torch.cat((-yhat, yhat), dim=1)
        
        y0 = y0.unsqueeze(1)
        y0 = torch.cat((-y0, y0), dim=1)

        error_lost1 = cross_entropy(y0, y, ignore_index=-1, weight=self.class_weights)
        error_lost = cross_entropy(yhat, y, ignore_index=-1, weight=self.class_weights)

        loss = (error_lost + self.loss_beta * error_lost1 + self.loss_l1 * l1_loss)

        return loss
    
    def fit(self, dataloader):
        self.train()
        metrics = {"loss": 0, "f1": 0}

        if self.verbose:
            dataloader = tqdm(dataloader)
        
        for batch in dataloader:

            y = batch["contact"].to(self.device)
            batch.pop("contact")
            self.optimizer.zero_grad()
            y_pred = self(batch)

            loss = self.loss(y_pred, y)

            if isinstance(y_pred, tuple):
                y_pred = y_pred[0]
            
            f1 = contact_f1(y.cpu(), y_pred.detach().cpu(), batch["length"], method="triangular")

            metrics["loss"] += loss.item()
            metrics["f1"] += f1

            loss.backward()
            self.optimizer.step()

            if self.scheduler_name == "cycle":
                self.scheduler.step()
        for i in metrics:
            metrics[i] /= len(dataloader)

        return metrics

    def test(self, dataloader):
        self.eval()
        metrics = {"loss": 0, "f1": 0, "f1_post": 0}

        if self.verbose:
            dataloader = tqdm(dataloader)

        with torch.no_grad():
            for batch in dataloader:
                y = batch["contact"].to(self.device)
                batch.pop("contact")
                lengths = batch["length"]

                y_pred = self(batch)
                loss = self.loss(y_pred, y)
                metrics["loss"] += loss.item()

                if isinstance(y_pred, tuple):
                    y_pred = y_pred[0]

                y_pred_post = utils.postprocessing(y_pred.cpu(), batch["canonical_mask"])

                f1 = contact_f1(y.cpu(), y_pred.cpu(), lengths, th=self.output_th, reduce=True, method="triangular")
                f1_post = contact_f1(
                    y.cpu(), y_pred_post.cpu(), lengths, th=self.output_th, reduce=True, method="triangular")

                metrics["f1"] += f1
                metrics["f1_post"] += f1_post

        for k in metrics:
            metrics[k] /= len(dataloader)

        if self.scheduler_name == "plateau":
            self.scheduler.step(metrics["f1_post"])

        return metrics  

    def pred(self, dataloader, logits=False):
        self.eval()

        if self.verbose:
            dataloader = tqdm(dataloader)

        predictions, logits_list = [], [] 
        with torch.no_grad():
            for batch in dataloader: 
                
                lengths = batch["length"]
                seqid = batch["id"]
                sequences = batch["sequence"]


                y_pred = self(batch)
                
                if isinstance(y_pred, tuple):
                    y_pred = y_pred[0]

                y_pred_post = utils.postprocessing(y_pred.cpu(), batch["canonical_mask"])

                for k in range(y_pred_post.shape[0]):
                    if logits:
                        logits_list.append(
                            (seqid[k],
                             y_pred[k, : lengths[k], : lengths[k]].squeeze().cpu(),
                             y_pred_post[k, : lengths[k], : lengths[k]].squeeze()
                            ))
                    predictions.append(
                        (seqid[k],
                         sequences[k],
                         utils.mat2bp(
                                y_pred_post[k, : lengths[k], : lengths[k]].squeeze()
                            )                         
                        )
                    )
        predictions = pd.DataFrame(predictions, columns=["id", "sequence", "base_pairs"])

        return predictions, logits_list

class ResidualLayer1D(nn.Module):
    '''
    This class defines a residual layer for 1D data
    '''
    def __init__(self, 
                 filters,
                 rb_factor,
                 kernel_size, 
                 dilation):
        
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm1d(filters),
            nn.ReLU(),
            nn.Conv1d(filters,
                      math.floor(rb_factor * filters),
                      kernel_size, 
                      dilation=dilation, 
                      padding="same"),
            nn.BatchNorm1d(math.floor(rb_factor * filters)),
            nn.ReLU(),
            nn.Conv1d(math.floor(rb_factor * filters), 
                      filters,
                      kernel_size, 
                      dilation=dilation, 
                      padding="same"),
        )

    def forward(self, x):
        return x + self.layer(x)
    
class ResidualLayer2D(nn.Module):
    '''
    This class defines a residual layer for 2D data
    '''
    def __init__(self, 
                 filters,
                 factors,
                 kernel_size, 
                 dilation):
        
        super().__init__()

        self.layer = nn.Sequential(
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.Conv2d(filters,
                      factors,
                      kernel_size, 
                      dilation=dilation, 
                      padding="same"),
            nn.BatchNorm2d(factors),
            nn.ReLU(),
            nn.Conv2d(factors, 
                      filters,
                      kernel_size, 
                      dilation=dilation, 
                      padding="same"),
        )

    def forward(self, x):
        return x + self.layer(x)