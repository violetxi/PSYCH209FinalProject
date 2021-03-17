import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# classes used for SimSiam is inspired by this implementation:
# https://github.com/leaderj1001/SimSiam/blob/main/model.py
# Image feature extractor
class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        resnet18 = models.resnet18(pretrained=False)
        self.encoder =  torch.nn.Sequential(*list(
            resnet18.children())[:-1])    # final output is 512-D

    def forward(self, x):
        x = self.encoder(x)
        return x


# 2-layer of non-linear transformation before projection & prediciton head
class NonLinearNeck(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(NonLinearNeck, self).__init__()
        self.neck = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),    # empirical study showed BN is important 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.neck(x)
        return x

    
class Head(nn.Module):
    def __init__(self, in_dim, embd_dim):
        super(Head, self).__init__()
        self.proj_head = nn.Sequential(
            nn.Linear(in_dim, embd_dim),
            nn.BatchNorm1d(embd_dim),
            nn.ReLU(),
        )
        self.pred_head = nn.Sequential(
            nn.Linear(embd_dim, embd_dim),
            nn.BatchNorm1d(embd_dim),
            nn.ReLU(),
            nn.Linear(embd_dim, embd_dim),
        )

    def forward(self, x):
        z = self.proj_head(x)
        p = self.pred_head(z)
        return z, p

    
class VisualModel(nn.Module):
    def __init__(self, neck_hidden=512, embd_dim=64):
        super(VisualModel, self).__init__()
        self.encoder = ResNetBackbone()
        self.neck = NonLinearNeck(512, neck_hidden)
        self.head = Head(neck_hidden, embd_dim)

    def get_embds(self, x):
        x = self.encoder(x).view(-1, 512)
        x = self.neck(x)        
        z, p = self.head(x)        
        return z, p

    def stop_grads_forward(self, p, z):
        z = z.detach()
        p = F.normalize(p, p=2, dim=1)
        z = F.normalize(z, p=2, dim=1)
        return 1 - (p * z).sum(dim=1).mean()    # minimize cosine distance
    
    def forward(self, x1, x2):    # x - (B,N,C,H,W)
        z1, p1 = self.get_embds(x1)
        z2, p2 = self.get_embds(x2)
        d1 = self.stop_grads_forward(p1, z2) / 2
        d2 = self.stop_grads_forward(p2, z1) / 2
        loss = d1 + d2
        return loss
    

class LanguageModel(nn.Module):
    def __init__(self, input_dim=140, hidden_dim=64):    # using Rogers implementation
        super(LanguageModel, self).__init__()
        self.fc = nn.Sequential(            
            nn.Linear(input_dim, hidden_dim),            
            nn.BatchNorm1d(hidden_dim),            
            nn.ReLU(),)

    def forward(self, x):
        x = self.fc(x)
        return x


class SemanticMemoryModel(nn.Module):
    def __init__(self, vision_input_dim=64, language_input_dim=64, hidden_dim=64):
        super(SemanticMemoryModel, self).__init__()
        self.vision_fc = nn.Sequential(            
            nn.Linear(hidden_dim, vision_input_dim),          
            nn.BatchNorm1d(vision_input_dim),            
            nn.ReLU(),)
        self.language_fc = nn.Sequential(
            nn.Linear(hidden_dim, language_input_dim),
            nn.BatchNorm1d(language_input_dim),
            nn.ReLU(),)

    def forward(self, x):
        vision_z = self.vision_fc(x)
        language_z = self.language_fc(x)
        return vision_z, language_z


class VisionLanguageModel(nn.Module):
    def __init__(self, neck_hidden=512, embd_dim=64,    # vision model
                 input_dim=140, hidden_dim=64,    # language model
                 mv_input_dim=64, ml_input_dim=64, m_hidden_dim=64):
        super(VisionLanguageModel, self).__init__()
        self.visual = VisualModel(neck_hidden, embd_dim)
        self.language = LanguageModel(input_dim, hidden_dim)
        self.memory = SemanticMemoryModel(mv_input_dim, ml_input_dim, m_hidden_dim)

    def stop_grads_forward(self, p, z):
        z = z.detach()
        p = F.normalize(p, p=2, dim=1)
        z = F.normalize(z, p=2, dim=1)
        return 1 - (p * z).sum(dim=1).mean()    # minimize cosine distance

    def forward(self, vis_x, lang_x):
        lang_x = lang_x.type(torch.cuda.FloatTensor)
        visual_z, _ = self.visual.get_embds(vis_x)      
        language_z = self.language(lang_x)
        memory_z = visual_z + language_z
        visual_recon, language_recon = self.memory(memory_z)
        d1 = self.stop_grads_forward(visual_recon, language_recon)
        d2 = self.stop_grads_forward(language_recon, visual_recon)
        # contrastive loss between visual and language features
        loss = d1 + d2
        return loss
        
