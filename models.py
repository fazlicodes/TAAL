import torch.nn as nn
import torch
import torch.nn.functional as F

class TripleSSLModel(nn.Module):
    def __init__(self, base_encoder, match_encoder, args):
         super().__init__()
         self.matchmodel = MatchEncoder(match_encoder, args)
         self.matchmodel_proj_dim = self.matchmodel.match_enc_feature_dim
         self.embmodel = EmbModel(base_encoder,self.matchmodel_proj_dim, args).apply(convert_to_hp)

    def forward(self, x, only_feats=False):
        B, C, H, W = x.shape

        op = {}
        
        op['ssl'] = self.embmodel(x, only_feats)
        with torch.no_grad():
            op['match'] = self.matchmodel(x[B//2:], only_feats)
        return op

def convert_to_hp(model):
    model.half()  # convert to half precision
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.float()

class MatchEncoder(nn.Module):

    def __init__(self, match_encoder, args):
        super().__init__()
        # Init matching encoder
        self.match_encoder = match_encoder(args['match_enc'])
        self.match_enc_feature_dim = self.match_encoder.fc.in_features
        self.projection_dim = args['projection_dim'] 
        self.proj_hidden = 512

        for p in self.match_encoder.parameters():
            p.requires_grad=False
        
        self.match_encoder.fc = nn.Identity() 

        self.match_enc_projector = nn.Sequential(nn.Linear(self.match_enc_feature_dim, self.proj_hidden),
                                       nn.ReLU(),
                                       nn.Linear(self.proj_hidden, self.projection_dim)).apply(convert_to_hp)
    
    def forward(self, x, only_feats=False):
        op = {}
        op['feat'] = self.match_encoder(x) 
        if not only_feats:        
            op['emb'] = self.match_enc_projector(op['feat'])

        return op

class EmbModel(nn.Module):
    '''
    Class used either on the training or inference stage of self-supervised learning model
    '''
    
    def __init__(self, base_encoder, args):
        super().__init__()
        self.enc = base_encoder(pretrained=args['pretrained'])
        try:
            self.feature_dim = self.enc.fc.in_features
        except:
            self.feature_dim = self.enc.heads.head.in_features

        self.projection_dim = args['projection_dim'] 
        self.proj_hidden = 512
        # self.match_projector = nn.Sequential(nn.Linear(self.feature_dim, match_enc_feature_dim))

        # remove final fully connected layer of the backbone
        self.enc.fc = nn.Identity()
        self.enc.heads = nn.Identity()

        if args['store_embeddings']:
            self.emb_memory = torch.zeros(args['num_train'], args['projection_dim'], 
                                          requires_grad=False, device=args['device'])
             

        # standard simclr projector
        # if args['half_precision']:
        #     self.projector = nn.Sequential(nn.Linear(self.feature_dim, self.proj_hidden),
        #                                 nn.ReLU(),
        #                                 nn.Linear(self.proj_hidden, self.projection_dim)).apply(convert_to_hp) 
        # else:
        self.projector = nn.Sequential(nn.Linear(self.feature_dim, self.proj_hidden),
                                    nn.ReLU(),
                                    nn.Linear(self.proj_hidden, self.projection_dim))
         
        
    def update_memory(self, inds, x):
        m = 0.9
        with torch.no_grad():
            self.emb_memory[inds] = m*self.emb_memory[inds] + (1.0-m)*F.normalize(x.detach().clone(), dim=1, p=2)
            self.emb_memory[inds] = F.normalize(self.emb_memory[inds], dim=1, p=2)        
    
    def forward(self, x, only_feats=False, context=None):
        op = {}
        op['feat'] = self.enc(x) 
        if not only_feats:        
            op['emb'] = self.projector(op['feat'])

        return op
    
    # def forward_match_encoder(self, x, only_feats=False):
    #     op = {}
    #     op['feat'] = self.match_encoder(x) 
    #     if not only_feats:        
    #         op['emb'] = self.match_enc_projector(op['feat'])

    #     return op


class AdapterMLP(nn.Module):
    '''
    MLP Network for low-shot adaptation (trained on top of frozen features)
    '''
    def __init__(self,num_classes,input_size,hidden_size):
        super(AdapterMLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, num_classes)
        )
    def forward(self, x):
        out = self.mlp(x)
        return out