"""model.py - Model and module class for ViT.
   They are built to mirror those in the official Jax implementation.
"""

import numpy as np
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo
from model.csv_encoder import *

class ImToSeqTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.labelEmbedding = nn.Linear(config.CLASS_N, config.D_MODEL)
        self.labelDecoding = nn.Linear(config.D_MODEL, config.CLASS_N)
        self.encoder = ViTEncoder(config)
        decoderCell = nn.TransformerDecoderLayer(config.D_MODEL, config.N_HEAD, config.FF_DIM, config.DROP_OUT, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoderCell, config.DECODER.NUM_LAYER)
        
    def forward(self, images, csv_feature, labels, *args, **kwargs):
        '''
        images (BxCxHxW)
        labels (BxL)
        '''
        enc = self.encoder(images)
        labels = labels.clone().detach()[:,:-1] # <E> Token 제거
        labels = F.one_hot(labels, num_classes=self.config.CLASS_N).to(torch.float32) # (BxL) -> (BxLxC)
        labels = self.labelEmbedding(labels) # (BxLxD)
        labels = PositionalEmbedding1D(self.config.LABEL_LEN, self.config.D_MODEL, False)(labels).to(torch.float32)
        labels_mask = torch.zeros((self.config.LABEL_LEN, self.config.LABEL_LEN))
        for i in range(self.config.LABEL_LEN):
            labels_mask[i, i+1:]=1.
        outputs = self.decoder(labels, enc, labels_mask.to(labels.device)) # (BxLxD)
        outputs = self.labelDecoding(outputs) # (BxLxC)
        return outputs
    
    def decode(self, images, csv_feature, *args, **kwargs):
        '''
        '''
        enc = self.encoder(images)
        labels = torch.zeros((images.shape[0], 1)).to(images.device) # (BxL) with '<S>'
        for i in range(self.config.LABEL_LEN-1):
            _labels = labels.clone().detach()
            _labels = one_hot_vector(_labels, self.config.CLASS_N)
            _labels = self.labelEmbedding(_labels) # (Bx(i+1)xD)
            _labels = PositionalEmbedding1D(i+1, self.config.D_MODEL, False)(_labels).to(torch.float32)
            outputs = self.decoder(_labels, enc) # (Bx(i+1)xD)
            outputs = self.labelDecoding(outputs)
            
            indices = torch.arange(labels.shape[0]).to(outputs.device)
            next_class = torch.zeros((labels.shape[0])).to(outputs.device)
            for k in range(1, self.config.CLASS_N + 1):
                topk_classes = outputs[indices].topk(k, dim=-1).indices
                next_class[indices] = topk_classes[:,i,-1].view(-1).to(torch.float32)
                indices = self.check_discon(labels, next_class, indices)
                if len(indices)==0:
                    break
            labels = torch.cat([labels, next_class.unsqueeze(-1)], dim=1)
        return labels
    
    def check_discon(self, labels, next_class, indices):
        disease_dict = {1:[],
                        2:[1], 
                        3:[3,6,9,10,11], 
                        4:[], 
                        5:[2,9,10,11], 
                        6:[4,5,7,8]}
        disease_dict = {key+1:[v+8 for v in value] for key, value in disease_dict.items()}
        
        dis_indices=[]
        for i in indices:
            ##### For Crop
            ## 
            if labels.shape[-1] == 1:
                if next_class[i] not in range(2,8):
                    dis_indices.append(i.item())
            ##### For Disease
            ## 종류에 맞는 disease가 아니면 에러
            elif labels.shape[-1] == 2:
                if (next_class[i] in range(8,20)) and (next_class[i]==8 or next_class[i] in disease_dict[int(labels[i][-1].item())]):
                    pass
                else:
                    dis_indices.append(i.item())
            ##### For Risk
            ## 00인데 risk가 0이 아니면 에러 / Risk가 아니면 에러
            elif labels.shape[-1] == 3:
                if (next_class[i] < 20) or ((labels[i][-1].item()==8.) != (next_class[i].item()==20)):
                    dis_indices.append(i.item())
        if dis_indices:
            # dis_indices = torch.stack(dis_indices)
            dis_indices = torch.tensor(dis_indices)
        return dis_indices

def one_hot_vector(x, C):
    output = torch.zeros((x.shape[0], x.shape[1], C)).to(x.device)
    for i in range(x.shape[0]):
        for l in range(x.shape[1]):
            output[i, l, x[i, l].to(torch.long)] = 1.
    return output
    
class ViTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vit = ViT(config, 'B_16', pretrained=True)
        for param in self.vit.parameters():
            param.requires_grad = True
        
    def forward(self, x, *args, **kwargs):
        """Breaks image into patches, applies pretrained transformer, applies custom MLP head.

        Args:
            x (tensor): `b,c,fh,fw`
        """
        b, c, fh, fw = x.shape
        x = self.vit.patch_embedding(x)  # b,d,gh,gw
        x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
        if hasattr(self, 'class_token'):
            x = torch.cat((self.vit.class_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d
        if hasattr(self, 'positional_embedding'): 
            x = self.vit.positional_embedding(x)  # b,gh*gw+1,d 
        x = self.vit.transformer(x)  # b,gh*gw+1,d
        return x

class ViT_tuned(nn.Module):
    def __init__(self, config):
        super(ViT_tuned, self).__init__()
        self.vit = ViT(config, 'B_16', pretrained=True)
        for param in self.vit.parameters():
            param.requires_grad = True
        # for param in self.vit.transformer.blocks[-1].pwff.parameters():
        #     param.requires_grad = True
        self.fc = nn.Linear(768, config.CLASS_N)
        
    def forward(self, x, *args, **kwargs):
        """Breaks image into patches, applies pretrained transformer, applies custom MLP head.

        Args:
            x (tensor): `b,c,fh,fw`
        """
        b, c, fh, fw = x.shape
        x = self.vit.patch_embedding(x)  # b,d,gh,gw
        x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
        if hasattr(self, 'class_token'):
            x = torch.cat((self.vit.class_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d
        if hasattr(self, 'positional_embedding'): 
            x = self.vit.positional_embedding(x)  # b,gh*gw+1,d 
        x = self.vit.transformer(x)  # b,gh*gw+1,d
        if hasattr(self, 'pre_logits'):
            x = self.vit.pre_logits(x)
            x = torch.tanh(x)
        if hasattr(self, 'fc'):
            x = self.vit.norm(x)[:, 0]  # b,d
            x = self.fc(x)  # b,num_classes
        return x
    
class MyViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.csv_on = config.USE_CSV
        self.vit = ViT(config, add_csv=self.csv_on,
                       image_size = (config['IMAGE_HEIGHT'], config['IMAGE_WIDTH']))
        self.in_channels = 4 if config.USE_SPOT else 3
        fh, fw = self.vit.fh, self.vit.fw
        self.vit.patch_embedding = nn.Conv2d(self.in_channels, config.D_MODEL, 
                                             kernel_size=(fh, fw), stride=(fh, fw))
        self.csv_extract = ResNet(config)
        self.norm = nn.LayerNorm(config.D_MODEL, eps=1e-6)
        self.fc = nn.Linear(config.D_MODEL, config.CLASS_N)
        
    def forward(self, img, csv_features, *args, **kwargs):
        # return self.vit(img)
        
        b, c, fh, fw = img.shape
        img = self.vit.patch_embedding(img)  # b,d,gh,gw
        img = img.flatten(2).transpose(1, 2)  # b,gh*gw,d
        
        if self.csv_on:
            csv_feat = self.csv_extract(csv_features) # b, 1, d
            feats = torch.cat((self.vit.class_token.expand(b, -1, -1), csv_feat.permute(0,2,1), img), dim=1)  # b,gh*gw+1+1,d
        else:
            feats = torch.cat((self.vit.class_token.expand(b, -1, -1), img), dim=1)  # b,gh*gw+1,d
        feats = self.vit.positional_embedding(feats)  # b,gh*gw+1+1,d 
        output = self.vit.transformer(feats)  # b,gh*gw+1+1,d
        output = self.norm(output)[:,0] # b,d
        outputs = self.fc(output)
        return outputs


class PositionalEmbedding1D(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, seq_len, dim, train=True):
        super().__init__()
        self.dim = dim
        if train:
            self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))
        else:
            self.pos_embedding = torch.stack([self.get_angle(pos, torch.arange(dim)) for pos in range(seq_len)])
    
    def forward(self, x):
        """Input has shape `(batch_size, seq_len, emb_dim)`"""
        return x + self.pos_embedding.to(x.device)
            
    def get_angle(self, position, i):
        angles = 1 / torch.float_power(10000, (2 * (i // 2)) / self.dim)
        return position * angles


class ViT(nn.Module):
    """
    Args:
        name (str): Model name, e.g. 'B_16'
        pretrained (bool): Load pretrained weights
        in_channels (int): Number of channels in input data
        num_classes (int): Number of classes, default 1000

    References:
        [1] https://openreview.net/forum?id=YicbFdNTTy
    """

    def __init__(
        self, 
        CONFIG,
        name: Optional[str] = None, 
        pretrained: bool = False, 
        patches: int = 16,
        dim: int = 768,
        ff_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        attention_dropout_rate: float = 0.0,
        dropout_rate: float = 0.1,
        representation_size: Optional[int] = None,
        load_repr_layer: bool = False,
        classifier: str = 'token',
        positional_embedding: str = '1d',
        in_channels: int = 3, 
        image_size: Optional[int] = None,
        num_classes: Optional[int] = None,
        resize_positional_embedding=False,
        add_csv=False
    ):
        super().__init__()
        # Configuration
        if name is None:
            check_msg = 'must specify name of pretrained model'
            assert not pretrained, check_msg
            assert not resize_positional_embedding, check_msg
            if image_size is None:
                image_size = (CONFIG['IMAGE_HEIGHT'], CONFIG['IMAGE_WIDTH'])
            num_classes = CONFIG['CLASS_N']
            patches = CONFIG['PATCHES']
            dim = CONFIG['D_MODEL']
            ff_dim = CONFIG['FF_DIM']
            num_heads = CONFIG['N_HEAD']
            num_layers = CONFIG.ENCODER['NUM_LAYER']
            dropout_rate = CONFIG['DROP_OUT']
        else:  # load pretrained model
            assert name in PRETRAINED_MODELS.keys(), \
                'name should be in: ' + ', '.join(PRETRAINED_MODELS.keys())
            config = PRETRAINED_MODELS[name]['config']
            patches = config['patches']
            dim = config['dim']
            ff_dim = config['ff_dim']
            num_heads = config['num_heads']
            num_layers = config['num_layers']
            attention_dropout_rate = config['attention_dropout_rate']
            dropout_rate = config['dropout_rate']
            representation_size = config['representation_size']
            classifier = config['classifier']
            if image_size is None:
                image_size = PRETRAINED_MODELS[name]['image_size']
            if num_classes is None:
                num_classes = PRETRAINED_MODELS[name]['num_classes']
        self.image_size = image_size                

        # Image and patch sizes
        h, w = as_tuple(image_size)  # image sizes
        self.fh, self.fw = as_tuple(patches)  # patch sizes
        gh, gw = h // self.fh, w // self.fw  # number of patches
        seq_len = gh * gw

        # Patch embedding
        self.patch_embedding = nn.Conv2d(in_channels, dim, kernel_size=(self.fh, self.fw), stride=(self.fh, self.fw))

        # Class token
        if classifier == 'token':
            self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
            seq_len += 1
        
        # CSV ADD
        if add_csv:
            seq_len += 1
        
        # Positional embedding
        if positional_embedding.lower() == '1d':
            self.positional_embedding = PositionalEmbedding1D(seq_len, dim)
        else:
            raise NotImplementedError()
        
        # Transformer
        self.transformer = Transformer(num_layers=num_layers, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate)
        
        # Representation layer
        if representation_size and load_repr_layer:
            self.pre_logits = nn.Linear(dim, representation_size)
            pre_logits_size = representation_size
        else:
            pre_logits_size = dim

        # Classifier head
        self.norm = nn.LayerNorm(pre_logits_size, eps=1e-6)
        self.fc = nn.Linear(pre_logits_size, num_classes)

        # Initialize weights
        self.init_weights()
        
        # Load pretrained model
        if pretrained:
            pretrained_num_channels = 3
            pretrained_num_classes = PRETRAINED_MODELS[name]['num_classes']
            pretrained_image_size = PRETRAINED_MODELS[name]['image_size']
            load_pretrained_weights(
                self, weights_path=CONFIG.PRE_PATH,
                load_first_conv=(in_channels == pretrained_num_channels),
                load_fc=(num_classes == pretrained_num_classes),
                load_repr_layer=load_repr_layer,
                resize_positional_embedding=(image_size != pretrained_image_size),
            )
        
    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)
        self.apply(_init)
        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.normal_(self.positional_embedding.pos_embedding, std=0.02)  # _trunc_normal(self.positional_embedding.pos_embedding, std=0.02)
        nn.init.constant_(self.class_token, 0)

    def forward(self, x):
        """Breaks image into patches, applies transformer, applies MLP head.

        Args:
            x (tensor): `b,c,fh,fw`
        """
        b, c, fh, fw = x.shape
        x = self.patch_embedding(x)  # b,d,gh,gw
        x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
        if hasattr(self, 'class_token'):
            x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d
        if hasattr(self, 'positional_embedding'): 
            x = self.positional_embedding(x)  # b,gh*gw+1,d 
        x = self.transformer(x)  # b,gh*gw+1,d
        if hasattr(self, 'pre_logits'):
            x = self.pre_logits(x)
            x = torch.tanh(x)
        if hasattr(self, 'fc'):
            x = self.norm(x)[:, 0]  # b,d
            x = self.fc(x)  # b,num_classes
        return x


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


class MultiHeadedSelfAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None # for visualization

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    """FeedForward Neural Networks for each position"""
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(dim, num_heads, dropout)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.drop(self.proj(self.attn(self.norm1(x), mask)))
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)))
        x = x + h
        return x


class Transformer(nn.Module):
    """Transformer with Self-Attentive Blocks"""
    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)
        return x


def load_pretrained_weights(
    model, 
    model_name=None, 
    weights_path=None, 
    load_first_conv=True, 
    load_fc=True, 
    load_repr_layer=False,
    resize_positional_embedding=False,
    verbose=True,
    strict=True,
):
    """Loads pretrained weights from weights path or download using url.
    Args:
        model (Module): Full model (a nn.Module)
        model_name (str): Model name (e.g. B_16)
        weights_path (None or str):
            str: path to pretrained weights file on the local disk.
            None: use pretrained weights downloaded from the Internet.
        load_first_conv (bool): Whether to load patch embedding.
        load_fc (bool): Whether to load pretrained weights for fc layer at the end of the model.
        resize_positional_embedding=False,
        verbose (bool): Whether to print on completion
    """
    assert bool(model_name) ^ bool(weights_path), 'Expected exactly one of model_name or weights_path'
    
    # Load or download weights
    if weights_path is None:
        url = PRETRAINED_MODELS[model_name]['url']
        if url:
            state_dict = model_zoo.load_url(url)
        else:
            raise ValueError(f'Pretrained model for {model_name} has not yet been released')
    else:
        state_dict = torch.load(weights_path)

    # Modifications to load partial state dict
    expected_missing_keys = []
    if not load_first_conv and 'patch_embedding.weight' in state_dict:
        expected_missing_keys += ['patch_embedding.weight', 'patch_embedding.bias']
    if not load_fc and 'fc.weight' in state_dict:
        expected_missing_keys += ['fc.weight', 'fc.bias']
    if not load_repr_layer and 'pre_logits.weight' in state_dict:
        expected_missing_keys += ['pre_logits.weight', 'pre_logits.bias']
    for key in expected_missing_keys:
        state_dict.pop(key)

    # Change size of positional embeddings
    if resize_positional_embedding: 
        posemb = state_dict['positional_embedding.pos_embedding']
        posemb_new = model.state_dict()['positional_embedding.pos_embedding']
        state_dict['positional_embedding.pos_embedding'] = \
            resize_positional_embedding_(posemb=posemb, posemb_new=posemb_new, 
                has_class_token=hasattr(model, 'class_token'))
        maybe_print('Resized positional embeddings from {} to {}'.format(
                    posemb.shape, posemb_new.shape), verbose)

    # Load state dict
    ret = model.load_state_dict(state_dict, strict=False)
    if strict:
        assert set(ret.missing_keys) == set(expected_missing_keys), \
            'Missing keys when loading pretrained weights: {}'.format(ret.missing_keys)
        assert not ret.unexpected_keys, \
            'Missing keys when loading pretrained weights: {}'.format(ret.unexpected_keys)
        maybe_print('Loaded pretrained weights.', verbose)
    else:
        maybe_print('Missing keys when loading pretrained weights: {}'.format(ret.missing_keys), verbose)
        maybe_print('Unexpected keys when loading pretrained weights: {}'.format(ret.unexpected_keys), verbose)
        return ret


def maybe_print(s: str, flag: bool):
    if flag:
        print(s)


def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)


def resize_positional_embedding_(posemb, posemb_new, has_class_token=True):
    """Rescale the grid of position embeddings in a sensible manner"""
    from scipy.ndimage import zoom

    # Deal with class token
    ntok_new = posemb_new.shape[1]
    if has_class_token:  # this means classifier == 'token'
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

    # Get old and new grid sizes
    gs_old = int(np.sqrt(len(posemb_grid)))
    gs_new = int(np.sqrt(ntok_new))
    posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

    # Rescale grid
    zoom_factor = (gs_new / gs_old, gs_new / gs_old, 1)
    posemb_grid = zoom(posemb_grid, zoom_factor, order=1)
    posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
    posemb_grid = torch.from_numpy(posemb_grid)

    # Deal with class token and return
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb

"""configs.py - ViT model configurations, based on:
https://github.com/google-research/vision_transformer/blob/master/vit_jax/configs.py
"""

def get_base_config():
    """Base ViT config ViT"""
    return dict(
      dim=768,
      ff_dim=3072,
      num_heads=12,
      num_layers=12,
      attention_dropout_rate=0.0,
      dropout_rate=0.1,
      representation_size=768,
      classifier='token'
    )

def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = get_base_config()
    config.update(dict(patches=(16, 16)))
    return config

def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.update(dict(patches=(32, 32)))
    return config

def get_l16_config():
    """Returns the ViT-L/16 configuration."""
    config = get_base_config()
    config.update(dict(
        patches=(16, 16),
        dim=1024,
        ff_dim=4096,
        num_heads=16,
        num_layers=24,
        attention_dropout_rate=0.0,
        dropout_rate=0.1,
        representation_size=1024
    ))
    return config

def get_l32_config():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config()
    config.update(dict(patches=(32, 32)))
    return config

def drop_head_variant(config):
    config.update(dict(representation_size=None))
    return config

PRETRAINED_MODELS = {
    'B_16': {
      'config': get_b16_config(),
      'num_classes': 21843,
      'image_size': (224, 224),
      'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16.pth"
    },
    'B_32': {
      'config': get_b32_config(),
      'num_classes': 21843,
      'image_size': (224, 224),
      'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_32.pth"
    },
    'L_16': {
      'config': get_l16_config(),
      'num_classes': 21843,
      'image_size': (224, 224),
      'url': None
    },
    'L_32': {
      'config': get_l32_config(),
      'num_classes': 21843,
      'image_size': (224, 224),
      'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/L_32.pth"
    },
    'B_16_imagenet1k': {
      'config': drop_head_variant(get_b16_config()),
      'num_classes': 1000,
      'image_size': (384, 384),
      'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth"
    },
    'B_32_imagenet1k': {
      'config': drop_head_variant(get_b32_config()),
      'num_classes': 1000,
      'image_size': (384, 384),
      'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_32_imagenet1k.pth"
    },
    'L_16_imagenet1k': {
      'config': drop_head_variant(get_l16_config()),
      'num_classes': 1000,
      'image_size': (384, 384),
      'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/L_16_imagenet1k.pth"
    },
    'L_32_imagenet1k': {
      'config': drop_head_variant(get_l32_config()),
      'num_classes': 1000,
      'image_size': (384, 384),
      'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/L_32_imagenet1k.pth"
    },
}
