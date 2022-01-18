import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerV1(nn.Module):
    '''
    ViT Model
    Inputs should be tensor images (B x C x 256 x 256)
    '''
    def __init__(self, config):
        self.image_size = config.IMAGE_SIZE
        self.seq_L = int(self.image_size/16)
        self.linearEmbedding = nn.Linear(3 * 16 * 16, config.D_MODEL)
        self.convEmbedding = nn.Conv2d(3, config.D_MODEL, kernel_size=16, stride=16)
        self.encoder = TransformerEncoder(config)
        
    def forward(self, inputs):
        ##### Image to 16x16 patches
        ### with Linear Embedding 
        # images = torch.zeros((inputs.shape[0], self.seq_L, 3, 16, 16))  # (B x L x 3 x 16 x 16)
        # for h in range(self.seq_L):
        #     for w in range(self.seq_L):
        #         images[:, h*self.seq_L + w] = inputs[:,:,h*16:(h+1)*16, w*16:(w+1)*16]
        # images = torch.flatten(images, start_dim=2)
        # images = self.linearEmbedding(images) # (B x L x D)
        ### with Convolution Embedding
        images = self.convEmbedding(inputs) # (B x D x 16 x 16)
        images = images.permute(0,2,3,1)    # (B x 16 x 16 x D)
        images = images.view(images.shape[0], -1, images.shape[-1]) # (B x L x D)
        
        ##### Positional Embedding
        
        
        ##### Encode
        outputs = self.encoder(inputs)
        ## Softmax
        
        return outputs
        
class TransformerV2(nn.Module):
    def __init__(self, config):
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        
    def forward(self, inputs, label, train=True):
        outputs = self.encoder(inputs)
        outputs = self.decoder(label, outputs)
        return outputs

class TransformerDecoder(nn.Module):
    def __init__(self, config):
        decoder_cell = nn.TransformerDecoderLayer(config.D_MODEL, config.N_HEAD, config.FF_DIM, config.DROP_OUT)
        self.decoder = nn.TransformerDecoder(decoder_cell, config.DECODER.NUM_LAYER)
        
    def forward(self, labels, memory, label_mask=None, memory_mask=None):
        return self.decoder(labels, memory, label_mask, memory_mask)

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        encoder_cell = nn.TransformerEncoderLayer(config.D_MODEL, config.N_HEAD, config.FF_DIM, config.DROP_OUT)
        self.encoder = nn.TransformerEncoder(encoder_cell, config.ENCODER.NUM_LAYER)
        
    def forward(self, inputs):
        return self.encoder(inputs)
    
class ImageEmbedder(nn.Module):
    def __init__(self, config):
        self.image_size = config.IMAGE_SIZE
        self.seq_L = int(self.image_size/16)
        self.linearEmbedding = nn.Linear(3 * 16 * 16, config.D_MODEL)
        self.convEmbedding = nn.Conv2d(3, config.D_MODEL, kernel_size=16, stride=16)
        self.encoder = TransformerEncoder(config)
        initial_position = torch.randn((self.seq_L**2 + 1, config.D_MODEL))
        self.positional = nn.Parameter(initial_position)
        
    def forward(self, inputs, mode='conv'):
        ##### Image to 16x16 patches
        if mode=='linear':
        ### with Linear Embedding 
            images = torch.zeros((inputs.shape[0], self.seq_L, 3, 16, 16))  # (B x L x 3 x 16 x 16)
            for h in range(self.seq_L):
                for w in range(self.seq_L):
                    images[:, h*self.seq_L + w] = inputs[:,:,h*16:(h+1)*16, w*16:(w+1)*16]
            images = torch.flatten(images, start_dim=2)
            images = self.linearEmbedding(images) # (B x L x D)
        ### with Convolution Embedding
        elif mode=='conv':
            images = self.convEmbedding(inputs) # (B x D x 16 x 16)
            images = images.permute(0,2,3,1)    # (B x 16 x 16 x D)
            images = images.view(images.shape[0], -1, images.shape[-1]) # (B x L x D)
        else:
            raise f'{mode} mode Embedding is not available!!'
        
        ##### Positional Encoding
        