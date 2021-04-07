import argparse
import deepspeed

import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision

dataset_dir = r"test_frames"


parser = argparse.ArgumentParser()
parser.add_argument('deepspeed_config')
args = parser.parse_args()



class Decoder(nn.Module):
    def __init__(self, dim, depth=6, heads=8, max_seq_len=16384, bucket_size=64):
        super(Decoder, self).__init__()
        
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.max_seq_len = max_seq_len
        self.bucket_size = bucket_size

        # Initializing a Reformer configuration
        self.configuration = ReformerConfig(attention_head_size=64, attn_layers=['local', 'lsh', 'local', 'lsh', 'local', 'lsh'], axial_norm_std=1.0, axial_pos_embds=True, axial_pos_shape=[256, 64], 
                                       axial_pos_embds_dim=[64, 192], chunk_size_lm_head=0, eos_token_id=2, feed_forward_size=256, hash_seed=None, hidden_act='relu', hidden_dropout_prob=0.05, 
                                       hidden_size=256, initializer_range=0.02, is_decoder=True, layer_norm_eps=1e-12, local_num_chunks_before=1, local_num_chunks_after=0, 
                                       local_attention_probs_dropout_prob=0.05, local_attn_chunk_length=64, lsh_attn_chunk_length=64, lsh_attention_probs_dropout_prob=0.0, lsh_num_chunks_before=1, 
                                       lsh_num_chunks_after=0, max_position_embeddings=16384, num_attention_heads=self.heads, num_buckets=None, num_hashes=1, pad_token_id=0, vocab_size=320, 
                                       tie_word_embeddings=False, use_cache=False, target_mapping=None)

        # Initializing a Reformer model
        self.decoder = ReformerModel(self.configuration)
        
        # self.pos_embedder = AxialPositionalEmbedding(256, (256, 64))
        self.fmap_embedder = AxialPositionalEmbedding(256, (256, 64))
    
    @autocast()
    def forward(self, x):
        
        # self.out = x + self.pos_embedder(x)
        self.out = x
        
        #Positional Embedding
        for b in range(len(self.out)): #batch
            for i in range(int(len(self.out[b])/64)): #vector embeddings in a batch
                self.out[b][i*64:(i+1)*64] = self.fmap_embedder(self.out[b][i*64:(i+1)*64].unsqueeze(0)).squeeze(0)
        
        print(self.out.shape)
        self.out = self.decoder(inputs_embeds=self.out)

        return self.out.last_hidden_state
    

class Input_Conv(nn.Module):
    def __init__(self):
        super(Input_Conv, self).__init__()
        
        # Initialize the DenseBlock, input shape is (n, 3, 256, 256), output shape is (n, 64, 16, 16)
        self.denseblock = torchvision.models.densenet121()
        self.denseblock.features.transition1.conv = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.denseblock.features.transition1.pool = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)
        self.denseblock = nn.Sequential(*list(self.denseblock.features.children())[:6])
    
    @autocast()
    def forward(self, x):
        return self.denseblock(x)
    

class Output_ConvTranspose(nn.Module):
    def __init__(self):
        super(Output_ConvTranspose, self).__init__()
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(scale_factor=2)
        
        self.conv1 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=[3,3], stride=1, padding=1)  
        self.conv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=[3,3], stride=1, padding=1)  
        self.conv3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=[3,3], stride=1, padding=1)  
        self.conv4 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=[3,3], stride=1, padding=1)
        self.conv5 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=[1,1], stride=1, padding=0)
    
    @autocast()
    def forward(self, x):
        # input size (1, 64, 16, 16)
        
        self.out = self.conv1(x)
        self.out = self.relu(self.out)  
        self.out = self.upsample(self.out)
        
        self.out = self.conv2(self.out)
        self.out = self.relu(self.out)  
        self.out = self.upsample(self.out)
        
        self.out = self.conv3(self.out)
        self.out = self.relu(self.out)  
        self.out = self.upsample(self.out)
        
        self.out = self.conv4(self.out)
        self.out = self.relu(self.out)  
        self.out = self.upsample(self.out)
        
        self.out = self.conv5(self.out)
        self.out = self.sigmoid(self.out)
        
        return self.out
        

class Generator(nn.Module):
    def __init__(self, dim, depth=6, heads=8, max_seq_len=16384, bucket_size=64):
        super(Generator, self).__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.max_seq_len = max_seq_len
        self.bucket_size = bucket_size
        
        self.inputconv = Input_Conv()
        self.reformer = Decoder(dim=self.dim, depth=self.depth, heads=self.heads, max_seq_len=self.max_seq_len, bucket_size=self.bucket_size)
        self.outputconvtranspose = Output_ConvTranspose()
    
    @autocast()
    def forward(self, x):
        #input shape is (b, n, c, h, w)
        self.out = []
        for b in x:
            for n in b:
                self.out.append(self.inputconv(n.unsqueeze(0)).squeeze(0).cpu().detach().numpy())
        self.out = torch.Tensor(self.out).cuda()
        
        self.unflattened_shape = self.out.shape
        self.out = self.out.view(x.shape[0], self.max_seq_len, self.dim) #TODO padding for variable sequence length input
        
        self.out = self.reformer(self.out)
        print(self.out.shape)
        self.out = self.out.view(1, 256, 128, 16, 16) #TODO this need to be changed for adaptive sizing
        
        self.outarray = []
        for b in self.out:
            for n in b:
                self.outarray.append(self.outputconvtranspose(n.unsqueeze(0)).squeeze(0))
        #self.out = torch.Tensor(self.outarray)
        
        return self.outarray     
    


class ReformerDataset(Dataset):

    def __init__(self, file_dir, transform=None, seq_len=1):

        self.dir = file_dir
        self.transform = transform
        self.seq_len = seq_len
        self.diction = [] # yes, yes, it is an array called diction

        idx = 0
        for filename in os.listdir(self.dir):
            if filename.endswith('jpg'):
                self.diction.append(filename)
                idx += 1

    def __len__(self):
        return len(self.diction) - 1


    def __getitem__(self, idx):
        start = time.time()
        readImage = lambda filename: self.transform(np.array(cv.imread(os.path.join(self.dir, filename)) / 255)) if self.transform else np.array(cv.imread(os.path.join(self.dir, filename)) / 255)
        
        x, y = self.diction[idx*self.seq_len : (idx+1)*self.seq_len], self.diction[idx*self.seq_len+1 : (idx+1)*self.seq_len+1]
        x, y = torch.Tensor(np.asarray(list(map(readImage, x)))), torch.Tensor(np.asarray(list(map(readImage, y))))
        return [x, y]


def HWC2CHW(x):
    return np.array(x).transpose(2, 0, 1)


dataset = ReformerDataset(file_dir=dataset_dir, transform=HWC2CHW, seq_len=256)

loader = DataLoader(dataset=dataset, batch_size=4, shuffle=False, drop_last=True, num_workers=0)

model = Generator()
model_engine, optimizer, _, _ = deepspeed.initialize(args=args, model=model, model_parameters=model.parameters)

MSE_loss = torch.nn.MSELoss()

for step, batch in enumerate(loader):
    #forward() method
    out = model_engine(batch[0])
    print(out.shape)
    
    loss = 0
    
    for i, img in enumerate(out):
        loss += MSE_loss(img, batch[1][i])
    
    #runs backpropagation
    model_engine.backward(loss)

    #weight update
    model_engine.step()