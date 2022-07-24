import torch
import torch.nn.functional as F
import torch.nn as nn

class WaveNet(nn.Module):
    def __init__(self, quantization_bins=256, channels=32, kernel_size=2, dilation_depth=5, blocks=1, initial_cond_size=768,
                 condition_size=512, global_initial_cond_size=768,
                 global_condition=False, local_condition=False, hugging_face_model='bert-base-uncased', use_bert=True):
        super(WaveNet, self).__init__()
        self.C = channels
        self.kernel_size = kernel_size
        self.bins = quantization_bins
        self.dilations = [2 ** i for i in range(dilation_depth)] * blocks
        self.receptive_field = sum([(self.kernel_size-1)*2 ** i for i in range(dilation_depth)] * blocks)
        self.causal_layers = nn.ModuleList()
        
        self.sentence_embedding = SentenceEmbedding(initial_cond_size, condition_size, hugging_face_model=hugging_face_model, use_bert=use_bert)
        
        self.input_embedding = nn.Embedding(self.bins,self.C)

        # Setup network
        self.pre_process_conv = nn.Conv1d(in_channels=self.C, out_channels=self.C, kernel_size=1)
        for d in self.dilations:
            self.causal_layers.append(ResidualLayer(in_channels=self.C,
                                                    out_channels=self.C,
                                                    dilation=d,
                                                    kernel_size=self.kernel_size, 
                                                    condition_size=condition_size, 
                                                    global_condition=global_condition, 
                                                    local_condition=local_condition))

            
        if local_condition or global_condition:
            self.condition_size = condition_size
        self.local_condition = local_condition
        self.global_condition = global_condition
        if global_condition:
            self.gc_initial = nn.Sequential(nn.Linear(global_initial_cond_size, condition_size), 
                                            nn.ReLU(), 
                                            nn.Linear(condition_size, condition_size), 
                                            nn.ReLU())
        if local_condition:
            self.lc_initial = nn.Sequential(nn.Conv1d(condition_size, condition_size, kernel_size=1), 
                                            nn.ReLU(), 
                                            nn.Conv1d(condition_size, condition_size, kernel_size=1), 
                                            nn.ReLU())
        self.post_process_conv1 = nn.Conv1d(self.C, self.C, kernel_size=1)
        self.post_process_conv2 = nn.Conv1d(self.C, self.bins, kernel_size=1)


    def forward(self, quantisized_x, gc=None, lc=None):
        """
        :param input: Mu- and one-hot-encoded waveform. Shape (batch_size, quantization_bins, samples)
        :return: distribution for prediction of next sample.
                 Shape (batch_size, quantization_bins, what's left after dilation, should be 1 at inference)
        NOTE: x most have at least the length of the model's receptive field
        """
        #embed = F.one_hot(quantisized_x, num_classes=self.bins).permute(0,2,1) #<--- one_hot encoding
        embed = self.input_embedding(quantisized_x).permute(0,2,1) #<--- Embedded encoding
        
        if self.global_condition and gc is not None:
            gc = self.gc_initial(gc)
        if self.local_condition and lc is not None:
            lc = self.lc_initial(lc)

        # (1) pre process
        x = self.pre_process_conv(embed) # shape --> (batch_size, channels, samples)

        # (2) Through the stack of dilated causal convolutions
        skips = []
        for layer in self.causal_layers:
            x, skip = layer(x, gc=gc, lc=lc)
            skips.append(skip)

        # (3) Post processes (-softmax)
        x = torch.stack([s[:, :, -skip.size(2):] for s in skips],0).sum(0) # adding up skip-connections
        x = F.relu(x)
        x = self.post_process_conv1(x) # shape --> (batch_size, channels, samples)
        x = F.relu(x)
        x = self.post_process_conv2(x)  # shape --> (batch_size, quantization_bins, samples)
        return x

class ResidualLayer(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, dilation:int, condition_size=None, global_condition:bool = False, local_condition:bool = False):
        super(ResidualLayer, self).__init__()
        self.conv_fg = nn.Conv1d(in_channels, 2*out_channels, kernel_size=kernel_size, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.dilation = dilation
        self.kernel_size = kernel_size
        if global_condition:
            self.gc_layer_fg = nn.Linear(condition_size, 2*out_channels)
        if local_condition:
            self.lc_layer_fg = nn.Conv1d(condition_size, 2*out_channels, 1)
        

    def forward(self, x, gc=None, lc=None):
        fg = self.conv_fg(x)
        
        if lc is not None and gc is not None:
            lc = self.lc_layer_fg(lc)[:,:,-fg.size(-1):]
            gc = self.gc_layer_fg(gc).view(fg.size(0),-1,1)
            fg = fg+lc+gc
        elif lc is not None:
            lc = self.lc_layer_fg(lc)[:,:,-fg.size(-1):]
            fg = fg+lc
        elif gc is not None:
            gc = self.gc_layer_fg(gc).view(fg.size(0),-1,1)
            fg = fg+gc
        
        f,g = torch.chunk(fg,2,dim=1)
        f = torch.tanh(f)
        g = torch.sigmoid(g)
        fg = f * g
        skip = self.conv_1x1(fg) # <-- TODO try with ReLU instead

        residual = x[:, :, -skip.size(2):] + skip
        return residual, skip
    
class SentenceEmbedding(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=8, stride=4, hugging_face_model='bert-base-uncased', use_bert=True):
        super(SentenceEmbedding, self).__init__()
        self.use_bert = use_bert
        if self.use_bert:
            self.bert = torch.hub.load('huggingface/pytorch-transformers', 'model', hugging_face_model)
        self.transposed_convs = nn.Sequential(nn.ConvTranspose1d(in_channels,out_channels,kernel_size,stride), nn.ReLU(),
                                              nn.ConvTranspose1d(out_channels,out_channels,kernel_size,stride), nn.ReLU(),
                                              nn.ConvTranspose1d(out_channels,out_channels,kernel_size,stride), nn.ReLU(),
                                              nn.ConvTranspose1d(out_channels,out_channels,kernel_size,stride), nn.ReLU())

    def forward(self, tokens):
        """
        :param tokens: Embedding of sentence shape (B, C, T+2) where T is sequence length and C is the size of the embedding dimension.
                       The additional items in the third dimension are a start token [CLS] and the end of sentence token [SEP].
        :return: [CLS] token from BERT embedding as global condition signal and upsampled local condition signal.
        """
        if self.use_bert:
            #Feed tokenized transcript into BERT.
            bert_out = self.bert(input_ids=tokens, return_dict=True)['last_hidden_state'] #shape: (B, C, T+2)

            #Get first token ([CLS]) as the global condition signal.
            gc_embed = bert_out[:,0]

            #Take the innermost items (between the [CLS] and [SEP] tokens) as the local condition signal.
            bert_lc = bert_out[:,1:-1,:].permute(0,2,1)
            #Feed into transposed convolution to perform learned upsampling.
            lc_embed = self.transposed_convs(bert_lc)
        else:
            gc_embed = None
            lc_embed = self.transposed_convs(tokens)
        
        return gc_embed, lc_embed