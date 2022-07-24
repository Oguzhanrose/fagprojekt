import numpy as np
from tqdm.notebook import tqdm
import torchaudio
import torch
import torch.nn.functional as F

@torch.no_grad()
def synthesize(transcripts, waveform_sizes, model, tokenizer, device, seed=None, temperature=1.0):
    T = max(waveform_sizes)
    #Tokenize the transcript with the BERT tokenizer
    with torch.no_grad():
        model.eval()
        tokens = tokenizer(transcripts, return_attention_mask=False, return_token_type_ids=False, return_tensors='pt', padding=True)['input_ids'].to(device)

        #Feed into sentence embedding class
        gc_embed, lc_embed = model.sentence_embedding(tokens)

        #Interpolate the locally conditioned signal from BERT so it fits with the waveform size and then trim the same portion of the signal as for the waveform.
        lc_embed = F.interpolate(lc_embed, size=T)
        lc_embed = F.pad(lc_embed, (model.receptive_field,0))

        rec_fld = model.receptive_field + 1

        if seed is not None:
            seed_T = seed.size(1)
        else:
            seed_T = 0
    
        generated = (torch.ones((len(transcripts),rec_fld+T), device=device, dtype=torch.int64)*torchaudio.transforms.MuLawEncoding(model.bins)(torch.tensor(0.0)).item())
        if seed is not None:
            generated[:, :seed_T] = seed
        with tqdm(range(seed_T if seed_T is not None else 0,T)) as t_bar:
            for n in t_bar:
                predictions = model(generated[:,n:rec_fld+n], lc=lc_embed[:,:,n:rec_fld+n], gc=gc_embed)
                predictions = torch.softmax(predictions/temperature, dim=1)
                generated[:,n+rec_fld] = torch.multinomial(predictions.squeeze(), 1).squeeze()
    generated = generated[:, rec_fld:]
    return generated