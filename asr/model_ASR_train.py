######################################### ASR not-pretained main script #################################################

# Import general libraries
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime

# Import stuff from other ASR modules
from asr.data import BaseDataset
from asr.data.preprocessors import SpectrogramPreprocessor, TextPreprocessor
from asr.modules import ASRModel
from asr.utils.training import batch_to_tensor, epochs, Logger
from asr.utils.text import greedy_ctc
from asr.utils.metrics import ErrorRateTracker, LossTracker

def train_ASR(train_IDs, test_IDs):
    """ Function: Train and test ASR model on input datasets
        Input:    2 txt-files with IDs to training and test files. One observation consists of a transcript
                  (txt-feature) and audio file (wav-target).
        Output:   Return best WER (validation) and save ASR model (model.pt) """


    """ Part 1: Load and preprocess data """
    train_source = train_IDs
    val_source = test_IDs

    # BLACK BOX
    spec_preprocessor = SpectrogramPreprocessor(output_format='NFT')
    text_preprocessor = TextPreprocessor()
    preprocessor = [spec_preprocessor, text_preprocessor]
    
    train_dataset = BaseDataset(source=train_source, preprocessor=preprocessor, sort_by=0)
    val_dataset = BaseDataset(source=val_source, preprocessor=preprocessor, sort_by=0)

    # Data loader
    train_loader = DataLoader(train_dataset, num_workers=4, pin_memory=True, collate_fn=train_dataset.collate, batch_size=16)
    val_loader = DataLoader(val_dataset, num_workers=4, pin_memory=True, collate_fn=val_dataset.collate, batch_size=16)

    """ Part 2: Setup model and loss """
    # Create instance of model
    asr_model = ASRModel()
    print(asr_model)
    print("Trainable parameters:", sum(p.numel() for p in asr_model.parameters() if p.requires_grad))
    asr_model.to(f'cuda')

    # Define loss, optimizer and learning rate scheduler
    ctc_loss = nn.CTCLoss(reduction='sum').cuda()
    optimizer = torch.optim.Adam(asr_model.parameters(), lr=3e-4)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=5e-5)

    """ Part 3: Train and evaluate model """
    # Variables to create and store performance metrics
    wer_metric = ErrorRateTracker(word_based=True)
    cer_metric = ErrorRateTracker(word_based=False)
    ctc_metric = LossTracker()

    train_logger = Logger('Training', ctc_metric, wer_metric, cer_metric)
    val_logger = Logger('Validation', ctc_metric, wer_metric, cer_metric)

    def forward_pass(batch):
        (x, x_sl), (y, y_sl) = batch_to_tensor(batch, device='cuda')  # For CPU: change 'cuda' to 'cpu'
        logits, output_sl = asr_model.forward(x, x_sl.cpu())
        log_probs = F.log_softmax(logits, dim=2)
        loss = ctc_loss(log_probs, y, output_sl, y_sl)

        hyp_encoded_batch = greedy_ctc(logits, output_sl)
        hyp_batch = text_preprocessor.decode_batch(hyp_encoded_batch)
        ref_batch = text_preprocessor.decode_batch(y, y_sl)

        wer_metric.update(ref_batch, hyp_batch)
        cer_metric.update(ref_batch, hyp_batch)
        ctc_metric.update(loss.item(), weight=output_sl.sum().item())

        return loss

    # Run 200 epochs
    model_name = f"{train_IDs}vs{test_IDs}{datetime.now().strftime('Y%Y-m%m-d%d-H%H-M%M')}" 
    with open(f"./results/{model_name}.csv", "a+") as o_f:
        for epoch in epochs(200):
            # Set PyTorch in training mode
            asr_model.train()

            # Train model on training set
            for batch, files in train_logger(train_loader):
                loss = forward_pass(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Set PyTorch in test mode
            asr_model.eval()

            # Test model using test set
            with torch.no_grad():
                for batch, files in val_logger(val_loader):
                    forward_pass(batch)

            # Store best WER and also save best model
            #for b in batch:
            wer = wer_metric.running
            cer = cer_metric.running
            ctc = ctc_metric.running
            print(f"{epoch}\t{wer}\t{cer}\t{ctc}", file=o_f)

            best_wer = 1000000
            if wer < best_wer:
                best_wer = wer
                torch.save(asr_model.state_dict(), f"./results/best_{model_name}.pt")

            if epoch >= 100:
                lr_scheduler.step()

    return best_wer
