import torch
import torch.nn as nn
import torch.optim as optim
import random 
import pandas as pd
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.nn import Transformer
from itertools import product


# Arguments
parser = argparse.ArgumentParser(); 
parser.add_argument("--train", type = bool, default = False); 
parser.add_argument("--generate", type = bool, default = False);
parser.add_argument("--dna", type = str, default = "CGGCGCTGGTGCCCAGGACGAGGATGGAGATT"); 
args = parser.parse_args();


# Initialize Vocaublary for Tokens
kmer_size = 3; 
dna_nucleotides = ['A', 'T', 'G', 'C']; 
start_of_sequence = '<sos>'; 
end_of_sequence = '<eos>'; 
pad_token = '<pad>'; 

kmer_vocab = [''.join(aPermutation) for aPermutation in product(dna_nucleotides, repeat = kmer_size)]; 
kmer_vocab += [start_of_sequence, end_of_sequence, pad_token]; 
vocab_size = len(kmer_vocab); 

string2index = {s: i for i, s in enumerate(kmer_vocab)}; 
index2string = {i: s for s, i in string2index.items()}; 


# Tokenization of inputs
def tokenizer(input_sequence): 
    return [input_sequence[i : i + kmer_size] for i in range(len(input_sequence) - kmer_size + 1)]; 


# Dataloader
class GRNADNADataset(Dataset): 
    def __init__(self, csv_path): 
        df = pd.read_csv(csv_path)
        self.data = []

        for dna_seq, grna_seq in zip(df["target_dna"], df["grna_seq"]):
            dna_tokens = tokenizer(dna_seq)
            grna_tokens = [start_of_sequence] + tokenizer(grna_seq) + [end_of_sequence]

            dna_ids = [string2index[token] for token in dna_tokens]
            grna_ids = [string2index[token] for token in grna_tokens]

            self.data.append((dna_ids, grna_ids))

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx): 
        return self.data[idx]


# Padding function
def padding_function(batch): 
    dna_batch, grna_batch = zip(*batch); 
    dna_lengths = [len(x) for x in dna_batch]; 
    grna_lengths = [len(x) for x in grna_batch]; 
    max_dna_length = max(dna_lengths); 
    max_grna_length = max(grna_lengths); 

    padding_id = string2index['<pad>']; 

    dna_padded = [x + [padding_id] * (max_dna_length - len(x)) for x in dna_batch]; 
    grna_padded = [x + [padding_id] * (max_grna_length - len(x)) for x in grna_batch]; 

    return torch.tensor(dna_padded, dtype = torch.long), torch.tensor(grna_padded, dtype = torch.long); 


class PositionalEncoding(nn.Module): 
    def __init__(self, model_dim, max_length = 5000): 
        super().__init__(); 
        pos_enc = torch.zeros(max_length, model_dim); 
        position = torch.arange(0, max_length).unsqueeze(1); 
        div = torch.exp(torch.arange(0, model_dim, 2) * -(math.log(10000.0) / model_dim)); 
        pos_enc[:, 0::2] = torch.sin(position * div); 
        pos_enc[:, 1::2] = torch.cos(position * div); 
        self.pos_enc = pos_enc.unsqueeze(0); 

    def forward(self, x): 
        return x + self.pos_enc[:, :x.size(1)].to(x.device); 


class GRNATransformer(nn.Module): 
    def __init__(self, vocab_size, model_dim = 256, nhead = 8, num_layers = 4, feedforward_dim = 512): 
        super().__init__(); 
        self.embedding = nn.Embedding(vocab_size, model_dim); 
        self.pos_enc = PositionalEncoding(model_dim); 
        self.transformer = nn.Transformer(
            d_model = model_dim,
            nhead = nhead, 
            num_encoder_layers = num_layers,
            num_decoder_layers = num_layers, 
            dim_feedforward = feedforward_dim, 
            dropout = 0.1, 
            batch_first = True
        ); 
        self.fully_connected = nn.Linear(model_dim, vocab_size); 

    def forward(self, dna, grna): 
        dna_mask = None; 
        grna_mask = nn.Transformer.generate_square_subsequent_mask(grna.size(1)).to(grna.device); 

        dna_emb = self.pos_enc(self.embedding(dna)); 
        grna_emb = self.pos_enc(self.embedding(grna)); 
        output = self.transformer(src = dna_emb, tgt = grna_emb, src_mask = None, tgt_mask = grna_mask); 
        return self.fully_connected(output); 


def train_model(model, dataloader, optimizer, criterion, device, num_epochs = 10): 
    model.train(); 
    epoch_losses = []; 

    for epoch in range(num_epochs): 
        epoch_loss = 0.0; 
        batch_count = 0; 
        
        for dna, grna, in dataloader: 
            dna, grna = dna.to(device), grna.to(device); 
            optimizer.zero_grad(); 
            output = model(dna, grna[:, :-1]); 
            loss = criterion(output.reshape(-1, vocab_size), grna[:, 1:].reshape(-1)); 
            loss.backward(); 
            optimizer.step(); 

            epoch_loss += loss.item(); 
            batch_count += 1; 

        avg_loss = epoch_loss / batch_count; 
        epoch_losses.append(avg_loss); 

        print(f"Epoch: {epoch + 1}, Loss: {loss.item():.4f}"); 

    plt.figure(); 
    plt.plot(range(1, num_epochs + 1), epoch_losses, linestyle='-'); 
    plt.title("Training Loss per Epoch"); 
    plt.xlabel("Epoch"); 
    plt.ylabel("Loss"); 
    plt.grid(True); 
    plt.savefig("training_loss_plot.png"); 
    plt.close(); 


def grna_generator(model, dna, max_length = 100): 
    model.eval(); 
    dna_tokens = tokenizer(dna); 
    dna_ids = torch.tensor([[string2index[token] for token in dna_tokens]]).to(next(model.parameters()).device); 

    generated = [string2index['<sos>']]; 
    for _ in range(max_length): 
        grna_tensor = torch.tensor([generated], dtype = torch.long).to(dna_ids.device); 
        with torch.no_grad(): 
            output = model(dna_ids, grna_tensor); 
        next_token = output[0, -1].argmax().item(); 
        if next_token == string2index['<eos>']:  
            break;  
        generated.append(next_token); 

    return [index2string[i] for i in generated[1:]]; 




if __name__ == '__main__': 

    # Hyperparameters 
    csv_path = "awesomeData.csv"; 
    batch_size = 32; 
    model_dim = 256; 
    nhead = 8; 
    num_layers = 4; 
    feedforward_dim = 512; 
    num_epochs = 50; 
    learning_rate = 1e-4; 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); 

    # Data
    dataset = GRNADNADataset(csv_path); 
    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = True,
        collate_fn = padding_function
    ); 

    # Model
    model = GRNATransformer(
        vocab_size = vocab_size,
        model_dim = model_dim,
        nhead = nhead,
        num_layers = num_layers,
        feedforward_dim = feedforward_dim
    ).to(device); 

    # Optimizer and Loss 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate); 
    criterion = nn.CrossEntropyLoss(ignore_index=string2index[pad_token]); 

    # Training 
    if(args.train): 
        print("Starting training...\n"); 
        train_model(model, dataloader, optimizer, criterion, device, num_epochs=num_epochs); 
        torch.save(model.state_dict(), 'transformer_model.pt'); 

    # Inference  
    if(args.generate):

        df = pd.read_csv("shortList.csv")
        dna = []; 
        grna = []; 

        for grna_seq, dna_seq in zip(df["Guide_sequence"], df["Target_sequence"]):
            dna.append(dna_seq); 
            grna.append(grna_seq); 

        
        model.load_state_dict(torch.load('transformer_model.pt')); 
        model.to(device); 
        model.eval(); 

        for i in range(len(dna)): 
            test_dna = dna[i]; 
            test_grna = grna[i]; 
            print("Target DNA:     " + test_dna); 
            print("Best gRNA:      " + test_grna)
            generated_grna = grna_generator(model, test_dna, max_length = len(test_dna)); 
            # print(generated_grna); 
            full_sequence = None; 
        
            if not generated_grna:
                full_sequence = ""; 
        
            full_sequence = generated_grna[0]; 
            for kmer in generated_grna[1:]:
                full_sequence += kmer[-1];  
            
            print("Generated gRNA: " + full_sequence + "\n"); 


