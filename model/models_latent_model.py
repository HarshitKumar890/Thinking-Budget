import torch
import torch.nn as nn
from data_generator import VOCAB, VOCAB_SIZE

class RecurrentLatentReasoner(nn.Module):
    """
    Recurrent Latent Reasoning Architecture.
    Iterates continuous hidden state via cross-attention to structured key-value input memory.
    Executes fixed k recurrent steps without emitting intermediate tokens.
    """
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=80, nhead=4, dim_feedforward=304, max_len=96):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(max_len, d_model))
        nn.init.normal_(self.pos_embedding, std=0.02)
        
        # Structured memory query & recurrent update
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.gru = nn.GRUCell(d_model, d_model)
        self.ln_h = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.ln_ffn = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def encode_memory(self, input_ids):
        """Encodes input into structured addressable memory."""
        B, L = input_ids.shape
        pos = self.pos_embedding[:L].unsqueeze(0)
        memory = self.embedding(input_ids) + pos
        return memory

    def step(self, h, memory, key_padding_mask=None):
        """
        Executes a single recurrent latent transition:
        Queries structured memory with current hidden state h, updates via GRU.
        """
        # Cross-attention: query is h [B, 1, d], key/value is memory [B, L, d]
        q = h.unsqueeze(1)
        attn_out, _ = self.cross_attn(q, memory, memory, key_padding_mask=key_padding_mask)
        attn_out = attn_out.squeeze(1)
        
        # GRU update
        h_next = self.gru(attn_out, h)
        h_next = self.ln_h(h_next)
        
        # Feedforward update
        ff_out = self.ffn(h_next)
        h_next = self.ln_ffn(h_next + ff_out)
        return h_next

    def forward(self, input_ids, reasoning_steps=8, input_mask=None):
        """
        Runs exactly reasoning_steps (k) recurrent transitions.
        Returns intermediate step logits (for aligned supervision) and final logits.
        """
        B, L = input_ids.shape
        device = input_ids.device
        
        # Key padding mask: True indicates ignored position
        pad_mask = (~input_mask) if input_mask is not None else None
        
        memory = self.encode_memory(input_ids)
        
        # Initialize hidden state from terminal input token
        h = memory[:, -1, :].clone()
        
        step_logits = []
        for s in range(reasoning_steps):
            h = self.step(h, memory, key_padding_mask=pad_mask)
            step_logits.append(self.head(h))
            
        final_logits = step_logits[-1]
        
        return {
            'final_logits': final_logits,
            'step_logits': torch.stack(step_logits, dim=1),  # [batch, reasoning_steps, vocab_size]
            'final_state': h
        }

    @torch.no_grad()
    def infer(self, input_ids, budget_k=8, input_mask=None):
        """
        Test-time inference: runs exactly budget_k recurrent transitions.
        """
        res = self.forward(input_ids, reasoning_steps=budget_k, input_mask=input_mask)
        final_ans = res['final_logits'].argmax(dim=-1)
        return {
            'final_ans': final_ans,
            'steps_executed': budget_k
        }