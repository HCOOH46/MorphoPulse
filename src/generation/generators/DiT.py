"""
Discrete Diffusion Transformer (DiT) for TimeVQVAE
Adapted to follow the MaskGIT interface and training pattern
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from pathlib import Path
from typing import Union
import copy
from einops import rearrange, repeat

from experiments.exp_stage1 import ExpStage1
from utils import freeze, timefreq_to_time, time_to_timefreq, quantize, zero_pad_low_freq, zero_pad_high_freq
import tempfile
from project_paths import GENERATION_CHECKPOINT_ROOT


# Helper: Sinusoidal Timestep Embedding ???
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DiscreteDiT(nn.Module):
    """
    Discrete Diffusion Transformer adapted for TimeVQVAE
    Follows the MaskGIT interface for compatibility with existing training pipeline
    """
    def __init__(self,
                 dataset_name: str,
                 in_channels: int,
                 input_length: int,
                 config: dict,
                 n_classes: int,
                 **kwargs):
        super().__init__()
        
        self.dataset_name = dataset_name
        self.in_channels = in_channels
        self.input_length = input_length
        self.config = config
        self.n_classes = n_classes
        self.n_fft = config['VQ-VAE']['n_fft']
        
        # Extract DiT configuration
        dit_config = config['DiT']
        self.choice_temperature_l = dit_config['choice_temperatures']['lf']
        self.choice_temperature_h = dit_config['choice_temperatures']['hf']
        self.T = dit_config['T']
        self.cfg_scale = dit_config['cfg_scale']
        self.p_unconditional = dit_config['p_unconditional']
        
        # DiT specific parameters from config
        self.dim = dit_config['dim']
        self.n_heads = dit_config['n_heads']
        self.n_layers = dit_config['n_layers']
        self.dropout = dit_config['dropout']
        self.gamma_mode = dit_config['gamma_mode']
        
        # load the staeg1 model
        self.stage1 = ExpStage1.load_from_checkpoint(os.path.join(GENERATION_CHECKPOINT_ROOT, f'stage1-{dataset_name}.ckpt'), 
                                                     in_channels=in_channels,
                                                     input_length=input_length, 
                                                     config=config,
                                                     map_location='cpu')
        freeze(self.stage1)
        self.stage1.eval()

        self.encoder_l = self.stage1.encoder_l
        self.decoder_l = self.stage1.decoder_l
        self.vq_model_l = self.stage1.vq_model_l
        self.encoder_h = self.stage1.encoder_h
        self.decoder_h = self.stage1.decoder_h
        self.vq_model_h = self.stage1.vq_model_h
        
        # Will be set after stage1 model is loaded
        self.mask_token_ids = {'lf': config['VQ-VAE']['codebook_sizes']['lf'], 
                              'hf': config['VQ-VAE']['codebook_sizes']['hf']}
        
        # Diffusion schedule
        self.num_timesteps = self.T['lf']  # Use same number of steps as MaskGIT
        self.gamma = self.gamma_func(self.gamma_mode)
        
        # Get token sequence lengths
        dummy_input = torch.randn(1, self.in_channels, self.input_length)
        with torch.no_grad():
            z_l = self.encoder_l(dummy_input)
            z_h = self.encoder_h(dummy_input)
            _, s_l, _, _ = quantize(z_l, self.vq_model_l)
            _, s_h, _, _ = quantize(z_h, self.vq_model_h)
            
        self.num_tokens_l = s_l.shape[1]
        self.num_tokens_h = s_h.shape[1]
        
        # Token embeddings (+1 for mask token)
        self.token_emb_l = nn.Embedding(self.mask_token_ids['lf'] + 1, self.dim)
        self.token_emb_h = nn.Embedding(self.mask_token_ids['hf'] + 1, self.dim)
        
        # Positional embeddings
        self.pos_emb_l = nn.Embedding(self.num_tokens_l, self.dim)
        self.pos_emb_h = nn.Embedding(self.num_tokens_h, self.dim)
        
        # Class condition embedding
        self.class_condition_emb = nn.Embedding(self.n_classes + 1, self.dim)  # +1 for unconditional
        
        # Timestep embedding
        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(self.dim),
            nn.Linear(self.dim, self.dim * 4),
            nn.GELU(),
            nn.Linear(self.dim * 4, self.dim)
        )
        
        # Transformer encoders for LF and HF
        encoder_layer_l = nn.TransformerEncoderLayer(
            d_model=self.dim, 
            nhead=self.n_heads, 
            dim_feedforward=self.dim * 4, 
            dropout=self.dropout, 
            batch_first=True
        )
        self.transformer_l = nn.TransformerEncoder(encoder_layer_l, num_layers=self.n_layers)
        
        encoder_layer_h = nn.TransformerEncoderLayer(
            d_model=self.dim, 
            nhead=self.n_heads, 
            dim_feedforward=self.dim * 4, 
            dropout=self.dropout, 
            batch_first=True
        )
        self.transformer_h = nn.TransformerEncoder(encoder_layer_h, num_layers=self.n_layers)
        
        # Output projection layers
        self.to_logits_l = nn.Linear(self.dim, self.mask_token_ids['lf'])
        self.to_logits_h = nn.Linear(self.dim, self.mask_token_ids['hf'])

    def load(self, model, dirname, fname):
        """
        model: instance
        path_to_saved_model_fname: path to the ckpt file (i.e., trained model)
        """
        try:
            model.load_state_dict(torch.load(dirname.joinpath(fname)))
        except FileNotFoundError:
            dirname = Path(tempfile.gettempdir())
            model.load_state_dict(torch.load(dirname.joinpath(fname)))


    # def load(self, model, dirname, fname):
    #     """Load stage1 model - following MaskGIT interface"""
    #     print('Loading Stage 1 model ...')
        
    #     # Load the checkpoint
    #     if isinstance(model, str):
    #         checkpoint_path = Path(dirname) / fname
    #         checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
    #         # Create ExpStage1 instance
    #         stage1_model = ExpStage1(
    #             in_channels=self.in_channels,
    #             input_length=self.input_length,
    #             config=self.config
    #         )
    #         stage1_model.load_state_dict(checkpoint['state_dict'])
    #     else:
    #         stage1_model = model
            
    #     # Extract components
    #     self.encoder_l = stage1_model.encoder_l
    #     self.encoder_h = stage1_model.encoder_h
    #     self.decoder_l = stage1_model.decoder_l
    #     self.decoder_h = stage1_model.decoder_h
    #     self.vq_model_l = stage1_model.vq_model_l
    #     self.vq_model_h = stage1_model.vq_model_h
        
    #     # Freeze stage1 components
    #     freeze(self.encoder_l)
    #     freeze(self.encoder_h)
    #     freeze(self.decoder_l)
    #     freeze(self.decoder_h)
    #     freeze(self.vq_model_l)
    #     freeze(self.vq_model_h)
        
    #     # Now initialize DiT components
    #     # self._init_dit_components()
        
    #     print('Stage 1 model loaded successfully!')

    # def _init_dit_components(self):
    #     """Initialize DiT components after stage1 model is loaded"""
    #     # Get token sequence lengths
    #     dummy_input = torch.randn(1, self.in_channels, self.input_length)
    #     with torch.no_grad():
    #         z_l = self.encoder_l(dummy_input)
    #         z_h = self.encoder_h(dummy_input)
    #         _, s_l, _, _ = quantize(z_l, self.vq_model_l)
    #         _, s_h, _, _ = quantize(z_h, self.vq_model_h)
            
    #     self.num_tokens_l = s_l.shape[1]
    #     self.num_tokens_h = s_h.shape[1]
        
    #     # Token embeddings (+1 for mask token)
    #     self.token_emb_l = nn.Embedding(self.mask_token_ids['lf'] + 1, self.dim)
    #     self.token_emb_h = nn.Embedding(self.mask_token_ids['hf'] + 1, self.dim)
        
    #     # Positional embeddings
    #     self.pos_emb_l = nn.Embedding(self.num_tokens_l, self.dim)
    #     self.pos_emb_h = nn.Embedding(self.num_tokens_h, self.dim)
        
    #     # Class condition embedding
    #     self.class_condition_emb = nn.Embedding(self.n_classes + 1, self.dim)  # +1 for unconditional
        
    #     # Timestep embedding
    #     self.time_emb = nn.Sequential(
    #         SinusoidalPositionEmbeddings(self.dim),
    #         nn.Linear(self.dim, self.dim * 4),
    #         nn.GELU(),
    #         nn.Linear(self.dim * 4, self.dim)
    #     )
        
    #     # Transformer encoders for LF and HF
    #     encoder_layer_l = nn.TransformerEncoderLayer(
    #         d_model=self.dim, 
    #         nhead=self.n_heads, 
    #         dim_feedforward=self.dim * 4, 
    #         dropout=self.dropout, 
    #         batch_first=True
    #     )
    #     self.transformer_l = nn.TransformerEncoder(encoder_layer_l, num_layers=self.n_layers)
        
    #     encoder_layer_h = nn.TransformerEncoderLayer(
    #         d_model=self.dim, 
    #         nhead=self.n_heads, 
    #         dim_feedforward=self.dim * 4, 
    #         dropout=self.dropout, 
    #         batch_first=True
    #     )
    #     self.transformer_h = nn.TransformerEncoder(encoder_layer_h, num_layers=self.n_layers)
        
    #     # Output projection layers
    #     self.to_logits_l = nn.Linear(self.dim, self.mask_token_ids['lf'])
    #     self.to_logits_h = nn.Linear(self.dim, self.mask_token_ids['hf'])

    def gamma_func(self, mode="cosine"):
        """Masking schedule function - following MaskGIT"""
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: torch.cos(r * math.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        elif mode == "cubic":
            return lambda r: 1 - r ** 3
        else:
            raise NotImplementedError(f"Gamma mode {mode} not implemented")

    def class_embedding(self, class_condition: Union[None, torch.Tensor], batch_size: int, device):
        """Class condition embedding - following MaskGIT interface"""
        cond_type = 'uncond' if isinstance(class_condition, type(None)) else 'class-cond'

        if cond_type == 'uncond':
            class_uncondition = repeat(torch.Tensor([self.n_classes]).long().to(device), 'i -> b i', b=batch_size)
            cls_emb = self.class_condition_emb(class_uncondition)  # (b dim), nn.Embedding的输出
            return cls_emb.unsqueeze(1)
        elif cond_type == 'class-cond':
            if self.training:
                ind = torch.rand(class_condition.shape).to(device) > self.p_unconditional
            else:
                ind = torch.ones_like(class_condition, dtype=torch.bool).to(device)
            class_condition = torch.where(ind, class_condition.long(), self.n_classes)
            cls_emb = self.class_condition_emb(class_condition)  # (b dim)
            return cls_emb.unsqueeze(1)

    def forward(self, x, y):
        """Forward pass following MaskGIT interface"""
        # 1. Get discrete tokens from the frozen stage1 model
        self.encoder_l.eval()
        self.vq_model_l.eval()
        self.encoder_h.eval()
        self.vq_model_h.eval()
        with torch.no_grad():
            # Extract time-frequency representations
            in_channels = x.shape[1]
            xf = time_to_timefreq(x, self.n_fft, in_channels)
            u_l = zero_pad_high_freq(xf)
            x_l = F.interpolate(timefreq_to_time(u_l, self.n_fft, in_channels), 
                              self.input_length, mode='linear')
            u_h = zero_pad_low_freq(xf)
            x_h = F.interpolate(timefreq_to_time(u_h, self.n_fft, in_channels), 
                              self.input_length, mode='linear')
            
            # Get quantized tokens
            z_l = self.encoder_l(x)
            z_h = self.encoder_h(x)
            _, s_l, _, _ = quantize(z_l, self.vq_model_l)
            _, s_h, _, _ = quantize(z_h, self.vq_model_h)
            
        b, device = s_l.shape[0], s_l.device

        # 2. Forward Diffusion: Sample timesteps and create masks
        t = torch.rand(b, device=device)  # Random timesteps [0, 1]
        
        # Get masking probabilities from schedule
        mask_prob_l = 1 - self.gamma(t)
        mask_prob_h = 1 - self.gamma(t)
        
        # Create random masks
        rand_matrix_l = torch.rand_like(s_l.float())
        rand_matrix_h = torch.rand_like(s_h.float())
        
        mask_l = rand_matrix_l < mask_prob_l.unsqueeze(1)
        mask_h = rand_matrix_h < mask_prob_h.unsqueeze(1)
        
        # Create noised tokens
        noised_s_l = torch.where(mask_l, self.mask_token_ids['lf'], s_l)
        noised_s_h = torch.where(mask_h, self.mask_token_ids['hf'], s_h)

        # 3. Process through DiT
        # Embeddings
        emb_l = self.token_emb_l(noised_s_l) + self.pos_emb_l(torch.arange(s_l.shape[1], device=device))
        emb_h = self.token_emb_h(noised_s_h) + self.pos_emb_h(torch.arange(s_h.shape[1], device=device))
        
        # Timestep and class embeddings
        t_emb = self.time_emb(t).unsqueeze(1)  # (b 1 dim)
        cls_emb = self.class_embedding(y, b, device)  # (b 1 dim)

        # Add conditioning
        x_l = emb_l + t_emb + cls_emb
        x_h = emb_h + t_emb + cls_emb
        
        # 4. Transformer forward pass
        out_l = self.transformer_l(x_l)
        out_h = self.transformer_h(x_h)
        
        # 5. Project to logits
        logits_l = self.to_logits_l(out_l)
        logits_h = self.to_logits_h(out_h)

        # 6. Calculate loss (only on masked tokens)
        loss_l = F.cross_entropy(
            rearrange(logits_l, 'b t c -> b c t'), 
            s_l, 
            reduction='none'
        )
        loss_l = (loss_l * mask_l).sum() / (mask_l.sum() + 1e-8)

        loss_h = F.cross_entropy(
            rearrange(logits_h, 'b t c -> b c t'),
            s_h,
            reduction='none'
        )
        loss_h = (loss_h * mask_h).sum() / (mask_h.sum() + 1e-8)

        total_loss = loss_l + loss_h
        
        return total_loss, (loss_l, loss_h)

    def mask_by_random_topk(self, mask_len, probs, temperature=1.0, device='cpu'):
        """Mask tokens by random top-k - following MaskGIT"""
        def log(t, eps=1e-20):
            return torch.log(t.clamp(min=eps))

        def gumbel_noise(t):
            noise = torch.zeros_like(t).uniform_(0, 1)
            return -log(-log(noise))

        confidence = torch.log(probs) + temperature * gumbel_noise(probs).to(device)
        sorted_confidence, _ = torch.sort(confidence, dim=-1)
        cut_off = torch.take_along_dim(sorted_confidence, mask_len.to(torch.long), dim=-1)
        masking = (confidence < cut_off)
        return masking

    @torch.no_grad()
    def iterative_decoding(self, s_init, frequency, class_condition, unknown_number_in_the_beginning, device):
        """Iterative decoding following MaskGIT pattern"""
        if frequency == 'lf':
            mask_token_id = self.mask_token_ids['lf']
            transformer = self.transformer_l
            token_emb = self.token_emb_l
            pos_emb = self.pos_emb_l
            to_logits = self.to_logits_l
            choice_temperature = self.choice_temperature_l
        else:
            mask_token_id = self.mask_token_ids['hf']
            transformer = self.transformer_h
            token_emb = self.token_emb_h
            pos_emb = self.pos_emb_h
            to_logits = self.to_logits_h
            choice_temperature = self.choice_temperature_h

        s = s_init
        T_steps = self.T[frequency]
        
        for t in range(T_steps):
            # Get embeddings
            emb = token_emb(s) + pos_emb(torch.arange(s.shape[1], device=device))
            
            # Add timestep and class conditioning
            t_ratio = (t + 1) / T_steps
            t_emb = self.time_emb(torch.full((s.shape[0],), t_ratio, device=device)).unsqueeze(1)
            cls_emb = self.class_embedding(class_condition, s.shape[0], device)
            
            x = emb + t_emb + cls_emb
            
            # Forward through transformer
            out = transformer(x)
            logits = to_logits(out)
            
            # Sample new tokens
            sampled_ids = torch.distributions.categorical.Categorical(logits=logits).sample()
            
            # Create mask for this timestep
            ratio = 1. * (t + 1) / T_steps
            mask_ratio = self.gamma(ratio)
            mask_len = torch.unsqueeze(torch.floor(unknown_number_in_the_beginning * mask_ratio), 1)
            mask_len = torch.clip(mask_len, min=0.)
            
            # Get confidence scores
            probs = torch.softmax(logits, dim=-1)
            confidence_scores = torch.gather(probs, 2, sampled_ids.unsqueeze(-1)).squeeze(-1)
            
            # Create masking
            masking = self.mask_by_random_topk(
                mask_len, 
                confidence_scores, 
                temperature=choice_temperature * (1. - ratio), 
                device=device
            )
            
            # Update tokens
            s = torch.where(masking, mask_token_id, sampled_ids)
            
        return s

    @torch.no_grad()
    def sample(self, num_samples, class_condition=None, device='cpu'):
        """Sample following MaskGIT interface"""
        self.eval()
        
        # Initialize with all mask tokens
        s_l = torch.full((num_samples, self.num_tokens_l), self.mask_token_ids['lf'], 
                        device=device, dtype=torch.long)
        s_h = torch.full((num_samples, self.num_tokens_h), self.mask_token_ids['hf'], 
                        device=device, dtype=torch.long)
        
        unknown_number_in_the_beginning_l = torch.FloatTensor([self.num_tokens_l]).to(device)
        unknown_number_in_the_beginning_h = torch.FloatTensor([self.num_tokens_h]).to(device)
        
        # First pass: decode LF tokens
        s_l = self.iterative_decoding(
            s_l, 'lf', class_condition, 
            unknown_number_in_the_beginning_l, device
        )
        
        # Second pass: decode HF tokens conditioned on LF
        s_h = self.iterative_decoding(
            s_h, 'hf', class_condition,
            unknown_number_in_the_beginning_h, device
        )
        
        return self.decode_token_ind_to_timeseries(s_l, s_h)

    def decode_token_ind_to_timeseries(self, s_l, s_h):
        """Decode tokens back to time series"""
        with torch.no_grad():
            # Get quantized representations
            z_q_l = F.embedding(s_l, self.vq_model_l._codebook.embed)
            z_q_h = F.embedding(s_h, self.vq_model_h._codebook.embed)
            
            # Decode to time series
            xhat_l = self.decoder_l(z_q_l)
            xhat_h = self.decoder_h(z_q_h)
            xhat = xhat_l + xhat_h
            
        return xhat_l, xhat_h, xhat

    def first_pass(self, s_l, unknown_number_in_the_beginning_l, class_condition, gamma, device):
        """First pass generation for LF tokens"""
        return self.iterative_decoding(s_l, 'lf', class_condition, unknown_number_in_the_beginning_l, device)

    def second_pass(self, s_l, s_h, unknown_number_in_the_beginning_h, class_condition, gamma, device):
        """Second pass generation for HF tokens"""
        return self.iterative_decoding(s_h, 'hf', class_condition, unknown_number_in_the_beginning_h, device)
