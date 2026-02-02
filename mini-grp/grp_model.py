import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def get_patches_fast(images, cfg):
    from einops import rearrange
    batch_size, height, width, channels = images.shape
    patch_size = cfg.patch_size ## n_patches = 8

    patches = rearrange(images[:,:,:,:3], 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size)
    if channels > 3:
        ## History stacking in the channel dimension for observations only, not goal images.
        patches = rearrange(images, 'b (h p1) (w p2) (c hs) -> b (h w hs) (p1 p2 c)', p1 = patch_size, p2 = patch_size, hs=cfg.policy.obs_stacking) ## Stack the history in the channel dimension
    return patches


def calc_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B,T,C = x.shape
        # TODO: 
        ## Provide the block masking logic for the attention head
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        #Apply mask
        if mask is not None:
            # Ensure mask is broadcastable to (B, T, T)
            if mask.dim() == 2:  # (T, T) -> (1, T, T)
                mask = mask.unsqueeze(0)
            # wei is (B, T, T), mask should be (B, T, T) or (1, T, T)
            wei = wei.masked_fill(mask == 0, float('-inf'))

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd=n_embd, dropout=dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        with torch.profiler.record_function("Self-Attention"):
            out = torch.cat([h(x, mask) for h in self.heads], dim=-1)
            out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd=n_embd, dropout=dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, mask=None):
        x = x + self.sa(self.ln1(x), mask)
        x = x + self.ffwd(self.ln2(x))
        return x


class GRP(nn.Module):
    def __init__(self, cfg, mlp_ratio=4):
        super(GRP, self).__init__()
        self._cfg = cfg
        chars = cfg.dataset.chars_list
        cfg.vocab_size = len(chars)

        # ===== CHOOSE MODE HERE =====
        # Read action representation from config (defaults to 'continuous' if missing)
        self.action_representation = getattr(cfg, 'action_representation', 'continuous')
        self.num_bins = 14 if self.action_representation == 'discrete' else None

        # Print to check at runtime if the correct mode is selected
        mode = "DISCRETE (14 bins)" if self.action_representation == 'discrete' else "CONTINUOUS"
        print(f"[GRP] Action mode: {mode}")
        # ============================

        # TODO: 
        ## Provide the logic for the GRP network

        #1) Vision Embedding
        patch_dim = (cfg.patch_size**2) * 3  # = 8*8*3 = 192
        self.patch_embedding = nn.Linear(patch_dim, cfg.n_embd)

        #Goal image projection
        self.goal_patch_embedding = nn.Linear((cfg.patch_size**2) * 3, cfg.n_embd)

        #Text/Goal embedding
        if not cfg.dataset.encode_with_t5:
            self.token_embedding_table = nn.Embedding(len(cfg.dataset.chars_list), cfg.n_embd)
        
        # 3) Learned Tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.n_embd))

        # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList([
            Block(cfg.n_embd, n_head=cfg.n_head, dropout=cfg.dropout) 
            for _ in range(cfg.n_blocks)
        ])
        self.ln_f = nn.LayerNorm(cfg.n_embd)

        # 5) Classification MLPk
        # Action head - different output sizes for discrete vs continuous
        if self.action_representation == 'discrete':
            print("DISCRETE MODE (14 bins) - Cross-Entropy Loss")
            # Output logits for each bin per action dimension per timestep
            self.action_head = nn.Sequential(
                nn.Linear(cfg.n_embd, cfg.n_embd * mlp_ratio),
                nn.ReLU(),
                nn.Linear(cfg.n_embd * mlp_ratio, 
                        cfg.action_dim * cfg.policy.action_stacking * self.num_bins)
            )
        else:  # continuous
            print("CONTINUOUS MODE - MSE Loss")
            self.action_head = nn.Sequential(
                nn.Linear(cfg.n_embd, cfg.n_embd * mlp_ratio),
                nn.ReLU(),
                nn.Linear(cfg.n_embd * mlp_ratio, 
                        cfg.action_dim * cfg.policy.action_stacking)
            )
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, images, goals_txt, goal_imgs, targets=None, pose=None, mask_=False):
        n, c, h, w = images.shape
        obs_patches = get_patches_fast(images, self._cfg)
        patches_g = get_patches_fast(goal_imgs, self._cfg)
        if self._cfg.dataset.encode_with_t5:
            goals_e = goals_txt
            B, T, E = goals_txt.shape
        else:
            goals_e = self.token_embedding_table(goals_txt)
            B, E = goals_txt.shape
            T = self._cfg.max_block_size

        # TODO: 
        ## Provide the logic to produce the output and loss for the GRP
        
        # Map the vector corresponding to each patch to the hidden size dimension
        obs_embeddings = self.patch_embedding(obs_patches)
        goal_img_embeddings = self.goal_patch_embedding(patches_g)

        # Adding classification and goal_img tokens to the tokens
        batch_size = images.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        x = torch.cat((cls_tokens, goals_e, goal_img_embeddings, obs_embeddings), dim=1)
        
        # ===== BLOCK MASKING =====
        #50/50 random dropout
        attn_mask = None
        
        # During TRAINING: create block mask to randomly mask language OR goal image
        if self.training and targets is not None:
            lang_len = goals_e.shape[1]
            goal_img_len = goal_img_embeddings.shape[1]
            total_seq_len = x.shape[1]
            
            # Create 3D attention mask (B, T, T) where 0 = block, 1 = allow
            attn_mask = torch.ones((batch_size, total_seq_len, total_seq_len), device=x.device)
            
            # Apply same mask to entire batch (50/50 for language vs goal image)
            if torch.rand(1).item() < 0.5:  # Mask language tokens
                attn_mask[:, :, 1:1+lang_len] = 0  # Block attention TO language tokens
            else:  # Mask goal image tokens
                start = 1 + lang_len
                end = start + goal_img_len
                attn_mask[:, :, start:end] = 0  # Block attention TO goal image tokens
        
        # During EVALUATION: use mask_ parameter if provided
        elif mask_:
            lang_len = goals_e.shape[1]
            goal_img_len = goal_img_embeddings.shape[1]
            total_seq_len = x.shape[1]
            
            attn_mask = torch.ones((batch_size, total_seq_len, total_seq_len), device=x.device)
            
            # For evaluation, randomly mask one modality (50/50) to test robustness
            if torch.rand(1).item() < 0.5 and lang_len > 0:
                attn_mask[:, :, 1:1+lang_len] = 0  # Mask language
            else:
                start = 1 + lang_len
                end = start + goal_img_len
                attn_mask[:, :, start:end] = 0  # Mask goal image
        # ====================================

        # Adding positional embedding
        pos_emb = calc_positional_embeddings(x.shape[1], self._cfg.n_embd).to(x.device)
        x = x + pos_emb

        # Transformer Blocks with mask
        for block in self.blocks:
            x = block(x, mask=attn_mask)
        x = self.ln_f(x)

        # Getting the classification token only
        cls_output = x[:, 0, :]

        # Compute output and loss
        out = self.action_head(cls_output)

        loss = None
        if targets is not None:
            if self.action_representation == 'discrete':
                # Reshape for cross-entropy: [B, T*action_dim, num_bins]
                B = out.shape[0]
                logits = out.view(B, self._cfg.policy.action_stacking, 
                                self._cfg.action_dim, self.num_bins)
                logits = logits.view(B * self._cfg.policy.action_stacking * self._cfg.action_dim, self.num_bins)
                
                # Targets should be bin indices [B * T * action_dim]
                targets_flat = targets.view(-1).long()
                loss = F.cross_entropy(logits, targets_flat)
            else:  # continuous
                loss = F.mse_loss(out, targets)
        
        return (out, loss)

    
    def resize_image(self, image):
        """
        Docstring for resize_image
        
        :param self: Description
        :param image: Description
        self._resize_state = lambda sf:   cv2.resize(np.array(sf, dtype=np.float32), (cfg.image_shape[0], cfg.image_shape[1]))  # resize state
        """
        import cv2
        import numpy as _np
        img = _np.array(image, dtype=_np.float32)
        img = cv2.resize(img, (self._cfg.image_shape[0], self._cfg.image_shape[1]))
        return img

    def normalize_state(self, image):
        """
        Docstring for preprocess_state
        
        :param self: Description
        :param image: Description
        self._encode_state = lambda af:   ((af/(255.0)*2.0)-1.0) # encoder: take a float, output an integer
        self._resize_state = lambda sf:   cv2.resize(np.array(sf, dtype=np.float32), (cfg.image_shape[0], cfg.image_shape[1]))  # resize state
        """
        # img = _np.array(image, dtype=_np.float32)
        # img = cv2.resize(img, (self._cfg.image_shape[0], self._cfg.image_shape[1]))
        enc = ((image / 255.0) * 2.0) - 1.0
        # t = _torch.tensor(enc, dtype=_torch.float32, device=self._cfg.device)
        return enc
    
    def preprocess_state(self, image):
        img = self.resize_image(image)
        img = self.normalize_state(img)
        return img

    def preprocess_goal_image(self, image):
        return self.preprocess_state(image)

    def encode_text_goal(self, goal, tokenizer=None, text_model=None):
        import numpy as _np
        import torch as _torch
        if self._cfg.dataset.encode_with_t5:
            if tokenizer is None or text_model is None:
                raise ValueError("tokenizer and text_model must be provided when using T5 encoding")
            # TODO:    
            ## Provide the logic converting text goal to T5 embedding tensor
            # Handle both single string and batch inputs
            # Handle precomputed embeddings (from buffer) vs raw text
            if isinstance(goal, torch.Tensor):  # Already embedded (from buffer)
                return goal.unsqueeze(0) if goal.dim() == 2 else goal  # Ensure [B, T, E]
            
            # Raw text input - encode on-the-fly (for eval/inference)
            inputs = tokenizer(
                goal,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self._cfg.max_block_size
            ).to(text_model.device)
            
            with torch.no_grad():
                outputs = text_model(**inputs)
                embeddings = outputs.last_hidden_state
            
            # Ensure exact shape
            if embeddings.shape[1] < self._cfg.max_block_size:
                pad = torch.zeros(
                    embeddings.shape[0],
                    self._cfg.max_block_size - embeddings.shape[1],
                    embeddings.shape[2],
                    device=embeddings.device
                )
                embeddings = torch.cat([embeddings, pad], dim=1)
            elif embeddings.shape[1] > self._cfg.max_block_size:
                embeddings = embeddings[:, :self._cfg.max_block_size]
            
            return embeddings  # Shape: [1, max_block_size, hidden_size]
                 
        else:
            pad = " " * self._cfg.max_block_size
            goal_ = goal[:self._cfg.max_block_size] + pad[len(goal):self._cfg.max_block_size]
            try:
                stoi = {c: i for i, c in enumerate(self._cfg.dataset.chars_list)}
                ids = [stoi.get(c, 0) for c in goal_]
            except Exception:
                ids = [0] * self._cfg.max_block_size
            return _torch.tensor(_np.expand_dims(_np.array(ids, dtype=_np.int64), axis=0), dtype=_torch.long, device=self._cfg.device)

    def process_text_embedding_for_buffer(self, goal, tokenizer=None, text_model=None):
        """
        Process text goal embedding for storing in the circular buffer.
        Returns a numpy array of shape (max_block_size, n_embd) without batch dimension.
        """
        import numpy as _np
        if tokenizer is None or text_model is None:
            raise ValueError("tokenizer and text_model must be provided when using T5 encoding")
        
        # Tokenize with proper padding/truncation
        inputs = tokenizer(
            goal,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self._cfg.max_block_size
        ).to(text_model.device)
        
        # Get embeddings (disable gradients)
        with torch.no_grad():
            encoder_outputs = text_model.encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            embeddings = encoder_outputs.last_hidden_state[:, 0, :]  # [CLS] token or first token
        
        # Convert to numpy and ensure exact shape
        embeddings_np = embeddings.cpu().numpy()
        
        # Pad/truncate to exact max_block_size
        if embeddings_np.shape[0] < self._cfg.max_block_size:
            pad = np.zeros(
                (self._cfg.max_block_size - embeddings_np.shape[0], embeddings_np.shape[1]),
                dtype=np.float32
            )
            embeddings_np = np.concatenate([embeddings_np, pad], axis=0)
        elif embeddings_np.shape[0] > self._cfg.max_block_size:
            embeddings_np = embeddings_np[:self._cfg.max_block_size]
        
        return embeddings_np  # Shape: [max_block_size, hidden_size]

    def decode_action(self, action_tensor):
        
        """
        Docstring for decode_action
        
        :param self: Description
        :param action_tensor: Description
        self._decode_action = lambda binN: (binN * action_std) + action_mean  # Undo mapping to [-1, 1]
        """
        import torch as _torch
        ## The action tensor is of shape (batch_size, action_dim * action_stacking) so we need to repeat the mean and std per action stacking
        action_mean = _torch.tensor(np.repeat(self._cfg.env.action_mean, self._cfg.policy.action_stacking), dtype=action_tensor.dtype, device=action_tensor.device)
        action_std = _torch.tensor(np.repeat(self._cfg.env.action_std, self._cfg.policy.action_stacking), dtype=action_tensor.dtype, device=action_tensor.device)
        return (action_tensor * action_std) + action_mean
    
    def encode_action(self, action_float):
        """
        Docstring for encode_action
        
        :param self: Description
        :param action_float: Description
        self._encode_action = lambda af:   (af - action_mean)/(action_std) # encoder: take a float, output an integer
        """
        import torch as _torch
        action_mean = _torch.tensor(self._cfg.env.action_mean, dtype=action_float.dtype, device=action_float.device)
        action_std = _torch.tensor(self._cfg.env.action_std, dtype=action_float.dtype, device=action_float.device)
        return (action_float - action_mean) / action_std
    
    def _continuous_to_bins(self, actions):
        """Convert continuous actions [-1, 1] to bin indices [0, num_bins-1]"""
        # Clip to [-1, 1] range first
        actions = torch.clamp(actions, -1.0, 1.0)
        # Map to [0, num_bins-1]
        bin_indices = ((actions + 1.0) / 2.0 * self.num_bins).long()
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)
        return bin_indices

    def _bins_to_continuous(self, bin_indices):
        """Convert bin indices to continuous values (bin centers)"""
        # Map bin index to center of bin in [-1, 1] range
        bin_centers = (bin_indices.float() + 0.5) / self.num_bins * 2.0 - 1.0
        return bin_centers

    def encode_action(self, action_float):
        """Encode continuous action to either normalized value or bin index"""
        import torch as _torch
        action_mean = _torch.tensor(self._cfg.dataset.action_mean, 
                                dtype=action_float.dtype, 
                                device=action_float.device)
        action_std = _torch.tensor(self._cfg.dataset.action_std, 
                                dtype=action_float.dtype, 
                                device=action_float.device)
        
        # First normalize to [-1, 1]
        normalized = (action_float - action_mean) / action_std
        
        if self.action_representation == 'discrete':
            return self._continuous_to_bins(normalized)
        else:
            return normalized

    def decode_action(self, action_tensor):
        """Decode action from model output to environment-ready values"""
        import torch as _torch
        
        if self.action_representation == 'discrete':
            # Reshape to [action_stacking, action_dim, num_bins]
            B = action_tensor.shape[0]
            logits = action_tensor.view(B, self._cfg.policy.action_stacking,
                                    self._cfg.action_dim, self.num_bins)
            # Get most probable bin per dimension
            bin_indices = torch.argmax(logits, dim=-1)  # [B, T, action_dim]
            # Convert to continuous values
            continuous = self._bins_to_continuous(bin_indices.float())
            # Flatten to [B, T*action_dim]
            continuous = continuous.view(B, -1)
            action_tensor = continuous
        
        # Denormalize to original action space
        action_mean = _torch.tensor(
            np.repeat(self._cfg.dataset.action_mean, self._cfg.policy.action_stacking),
            dtype=action_tensor.dtype, 
            device=action_tensor.device
        )
        action_std = _torch.tensor(
            np.repeat(self._cfg.dataset.action_std, self._cfg.policy.action_stacking),
            dtype=action_tensor.dtype, 
            device=action_tensor.device
        )
        return (action_tensor * action_std) + action_mean

@torch.no_grad()
def estimate_loss(model, dataset):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(model._cfg.eval_iters)
        for k in range(model._cfg.eval_iters):
            X, x_pose, x_goal, x_goal_img, Y = dataset.get_batch_grp(split, model._cfg, model._cfg.batch_size)
            logits, loss = model(X, x_goal, x_goal_img, Y, pose=x_pose)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
