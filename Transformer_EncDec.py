import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class TSKFuzzyGraphGenerator(nn.Module):
    def __init__(self, d_model, n_rules=3):
        super().__init__()
        self.d_model = d_model
        self.n_rules = n_rules

        # 1. 前件: 规则中心
        self.centers = nn.Parameter(torch.randn(n_rules, d_model))

        # self.initialized = False
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

        self.sigma = nn.Parameter(torch.ones(n_rules) * 1.0)

        # 2. 后件: 规则对应的基准图
        # [K, D, D]
        self.rule_adjs = nn.Parameter(torch.zeros(n_rules, d_model, d_model))
        nn.init.xavier_uniform_(self.rule_adjs, gain=0.01)

    def _initialize_centers(self, x):
        with torch.no_grad():
            if x.size(0) >= self.n_rules:
                indices = torch.randperm(x.size(0))[:self.n_rules]
                selected = x[indices]
            else:
                indices = torch.randint(0, x.size(0), (self.n_rules,))
                selected = x[indices]
            self.centers.copy_(selected)
            self.initialized.fill_(True)
            # self.centers.data.copy_(selected)
            # self.initialized = True

    def forward(self, x_ctx):
        """
        return:
            firing_strength: [B, K]
            rule_adjs: [K, D, D] (直接返回原始参数，不做混合)
        """
        if self.training and (not self.initialized.item()):
            self._initialize_centers(x_ctx)

        # 计算激活度
        dist = torch.norm(x_ctx.unsqueeze(1) - self.centers.unsqueeze(0), dim=-1)
        sigma = torch.clamp(self.sigma, min=0.1)
        membership = torch.exp(- (dist ** 2) / (2 * sigma ** 2))
        firing_strength = membership / (torch.sum(membership, dim=1, keepdim=True) + 1e-8)

        # dist2 = ((x_ctx.unsqueeze(1) - self.centers.unsqueeze(0)) ** 2).mean(dim=-1)
        # sigma2 = self.sigma.clamp_min(0.1).pow(2).unsqueeze(0)
        # logits = - dist2 / (2 * sigma2)
        # firing_strength = torch.softmax(logits, dim=1)

        return firing_strength, self.rule_adjs


class DynamicGraphConv(nn.Module):
    def __init__(self, d_model: int, n_rules=3):
        super().__init__()
        self.d_model = d_model
        self.tsk_gen = TSKFuzzyGraphGenerator(d_model, n_rules=n_rules)
        self.phi = nn.Linear(d_model, d_model)
        self.theta = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.alpha = nn.Parameter(torch.full((d_model,), -2.0))
        self.norm_ctx = nn.LayerNorm(d_model)

        self.firing_history = []

        self.record_firing = False
        self.vis_firing_history = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        x_ctx = x.mean(dim=1)
        x_ctx_norm = self.norm_ctx(x_ctx)

        # 1. TSK Firing
        firing, rule_adjs = self.tsk_gen(x_ctx_norm)

        # ★ 新增：缓存 firing strength
        # 因为 Encoder 可能会循环调用多次，我们需要把每次的都存下来
        if self.training:
            self.firing_history.append(firing)

        if self.record_firing:
            self.vis_firing_history.append(firing.detach().cpu())

        # 2. Dynamic Component
        phi_x = self.phi(x_ctx)
        theta_x = self.theta(x_ctx)
        dynamic_adj = torch.bmm(phi_x.unsqueeze(2), theta_x.unsqueeze(1))

        # 3. Function-Space Mixture
        total_adj = dynamic_adj.unsqueeze(1) + rule_adjs.unsqueeze(0)
        adj_k = F.softmax(total_adj, dim=-1)
        eye = torch.eye(D, device=x.device).view(1, 1, D, D)
        adj_k = 0.5 * adj_k + 0.5 * eye

        x_graph_k = torch.matmul(x.unsqueeze(1), adj_k)
        x_graph = (x_graph_k * firing.view(B, -1, 1, 1)).sum(dim=1)

        x_graph = self.proj(x_graph)
        gate = torch.sigmoid(self.alpha).view(1, 1, -1)
        return x_graph * gate

    def get_tsk_loss(self):
        """计算 TSK 辅助损失并清空缓存"""
        if not self.firing_history:
            return torch.tensor(0.0).to(self.alpha.device)

        # 拼接所有 step 和所有 batch 的 firing strength
        # shape: [Total_Steps * B, K]
        P = torch.cat(self.firing_history, dim=0)

        # 清空缓存，防止累积到下一个 epoch
        self.firing_history = []

        # ---------------------------------------------------
        # 1. 规则置信度损失 (Rule Confidence Loss)
        # 最小化样本熵：希望每个样本只激活一个规则
        # H(p) = -sum(p * log(p))
        # ---------------------------------------------------
        epsilon = 1e-8
        entropy_sample = -torch.sum(P * torch.log(P + epsilon), dim=1).mean()

        # ---------------------------------------------------
        # 2. 规则均衡损失 (Rule Balance Loss)
        # 最大化 Batch 平均熵：希望所有规则被均匀使用
        # ---------------------------------------------------
        # 计算 Batch 内的平均激活度: [K]
        p_avg = P.mean(dim=0)
        # 这是一个负熵项，我们希望熵最大，即 minimize (sum(p_avg * log(p_avg)))
        # 因为 entropy = -sum(...), maximize entropy => minimize -entropy => minimize sum(...)
        entropy_batch = torch.sum(p_avg * torch.log(p_avg + epsilon))

        # 总损失 = 置信度(越小越好) + 均衡性(越小越好)
        return 0.2 * entropy_sample + 0.05 * entropy_batch

# -----------------------------------------------------------
# 模块 2: Learnable Spectral Filter (保持不变)
# -----------------------------------------------------------
class TSKFuzzySpectralFilter(nn.Module):
    def __init__(self, seq_len: int, d_model: int, dropout: float = 0.1, n_rules: int = 3):
        super().__init__()
        self.freq_len = seq_len // 2 + 1
        self.d_model = d_model
        self.n_rules = n_rules
        self.centers = nn.Parameter(torch.randn(n_rules, self.freq_len))
        self.sigma = nn.Parameter(torch.ones(n_rules) * 1.0)
        self.initialized = False
        self.rule_complex_weights = nn.Parameter(
            torch.randn(n_rules, self.freq_len, d_model, 2) * 0.01
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.full((d_model,), -2.0))

        # ★ 新增：缓存
        self.firing_history = []

        self.record_firing = False
        self.vis_firing_history = []
    def _initialize_centers(self, x_fft_mag):
        with torch.no_grad():
            if x_fft_mag.size(0) >= self.n_rules:
                indices = torch.randperm(x_fft_mag.size(0))[:self.n_rules]
                selected = x_fft_mag[indices]
            else:
                indices = torch.randint(0, x_fft_mag.size(0), (self.n_rules,))
                selected = x_fft_mag[indices]
            self.centers.data.copy_(selected)
            self.initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        x_fft = torch.fft.rfft(x, dim=1, norm="ortho")
        cur_len = x_fft.shape[1]

        x_fft_mag = x_fft.abs().mean(dim=-1)
        if cur_len != self.freq_len:
            x_fft_mag_ctx = F.interpolate(
                x_fft_mag.unsqueeze(1), size=self.freq_len, mode='linear', align_corners=True
            ).squeeze(1)
        else:
            x_fft_mag_ctx = x_fft_mag

        if self.training and not self.initialized:
            self._initialize_centers(x_fft_mag_ctx)

        dist = torch.norm(x_fft_mag_ctx.unsqueeze(1) - self.centers.unsqueeze(0), dim=-1)
        sigma = torch.clamp(self.sigma, min=0.1)
        membership = torch.exp(- (dist ** 2) / (2 * sigma ** 2))
        firing_strength = membership / (torch.sum(membership, dim=1, keepdim=True) + 1e-8)

        # dist2 = ((x_fft_mag_ctx.unsqueeze(1) - self.centers.unsqueeze(0)) ** 2).mean(dim=-1)
        # sigma2 = self.sigma.clamp_min(0.1).pow(2).unsqueeze(0)
        # logits = - dist2 / (2 * sigma2)
        # firing_strength = torch.softmax(logits, dim=1)

        # ★ 新增：缓存 firing
        if self.training:
            self.firing_history.append(firing_strength)

        if self.record_firing:
            self.vis_firing_history.append(firing_strength.detach().cpu())

        # ... (Function-Space Mixture logic continued ...)
        raw_weights = self.rule_complex_weights
        if cur_len != self.freq_len:
            K, F0, D, two = raw_weights.shape
            raw_weights = raw_weights.permute(0, 2, 3, 1).reshape(K * D, 2, F0)  # [K*D, 2, F0]
            raw_weights = F.interpolate(raw_weights, size=cur_len, mode='linear', align_corners=True)
            raw_weights = raw_weights.reshape(K, D, 2, cur_len).permute(0, 3, 1, 2).contiguous()  # [K, cur_len, D, 2]
            # raw_weights = raw_weights.permute(0, 3, 2, 1)
            # raw_weights = F.interpolate(raw_weights, size=cur_len, mode='linear', align_corners=True)
            # raw_weights = raw_weights.permute(0, 3, 2, 1)

        # rules_complex = torch.view_as_complex(raw_weights)
        rules_complex = torch.view_as_complex(raw_weights.contiguous())
        mag = torch.tanh(torch.abs(rules_complex))
        phase = torch.angle(rules_complex)
        valid_rules = mag * torch.exp(1j * phase)
        freq_decay = torch.linspace(1.0, 0.2, cur_len, device=x.device).view(1, -1, 1)
        valid_rules = valid_rules * freq_decay

        effective_filter = torch.einsum('bk,kfd->bfd', firing_strength.type_as(valid_rules), valid_rules)

        x_fft_modulated = x_fft * effective_filter
        x_out = torch.fft.irfft(x_fft_modulated, n=L, dim=1, norm="ortho")
        x_out = self.norm(x_out)
        gate = torch.sigmoid(self.alpha).view(1, 1, -1)
        return gate * self.dropout(x_out)

    def get_tsk_loss(self):
        """计算 TSK 辅助损失并清空缓存"""
        if not self.firing_history:
            return torch.tensor(0.0).to(self.alpha.device)

        P = torch.cat(self.firing_history, dim=0)
        self.firing_history = []

        epsilon = 1e-8
        # 1. L_conf
        entropy_sample = -torch.sum(P * torch.log(P + epsilon), dim=1).mean()
        # 2. L_bal
        p_avg = P.mean(dim=0)
        entropy_batch = torch.sum(p_avg * torch.log(p_avg + epsilon))

        return 0.1 * entropy_sample + 0.02 * entropy_batch


# -----------------------------------------------------------
# Encoder Layer (集成 TSK)
# -----------------------------------------------------------
class EncoderLayer1(nn.Module):
    def __init__(
            self,
            attention,
            d_model,
            d_ff=None,
            dropout=0.1,
            activation="relu",
            seq_len=256,
            n_refine_steps=3,
            n_tsk_rules=3
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.graph_conv = DynamicGraphConv(d_model, n_rules=n_tsk_rules)
        self.spectral_filter = TSKFuzzySpectralFilter(seq_len=seq_len, d_model=d_model, dropout=dropout, n_rules=n_tsk_rules)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.n_refine_steps = n_refine_steps

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # ... (保持原有的 forward 逻辑不变) ...
        # 1. Temporal Attention
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = self.norm1(x + self.dropout(new_x))

        # 2. Recurrent Refinement
        for _ in range(self.n_refine_steps):
            x_res = x
            feat_graph = self.graph_conv(x)
            feat_spec = self.spectral_filter(x)
            x = self.norm2(x_res + feat_graph + feat_spec)

        # 3. FFN
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y), attn

    def get_extra_loss(self):
        """递归收集该层所有组件的 TSK 损失"""
        loss = 0.0
        loss += self.graph_conv.get_tsk_loss()
        loss += self.spectral_filter.get_tsk_loss()
        return loss


class Encoder1(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder1, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(
                    zip(self.attn_layers, self.conv_layers)
            ):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

    # ★★★ 新增：辅助损失收集接口 ★★★
    def get_extra_loss(self):
        """
        遍历所有 EncoderLayer，收集 TSK 产生的辅助损失 (置信度 + 均衡性)
        """
        # total_loss = 0.0
        total_loss = torch.zeros((), device=next(self.parameters()).device)
        # 遍历每一层 Attention Layer (即 EncoderLayer1 实例)
        for layer in self.attn_layers:
            # 检查该层是否有 get_extra_loss 方法
            # (标准的 EncoderLayer 可能没有，但你的 EncoderLayer1 有)
            if hasattr(layer, 'get_extra_loss'):
                total_loss += layer.get_extra_loss()

        return total_loss




class DecoderLayer(nn.Module):
    def __init__(
        self,
        self_attention,
        cross_attention,
        d_model,
        d_ff=None,
        dropout=0.1,
        activation="relu",
    ):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(
            self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0]
        )
        x = self.norm1(x)

        x = x + self.dropout(
            self.cross_attention(
                x, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta
            )[0]
        )

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(
                x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta
            )

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
