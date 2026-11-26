import torch


def feature_loss(fmap_r, fmap_g):
    loss: torch.Tensor | None = None
    for dr, dg in zip(fmap_r, fmap_g, strict=True):
        for rl, gl in zip(dr, dg, strict=True):
            rl = rl.float().detach()
            gl = gl.float()
            term = torch.mean(torch.abs(rl - gl))
            loss = term if loss is None else loss + term

    if loss is None:
        raise ValueError("feature_loss received no feature maps")
    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss: torch.Tensor | None = None
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs, strict=True):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        term = r_loss + g_loss
        loss = term if loss is None else loss + term
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    if loss is None:
        raise ValueError("discriminator_loss received no discriminator outputs")
    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss: torch.Tensor | None = None
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        item_loss = torch.mean((1 - dg) ** 2)
        gen_losses.append(item_loss)
        loss = item_loss if loss is None else loss + item_loss

    if loss is None:
        raise ValueError("generator_loss received no discriminator outputs")
    return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    loss = kl / torch.sum(z_mask)
    return loss


class WavLMLoss(torch.nn.Module):
    def __init__(self, model, wd, model_sr, slm_sr=16000):
        super().__init__()
        import torchaudio  # pyright: ignore[reportMissingImports]
        from transformers import AutoModel

        self.wavlm = AutoModel.from_pretrained(model)
        self.wd = wd
        self.resample = torchaudio.transforms.Resample(model_sr, slm_sr)
        self.wavlm.eval()
        for param in self.wavlm.parameters():
            param.requires_grad = False

    def _wavlm_hidden_states(self, wav_16: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # Torch 2.10/cuDNN can select very large-workspace conv1d kernels for WavLM's
        # feature extractor. The non-cuDNN path is much faster for these small inputs.
        with torch.backends.cudnn.flags(enabled=False):
            hidden_states = self.wavlm(input_values=wav_16, output_hidden_states=True).hidden_states
        if hidden_states is None:
            raise ValueError("WavLM returned no hidden states")
        return tuple(hidden_states)

    def forward(self, wav, y_rec):
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self._wavlm_hidden_states(wav_16)
        y_rec_16 = self.resample(y_rec)
        y_rec_embeddings = self._wavlm_hidden_states(y_rec_16)

        floss: torch.Tensor | None = None
        for er, eg in zip(wav_embeddings, y_rec_embeddings, strict=True):
            term = torch.mean(torch.abs(er - eg))
            floss = term if floss is None else floss + term

        if floss is None:
            raise ValueError("WavLMLoss received no hidden states")
        return floss

    def generator(self, y_rec):
        y_rec_16 = self.resample(y_rec)
        y_rec_embeddings = self._wavlm_hidden_states(y_rec_16)
        y_rec_embeddings = torch.stack(y_rec_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)
        y_df_hat_g = self.wd(y_rec_embeddings)
        loss_gen = torch.mean((1 - y_df_hat_g) ** 2)

        return loss_gen

    def discriminator(self, wav, y_rec):
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self._wavlm_hidden_states(wav_16)
            y_rec_16 = self.resample(y_rec)
            y_rec_embeddings = self._wavlm_hidden_states(y_rec_16)

            y_embeddings = torch.stack(wav_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)
            y_rec_embeddings = torch.stack(y_rec_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)

        y_d_rs = self.wd(y_embeddings)
        y_d_gs = self.wd(y_rec_embeddings)

        y_df_hat_r, y_df_hat_g = y_d_rs, y_d_gs

        r_loss = torch.mean((1 - y_df_hat_r) ** 2)
        g_loss = torch.mean((y_df_hat_g) ** 2)

        loss_disc_f = r_loss + g_loss

        return loss_disc_f.mean()

    def discriminator_forward(self, wav):
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self._wavlm_hidden_states(wav_16)
            y_embeddings = torch.stack(wav_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)

        y_d_rs = self.wd(y_embeddings)

        return y_d_rs
