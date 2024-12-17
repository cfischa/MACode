# modules/diffusion_training.py
import copy
import torch
from ModelCode.timediffusion import BiRNN_score, EMA, get_loss

def train_diffusion(latent_features, time_info, params, device, progress_bar=None):
    """
    Train a diffusion model with Exponential Moving Average (EMA).

    Parameters:
    - latent_features (Tensor): Latent features from the autoencoder.
    - time_info (Tensor): Time-related features.
    - params (dict): Training parameters.
    - device (torch.device): Device for training ('cpu' or 'cuda').
    - progress_bar: Streamlit progress bar object (optional).

    Returns:
    - ema_model (Model): EMA of the trained diffusion model.
    - losses (list): List of training losses per epoch (optional).
    """
    # Initialize the main model
    input_size = latent_features.shape[2]
    model = BiRNN_score(
        input_size=input_size,
        hidden_size=params['hidden_dim'],
        num_layers=params['num_layers'],
        diffusion_steps=params['diffusion_steps']
    ).to(device)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params.get('lr', 1e-4))
    ema = EMA(beta=0.995)

    # Reinitialize ema_model to match model's architecture
    ema_model = BiRNN_score(
        input_size=input_size,
        hidden_size=params['hidden_dim'],
        num_layers=params['num_layers'],
        diffusion_steps=params['diffusion_steps']
    ).to(device)
    ema_model.load_state_dict(model.state_dict())  # Match state_dicts

    # Prepare inputs
    x = latent_features.detach().to(device)
    N, T, K = latent_features.shape
    all_indices = list(range(N))

    n_epochs = params['n_epochs']
    losses = []

    # Training loop
    for epoch in range(n_epochs):
        batch_indices = torch.randint(0, len(all_indices), size=(params['batch_size'],))
        t = torch.rand(len(batch_indices), T, 1).to(device)
        loss = get_loss(model, x[batch_indices], t, time_info[batch_indices])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update EMA model
        ema.step_ema(ema_model, model)

        # Record the loss for monitoring (optional)
        losses.append(loss.item())

        # Update the progress bar if provided
        if progress_bar is not None:
            progress = (epoch + 1) / n_epochs
            progress_bar.progress(progress)
            # Optionally display loss in Streamlit
            # st.write(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")

    return ema_model, losses  # Return losses if you want to plot them later
