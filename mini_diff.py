 # Imports
import torch
from torch.optim import Adam
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image

class MinUNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        # Define a convolutional block
        conv_block = lambda in_channels, out_channels: nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

        self.blocks = nn.Sequential(
            conv_block(1, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512),
            conv_block(512, 256),
            conv_block(256, 128),
            conv_block(128, 64),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.blocks(x)

class GaussianDiffusion:

    def __init__(self, num_timesteps, device):
        
        self.device = device

        # Construct the variance schedules
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(start=0.0001, end=0.02, steps=num_timesteps, device=device, dtype=torch.float64)
        self.alphas = 1.0 - self.betas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)

    @torch.no_grad()
    def q_sample(self, x_0, timesteps, noise=None):
        """ q(x_t | x_0)

        Sample from the diffusion process at time t
        """

        # Construct mean and std of x_t
        # Give the alpha bars dummy dimensions so they can be broadcasted
        mean = torch.sqrt(self.alpha_bars[timesteps, None, None, None]) * x_0
        std = torch.sqrt(1 - self.alpha_bars[timesteps, None, None, None])

        # Construct optional noise (normal distribution between 0 and 1)
        if noise is None:
            noise = torch.randn_like(mean)

        return mean + std * noise

    @torch.no_grad()
    def p_mean_variance(self, model : MinUNet, x_t : torch.Tensor, t : int):
        """ p(x_{t-1} | x_t)
        
        Calculates the mean and variance of the posterior (x_{t-1}) given the current state
        """

        # Retrieve relevant constants
        beta_t = self.betas[t]
        sqrt_alpha_t = self.sqrt_alphas[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t]

        # Predict the noise
        pred_noise = model(x_t)
        
        # Calculate the mean and variance of the posterior (x_{t-1})
        posterior_mean = (1 / sqrt_alpha_t) * (x_t - beta_t / sqrt_one_minus_alpha_bar_t * pred_noise)
        posterior_variance = beta_t

        return posterior_mean, posterior_variance
    
    @torch.no_grad()
    def p_sample(self, model : MinUNet, x_t : torch.Tensor, t : int):
        """ p(x_{t-1} | x_t)
        
        Sample from the posterior (x_{t-1}) given the current state
        """
        
        # Don't add noise if we are on last timestep
        noise = torch.randn_like(x_t) if t > 0 else 0

        # Calculate the mean and variance of the posterior
        posterior_mean, posterior_variance = self.p_mean_variance(model, x_t, t)

        # Sample from the posterior
        return noise * torch.sqrt(posterior_variance) + posterior_mean
    
    @torch.no_grad()
    def p_sample_loop(self, model : MinUNet, shape : list[int]):
        """ Denoising Loop.

        Samples from the posterior at each timestep to denoise the image
        """

        # Create noise if not provided
        x_t = torch.randn(shape, device=self.device)

        # Denoising loop
        for t in range(self.num_timesteps, -1, -1):
            x_t = self.p_sample(model, x_t, t)

        # Return the denoised image
        return x_t

def main():

    # Define Hyperparameters
    epochs = 10
    report_freq = 100
    batch_size = 128
    lr=2e-4
    num_timesteps = 500
    device = 0

    # Load the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        lambda x : x * 2 -1 # Normalize to [-1, 1]
    ])
    dataset = MNIST(root='data', download=True, train=True, transform=transform)

    # Create dataloader that normalizes data
    dataloader =  DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Define denormalization function
    denorm = lambda x : ((x + 1) / 2).clamp(-1, 1)

    # Create diffuser object which handles equations
    diffuser = GaussianDiffusion(num_timesteps=num_timesteps, device=device)

    # Create model
    model = MinUNet().to(device)

    # Define optimizer and loss
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Keep track of running loss
    running_loss = 0.0

    # Train the model
    for epoch in range(epochs):

        for i, (x, _) in enumerate(dataloader):
            
            # Cast image to device
            x = x.to(device)
            
            # Sample noise
            timesteps = torch.randint(low=0, high=num_timesteps, size=(x.shape[0],))
            noise = torch.randn_like(x)

            # Sample from the diffusion process
            x_t = diffuser.q_sample(x_0=x, timesteps=timesteps, noise=noise)

            # Calculate the loss
            pred_noise = model(x_t)
            loss = criterion(pred_noise, noise)
            running_loss += loss.item()

            # Backpropagate and update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Evaluate progress
            if (i + 1) % report_freq == 0:
                
                # Calculate average loss and reset running loss
                avg_loss = running_loss / report_freq
                running_loss = 0.0

                # Generate an image and save it
                gen_img = denorm(diffuser.p_sample_loop(model, shape=(1, 1, 28, 28)))
                save_image(gen_img, f'./images/{epoch}_{i}.png')

                # Print statistics
                print(f'Epoch [{epoch}/{epochs}] Batch [{i}/{len(dataloader)}] Loss: {avg_loss:.4f}')

if __name__ == '__main__':
    main()