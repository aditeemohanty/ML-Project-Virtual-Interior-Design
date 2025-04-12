import torch
from dataset import FurnishedUnfurnishedDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
import albumentations as A
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator


def train_fn(
    disc_F, disc_U, gen_U, gen_F, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler
):
    F_reals = 0
    F_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (unfurnished, furnished) in enumerate(loop):
        unfurnished = unfurnished.to(config.DEVICE)
        furnished = furnished.to(config.DEVICE)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_furnished = gen_F(unfurnished)
            D_F_real = disc_F(furnished)
            D_F_fake = disc_F(fake_furnished.detach())
            F_reals += D_F_real.mean().item()
            F_fakes += D_F_fake.mean().item()
            D_F_real_loss = mse(D_F_real, torch.ones_like(D_F_real))
            D_F_fake_loss = mse(D_F_fake, torch.zeros_like(D_F_fake))
            D_F_loss = D_F_real_loss + D_F_fake_loss

            fake_unfurnished = gen_U(furnished)
            D_U_real = disc_U(unfurnished)
            D_U_fake = disc_U(fake_unfurnished.detach())
            D_U_real_loss = mse(D_U_real, torch.ones_like(D_U_real))
            D_U_fake_loss = mse(D_U_fake, torch.zeros_like(D_U_fake))
            D_U_loss = D_U_real_loss + D_U_fake_loss

            # put it togethor
            D_loss = (D_F_loss + D_U_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_F_fake = disc_F(fake_furnished)
            D_F_fake = disc_U(fake_unfurnished)
            loss_G_F = mse(D_F_fake, torch.ones_like(D_F_fake))
            loss_G_U = mse(D_U_fake, torch.ones_like(D_U_fake))

            # cycle loss
            cycle_unfurnished = gen_U(fake_furnished)
            cycle_furnished = gen_F(fake_unfurnished)
            cycle_unfurnished_loss = l1(unfurnished, cycle_unfurnished)
            cycle_furnished_loss = l1(furnished, cycle_furnished)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_unfurnished = gen_U(unfurnished)
            identity_furnished = gen_F(furnished)
            identity_unfurnished_loss = l1(unfurnished, identity_unfurnished)
            identity_furnished_loss = l1(furnished, identity_furnished)

            # add all togethor
            G_loss = (
                loss_G_U
                + loss_G_F
                + cycle_unfurnished_loss * config.LAMBDA_CYCLE
                + cycle_furnished_loss * config.LAMBDA_CYCLE
                + identity_furnished_loss * config.LAMBDA_IDENTITY
                + identity_unfurnished_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_furnished * 0.5 + 0.5, f"saved_images/furnished_{idx}.png")
            save_image(fake_unfurnished * 0.5 + 0.5, f"saved_images/unfurnished_{idx}.png")

        loop.set_postfix(F_real=F_reals / (idx + 1), F_fake=F_fakes / (idx + 1))


def main():
    disc_F = Discriminator(in_channels=3).to(config.DEVICE)
    disc_U = Discriminator(in_channels=3).to(config.DEVICE)
    gen_U = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_F = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_F.parameters()) + list(disc_U.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_U.parameters()) + list(gen_F.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_F,
            gen_F,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_U,
            gen_U,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_F,
            disc_F,
            opt_disc,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_U,
            disc_U,
            opt_disc,
            config.LEARNING_RATE,
        )

    dataset = FurnishedUnfurnishedDataset(
        root_furnished="data/train/furnished",
        root_unfurnished="data/train/unfurnished",
        transform=config.transforms,
    )
    val_dataset = FurnishedUnfurnishedDataset(
        root_furnished="data/val/furnished",
        root_unfurnished="data/val/unfurnished",
        transform=config.transforms,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc_F,
            disc_U,
            gen_U,
            gen_F,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
        )

        if config.SAVE_MODEL:
            save_checkpoint(gen_F, opt_gen, filename=config.CHECKPOINT_GEN_F)
            save_checkpoint(gen_U, opt_gen, filename=config.CHECKPOINT_GEN_U)
            save_checkpoint(disc_F, opt_disc, filename=config.CHECKPOINT_CRITIC_F)
            save_checkpoint(disc_U, opt_disc, filename=config.CHECKPOINT_CRITIC_U)


if __name__ == "__main__":
    main()