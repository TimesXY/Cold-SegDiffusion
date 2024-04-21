import os
import torch
import wandb
import random
import datetime
import torchvision.transforms as transforms

from tqdm import tqdm
from torch.optim import AdamW
from accelerate import Accelerator
from ColdSeg.dataset import ISICDataset
from torch.utils.data import DataLoader
from ColdSeg.SegDiffusion import Unet, MedSegDiff
from ColdSeg.SegDiffusion import create_lr_scheduler

if __name__ == '__main__':

    # Hyper Parameter Setting
    dim = 64  # Foundation dimensions of the UNet network
    epochs = 6000  # Iterations
    img_size = 256  # Input Image Size, or 128
    batch_size = 32  # Batch Size, or 16
    save_every = 100  # Iterate X times, save the model
    time_steps = 50  # Time step - number of noise stacks
    adam_beta_1 = 0.950  # Adam Parameter 1
    adam_beta_2 = 0.999  # Adam Parameter 2
    adam_epsilon = 1e-8  # eps value
    weight_decay = 1e-6  # regularization factor
    learning_rate = 2e-5  # learning rate
    mask_channels = 3  # Number of mask channels
    self_condition = False  # self-conditional input
    input_img_channels = 3  # Number of input image channels

    # data path
    data_path = r"D:\MyDataSet\ISIC_Med"
    load_model_from = None  # Pre-training model paths

    # Create path - for saving training results
    checkpoint_dir = "output/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create path - for saving training logs
    logging_dir = "output/logs"
    os.makedirs(logging_dir, exist_ok=True)

    # Accelerating the training process with gas pedals
    # (gradient accumulation steps, blending accuracy, recording positions)
    accelerator = Accelerator(gradient_accumulation_steps=4,
                              mixed_precision="no",
                              log_with=["wandb"],
                              project_dir="logs")

    # Establishment of the Wandb project
    if accelerator.is_main_process:
        accelerator.init_trackers("ColdSegDiffusion")

    # Network Architecture
    model = Unet(dim=dim, image_size=img_size,
                 dim_mult=(1, 2, 4, 8), mask_channels=mask_channels,
                 input_img_channels=input_img_channels, self_condition=self_condition)

    # Data Augmentation
    transform_data = transforms.Compose([transforms.Resize(img_size),
                                         transforms.CenterCrop(img_size),
                                         transforms.ToTensor()])

    # Data Loading
    is_folder = "ISBI2016_ISIC_Dataset"
    dataset_train = ISICDataset(data_path, "train.txt", is_folder, transform=transform_data, training=True, flip_p=0.5)
    dataset_valid = ISICDataset(data_path, "valid.txt", is_folder, transform=transform_data, training=False, flip_p=0.0)

    # Data Processing
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)

    # Initializing the Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate,
                      betas=(adam_beta_1, adam_beta_2), weight_decay=weight_decay, eps=adam_epsilon)

    # Data acceleration and mixing accuracy settings
    model, optimizer, train_loader, valid_loader = accelerator.prepare([model, optimizer, train_loader, valid_loader])

    # Training Diffusion Networks
    diffusion = MedSegDiff(model, time_steps=time_steps, objective='predict_x0').to(accelerator.device)

    # Loading pre-trained models
    if load_model_from is not None:
        save_dict = torch.load(load_model_from)
        diffusion.model.load_state_dict(save_dict['model_state_dict'])
        optimizer.load_state_dict(save_dict['optimizer_state_dict'])
        accelerator.print(f'Loaded from {load_model_from}')

    # Create a learning rate update strategy, in this case once per step (not per epoch).
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), epochs, warmup=True)

    # Save information during training and validation
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # training process
    for epoch in range(epochs):
        # Recorded losses
        running_train_loss = 0.0
        running_valid_loss = 0.0
        print('Epoch {}/{}'.format(epoch + 1, epochs))

        # Initialize images and masks
        train_img = None
        valid_img = None
        train_mask = None
        valid_mask = None

        # Read image and mask Gradient optimization process
        model.train()
        for (train_img, train_mask) in tqdm(train_loader):
            with accelerator.accumulate(model):
                # Calculating and recording losses
                train_loss = diffusion(train_mask, train_img, epoch, epochs)

                # backpropagation gradient
                accelerator.backward(train_loss)

                # Update Gradient
                optimizer.step()
                optimizer.zero_grad()

                # Updated learning rate
                lr_scheduler.step()
                lr = optimizer.param_groups[0]["lr"]

        # Record training loss results
        running_train_loss += train_loss.item() * train_img.size(0)
        epoch_train_loss = running_train_loss / len(train_loader)
        print('Train Loss : {:.7f}'.format(epoch_train_loss))

        # Model Validation Process
        model.eval()
        for (valid_img, valid_mask, _) in tqdm(valid_loader):
            with torch.no_grad():
                with accelerator.accumulate(model):
                    # 计算损失并记录
                    valid_loss = diffusion(valid_mask, valid_img, epoch, epochs)

        # Record training loss results
        running_valid_loss += valid_loss.item() * valid_img.size(0)
        epoch_valid_loss = running_valid_loss / len(valid_loader)
        print('Valid Loss : {:.7f}'.format(epoch_valid_loss))

        # wandb Record results Training loss Validation loss
        accelerator.log({
            'lr': lr,
            'train_loss': epoch_train_loss,
            'valid_loss': epoch_valid_loss})

        # Record the train_loss, valid_loss, and lr for each epoch.
        with open(results_file, "a") as f:
            info = f"[lr: {lr}]\n" \
                   f"[epoch: {epoch}]\n" \
                   f"train_loss: {train_loss:.7f}\n" \
                   f"valid_loss: {valid_loss:.7f}\n"

            f.write(info + "\n\n")

        # Reasoning process Preservation model
        if epoch % save_every == 0:
            torch.save({'epoch': epoch, 'model_state_dict': diffusion.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'loss': valid_loss,
                        }, os.path.join(checkpoint_dir, f'state_dict_epoch_{epoch}_loss_{epoch_valid_loss}.pt'))

            # Getting Sampling Results
            predict_img = diffusion.sample(valid_img).cpu().detach().numpy()
            predict_img = torch.as_tensor(predict_img)

            # Logging to the wandb page
            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    # create rand int
                    key_int = random.randint(0, predict_img.shape[0] - 1)
                    tracker.log(
                        {'predict-image-mask': [wandb.Image(predict_img[key_int, :, :, :]),
                                                wandb.Image(valid_img[key_int, :, :, :]),
                                                wandb.Image(valid_mask[key_int, :, :, :])]})
