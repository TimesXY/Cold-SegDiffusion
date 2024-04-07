import os
import torch
import numpy as np
import skimage.io as io
import torchvision.transforms as transforms

from tqdm import tqdm
from skimage.color import rgb2gray
from accelerate import Accelerator
from ColdSeg.dataset import ISICDataset
from torch.utils.data import DataLoader
from ColdSeg.SegDiffusion import Unet, MedSegDiff

if __name__ == '__main__':

    # Hyper Parameter Setting
    dim = 64  # Foundation dimensions of the UNet network
    img_size = 256  # Input Image Size, or 128
    batch_size = 32  # Batch Size
    time_steps = 50  # Time step - number of noise stacks
    mask_channels = 3  # Number of mask channels, will be
    self_condition = False  # self-conditional input
    save_uncertainty = False  # Preservation of uncertainty
    input_img_channels = 3  # Number of input image channels

    # data path
    data_path = r"D:\MyDataSet\ISIC_Med"
    load_model_from = r"\best_model.pt"

    # Reasoning Procedure Storage Path
    inference_dir = "output/inference"
    os.makedirs(inference_dir, exist_ok=True)

    # Acceleration of the training process using gas pedals (mixed precision)
    accelerator = Accelerator(mixed_precision="no")

    # Establishment of the UNet
    model = Unet(dim=dim, image_size=img_size,
                 dim_mult=(1, 2, 4, 8), mask_channels=mask_channels,
                 input_img_channels=input_img_channels, self_condition=self_condition)

    # Data Augmentation
    transform_test = transforms.Compose([transforms.Resize(img_size),
                                         transforms.CenterCrop(img_size),
                                         transforms.ToTensor()])

    # Data Loading
    is_folder = "ISBI2016_ISIC_Dataset"
    dataset = ISICDataset(data_path, "test.txt", is_folder, transform=transform_test, training=False, flip_p=0.0)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Diffusion Model
    diffusion = MedSegDiff(model, time_steps=time_steps).to(accelerator.device)

    # Read saved model parameters
    if load_model_from is not None:
        save_dict = torch.load(load_model_from)
        new_state_dict = {}
        for k, v in save_dict['model_state_dict'].items():
            new_state_dict[k[7:]] = v
        diffusion.model.load_state_dict(new_state_dict)

    diffusion.model.eval()

    # Loop through the data, make a prediction
    for (images, masks, file_names) in tqdm(data_loader):

        # Projected results
        predict_images = diffusion.sample(images).cpu().detach()

        # Preserving the results of reasoning
        for idx in range(predict_images.shape[0]):
            save_image = np.transpose(predict_images[idx, :, :, :], (1, 2, 0))
            save_image = rgb2gray(save_image)
            save_image = np.where(save_image >= 0.5, 1, 0)
            save_image = (save_image * 255.0).astype('uint8')
            io.imsave(os.path.join(inference_dir, file_names[idx][:-4] + "_prediction.png"), save_image)

            save_mask = np.transpose(masks[idx, :, :, :], (1, 2, 0))
            save_mask = rgb2gray(save_mask)
            save_mask = np.where(save_mask >= 0.5, 1, 0)
            save_mask = (save_mask * 255.0).astype('uint8')
            io.imsave(os.path.join(inference_dir, file_names[idx]), save_mask)
