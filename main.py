from math import gamma
import torch
import torchvision
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import argparse
import os
import utils
import OTB100_data
import json

import torchattacks
from torchvision.transforms import ToPILImage, ToTensor


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--vis', type=lambda x: (str(x).lower() == 'true'), default=False, help='Visualization flag')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device ID')
    parser.add_argument('--adv_path', type=str, default='', help='Directory to save adversarial examples')
    return parser.parse_args()


def test_transform():
    """Define image transformation pipeline for testing"""
    default_mean = [0.485, 0.456, 0.406]  # ImageNet mean values
    default_std = [0.229, 0.224, 0.225]  # ImageNet std values

    norm_method = OTB100_data.Normalize(default_mean, default_std)
    spatial_transform = OTB100_data.spatial_Compose([
        OTB100_data.ToTensor(),  # Convert to tensor
        norm_method  # Apply normalization
    ])
    return spatial_transform


def transform_video(video, mode='forward'):
    r'''
    Transform video between normalized and original pixel space
    Args:
        video: Input tensor to transform
        mode: 'forward' (normalize) or 'back' (denormalize)
    '''
    dtype = video.dtype
    mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=dtype).cuda()
    std = torch.as_tensor([0.229, 0.224, 0.225], dtype=dtype).cuda()

    if mode == 'forward':
        video.sub_(mean[:, None, None]).div_(std[:, None, None])  # Normalize
    elif mode == 'back':
        video.mul_(std[:, None, None]).add_(mean[:, None, None])  # Denormalize
    return video


if __name__ == '__main__':
    # Configuration and initialization
    args = parse_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # Set GPU device

    # Ensure reproducibility
    cudnn.benchmark = False
    cudnn.deterministic = True
    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained models with normalization layers
    model_alex = nn.Sequential(
        utils.Normalize('imagenet'),
        torchvision.models.alexnet(pretrained=True)
    ).cuda().eval()  # AlexNet

    model_res50 = nn.Sequential(
        utils.Normalize('imagenet'),
        torchvision.models.resnet50(pretrained=True)
    ).cuda().eval()  # ResNet50

    # Initialize transformation
    test_spa_trans = test_transform()

    # Define attack method (LCMC ensemble attack)
    ADAMTAP10 = torchattacks.LCMC(
        [model_res50, model_alex],
        eps=16.0 / 255,  # Perturbation budget
        alpha=0.005,  # Step size
        steps=60,  # Number of iterations
        momentum=1.0  # Momentum factor
    )
    attacks = [ADAMTAP10]

    # Process each attack method
    for attack_id, attack in enumerate(attacks):
        # Load image metadata from JSON
        json_path = './output.json'
        with open(json_path, 'r') as json_file:
            image_data = json.load(json_file)

        # Process each video sequence
        for entry in image_data:
            image_name = entry['directory']
            print(f'Processing sequence: {image_name}')

            # Create dataset and dataloader for current sequence
            test_dataset = OTB100_data.attack_otb100(
                spatial_transform=test_spa_trans,
                dir_name=image_name
            )
            val_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )

            # Prepare output directory
            current_folder = os.path.join(args.adv_path, image_name)
            if not os.path.exists(current_folder):
                os.makedirs(current_folder)
                count = 0  # Counter for saved images

                # Process batches
                for batch_image, target, duration in val_loader:
                    # Prepare input batch
                    X_batch = batch_image[0].to(device)  # Get image batch
                    b, c, h, w = X_batch.shape
                    X_batch = transform_video(X_batch.clone().detach(), mode='back')  # Denormalize

                    # Handle batch size mismatch with padding
                    padding_count = 0
                    if X_batch.shape[0] < args.batch_size:
                        padding_count = args.batch_size - X_batch.shape[0]
                        padding = torch.zeros((padding_count, *X_batch.shape[1:])).cuda()
                        input_batch = torch.cat((X_batch, padding), dim=0)
                        adv_X = attack(input_batch, args.batch_size)
                        adv_X = adv_X[:-padding_count]  # Remove padding
                    else:
                        adv_X = attack(X_batch, args.batch_size)

                    # Save adversarial images
                    for k in range(adv_X.shape[0]):
                        # Convert tensor to PIL image and save
                        adv_img = ToPILImage()(adv_X[k].detach().cpu().squeeze())
                        adv_img = adv_img.resize((w, h))
                        count += 1
                        adv_img.save(os.path.join(current_folder, f'%04d.jpg' % count), quality=200)

                    # Print progress
                    progress = 100.0 * (count / duration[0])
                    print(f'{current_folder} - Progress: {progress:.2f}%')
            else:
                print(f'Directory exists, skipping: {current_folder}')