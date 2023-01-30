import torch
import nibabel as nib
import numpy as np
import pdb
import torchvision
from tqdm import tqdm
from einops import rearrange
import glob

def wmh_seg(model, in_path, out_path, train_transforms):
    img = nib.load(in_path)
    input = img.get_fdata()
    origin_size = img.get_fdata().shape
    affine = img.affine
    prediction = np.zeros((224,256,input.shape[-1]))

    input = train_transforms(input)
    input = torch.unsqueeze(input, 1)
    prediction_input = input/torch.max(input)
    for idx in tqdm(range(input.shape[0])):
        prediction[:, :, idx] = model(torch.unsqueeze(prediction_input[idx], 0).float()).squeeze().detach().numpy()

    #saving images
    arg = prediction > 0.5
    out = np.zeros(prediction.shape)
    out[arg] = 1
    img_fit = input.squeeze().detach().numpy()
    img_fit = rearrange(img_fit, 'd0 d1 d2 -> d1 d2 d0')
    train_transforms = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.CenterCrop((img.get_fdata().shape[0], img.get_fdata().shape[1],)),])
    img_fit = train_transforms(img_fit).detach().cpu().numpy()
    out = train_transforms(out).detach().cpu().numpy()

    img_fit = rearrange(img_fit, 'd0 d1 d2 -> d1 d2 d0')
    out = rearrange(out, 'd0 d1 d2 -> d1 d2 d0')

    nii_seg = nib.Nifti1Image(out, affine=affine)
    nib.save(nii_seg, out_path)

def wmh(data_path, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_transforms = torchvision.transforms.Compose([ 
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.CenterCrop((224, 256,)),
                    ])
    model = torch.load(f"{model_path}")
    model.eval()
    model.to(device)    

    in_paths = glob.glob(f'{data_path}/*.nii.gz')            

    for in_path in in_paths:
        filename=in_path.split('/')[-1]
        ID=filename.split('_')[0]
        date=filename.split('_')[-1].split('.')[0]
        out_path=f'{data_path}/{ID}_wmh_{date}.nii.gz'
        wmh_seg(in_path, out_path, train_transforms)


# basepath='/Users/jinghangli/Library/CloudStorage/OneDrive-UniversityofPittsburgh/02-Unet_segmentation/7T_data/KLU-APC2'
# in_paths=glob.glob(f'{basepath}/*.nii.gz')




