import os
import pdb
import glob
import numpy as np
from natsort import natsorted
from tqdm import tqdm 
import torch
import pydicom
import nibabel as nib

def pulldata(storinator, study, output_path, modality, sequence=-1):
    
    try:
        os.mkdir(f'{output_path}/{modality}')
    except:
        pass

    files = glob.glob(f'{storinator}/scans/{study}/*/*/*{modality}*')
    scanID = [p.split('/')[6] for p in files]
    dates = [p.split('/')[5] for p in files]

    scanID = np.unique(scanID)
    dates = np.unique(dates)

    desired_path = []

    for idx in range(len(dates)):
        date = dates[idx]
        modality_files = glob.glob(f'{storinator}/scans/{study}/{date}/*/*{modality}*')
        subIndx = np.int16([p.split('.')[-1] for p in modality_files])
        subIndx = natsorted(subIndx)
        desired_path.append(glob.glob(f'{storinator}/scans/{study}/{date}/*/*{modality}*.{subIndx[sequence]}')[0])

    if desired_path is not None:
        print('Pulling storinator data now.......')
    for idx in tqdm(range(len(desired_path))):
        os.chdir(f'{desired_path[idx]}')
        file = desired_path[idx].split('/')[-1]
        os.system(f'rsync -a {desired_path[idx]} {output_path}/{modality}')
        os.chdir(f'{output_path}/{modality}/{file}')
        os.system(f'dcm2niix -z y -f %i_%t -o {output_path}/{modality} * ')
        os.chdir('..')
        os.system(f'rm -r {file}')
        os.system(f'rm *json*')

def crop(image, size):
    new_h, new_w = size
    h, w = image.shape
    if h < new_h:
            pad_size = int((new_h - h)/2)
            image = np.pad(image, ((pad_size, pad_size), (0,0)))
    elif h > new_h:
        crop_size = int((h - new_h)/2)
        image = image[crop_size:h-crop_size, :]    
    if w < new_w:
        pad_size = int((new_w - w)/2)
        image = np.pad(image, ((0,0),  (pad_size, pad_size)))
    elif w > new_w:
        crop_size = int((w - new_w)/2)
        image = image[:, crop_size:w-crop_size]
    image = np.stack((image,)*3, axis=0)    
    return image

def quality_rating(storinator, study, output='.', inpath=None, mode='dcm'):
    
    model = torch.load('./trained_model/motion_resnet18_model.pth')
    model.eval()
    
    if mode == 'dcm':

        dcm_path = f'{storinator}/scans/{study}/20*'
        img_path = glob.glob(f'{dcm_path}/*/*TSE*_ND*/MR*')
        img_path.sort()
        subject_list = [i.split('/')[-3] for i in img_path]
        subject_list = np.unique(subject_list)
        study = dcm_path.split('/')[-2]

        if subject_list is not None: 
            print('Rating dcm images now.......')
            f = open(f"{output}/{study}_TSE_rating.txt", "w")   

        for idx in tqdm(range(len(subject_list))):
            path = glob.glob(f'{dcm_path}/{subject_list[idx]}/*TSE*/MR*')
            ds = pydicom.filereader.dcmread(path[0])
            _, row, col, _ = ds.AcquisitionMatrix
            img_data = [pydicom.filereader.dcmread(i).pixel_array for i in path]
            img_data = [np.expand_dims(crop(i, (512,512)), 0) for i in img_data]
            img_data = [i/np.max(i) for i in img_data]

            output = [model(torch.tensor(i).float()) for i in img_data]
            output = [int(np.argmax(i.detach().numpy(), axis = 1)) for i in output]
            rating = sum(output)/len(output)
            name = ds.StudyDescription.split('^')[-1]
            name = name + '_' + ds.PatientID
            path = path[0].split('/')[:-1]
            path.insert(1, '/')
            path = os.path.join(*path)
            print(f'Coil: {ds.TransmitCoilName} | Patient: {ds.StudyDate}_{name} | Sex: {ds.PatientSex} | DOB: {ds.PatientBirthDate} | MR Sequence: {ds.ProtocolName} | Rating: {rating}')
            f.write(f'Coil: {ds.TransmitCoilName} | Patient: {ds.StudyDate}_{name} | Sex: {ds.PatientSex} | DOB: {ds.PatientBirthDate} | MR Sequence: {ds.ProtocolName} | Rating: {rating} | Path: {path}\n')

    elif mode == 'nii':
        assert(inpath is not None)
        
        path = glob.glob(f'{inpath}/*')
        if len(path) > 0:
            print('Rating nii images now......')
            f = open(f"{output}/TSE_rating.txt", "w") 
        else:
            print('Check your input path. No images found')
        img_data = [nib.load(i).get_fdata() for i in path]
        cropped_img = []
        for idx in tqdm(range(len(img_data))):
            for z in range(np.shape(img_data[idx])[-1]):
                cropped_img.append(np.expand_dims(crop(img_data[idx][:,:,z], (512,512)),0))
            cropped_img = [i/np.max(i) for i in cropped_img]    
            output = [model(torch.tensor(i).float()) for i in cropped_img]
            output = [int(np.argmax(i.detach().numpy(), axis = 1)) for i in output]
            cropped_img = []
            rating = sum(output)/len(output)
            file_name = path[idx].split('/')[-1]
            print(f'File: {file_name} | Rating: {rating}')
            f.write(f'File: {file_name} | Rating: {rating}\n')

def icv(inpath, voxel_size, core=6):
    os.system(f'gunzip {inpath}/*')
    skull_strip = 'mri_synthstrip -i {} -o {.}_skullstripped.nii.gz -m {.}_icv_mask.nii.gz'
    os.system(f'ls {inpath}/* | parallel --jobs {core} {skull_strip}')

    icv_files = glob.glob(f'{inpath}/*mask.nii.gz')
    icv_files = natsorted(icv_files)

    icv_nii = [nib.load(p).get_fdata() for p in icv_files]
    icv_count = [np.sum(nii) for nii in icv_nii]
    voxel_size = voxel_size[0] * voxel_size[1] * voxel_size[2]
    icv = [icv * voxel_size for icv in icv_count]
    
    return icv, icv_count, [file.split('/')[-1] for file in icv_files]


            
            
