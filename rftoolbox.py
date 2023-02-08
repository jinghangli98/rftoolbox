import os
import pdb
import glob
import numpy as np
from natsort import natsorted
from tqdm import tqdm 
import torch
import pydicom
import nibabel as nib
# import ants
# import torchio as tio
# from intensity_normalization.normalize.nyul import NyulNormalize
# from intensity_normalization.typing import Modality, TissueType
# from intensity_normalization.normalize.fcm import FCMNormalize
# import psutil

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
        if sequence is None:
            for ind in subIndx:
                desired_path.append(glob.glob(f'{storinator}/scans/{study}/{date}/*/*{modality}*.{ind}')[0])
        else:
            desired_path.append(glob.glob(f'{storinator}/scans/{study}/{date}/*/*{modality}*.{subIndx[sequence]}')[0])
            

    if desired_path is not None:
        print('Pulling storinator data now.......')
    for idx in tqdm(range(len(desired_path))):
        os.chdir(f'{desired_path[idx]}')
        file = desired_path[idx].split('/')[-1]
        try:
            os.mkdir(f'{output_path}')
        except:
            pass
        os.system(f'rsync -a {desired_path[idx]} {output_path}/{modality}')
        os.chdir(f'{output_path}/{modality}/{file}')
        os.system(f'dcm2niix -z y -f %i_%t -o {output_path}/{modality} * ')
        os.chdir('..')
        os.system(f'rm -r {file}')
        os.system(f'rm *json*')
        
def pulldata_globus(storinator, study, output_path, scanID='*'):
    try:
        os.mkdir(f'{output_path}')
    except:
        pass
    os.chdir(f'{storinator}/scans')
    files = glob.glob(f'{study}/*/{scanID}/*')
    scanID = [p.split('/')[2] for p in files]
    scanID = np.unique(scanID)
    dates = [p.split('/')[1] for p in files]
    dates = np.unique(dates)
    
    
    for idx in tqdm(range(len(files))):
        file = files[idx].split('/')[-1]
        try:
            os.mkdir(f'{output_path}')
        except:
            pass
        
        os.chdir(f'{storinator}/scans')
        os.system(f'rsync -a -R {files[idx]} {output_path}')
        os.chdir(f'{output_path}/{files[idx]}')
        os.system(f'dcm2niix -z y * ')
        os.system(f'cp *.nii.gz ..')
        
        os.chdir('..')
        os.system(f'rm -r {file}')
        
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

def icv(inpath, voxel_size=None, core=6):
    os.system(f'gunzip {inpath}')
    skull_strip = 'mri_synthstrip -i {} -o {.}_skullstripped.nii.gz -m {.}_icv_mask.nii.gz'
    os.system(f'ls {inpath} | parallel --jobs {core} {skull_strip}')

    if voxel_size is not None:
        icv_files = glob.glob(f'{inpath}/*mask.nii.gz')
        icv_files = natsorted(icv_files)

        icv_nii = [nib.load(p).get_fdata() for p in icv_files]
        icv_count = [np.sum(nii) for nii in icv_nii]
        voxel_size = voxel_size[0] * voxel_size[1] * voxel_size[2]
        icv = [icv * voxel_size for icv in icv_count]
        
        return icv, icv_count, [file.split('/')[-1] for file in icv_files]
    else:
        icv_files = glob.glob(f'{inpath}/*mask.nii.gz')
        icv_files = natsorted(icv_files)
        icv_nii = [nib.load(p).get_fdata() for p in icv_files]
        
        return icv_nii

def n4_writepath(path):

    filename = path.split('/')[-1]
    path = path.split('/')[1:-1]
    path.insert(0,'/')
    prefix_path = os.path.join(*path)
    path.append(f'n4_{filename}')
    path = os.path.join(*path)

    return path, prefix_path

def train_nyul(img_paths, standard_histogram_out):
    nyul_normalizer = NyulNormalize()
    images = []

    for i in tqdm(range(len(img_paths))):
        
        images.append(nib.load(img_paths[i]).get_fdata())
        # if i % 10 == 0:
        #     print(f'RAM memory % used: {psutil.virtual_memory()[2]}')
            
        if psutil.virtual_memory()[2] > 90:
            break
    nyul_normalizer.fit(images)
    nyul_normalizer.save_standard_histogram(f"{standard_histogram_out}")

def apply_nyul(img, npy, out):
    nyul_normalizer = NyulNormalize()
    nyul_normalizer.load_standard_histogram(npy)
    norm_img = nyul_normalizer(img.to_nibabel().get_fdata())
    norm_img = nib.Nifti1Image(norm_img, img.to_nibabel().affine)
    nib.save(norm_img, out)

    return norm_img.get_fdata()

# def apply_fcm(img, m, t, out):
#     pdb.set_trace()
#     fcm_norm = FCMNormalize(tissue_type=TissueType.t)
#     normalized = fcm_norm(img.to_nibabel().get_fdata(), modality=Modality.m)
#     normalized = nib.Nifti1Image(normalized, img.to_nibabel().affine)
#     nib.save(normalized, out)

#     return normalized.get_fdata()
    

def ratioCalc(T1, T2, mask, quantile):
    
    # T2_mask = np.zeros_like(T2)
    # T2_template_ind = np.argwhere(T2 >= 0.001)
    # for idx in range(len(T2_template_ind)):
    #     x,y,z = T2_template_ind[idx]
    #     T2_mask[x,y,z] = 1
    
    ratio = T1/(T2)
    ratio[np.isnan(ratio)] = 0

    upper_lim = np.quantile(ratio.flatten(), quantile)
    ratio[ratio >= upper_lim] = 0

    return ratio * mask

def myelinmap(T1_path, T2_path, T1_npy=None, T2_npy=None, ratio_map_output=None):
    print('Calculating T1-T2 Ratio Map.......')

    #bias correction
    T1_N4 = ants.n4_bias_field_correction(ants.image_read(T1_path))
    T2_N4 = ants.n4_bias_field_correction(ants.image_read(T2_path))
    print('(1) Bias correction done!')
    
    T1_path, prefix_T1 = n4_writepath(T1_path)
    T2_path, prefix_T2 = n4_writepath(T2_path)

    ants.image_write(T1_N4, f'{T1_path}')
    ants.image_write(T2_N4, f'{T2_path}')
    
    #skull stripping
    icv(T1_path, core=6)
    icv(T2_path, core=6)
    
    T1_skull_stripped = glob.glob(T1_path.split('.nii')[0] + '_skullstripped.nii.gz')[0]
    T1_brain_mask = glob.glob(T1_path.split('.nii')[0] + '_icv_mask.nii.gz')[0]
    T2_skull_stripped = glob.glob(T2_path.split('.nii')[0] + '_skullstripped.nii.gz')[0]
    T2_brain_mask = glob.glob(T2_path.split('.nii')[0] + '_icv_mask.nii.gz')[0]
    print('(2) Skull stripping done!')

    T1_spacing = np.round(T1_N4.spacing, 2)
    T2_spacing = np.round(T2_N4.spacing, 2)
    
    #reslicing to the lowest resolution
    resample_spacing = (np.max((T1_spacing[0], T2_spacing[0])),
    np.max((T1_spacing[1], T2_spacing[1])),
    np.max((T1_spacing[2], T2_spacing[2])))
    transform = tio.Resample(resample_spacing)
    rT1 = transform(nib.load(T1_skull_stripped))
    rT2 = transform(nib.load(T2_skull_stripped))
    rmask = transform(nib.load(T1_brain_mask))
    print('(3) Reslicing done!')

    #Registration
    T1 = ants.from_nibabel(rT1)
    T2 = ants.from_nibabel(rT2)
    registered_T2 = ants.registration(fixed=T1 , moving=T2, type_of_transform='Rigid')['warpedmovout']
    print('(4) Registration done!')

    # nyul_T1 = apply_nyul(T1, T1_npy, f'{prefix_T1}/nyul_T1.nii.gz')
    # nyul_T2 = apply_nyul(T2, T2_npy, f'{prefix_T2}/nyul_T2.nii.gz')
    if T1_npy is None and T1_npy is None:
        print('Applying FCM normalization')
        fcm_norm = FCMNormalize(tissue_type=TissueType.WM)
        T1_normalized = fcm_norm(T1.to_nibabel().get_fdata(), modality=Modality.T1)
        T2_normalized = fcm_norm(registered_T2.to_nibabel().get_fdata(), modality=Modality.T2)
        T1_normalized = nib.Nifti1Image(T1_normalized, T1.to_nibabel().affine)
        T2_normalized = nib.Nifti1Image(T2_normalized, registered_T2.to_nibabel().affine)

        nib.save(T1_normalized, f'{prefix_T1}/fcm_T1.nii.gz')
        nib.save(T2_normalized, f'{prefix_T2}/fcm_T2.nii.gz')
        T1_normalized = T1_normalized.get_fdata()
        T2_normalized = T2_normalized.get_fdata()
        print('(5) Normalization done!')
    else: 
        print('Applying Nyul normalization')
        T1_normalized = apply_nyul(T1, T1_npy, f'{prefix_T1}/nyul_T1.nii.gz')
        T2_normalized = apply_nyul(registered_T2, T2_npy, f'{prefix_T2}/nyul_T2.nii.gz')
        print('(5) Normalization done!')

    
    if ratio_map_output is None:
        ratio_map_output = prefix_T2
        if T1_npy is None and T1_npy is None: 
            ratio_map_output += '/fcm_ratio_map.nii.gz'
        else:
            ratio_map_output += '/nyul_ratio_map.nii.gz'
    
    
    ratio_map = ratioCalc(T1_normalized, T2_normalized, rmask.get_fdata(), 0.995)
    ratio_map = nib.Nifti1Image(ratio_map, affine=registered_T2.to_nibabel().affine)
    nib.save(ratio_map, ratio_map_output)
    print(f'Ratio map saved to {ratio_map_output}')
    



