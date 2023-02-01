from rftoolbox import *

storinator='/Volumes/storinator'
study='WPC-8521'
output_path=f'/Users/jinghangli/Library/CloudStorage/OneDrive-UniversityofPittsburgh/02-Unet_segmentation/7T_data/{study}'
modality='FLAIR'

# pulldata(storinator, study, output_path, modality)
# quality_rating(storinator, study, output='.', inpath = '/Users/jinghangli/Library/CloudStorage/OneDrive-UniversityofPittsburgh/02-Unet_segmentation/7T_data/COV-IBR/TSE', mode='nii')
# quality_rating(storinator, study, output='.icv(inpath = '/Users/jinghangli/Library/CloudStorage/OneDrive-UniversityofPittsburgh/02-Unet_segmentation/7T_data/COV-IBR/TSE')', mode='dcm')
# icv(inpath = '/Users/jinghangli/Library/CloudStorage/OneDrive-UniversityofPittsburgh/02-Unet_segmentation/7T_data/COV-IBR/TSE', voxel_size=(0.375,0.375,1.5))

T1_path = glob.glob('/mnt/Mercury/camcan/Public/Analysis/data/anat/sub-CC110033/anat/sub*T1*')[0]
T2_path = glob.glob('/mnt/Mercury/camcan/Public/Analysis/data/anat/sub-CC110033/anat/sub*T2*')[0]
ratio_map_output = '/mnt/Mercury/camcan/Public/Analysis/data/anat/sub-CC110033/anat'

T1_npy = './nyul_histogram/camcan_T1_histogram_backup.npy'
T2_npy = './nyul_histogram/camcan_T2_histogram_backup.npy'

# train_nyul(glob.glob('/mnt/Mercury/camcan/Public/Analysis/data/anat/*/anat/sub*T1*.nii'), T1_npy)
# train_nyul(glob.glob('/mnt/Mercury/camcan/Public/Analysis/data/anat/*/anat/sub*T2*.nii'), T2_npy)

myelinmap(T1_path, T2_path)
