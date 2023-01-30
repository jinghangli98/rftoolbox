from rftoolbox import *

storinator='/Volumes/storinator'
study='WPC-8521'
output_path=f'/Users/jinghangli/Library/CloudStorage/OneDrive-UniversityofPittsburgh/02-Unet_segmentation/7T_data/{study}'
modality='FLAIR'

pulldata(storinator, study, output_path, modality)
# quality_rating(storinator, study, output='.', inpath = '/Users/jinghangli/Library/CloudStorage/OneDrive-UniversityofPittsburgh/02-Unet_segmentation/7T_data/COV-IBR/TSE', mode='nii')
# quality_rating(storinator, study, output='.icv(inpath = '/Users/jinghangli/Library/CloudStorage/OneDrive-UniversityofPittsburgh/02-Unet_segmentation/7T_data/COV-IBR/TSE')', mode='dcm')
# icv(inpath = '/Users/jinghangli/Library/CloudStorage/OneDrive-UniversityofPittsburgh/02-Unet_segmentation/7T_data/COV-IBR/TSE', voxel_size=(0.375,0.375,1.5))