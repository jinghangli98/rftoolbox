from toolbox import *

storinator='/Volumes/storinator'
study='COV-IBR'
output_path='/Users/jinghangli/Library/CloudStorage/OneDrive-UniversityofPittsburgh/02-Unet_segmentation/7T_data/COV-IBR'
modality='TSE'

pulldata(storinator, study, output_path, modality)
# quality_rating(storinator, study, output='.', inpath = '/Users/jinghangli/Library/CloudStorage/OneDrive-UniversityofPittsburgh/02-Unet_segmentation/7T_data/COV-IBR/UNI-Images', mode='nii')