import os
from pathlib import Path

import imageio
import numpy as np
import nibabel as nib
import h5py
from pytorch3dunet.augment import transforms
from pytorch3dunet.datasets.utils import ConfigDataset, calculate_stats
from pytorch3dunet.unet3d.utils import get_logger
from pytorch3dunet.datasets.hdf5 import AbstractHDF5Dataset  

from scipy import ndimage

import ipdb

logger = get_logger('Kits19Dataset')


class Kits19Dataset(AbstractHDF5Dataset):
    def __init__(self, file_path, phase, slice_builder_config, transformer_config, mirror_padding=(16, 32, 32),
                 raw_internal_path='raw', label_internal_path='label', weight_internal_path=None):
        super().__init__(file_path=file_path,
                         phase=phase,
                         slice_builder_config=slice_builder_config,
                         transformer_config=transformer_config,
                         mirror_padding=mirror_padding,
                         raw_internal_path=raw_internal_path,
                         label_internal_path=label_internal_path,
                         weight_internal_path=weight_internal_path)
                
    @classmethod
    def create_datasets(cls, dataset_config, phase):
        # prerocess each patient case
        
        # exist folder for storing preprocessed files?
        cls.original_dir = Path(dataset_config['original_data_dir'])
        cls.train_dir = Path(dataset_config[phase]['file_paths'][0])
        if cls.train_dir.is_dir() == False: 
            cls.train_dir.mkdir()

        # exist already preprocessed data?
        files = os.listdir(cls.train_dir)
        need_processing = True 

        if ( (phase == 'train' and len(files) == 200 ) or 
           (phase == 'val' and len(files) == 10) ):
            need_processing = False

        if phase == 'train':
            id_range = range(0, 200)
        if phase == 'val':
            id_range = range(200, 210)

        if need_processing:
            # deleted existed files
            for f in files:
                    os.remove(Path(cls.train_dir)/f)
            #  preprocesing each patient's data
            for case_id in id_range:
                print(f'Preprocessing {phase} case {case_id+1}/{len(id_range)},')
                vol, seg = cls.load_case(cls, case_id)
                #spacing = vol.affine
                spacing = vol.header.get_zooms()
                print('\t reading data ...,')
                vol = vol.get_data()
                seg = seg.get_data()
                seg = seg.astype(np.int32)
                # resample (or re-slice) for isotropic voxel
                new_spacing = [spacing[1],spacing[1],spacing[2]]
                resize_factor = np.array(spacing) / new_spacing
                print(f'\t resample Z spacing from {spacing[0]} to {spacing[1]}') 
                vol = ndimage.zoom(vol, resize_factor, order=2, mode='nearest')
                seg = ndimage.zoom(seg, resize_factor, order=2, mode='nearest')

                # 3D ROI, CT slices only contain masks will be preserved
                ior_z = []
                for i in range( seg.shape[0]):
                  unique_list = np.unique(seg[i,:,:])
                  assert( all( [ element in (0,1,2) for element in unique_list] ))
                  if len(unique_list) > 1 and len(unique_list)<=3:
                         ior_z.append(i)
                ior_zmin = min(ior_z)
                ior_zmax = max(ior_z)
                vol = vol[ior_zmin:ior_zmax+1,:,:]
                seg = seg[ior_zmin:ior_zmax+1,:,:]
                print('\t done.')

                # store as a h5d file
                f =  h5py.File(cls.train_dir/'case_{:05d}.h5'.format(case_id),'w')
                f.create_dataset('raw', data = vol)
                f.create_dataset('label', data = seg)
                f.close()
            
        return super().create_datasets( dataset_config, phase )
        
    def _load_files(dir, expand_dims):
        files_data = []
        paths = []
        for file in os.listdir(dir):
            path = os.path.join(dir, file)
            img = np.asarray(imageio.imread(path))
            if expand_dims:
                img = np.expand_dims(img, axis=0)

            files_data.append(img)
            paths.append(path)

        return files_data, paths 


    @staticmethod 
    def fetch_datasets(input_file_h5, internal_paths):
        return [input_file_h5[internal_path] for internal_path in internal_paths]

   
   
    @staticmethod
    def create_h5_file(file_path, internal_paths):
        return h5py.File(file_path, 'r')
    
    def ds_stats(self):
        # Do not calculate stats on the whole stacks when using lazy loader,
        # they min, max, mean, std should be provided in the config
        logger.info(
            'Using LazyHDF5Dataset. Make sure that the min/max/mean/std values are provided in the loaders config')
        return -800, 600, 0, 100
        #return None, None, None, None
    
    def load_case(self,cid):
        # Resolve location where data should be living
        # data_path = Path(__file__).parent.parent / "data"
        data_path = Path(self.original_dir)
        if not data_path.exists():
            raise IOError(
                    "Data path, {}, could not be resolved".format(str(data_path))
                    )

        # Get case_id from provided cid
        try:
            cid = int(cid)
            case_id = "case_{:05d}".format(cid)
        except ValueError:
            case_id = cid

        # Make sure that case_id exists under the data_path
        case_path = data_path / case_id
        if not case_path.exists():
            raise ValueError(
                    "Case could not be found \"{}\"".format(case_path.name)
                    )
        if (case_path/"imaging.nii.gz").exists() and (case_path/"segmentation.nii.gz").exists():
            vol = nib.load(str(case_path / "imaging.nii.gz"))
            seg = nib.load(str(case_path / "segmentation.nii.gz"))
            return vol, seg
        if (case_path/'imaging.nii.gz').exists() and not (case_path/'segmentation.nii.gz').exists():
            vol = nib.load(str(case_path / "imaging.nii.gz"))
            return vol
    
