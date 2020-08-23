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
from pytorch3dunet.datasets.visualize import visualizer
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
        
        if (phase == 'train' and len(files) == 200 ) or \
           (phase == 'val' and len(files) == 10) or \
           (phase == 'test' and len(files) == 90):
             need_processing = False

        if phase == 'train':
            id_range = range(0, 200)
        if phase == 'val':
            id_range = range(200, 210)
        if phase == 'test':
            id_range = range(210, 300)
        
        if need_processing:
            # deleted existed files
            for f in files:
                    os.remove(Path(cls.train_dir)/f)
            #  preprocesing each patient's data
            save_img = visualizer()
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
                new_spacing = [2,2,2]

                resize_factor = np.array(spacing) / new_spacing
                new_real_shape = vol.shape * resize_factor
                new_shape = np.round(new_real_shape)   
                real_resize_factor = new_shape / vol.shape
                new_spacing = spacing / real_resize_factor


                resize_factor = np.array(spacing) / new_spacing
                print(f'\t original shape: {vol.shape}')
                print(f'\t resample Z spacing from {spacing} to {new_spacing}') 
                vol = ndimage.zoom(vol, resize_factor, order=1 )
                seg = ndimage.zoom(seg, resize_factor, order=1 )

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

                # save data to disk
                # save_img.save( vol, seg, '/mnt/sda2/kits19_processed/img/'+str(case_id))
                print(f'\t new shape: {vol.shape}')
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
        return -1000, 400, None, None 
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
   

    def plot_3d(image, threshold=-300):
        # Position the scan upright,
        # so the head of the patient would be at the top facing the camera
        p = image.transpose(2,1,0)  #将扫描件竖直放置
        verts, faces = measure.marching_cubes(p, threshold) #Liner推进立方体算法来查找3D体积数据中的曲面。
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        mesh = Poly3DCollection(verts[faces], alpha=0.1)  #创建3Dpoly
        face_color = [0.5, 0.5, 1]
        mesh.set_facecolor(face_color)  #设置颜色
        ax.add_collection3d(mesh)
        ax.set_xlim(0, p.shape[0])
        ax.set_ylim(0, p.shape[1])
        ax.set_zlim(0, p.shape[2])
        plt.show()
