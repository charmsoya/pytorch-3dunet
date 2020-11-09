import os
from pathlib import Path

import imageio
import numpy as np
import SimpleITK as sitk
import h5py
from pytorch3dunet.augment import transforms
from pytorch3dunet.datasets.utils import ConfigDataset, calculate_stats
from pytorch3dunet.unet3d.utils import get_logger, expand_as_one_hot 
from pytorch3dunet.datasets.hdf5 import AbstractHDF5Dataset  
from pytorch3dunet.datasets.visualize import visualizer
from scipy import ndimage

import re
import yaml
import ipdb
import torch

logger = get_logger('SXTHDataset')


class SXTHDataset(AbstractHDF5Dataset):
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
        sxth_case_loader = SXTHCaseLoader( dataset_config['original_data_dir']  )
        sxth_case_loader.set_phase(phase)
        cls.original_dir = Path(dataset_config['original_data_dir'])
        cls.train_dir = Path(dataset_config[phase]['file_paths'][0])
        if cls.train_dir.is_dir() == False: 
            cls.train_dir.mkdir( parents = True )

        # exist already preprocessed data?
        files = os.listdir(cls.train_dir)
        need_processing = True 

        if need_processing:
            # deleted existed files
            for f in files:
                    os.remove(Path(cls.train_dir)/f)

            # We use the following step to carry out the pre-process operation
            # STEP 1: loop all cases for their three axis' voxel space, determin a suitable voxel space
            # STEP 2: for each case, 
            #   a. remove dark borders (air) of each slice, crop each slice to human body
            #   b. resample each slice according to the above determined suitable voxel space
            #   c. only keep the slices from the first  to the last slice with mask
            # STEP 3: loop all cases for a suitable path size   
    
            # STEP 1:
            #case_info_dict = {}  
            #rt_info_file = open("runtime_info.yml",'w',encoding='utf-8')
            seg_info = []
            foreground_vol = np.array([])
            one_case_visualizer = visualizer()

            for vol,seg,img_info in sxth_case_loader: 

                spacing = img_info['Spacing']
                case_id = img_info['Case_id']
                #if phase != 'test':
                #    crop_border = cls.crop_image_only_outside(cls, vol, -1000)
                #    seg = seg[:,crop_border[0]:crop_border[1], crop_border[2]:crop_border[3]]
                #    vol = vol[:,crop_border[0]:crop_border[1], crop_border[2]:crop_border[3]]

                # resample (or re-slice) for isotropic voxel
                new_spacing = [2, 0.8, 0.8]
                resize_factor = np.array(spacing) / new_spacing
                new_real_shape = vol.shape * resize_factor
                new_shape = np.round(new_real_shape)   
                real_resize_factor = new_shape / vol.shape
                new_spacing = spacing / real_resize_factor

                #case_info = { 'case_'+str(case_id): np.array( spacing ) }
                #case_info_dict.update( case_info )
                resize_factor = np.array(spacing) / new_spacing
                print(f'\t shape after crop: {vol.shape}')
                print(f'\t resampling image voxel spacing from {spacing} to {new_spacing} ...')
                vol = ndimage.zoom(vol, resize_factor, order=0 )
                print(f'\t shape after resample: {vol.shape}') 
                
                print('\t resampling segmetation voxel ...')
                nclass = 2 
                seg_resampled = np.zeros( ( nclass, vol.shape[0], vol.shape[1], vol.shape[2] ), dtype='int32' )
                if phase != 'test':
                    seg_expanded = torch.from_numpy( np.expand_dims(seg.astype('int64'),0) )
                    seg_onehot = expand_as_one_hot( seg_expanded, nclass).squeeze(0).numpy()
                    for ilabel in range( nclass):
                        seg_resampled[ilabel,] = ndimage.zoom( seg_onehot[ilabel,], resize_factor, order= 0)
                    seg = np.argmax( seg_resampled, axis = 0)
		    # collect all fore-ground voxels for further compute  max/min/mean/std value
                    foreground_vol = np.concatenate((foreground_vol, vol[seg>0]),axis = 0)
                #one_case_visualizer.save(vol,seg,'/mnt/sda2/sxth_processed/img') 
                print(f'\t after removing slices without masks, image & segmentation data shape is {vol.shape} ')
                print('\t done.')

                # store as a h5d file
                f =  h5py.File(cls.train_dir/'case_{:05d}.h5'.format(case_id),'w')
                f.create_dataset('raw', data = vol)
                if phase != 'test':
                    f.create_dataset('label', data = seg)
                    f.create_dataset('weight', data = np.ones( seg.shape ) )
                f.close()
                seg_info.append(vol.shape)
            
        # save data to disk
        #save_img = visualizer(vol_folder = dataset_config.get(phase)['file_paths'][0]) 
        #save_img.save_train_cases( '/mnt/sda2/sxth_processed/img/')
        #ipdb.set_trace()
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
        #return 40, 250, 110, 31 
        return 0, 600, None, None
    
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
    
    def crop_image_only_outside(self, img, tol):
	# img is 2D or 3D image data
	# tol  is tolerance
        mask = img>tol
        if img.ndim==3:
            mask = mask.all(0)
        m,n = mask.shape
        mask0,mask1 = mask.any(0),mask.any(1)
        col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
        row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()
        return row_start,row_end,col_start,col_end

    def plot_3d(self, image, threshold=-300):
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


class SXTHCaseLoader():
    def __init__(self, data_folder):
        self.img_folder = Path(data_folder)/'T2_case1-200'
        self.seg_folder = Path(data_folder)/'ROI_T2'
        self.reader = sitk.ImageSeriesReader()
        if not self.img_folder.exists() or not self.seg_folder.exists():
            raise IOError( "IMG or SEG path, {}, could not be resolved".format(str(img_folder)) )
        self.cases_img = os.listdir( self.img_folder )
        self.cases_seg = os.listdir( self.seg_folder )
        assert len(self.cases_img)==len(self.cases_seg)
        self.n_train = round( len(self.cases_img) * 0.8 )
        self.n_val = len(self.cases_img) - self.n_train
           
    def __iter__(self):
        assert self.phase == 'train' or self.phase == 'val'
        self.n_cases = self.n_train if self.phase == 'train' else self.n_val
        self.i_case = 0
        return self

    def __next__(self):
        print( f'current case: {self.i_case}/{self.n_cases}' )
        if self.i_case >= self.n_cases: 
            raise StopIteration 
        img_names = self.reader.GetGDCMSeriesFileNames( str(self.img_path[ self.i_case ] ) )

        self.reader.SetFileNames( img_names )
        image = self.reader.Execute( )
        image_array = sitk.GetArrayFromImage( image )
        info_spacing = image.GetSpacing()
        image_info = {'Case_id': self.i_case,'Spacing': (info_spacing[2], info_spacing[0], info_spacing[1]) }

        seg = sitk.ReadImage( str( self.seg_path[ self.i_case ] ) )
        seg_array = sitk.GetArrayFromImage( seg )

        self.i_case+=1
        return image_array, seg_array, image_info

    def set_phase( self, phase ):
        assert phase == 'train' or phase == 'val'
        self.phase = phase
        self.img_path = []
        self.seg_path = []
        cases_img_selected = self.cases_img[0:self.n_train] if self.phase == 'train' else self.cases_img[self.n_train:]
        for case_name in cases_img_selected:
            idx = re.findall('\d*.?\d', case_name)[0]
            sub_folder = os.listdir( self.img_folder/case_name )[0]
            self.img_path.append( self.img_folder/case_name/sub_folder )
            self.seg_path.append( self.seg_folder/(idx+'hrT2a_Merge.nii') )


        

