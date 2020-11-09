from pathlib import Path
import os 

import imageio
import numpy as np

import ipdb
from PIL import Image,ImageDraw
import h5py

class visualizer:
    """ Visualize the original CT slices and segmentation label
     
        .
    """
    def __init__(self, vol_folder = None, seg_folder = None):
        # Constants
        self.label_color =[ [0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 165, 0 ] ]
        self.hu_max = 512
        self.hu_min = -512
        self.overlay_alpha = 0.4
        self.plane = "axial"
        # File folder
        self.vol_folder = vol_folder 
        self.seg_folder = seg_folder

    def hu_to_grayscale(self, volume, hu_min, hu_max):
        # Clip at max and min values if specified
        if hu_min is not None or hu_max is not None:
            volume = np.clip(volume, hu_min, hu_max)

        # Scale to values between 0 and 1
        mxval = np.max(volume)
        mnval = np.min(volume)
        im_volume = (volume - mnval)/max(mxval - mnval, 1e-3)

        # Return values scaled to 0-255 range, but *not cast to uint8*
        # Repeat three times to make compatible with color overlay
        im_volume = (255*im_volume).astype('uint8')
        return  np.stack((im_volume, im_volume, im_volume), axis=-1)

    def class_to_color(self, segmentation, k_color, t_color):
        # initialize output to zeros
        shp = segmentation.shape
        seg_color = np.zeros((shp[0], shp[1], shp[2], 3), dtype=np.float32)

        # set output to appropriate color at each location
        seg_color[np.equal(segmentation,1)] = k_color
        seg_color[np.equal(segmentation,2)] = t_color
        return seg_color


    def overlay(self, volume_ims, segmentation_ims, segmentation, alpha):
        # Get binary array for places where an ROI lives
        segbin = np.greater(segmentation, 0)
        repeated_segbin = np.stack((segbin, segbin, segbin), axis=-1)
        # Weighted sum where there's a value to overlay
        overlayed = np.where(
            repeated_segbin,
            np.round(alpha*segmentation_ims+(1-alpha)*volume_ims).astype(np.uint8),
            np.round(volume_ims).astype(np.uint8)
        )
        return overlayed

    def save_predict_cases(self, destination):
        vol_files = sorted(os.listdir(self.vol_folder))
        seg_files = sorted(os.listdir(self.seg_folder))
        assert len(vol_files) == len(seg_files), 'Num.of data volum is not equal to Num. of mask segmentation'
        for vol_file_name, seg_file_name in zip(vol_files, seg_files   ):
            vol = h5py.File( Path(self.vol_folder)/vol_file_name, 'r' )['raw'][()]
            seg = h5py.File( Path(self.seg_folder)/seg_file_name, 'r' )['predictions'][()]
            case_dir = Path(destination)/vol_file_name[5:10]
            if case_dir.is_dir() == False: 
               case_dir.mkdir()
            print('saving prediction result:' + vol_file_name + ' & ' + seg_file_name + ' to ' + str(case_dir) )
            self.save(vol, seg, case_dir)
    def save_train_cases(self, destination):
        files = sorted(os.listdir(self.vol_folder))
        for file_name in files:
            vol = h5py.File( Path(self.vol_folder)/file_name, 'r' )['raw'][()]
            seg = h5py.File( Path(self.vol_folder)/file_name, 'r' )['label'][()]
            case_dir = Path(destination)/file_name[5:10]
            if case_dir.is_dir() == False: 
               case_dir.mkdir()
            print('saving labeled train file:' + file_name + ' to ' + str(case_dir) )
            self.save(vol, seg, case_dir)       
    def save(self, vol, seg, destination):

        plane = self.plane.lower()

        plane_opts = ["axial", "coronal", "sagittal"]
        if plane not in plane_opts:
            raise ValueError((
                "Plane \"{}\" not understood. " 
                "Must be one of the following\n\n\t{}\n"
            ).format(plane, plane_opts))

        # Prepare output location
        out_path = Path(destination)
        if not out_path.exists():
            out_path.mkdir()  

        # Load segmentation and volume
        # spacing = vol.affine
        
        # Convert to a visual format
        vol_ims = self.hu_to_grayscale(vol, self.hu_min, self.hu_max)
        seg_ims = self.class_to_color(seg, self.label_color[0], self.label_color[1])
        
        # Save individual images to disk
        if plane == plane_opts[0]:
            # Overlay the segmentation colors
            viz_ims = self.overlay(vol_ims, seg_ims, seg, self.overlay_alpha)
            for i in range(viz_ims.shape[0]):
                fpath = out_path / ("{:05d}.png".format(i))
                imageio.imwrite(str(fpath), viz_ims[i])

        if plane == plane_opts[1]:
            # I use sum here to account for both legacy (incorrect) and 
            # fixed affine matrices
            spc_ratio = np.abs(np.sum(spacing[2,:]))/np.abs(np.sum(spacing[0,:]))
            for i in range(vol_ims.shape[1]):
                fpath = out_path / ("{:05d}.png".format(i))
                vol_im = scipy.misc.imresize(
                    vol_ims[:,i,:], (
                        int(vol_ims.shape[0]*spc_ratio),
                        int(vol_ims.shape[2])
                    ), interp="bicubic"
                )
                seg_im = scipy.misc.imresize(
                    seg_ims[:,i,:], (
                        int(vol_ims.shape[0]*spc_ratio),
                        int(vol_ims.shape[2])
                    ), interp="nearest"
                )
                sim = scipy.misc.imresize(
                    seg[:,i,:], (
                        int(vol_ims.shape[0]*spc_ratio),
                        int(vol_ims.shape[2])
                    ), interp="nearest"
                )
                viz_im = overlay(vol_im, seg_im, sim, alpha)
                scipy.misc.imsave(str(fpath), viz_im)

        if plane == plane_opts[2]:
            # I use sum here to account for both legacy (incorrect) and 
            # fixed affine matrices
            spc_ratio = np.abs(np.sum(spacing[2,:]))/np.abs(np.sum(spacing[1,:]))
            for i in range(vol_ims.shape[2]):
                fpath = out_path / ("{:05d}.png".format(i))
                vol_im = scipy.misc.imresize(
                    vol_ims[:,:,i], (
                        int(vol_ims.shape[0]*spc_ratio),
                        int(vol_ims.shape[1])
                    ), interp="bicubic"
                )
                seg_im = scipy.misc.imresize(
                    seg_ims[:,:,i], (
                        int(vol_ims.shape[0]*spc_ratio),
                        int(vol_ims.shape[1])
                    ), interp="nearest"
                )
                sim = scipy.misc.imresize(
                    seg[:,:,i], (
                        int(vol_ims.shape[0]*spc_ratio),
                        int(vol_ims.shape[1])
                    ), interp="nearest"
                )
                viz_im = overlay(vol_im, seg_im, sim, alpha)
                scipy.misc.imsave(str(fpath), viz_im)
           


