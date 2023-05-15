import json
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import os.path as osp

import cv2
from PIL import Image
import numpy as np
import numpy.linalg as LA
import torch
from torch.utils.data import Dataset
from multiprocessing.pool import ThreadPool

def get_dataset():
    return DiffuseData

def imread_rgb(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def load_vid(K_path, Rt_path, image_path, depth_path):

    # K
    K = np.genfromtxt(K_path).astype(np.float32)

    # Rt
    param = np.genfromtxt(Rt_path)
    w2c = np.eye(4)
    w2c[:3, :4] = param[:3, :4]
    w2c = w2c.astype(np.float32)

    # Image depths, mask
    image = np.array(Image.open(image_path).convert('RGBA')) / 255.
    image = image[:, :, :3]

    minrange = 0.5
    maxrange = 3.5

    depths = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)[:, :, 0]
    depths = np.minimum(depths, maxrange)

    mask = (depths != maxrange).astype(np.bool)
    depths = depths[mask].astype(np.float32).reshape(-1,1)
    
    image = image[mask].astype(np.float32)

    return K, w2c, image, mask, depths

TRAIN_SPLIT = 0.9
class DiffuseData(Dataset):
    def __init__(self, args):
        self.input_dir = args.input_dir
        self.diffuse_dir = args.diffuse_dir
        self.datalist = args.datalist
        self.batch_size = args.batch_size

        with open(self.datalist, 'r') as f:
            all_subjects = json.load(f)
        subjects = sorted(list(all_subjects.keys()))
        print(subjects)
        exit()

        train_split = int(len(subjects)*TRAIN_SPLIT)
        subjects = subjects[:train_split] if mode == 'train' else subjects[train_split:]
       
        self.filelist = []
        for subject in subjects:
            for num in all_subjects[subject]:
                for animation in all_subjects[subject][num]:
                    frame_id = np.random.choice(all_subjects[subject][num][animation])
                    self.filelist.append((subject,num,animation,frame_id))
        self.size = len(self.filelist)
        
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        subject, num, anim_dir, frame_id = self.filelist[idx]

        rp_dir = f'rp_{subject}_rigged_{num}'
        frame_dir = f'frame_{frame_id:03d}'

        # vids = [4,6,8,10]
        vids = list(range(self.num_ref_views))

        Ks = []
        Rts = []
        images = []
        masks = []
        depths = []

        args = []
        for i in range(len(vids)):
            vid = vids[i]
            K_path = osp.join(self.data_dir, f'camera/cam_K.txt')
            Rt_path = osp.join(self.data_dir, f'camera/cam_RT_{vid+1:03d}.txt')
            image_path = osp.join(self.data_dir, f'image/{rp_dir}/{anim_dir}/{frame_dir}/color_{vid+1:03d}.png')
            depth_path = osp.join(self.data_dir, f'depth/{rp_dir}/{anim_dir}/{frame_dir}/depth_{vid+1:03d}.exr')

            args.append((K_path, Rt_path, image_path, depth_path))

        pool = ThreadPool()
        results = pool.starmap(load_vid, args)
        pool.close()
        pool.join()

        geo_path = osp.join(self.data_dir, f'geometry/{rp_dir}/{anim_dir}/{frame_dir}.obj')
        geo_file = open(geo_path,"r")
        geo = trimesh.exchange.obj.load_obj(geo_file)

        # Ref data
        ref_data = results
        for (K,w2c,image,mask,depth) in ref_data:
            Ks.append(K)
            Rts.append(w2c)
            images.append(image) # range: [-1, 1]
            masks.append(mask)
            depths.append(depth)
            
        ref_K = np.array(Ks)
        ref_Rts = np.array(Rts)
        ref_images = np.concatenate(images,0)
        ref_depths = np.concatenate(depths,0)
        ref_masks = np.array(masks)

        # verts = np.array(geo['vertices']).astype(np.float32)
        # normals = np.array(geo['vertex_normals']).astype(np.float32)

        # mesh = trimesh.Trimesh(geo['vertices'],geo['faces'],vertex_normals=geo['vertex_normals'])
        # n_points = 100000
        # vertices, faces = mesh.sample(n_points, return_index=True)
        # normals = mesh.face_normals[faces]

        vertices = np.array(geo['vertices']).astype(np.float32)
        faces = np.array(geo['faces']).astype(np.int32)
        
        data = {
            'ref_K': ref_K,
            'ref_Rts': ref_Rts,
            'ref_images': ref_images,
            'ref_masks': ref_masks,
            'ref_depths': ref_depths,
            'verts': vertices,
            'faces': faces
        }

        return data

    def log(self, *args, **kwargs):
        if self.logger is not None:
            self.logger.log(*args, **kwargs)