import argparse
import os
import platform
from pathlib import Path

from xlib import appargs as lib_appargs
from xlib import os as lib_os

# onnxruntime==1.8.0 requires CUDA_PATH_V11_2, but 1.8.1 don't
# keep the code if they return that behaviour
# if __name__ == '__main__':
#     if platform.system() == 'Windows':
#         if 'CUDA_PATH' not in os.environ:
#             raise Exception('CUDA_PATH should be set to environ')
#         # set environ for onnxruntime
#         # os.environ['CUDA_PATH_V11_2'] = os.environ['CUDA_PATH']

import numpy as np
from xlib import cv as lib_cv2
import cv2

def checkerboard(shape):
    return np.indices(shape).sum(axis=0) % 2


def dct_np(x, ps=8, ops=8):
    """N H W"""
    xic = np.arange(ps, dtype=np.float32)[None,None,None,:]
    yic = np.arange(ps, dtype=np.float32)[None,None,:,None]
    xoc = np.arange(ops, dtype=np.float32)[None,:,None,None]
    yoc = np.arange(ops, dtype=np.float32)[:,None,None,None]

    krn = np.cos( (np.pi/ps) * (xic+0.5) * xoc) * \
          np.cos( (np.pi/ps) * (yic+0.5) * yoc)

    krn = krn.reshape(ops,ops,ps,ps)
    
    N,H,W = x.shape

    x = x.reshape(N,H//ps,ps,W//ps,ps)
    x = x.transpose((0, 1, 3, 2, 4))[...,None,None,:,:]

    x = x * krn
    x = x.sum((-2,-1)) / (ps*ps)
    return x
    
def idct_np(x, ps=8):
    """N H W K K"""
    xic = np.arange(ps, dtype=np.float32)[None,None,None,:]
    yic = np.arange(ps, dtype=np.float32)[None,None,:,None]
    xoc = np.arange(ps, dtype=np.float32)[None,:,None,None]
    yoc = np.arange(ps, dtype=np.float32)[:,None,None,None]

    krn = np.cos( (np.pi/ps) * (xic+0.5) * xoc) * \
          np.cos( (np.pi/ps) * (yic+0.5) * yoc)

    krn = krn.reshape(ps,ps,ps,ps).transpose((2,3,0,1))

    mod_x = np.full( (ps,), 0.5, np.float32)
    mod_x[0] *= 0.5
    mod_y = np.full( (ps,), 0.5, np.float32)
    mod_y[0] *= 0.5
    inv_mod = mod_y[None,None,:,None]*mod_x[None,None,None,:]
    #import code
    #code.interact(local=dict(globals(), **locals()))
    
    x = x*(ps*ps)

    N,H,W,KH,KW = x.shape
    
    
    x = x[...,None,None,:,:]
    #import code
    #code.interact(local=dict(globals(), **locals()))
    x = x * (krn*inv_mod)[None,None,None,...]
    #x = x * krn[None,None,None,...]
    x = x.sum((-2,-1)) / 4.0

    x = x.transpose((0, 1, 3, 2, 4))
    x = x.reshape(N, H*KH,W*KW)
    return x
    
    

# def cosine_resize(x, size):
#     """HWC"""
#     x = x.transpose(2,0,1)
#     C,H,W = x.shape
#     NW, NH = size
    
    
#     xic = np.arange(W, dtype=np.float32)[None,:,None]
#     yic = np.arange(H, dtype=np.float32)[:,None,None]
#     xoc = np.arange(W, dtype=np.float32)[None,None,:]
    
#     imm = np.empty( (C,H,W), dtype=x.dtype)
    
#     x = x.reshape(C, -1)
#     for j in range(H):
#         krn = np.cos( (np.pi/W) * (xic+0.5) * xoc) * \
#               np.cos( (np.pi/H) * (yic+0.5) * j)
                      
#         np.matmul(x, krn.reshape(H*W,W), out=imm[:,j,:])
    
    
#     imm[:,0,:] /= 2
#     imm[:,:,0] /= 2
#     imm /= 4
    
#     #import code
#     #code.interact(local=dict(globals(), **locals()))
    
#     xic = np.arange(W, dtype=np.float32)[None,:,None]
#     yic = np.arange(H, dtype=np.float32)[:,None,None]
#     xoc = np.arange(NW, dtype=np.float32)[None,None,:]
    
#     # mod_x = np.full( (NW,), 0.5, np.float32)
#     # mod_x[0] *= 0.5
#     # mod_y = np.full( (NH,), 0.5, np.float32)
#     # mod_y[0] *= 0.5
#     # inv_mod = mod_y[:,None,None]*mod_x[None,:,None]
    
#     out = np.zeros( (C,NH,NW), dtype=x.dtype)
#     imm = imm.reshape(C, -1)
#     for j in range(NH):
#         krn = np.cos( (np.pi/NW) * (xoc+0.5) * xic) * \
#               np.cos( (np.pi/NH) * (j  +0.5) * yic)
#         #krn *= inv_mod
        
        
#         np.matmul(imm, krn.reshape(H*W,NW), out=out[:,j,:])      
    
#     #out /= ( max(H,NH)*max(W,NW) ) / 16.0
#     out /= ( H*W ) / 16.0
#     print('max ', out.max())
    
    
    
#     #import code
#     #code.interact(local=dict(globals(), **locals()))
#     #out = out/H/W
    
#     #out /= 4
#     #out /= 256
    
#     #out /= 4096
#     #import code
#     #code.interact(local=dict(globals(), **locals()))
    
    
#     return out.transpose(1,2,0)

def cosine_resize(x, size):
    """HWC"""
    x = x.transpose(2,0,1)
    C,H,W = x.shape
    NW, NH = size
    
    
    xic = np.arange(W, dtype=np.float32)[None,:,None]
    yic = np.arange(H, dtype=np.float32)[:,None,None]
    xoc = np.arange(NW, dtype=np.float32)[None,None,:]
    
    imm = np.empty( (C,NH,NW), dtype=x.dtype)
    
    x = x.reshape(C, -1)
    for j in range(NH):
        krn = np.cos( (np.pi/W) * (xic+0.5) * xoc) * \
              np.cos( (np.pi/H) * (yic+0.5) * j)
                      
        np.matmul(x, krn.reshape(H*W,NW), out=imm[:,j,:])
    
    imm[:,0,:] /= 2
    imm[:,:,0] /= 2
    #imm /= 4
    #import code
    #code.interact(local=dict(globals(), **locals()))
    
    xic = np.arange(NW, dtype=np.float32)[None,:,None]
    yic = np.arange(NH, dtype=np.float32)[:,None,None]
    xoc = np.arange(NW, dtype=np.float32)[None,None,:]
    
    # mod_x = np.full( (NW,), 0.5, np.float32)
    # mod_x[0] *= 0.5
    # mod_y = np.full( (NH,), 0.5, np.float32)
    # mod_y[0] *= 0.5
    # inv_mod = mod_y[:,None,None]*mod_x[None,:,None]
    
    out = np.zeros( (C,NH,NW), dtype=x.dtype)
    imm = imm.reshape(C, -1)
    for j in range(NH):
        krn = np.cos( (np.pi/NW) * (xoc+0.5) * xic) * \
              np.cos( (np.pi/NH) * (j  +0.5) * yic)
        #krn *= inv_mod        
        
        np.matmul(imm, krn.reshape(NH*NW,NW), out=out[:,j,:])      
    
    out /= ( max(H,NH)*max(W,NW) ) / 4.0
    print('max ', out.max())
    
    
    
    #import code
    #code.interact(local=dict(globals(), **locals()))
    #out = out/H/W
    
    #out /= 4
    #out /= 256
    
    #out /= 4096
    #import code
    #code.interact(local=dict(globals(), **locals()))
    
    
    return out.transpose(1,2,0)


from xlib.image import ImageProcessor

    
# img = img_def = lib_cv2.imread(r'D:\DevelopPython\test\00000.png')
# img = img.astype(np.float32) / 255.0
# #img = img[None,...].transpose(0,3,1,2) / 255.0
# img = cv2.resize(img, (256,256))


# #x = np.clip(x, 0, 1)

# #x = idct_np(x[:,None,None,:,:], ps=64).transpose((1,2,0))
# #import code
# #code.interact(local=dict(globals(), **locals()))
    
    
# # x = dct_np(img.transpose(2,0,1), ps=128, ops=64)
# # def_shape = x.shape[:3]
# #new_ps = 64
# #x = idct_np(x, ps=new_ps).transpose((1,2,0))
# #x /= new_ps

# #img = checkerboard((128,128)).astype(np.float32)
# #img = cv2.resize(img, (256,256), interpolation=cv2.INTER_NEAREST)[...,None]

# for i in range(64, 0, -1):        
#     x = cosine_resize(img, (i,i))
#     x = np.clip(x, 0, 1)
    
    
    
#     ip = ImageProcessor( np.concatenate([cv2.resize(img, (i,i), interpolation=cv2.INTER_LANCZOS4), x ], axis=1) )
#     ip.fit_in(256,256, pad_to_target=True)
#     x = ip.get_image('HWC')
    
    

#     x = np.concatenate([img,x ], axis=1)
#     #cv2.imshow('', img)
#     #cv2.waitKey(0)
#     cv2.imshow('', x)
#     cv2.waitKey(0)
# import code
# code.interact(local=dict(globals(), **locals()))
    
    

#x = np.stack([ cv2.resize(i, (64,64), interpolation=cv2.INTER_LINEAR) for i in x.reshape((-1,128,128)) ], 0)
#x = x[...,:new_ps,:new_ps]
# x = x.reshape(def_shape+x.shape[-2:])

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    run_parser = subparsers.add_parser( "run", help="Run the application.")
    run_subparsers = run_parser.add_subparsers()

    def run_DeepFaceLive(args):
        userdata_path = Path(args.userdata_dir)
        lib_appargs.set_arg_bool('NO_CUDA', args.no_cuda)

        print('Running DeepFaceLive.')
        from apps.DeepFaceLive.DeepFaceLiveApp import DeepFaceLiveApp
        DeepFaceLiveApp(userdata_path=userdata_path).run()

    p = run_subparsers.add_parser('DeepFaceLive')
    p.add_argument('--userdata-dir', default=None, action=fixPathAction, help="Workspace directory.")
    p.add_argument('--no-cuda', action="store_true", default=False, help="Disable CUDA.")
    p.set_defaults(func=run_DeepFaceLive)

    dev_parser = subparsers.add_parser("dev")
    dev_subparsers = dev_parser.add_subparsers()

    def run_split_large_files(args):
        from scripts import dev
        dev.split_large_files()

    p = dev_subparsers.add_parser('split_large_files')
    p.set_defaults(func=run_split_large_files)

    def run_merge_large_files(args):
        from scripts import dev
        dev.merge_large_files(delete_parts=args.delete_parts)

    p = dev_subparsers.add_parser('merge_large_files')
    p.add_argument('--delete-parts', action="store_true", default=False)
    p.set_defaults(func=run_merge_large_files)

    def run_extract_FaceSynthetics(args):
        from scripts import dev

        inputdir_path = Path(args.input_dir)
        faceset_path = Path(args.faceset_path)

        dev.extract_FaceSynthetics(inputdir_path, faceset_path)

    p = dev_subparsers.add_parser('extract_FaceSynthetics')
    p.add_argument('--input-dir', default=None, action=fixPathAction, help="FaceSynthetics directory.")
    p.add_argument('--faceset-path', default=None, action=fixPathAction, help="output .dfs path")
    p.set_defaults(func=run_extract_FaceSynthetics)

    train_parser = subparsers.add_parser( "train", help="Train neural network.")
    train_parsers = train_parser.add_subparsers()

    def train_FaceAligner(args):
        lib_os.set_process_priority(lib_os.ProcessPriority.IDLE)
        from apps.trainers.FaceAligner.FaceAlignerTrainerApp import FaceAlignerTrainerApp
        FaceAlignerTrainerApp(workspace_path=Path(args.workspace_dir), faceset_path=Path(args.faceset_path))

    p = train_parsers.add_parser('FaceAligner')
    p.add_argument('--workspace-dir', default=None, action=fixPathAction, help="Workspace directory.")
    p.add_argument('--faceset-path', default=None, action=fixPathAction, help=".dfs path")
    p.set_defaults(func=train_FaceAligner)

    def bad_args(arguments):
        parser.print_help()
        exit(0)
    parser.set_defaults(func=bad_args)

    args = parser.parse_args()
    args.func(args)

class fixPathAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))

if __name__ == '__main__':
    main()

# import code
# code.interact(local=dict(globals(), **locals()))
