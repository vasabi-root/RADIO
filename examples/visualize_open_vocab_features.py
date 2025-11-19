# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import argparse
from collections import defaultdict
from functools import partial
import gc
import math
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random
from tqdm import tqdm
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.io import read_video, write_video
import torchvision.transforms.functional as F

from einops import rearrange

from datasets import load_dataset_builder, load_dataset
from datasets.distributed import split_dataset_by_node

from RADIO.examples.common import rank_print, load_model, get_standard_transform, collate
from RADIO.radio.input_conditioner import InputConditioner
from .visualize_features import get_robust_pca, get_pca_map

def read_frames_from_dir(dir):
    dir = Path(dir)
    frame_names = [name for name in sorted(os.listdir(dir)) if name[-3:].lower() in ['jpg', 'png', 'jpeg']]
    if len(frame_names) == 0:
        raise FileNotFoundError(f'There are no supported frames in a dir {dir!r}')
    
    return torch.stack([F.pil_to_tensor(Image.open(dir/name)) for name in frame_names])
    

def cat_frames(frames: torch.Tensor, direction='vertiacal'):
    '''
    Concatenates frames by direction.  
    `direction` can be 'vertical' or 'horizontal'.  
    Frames are concatenated starting from the frames[0] and up/left direction
    '''
    # assert a.shape == b.shape
    match direction:
        case 'horizontal':
            return rearrange(frames, 'n h w c -> h (n w) c').float().cpu()
            # return torch.cat(frames, dim=1)
        case 'vertical':
            return rearrange(frames, 'n h w c -> (n h) w c').float().cpu()
        case _:
            raise Exception(f'Unsupported direction: {direction}')
        
def cat_frames_into_grid(frames, rows, cols):
    assert rows*cols == len(frames)
    
    grid = torch.Tensor()
    rows_list = []
    for i in range(rows):
        row = frames[i*cols:(i+1)*cols]
        rows_list.append(cat_frames(row, 'horizontal'))
    
    return cat_frames(rows_list, 'vertical')

def calc_grid(frame_shape, frame_num) -> Tuple[int]:
    '''Returns (rows, cols)'''
    height = frame_shape[-2]
    width = frame_shape[-1]
    
    if height > width:
        return 1, frame_num
    if frame_num < 4:
        return frame_num, 1
    return frame_num // 2, 2

def add_title_to_frame(frame: torch.Tensor, title: str):
    img = F.to_pil_image(frame.permute(2, 0, 1)/255, 'RGB')
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 16)
    d = ImageDraw.Draw(img)
    height = int(frame.shape[0])
    width = int(frame.shape[1])
    d.text((width//2, 0), title, anchor='ma', font=font, fill='white')
    return F.pil_to_tensor(img).permute(1, 2, 0)

@torch.no_grad
def get_query_map(adaptor: torch.nn.Module, query: List[str], features: torch.Tensor, cos_threshold=0.03) -> torch.Tensor:
    tokenizer = adaptor.tokenizer

    tokens = tokenizer([query[0]])
    tokens['input_ids'] = tokens['input_ids'].cuda()

    text_features = adaptor.encode_text(tokens).cuda()
    
    similarity = torch.cosine_similarity(features.cuda(), text_features, dim=-1).unsqueeze(-1)
    # positive = similarity-(-1)
    # similarity = positive / 2
    similarity = torch.where(similarity > cos_threshold, 1.0, 0.0)
    return similarity.expand(*similarity.shape[:3], 3)


@torch.inference_mode()
def main(rank: int = 0, world_size: int = 1):
    '''
    Computes the PCA features for every frame in a supplied video and renders them into a new video.
    '''

    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    cv2.setNumThreads(1)

    device = torch.device('cuda', local_rank)
    parser = argparse.ArgumentParser(description='Visual Model Features in Video')
    parser.add_argument('-v', '--model-version', default='c-radio_v3-h',
                        help='Which radio model to load.'
    )
    parser.add_argument('--video', type=str, required=True,
                        help='Path to the video. Can be a directory containing frames (in this case, set --fps key also)')
    parser.add_argument('--fps', type=int, default=None, 
                        help='Desired FPS rate of a video. Requred if video is a directory with frames. ')
    parser.add_argument('--query', type=str, required=True,
                        help='Query to the image. Objects should be separated by \'; \' delimiter')
    parser.add_argument('--output', type=str, required=True,
                        help='Where to store the output video')
    parser.add_argument('-r', '--resolution', nargs='+', type=int, default=None,
                        help='The input image resolution.'
                             ' If one value is specified, the shortest dimension is resized to this.'
                             ' If two, the image is center cropped.'
                             ' If not specified, center cropped 378px is used.'
                             ' Default: The RADIO model\'s preferred resolution.'
    )
    parser.add_argument('--max-dim', default=False, action='store_true', help='Resize the max dimension to the specified resolution')
    parser.add_argument('--resize-multiple', type=int, default=16,
                        help='Resize images with dimensions a multiple of this value.'
                             ' This should be equal to the patch size of a ViT (e.g. RADIOv1)'
    )
    parser.add_argument('--vitdet-window-size', default=None, type=int, help='Enable ViTDet at the specific window size')
    parser.add_argument('--patch-size', default=16, type=int, help='The model patch size')
    parser.add_argument('--torchhub-repo',
                        help="Path to the Torchhub repo", default="NVlabs/RADIO"
    )
    parser.add_argument('--side-by-side', default=False, action='store_true',
                        help='Render the original frame and the PCA frame side-by-side')
    parser.add_argument('--audio', default=False, action='store_true',
                        help='Encode the audio in the output video')
    parser.add_argument('--video-codec', default='libx264', type=str, help='The video codec to use')
    parser.add_argument('--batch-size', type=int, default=16, help='The processing batch size')
    parser.add_argument('--force-reload', default=False, action='store_true', help='Reload the torch.hub codebase')

    args, _ = parser.parse_known_args()

    adaptor_names = ['siglip2-g']
    rank_print(f'Loading model: "{args.model_version}", ViTDet: {args.vitdet_window_size}, Adaptor: "{adaptor_names}", Resolution: {args.resolution}, Max: {args.max_dim}...')
    model, preprocessor, info = load_model(args.model_version, vitdet_window_size=args.vitdet_window_size, adaptor_names=adaptor_names,
                                           torchhub_repo=args.torchhub_repo, force_reload=args.force_reload)
    model.to(device=device).eval()
    if isinstance(preprocessor, nn.Module):
        preprocessor.to(device).eval()
    rank_print('Done')
    
    adaptor = model.adaptors['siglip2-g'] # siglip2-g
    query = [args.query] #.split('; ')

    if args.resolution is None:
        args.resolution = (model.preferred_resolution.height, model.preferred_resolution.width)

    patch_size = getattr(model, 'patch_size', None) or args.patch_size

    if args.resize_multiple is None:
        args.resize_multiple = getattr(model, 'min_resolution_step', patch_size)

    transform = get_standard_transform(args.resolution, args.resize_multiple, max_dim=args.max_dim,
                                       pad_mean=preprocessor.norm_mean if isinstance(preprocessor, InputConditioner) else None,)

    if os.path.isfile(args.video):
        input_video = read_video(args.video, output_format='TCHW')
        input_frames = input_video[0]
        fps = input_video[2]['video_fps']
    elif os.path.isdir(args.video):
        input_frames = read_frames_from_dir(args.video)
        fps = args.fps

    all_features = []
    tx_frames = []

    batch_size = args.batch_size
    
    num_frames = len(adaptor_names)+1
    if args.side_by_side: num_frames += 1
    
    rows, cols = calc_grid(input_frames[0].shape, num_frames)
    
    features_map = {adaptor_name: [] for adaptor_name in adaptor_names}
    features_map[query[0]] = []
    
    for b in tqdm(range(0, len(input_frames), batch_size)):
        curr_frames = input_frames[b:b+batch_size]
        curr_frames = transform(curr_frames)

        tx_frames.append(curr_frames)

        curr_frames = curr_frames.cuda()
        
        num_rows = curr_frames.shape[-2] // patch_size
        num_cols = curr_frames.shape[-1] // patch_size
        if b >= len(input_frames) - batch_size:
            pass

        with torch.autocast(device.type, dtype=torch.bfloat16):
            p_frames = preprocessor(curr_frames)

            output = model(p_frames)
            for adaptor_name in adaptor_names:
                if adaptor_name:
                    features = output[adaptor_name].features
                else:
                    features = output[1]

                features = rearrange(features, 'b (h w) c -> b h w c', h=num_rows, w=num_cols).float()
                query_map = get_query_map(adaptor, query, features)
                
                features_map[adaptor_name].append(features.cpu())
                features_map[query[0]].append(query_map.cpu())
            
            
    tx_frames = torch.cat(tx_frames)
    
    colored_frames = []
    if args.side_by_side:
        original_frames = []
        for frame in tx_frames.permute(0, 2, 3, 1) * 255:
            original_frames.append(add_title_to_frame(frame, 'original'))
        colored_frames.append(torch.stack(original_frames))
        
    for adaptor_name, all_features in features_map.items():
        all_features = torch.cat(all_features)
        
        num_keyframes = 30
        kf_stride = max(all_features.shape[0] // num_keyframes, 1)

        # We'll use this to compute the PCA
        sub_features = all_features[::kf_stride]
        pca_stats = get_robust_pca(sub_features.flatten(0, 2))

        output_frames = []
        for raw_frame, features in zip(tx_frames, all_features):
            img_size = raw_frame.shape[-2:]
            if all_features.shape[-1] != 3:
                pca_features = torch.from_numpy(get_pca_map(features, img_size, pca_stats=pca_stats, interpolation='bilinear'))
            else:
                pca_features = torch.nn.functional.interpolate(
                    features[None].permute(0, 3, 1, 2),
                    size=img_size,
                    mode='bilinear',
                ).permute(0, 2, 3, 1).squeeze(0)
            pca_features = pca_features.mul_(255).byte()
            titled = add_title_to_frame(pca_features, adaptor_name)
            output_frames.append(titled)

        output_frames = torch.stack(output_frames)
        colored_frames.append(output_frames)
    
    del features_map
    gc.collect()
    torch.cuda.empty_cache()
    
    colored_frames = torch.stack(colored_frames, dim=1)
    grid_frames = []
    for frames in colored_frames:
        grid_frames.append(cat_frames_into_grid(frames, rows, cols))
    
    del colored_frames
    gc.collect()
    torch.cuda.empty_cache()
    
    grid_frames = torch.stack(grid_frames, dim=0)
    
    extra_args = dict()
    if args.audio:
        extra_args.update(dict(
            audio_array=input_video[1],
            audio_fps=input_video[2]['audio_fps'],
        ))

    dirname = os.path.dirname(args.output)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    options = {
        'crf': '18',  # Lower CRF for better quality
        'preset': 'slow',  # Use a slower preset for better compression efficiency
        'profile': 'high',  # Use high profile for advanced features
    }
    write_video(args.output, grid_frames, fps, video_codec=args.video_codec, options=options, **extra_args)


if __name__ == '__main__':
    main()
