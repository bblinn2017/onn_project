import os
from tqdm.auto import tqdm
import numpy as np
from opt import train_config_parser as config_parser
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib

import matplotlib.pyplot as plt
from PIL import Image as im

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import datetime

from model import get_model
from dataloader import get_dataset, collate_fn

import shutil

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('>>> Using', device)

def sigfig(val, num):

    if val < 1e-4: return 0.

    try:
        exp = int(np.floor(np.log(val) / np.log(10.)))
        val_exp = val * pow(10,-exp)
        out = '{:.{prec}}'.format(val_exp, prec=num)
    except:
        return 'N/A'
    return f'{out}e{exp}'

def get_PSNR(original, compressed):
    mse = np.mean(((original - compressed)*255) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def multi_gpu_collate_fn(batch, device):
    for key in batch.keys():
        if type(batch[key]) == torch.Tensor:
            batch[key] = batch[key].to(device)
    return batch

def run_iter(model,optimizer,batch):
    #----- Train -----#
    model.train()

    result = model(batch) # forward pass

    loss = result
    if loss.requires_grad:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_np = loss.detach().cpu().numpy()
    del loss

    return loss_np

def train(args):
    train_dataset = get_dataset()(args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    if args.load_checkpoint != None:
        spec = importlib.util.spec_from_file_location("model", f'{args.load_checkpoint}/model.py')
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        model = nn.DataParallel(foo.OmniNet(args, device).cuda())
    else:
        model = nn.DataParallel(get_model()(args, device).cuda())
    if args.load_checkpoint != None:
        # Load pre-trained model to finetune
        print(f">>> Load checkpoint: {args.load_checkpoint}")
        model.load_state_dict(torch.load(f'{args.load_checkpoint}/checkpoints')["model_state_dict"])
    model.to(device)
    
    if args.load_checkpoint == None:
        logfolder = f'{args.base_dir}/{args.exp_name}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.load_checkpoint.split("-")[0]}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    summary_writer = SummaryWriter(logfolder)

    # TODO
    # grad_vars = [
    #             {'params': model.module.render_net.parameters(), 'lr': args.lr_nerf_renderer}
    #             ]
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.75)
     
    pbar = tqdm(range(args.iter), desc=">>> Training", position=0, leave=True)
    for epoch in pbar:

        niters = int(np.ceil(float(len(train_dataset) / args.batch_size)))
        pbar_iter = tqdm(total=niters)

        for step, batch in enumerate(train_dataloader):
            loss = run_iter(model,optimizer,batch)

            pbar_iter.set_description(
                sigfig(loss,5)
            )

            summary_writer.add_scalar('train/loss', loss, global_step=epoch * len(train_dataset) + step * args.batch_size)

            #----- Save checkpoint -----# 
            if step % args.save_checkpoint_freq == 0 or step == len(train_dataset) - 1:
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            }, os.path.join(logfolder, 'checkpoints'))
            # scheduler.step(loss)

            pbar_iter.update(1)
            torch.cuda.empty_cache()

if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')

    args = config_parser()
    train(args)
    print('>>> Completed')
