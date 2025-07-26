import os
import time
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather, get_dice_score,get_dice_selected_classes,remap_labels,resample_3d,ORGAN_NAME,dice
import gc
import torch.nn.functional as F
import matplotlib.pyplot as plt



def dice_similarity(pred_a, pred_b, eps=1e-8):
    batch_size = pred_a.shape[0]

    pred_a = F.softmax(pred_a, dim=1)  # (B, C, D, H, W)
    pred_b = F.softmax(pred_b, dim=1)

    intersection = torch.sum(pred_a * pred_b, dim=(1, 2, 3, 4))
    union = torch.sum(pred_a, dim=(1, 2, 3, 4)) + torch.sum(pred_b, dim=(1, 2, 3, 4))

    dice = (2.0 * intersection + eps) / (union + eps)

    dice_matrix = dice.unsqueeze(0).repeat(batch_size, 1)

    return dice_matrix


def scr_loss(CT_out, MRI_out):
    batch_size = CT_out.shape[0]
    z = torch.cat((CT_out, MRI_out), dim=0)
    sim_matrix = dice_similarity(z, z)
    sim_pos_1 = torch.diag(sim_matrix, batch_size)
    sim_pos_2 = torch.diag(sim_matrix, -batch_size)

    sim_pos = torch.cat((sim_pos_1, sim_pos_2), dim=0).reshape(2 * batch_size, 1)

    mask = torch.ones((2 * batch_size, 2 * batch_size), dtype=bool, device=CT_out.device)
    mask.fill_diagonal_(0)
    for i in range(batch_size):
        mask[i, batch_size + i] = 0
        mask[batch_size + i, i] = 0

    sim_neg = sim_matrix[mask].reshape(2 * batch_size, -1)

    exp_pos = torch.exp(sim_pos)
    exp_neg = torch.sum(torch.exp(sim_neg), dim=1, keepdim=True)
    loss = -torch.log(exp_pos / (exp_pos + exp_neg + 1e-8))

    return loss.mean()


def my_sup_loss(loss_func1,args, CT_seg_out, MRI_seg_out, CT_seg, MRI_seg, CT_img_F_ds, MRI_img_F_ds, label_id):

    sup_losses = 0.0
    sar_losses = 0.0
    scr_losses = 0.0
    label_id_cnt = 0
    unlabel_id_cnt = 0

    for i in range(len(label_id)):
        if label_id[i] == 1:
            label_id_cnt += 1
            CT_seg_out = CT_seg_out.to(torch.float32)
            MRI_seg_out = MRI_seg_out.to(torch.float32)
            CT_sup_loss = loss_func1(CT_seg_out, CT_seg)
            MRI_sup_loss = loss_func1(MRI_seg_out, MRI_seg)
            sup_loss = (CT_sup_loss + MRI_sup_loss) / 2
            sup_losses += sup_loss
            sar_loss = my_sar_loss(CT_img_F_ds[i], MRI_img_F_ds[i])
            sar_losses += sar_loss
        else:
            unlabel_id_cnt += 1
            scr_loss_val = scr_loss(CT_seg_out, MRI_seg_out)
            scr_losses += scr_loss_val

    sup_losses = sup_losses / max(1, label_id_cnt)
    sar_losses = sar_losses / max(1, label_id_cnt)
    scr_losses = scr_losses / max(1, unlabel_id_cnt)

    return sup_losses, sar_losses,scr_losses

def my_sar_loss(CT_out, MRI_out):

    channel_losses = 0.0
    num_channels = CT_out.shape[0]
    for c in range(num_channels):
        ct_channel = CT_out[c].flatten()
        mri_channel = MRI_out[c].flatten()
        ct_channel = F.normalize(ct_channel, dim=0)
        mri_channel = F.normalize(mri_channel, dim=0)
        cosine_similarity = torch.dot(ct_channel, mri_channel)
        channel_loss = -torch.log(cosine_similarity + 1e-8)
        channel_losses += channel_loss
    return channel_losses / num_channels


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def get_current_consistency_weight(args, cons_ramp_type, epoch):
    if cons_ramp_type == 'sig_ram':
        if epoch < args.fusion_start_epoch:
            return args.smooth_nr
        else:
            return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    elif cons_ramp_type == 'lin_ram':
        return args.consistency * linear_rampup(epoch, args.consistency_rampup)
    elif cons_ramp_type == 'cos_ram':
        return args.consistency * cosine_rampdown(epoch, args.consistency_rampup)
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func1, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):

        CT_image, CT_seg, MRI_image, MRI_seg, label_id = batch_data["CT_image"], batch_data["CT_seg"], batch_data[
            "MRI_image"], batch_data["MRI_seg"], batch_data["label"]
        CT_image, CT_seg, MRI_image, MRI_seg = CT_image.cuda(args.rank), CT_seg.cuda(args.rank), MRI_image.cuda(
            args.rank), MRI_seg.cuda(args.rank)

        with autocast(enabled=True):
            CT_img_F_ds, MRI_img_F_ds, CT_seg_out, MRI_seg_out = model(CT_image, MRI_image)
            CT_img_F_ds, MRI_img_F_ds, CT_seg_out, MRI_seg_out = CT_img_F_ds.cuda(args.rank), MRI_img_F_ds.cuda(
                args.rank), CT_seg_out.cuda(args.rank), MRI_seg_out.cuda(args.rank)

            consistency_weight = get_current_consistency_weight(args, 'sig_ram', epoch)
            contra_weight = get_current_consistency_weight(args, 'cos_ram', epoch)
            sup_loss, sar_loss, scr_loss = my_sup_loss(loss_func1,args,CT_seg_out, MRI_seg_out,
                                                       CT_seg, MRI_seg, CT_img_F_ds, MRI_img_F_ds, label_id)
            loss = sup_loss + consistency_weight * sar_loss + contra_weight * scr_loss

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        torch.cuda.empty_cache()

        if args.dist:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    return run_loss.avg

def val_epoch(model, loader, save_root, epoch, args):
    model.eval()
    args.amp = True
    all_avg_dice = []
    num_class = args.out_channels
    log_path = os.path.join(save_root, 'log1_20.txt')

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            CT_image, CT_seg, MRI_image, MRI_seg, label_id = batch_data["CT_image"], batch_data["CT_seg"], \
                batch_data["MRI_image"], batch_data["MRI_seg"], batch_data["label"]
            CT_image, CT_seg, MRI_image, MRI_seg = CT_image.cuda(args.rank), CT_seg.cuda(args.rank), MRI_image.cuda(
                args.rank), MRI_seg.cuda(args.rank)
            _, _, h, w, d = CT_seg.shape
            target_shape = (h, w, d)

            with autocast(enabled=args.amp):
                _, _, CT_seg_out, MRI_seg_out = model(CT_image, MRI_image)

            if not CT_seg_out.is_cuda:
                CT_seg, MRI_seg = CT_seg.cpu(), MRI_seg.cpu()

            val_outputs = torch.softmax(CT_seg_out, 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)[0]
            val_labels = CT_seg.cpu().numpy()[0, 0, :, :, :]
            val_outputs = resample_3d(val_outputs, target_shape)

            FOCUS_ORGANS = ['Spleen', 'Right Kidney', 'Left Kidney', 'Liver']
            organ_dice = []

            with open(log_path, 'a') as f:
                print(f"\nEpoch {epoch}, Sample {idx}:", file=f)

                for i in range(1, num_class):
                    organ_name = ORGAN_NAME[i - 1]
                    if organ_name in FOCUS_ORGANS:
                        organ_dice_i = dice(val_outputs == i, val_labels == i)
                        organ_dice.append(organ_dice_i)
                        print(f"{organ_name} dice: {organ_dice_i:.4f}", file=f)
                        print(f"{organ_name} dice:", organ_dice_i)

                avg_dice = np.mean(organ_dice)
                print("avg_dice: {:.4f}".format(avg_dice))
                print("avg_dice: {:.4f}".format(avg_dice), file=f)

            all_avg_dice.append(avg_dice)

    mean_dice_epoch = np.mean(all_avg_dice)

    with open(log_path, 'a') as f:
        print(f"\n=== Epoch {epoch} Summary ===", file=f)
        print("Mean Dice across samples: {:.4f}".format(mean_dice_epoch), file=f)

    return mean_dice_epoch

def save_checkpoint(model, epoch, args, optimizer, scheduler, filename="model.pt"):
    checkpoint = {
        "net": model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        "epoch": epoch
    }
    checkpoint_filename = os.path.join(args.logdir, filename)
    torch.save(checkpoint, checkpoint_filename)
    print("Saving checkpoint", checkpoint_filename)

def run_training(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_func,
        args,
        scheduler=None,
        start_epoch=0,
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        print("Writing Tensorboard logs to ", args.logdir)

    scaler = GradScaler() if args.amp else None
    best_dice = 0.0
    save_root = os.path.join(args.logdir, args.backbone)
    os.makedirs(save_root, exist_ok=True)

    for epoch in range(start_epoch, args.max_epochs):
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer,
            scaler=scaler, epoch=epoch,
            loss_func1=loss_func, args=args
        )

        print(f"Final training  {epoch}/{args.max_epochs - 1} loss: {train_loss:.4f}, time: {time.time() - epoch_time:.2f}s")
        with open(os.path.join(save_root, 'log1_20.txt'), 'a') as f:
            print(f"Final training  {epoch}/{args.max_epochs - 1} loss: {train_loss:.4f}, time: {time.time() - epoch_time:.2f}s", file=f)

        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)

        if (epoch + 1) % args.val_every == 0:
            if args.dist:
                torch.distributed.barrier()

            val_avg_acc = val_epoch(
                model,
                val_loader,
                save_root,
                epoch=epoch,
                args=args
            )
            val_avg_acc = np.mean(val_avg_acc)

            if writer is not None:
                writer.add_scalar("val_avg_acc", val_avg_acc, epoch)

            if epoch == args.max_epochs:
                save_checkpoint(model, epoch, args, optimizer, scheduler, filename="final_model.pt")

            if val_avg_acc > best_dice:
                print(f"New best Dice: {best_dice:.4f} -> {val_avg_acc:.4f}")
                best_dice = val_avg_acc
                save_checkpoint(model, epoch, args, optimizer, scheduler, filename="best_model1_20.pt")

                with open(os.path.join(save_root, 'log1_20.txt'), 'a') as f:
                    print(f"New best Dice: {best_dice:.4f} -> {val_avg_acc:.4f}", file=f)
                    print("Saved best model.", file=f)

        if scheduler is not None:
            scheduler.step()

    print("Training Finished! Best Dice:", best_dice)
