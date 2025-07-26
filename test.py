import os, torch, random
import numpy as np
from models.Semi_SM_model import Semi_SM_model
import warnings
warnings.filterwarnings("ignore")
from utils.data_utils_mm import get_loader
from utils.utils import  ORGAN_NAME,resample_3d,dice
import nibabel as nib

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def validation(model, test_loader, args):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    num_class = args.out_channels

    with torch.no_grad():
        ct_spl_dice_all = []
        mri_spl_dice_all = []
        ct_Rkid_dice_all = []
        mri_Rkid_dice_all = []
        ct_Lkid_dice_all = []
        mri_Lkid_dice_all = []
        ct_Liver_dice_all = []
        mri_Liver_dice_all = []
        for idx, batch_data in enumerate(test_loader):
            CT_image, CT_seg, MRI_image, MRI_seg, label_id = batch_data["CT_image"], batch_data["CT_seg"], batch_data["MRI_image"], batch_data["MRI_seg"], batch_data["label"]
            CT_image, CT_seg, MRI_image, MRI_seg = CT_image.cuda(args.rank), CT_seg.cuda(args.rank), MRI_image.cuda(args.rank), MRI_seg.cuda(args.rank)
            ct_img_name = batch_data["CT_filename"]
            mri_img_name = batch_data["MRI_filename"]

            if isinstance(ct_img_name, list):
                ct_img_name = ct_img_name[0]
            if isinstance(mri_img_name, list):
                mri_img_name = mri_img_name[0]

            _, _, CT_seg_out, MRI_seg_out = model(CT_image, MRI_image)

            pre_ct_outputs = torch.softmax(CT_seg_out, 1).cpu().numpy()
            pre_ct_outputs = np.argmax(pre_ct_outputs, axis=1).astype(np.uint8)[0]
            pre_mri_outputs = torch.softmax(MRI_seg_out, 1).cpu().numpy()
            pre_mri_outputs = np.argmax(pre_mri_outputs, axis=1).astype(np.uint8)[0]

            _, _, h, w, d = CT_seg.shape
            _, _, h1, w1, d1 = MRI_seg.shape
            CT_seg_shape = (h, w, d)
            MRI_seg_shape = (h1, w1, d1)

            ori_ct_data = CT_image[0, 0].detach().cpu().numpy()
            ori_mri_data = MRI_image[0, 0].detach().cpu().numpy()

            ct_labels = CT_seg[0, 0].cpu().numpy()
            mri_labels = MRI_seg[0, 0].cpu().numpy()

            pre_ct_outputs = resample_3d(pre_ct_outputs, CT_seg_shape)
            ori_ct_data = resample_3d(ori_ct_data, CT_seg_shape)
            pre_mri_outputs = resample_3d(pre_mri_outputs, MRI_seg_shape)
            ori_mri_data = resample_3d(ori_mri_data, MRI_seg_shape)

            ct_Rkid_dice = 0
            ct_Lkid_dice = 0
            ct_spl_dice = 0
            ct_Liver_dice = 0
            mri_Rkid_dice = 0
            mri_Lkid_dice = 0
            mri_spl_dice = 0
            mri_Liver_dice = 0

            for i in range(1, num_class):
                organ_name = ORGAN_NAME[i - 1]
                if organ_name == 'Spleen':
                    ct_spl_dice = dice(pre_ct_outputs == i, ct_labels == i)
                    mri_spl_dice = dice(pre_mri_outputs == i, mri_labels == i)
                    ct_spl_dice_all.append(ct_spl_dice)
                    mri_spl_dice_all.append(mri_spl_dice)
                elif organ_name == 'Right Kidney':
                    ct_Rkid_dice = dice(pre_ct_outputs == i, ct_labels == i)
                    mri_Rkid_dice = dice(pre_mri_outputs == i, mri_labels == i)
                    ct_Rkid_dice_all.append(ct_Rkid_dice)
                    mri_Rkid_dice_all.append(mri_Rkid_dice)
                elif organ_name == 'Left Kidney':
                    ct_Lkid_dice = dice(pre_ct_outputs == i, ct_labels == i)
                    mri_Lkid_dice = dice(pre_mri_outputs == i, mri_labels == i)
                    ct_Lkid_dice_all.append(ct_Lkid_dice)
                    mri_Lkid_dice_all.append(mri_Lkid_dice)
                elif organ_name == 'Liver':
                    ct_Liver_dice = dice(pre_ct_outputs == i, ct_labels == i)
                    mri_Liver_dice = dice(pre_mri_outputs == i, mri_labels == i)
                    ct_Liver_dice_all.append(ct_Liver_dice)
                    mri_Liver_dice_all.append(mri_Liver_dice)
                elif i > 8:
                    break

            print(
                "CT case name:{}, spleen dice: {}, R kidney dice: {}, L kidney dice: {}, Liver dice:{}, avg organ dice:{}".format(
                    ct_img_name, ct_spl_dice, ct_Rkid_dice, ct_Lkid_dice, ct_Liver_dice,
                    np.mean([ct_spl_dice, ct_Rkid_dice, ct_Lkid_dice, ct_Liver_dice])))
            print(
                "MRI case name:{}, spleen dice: {}, R kidney dice: {}, L kidney dice: {}, Liver dice:{}, avg organ dice:{}".format(
                    mri_img_name, mri_spl_dice, mri_Rkid_dice, mri_Lkid_dice, mri_Liver_dice,
                    np.mean([mri_spl_dice, mri_Rkid_dice, mri_Lkid_dice, mri_Liver_dice])))

        ct_spleen_dice_avg = np.mean(ct_spl_dice_all)
        mri_spleen_dice_avg = np.mean(mri_spl_dice_all)
        ct_R_kidney_dice_avg = np.mean(ct_Rkid_dice_all)
        mri_R_kidney_dice_avg = np.mean(mri_Rkid_dice_all)
        ct_L_kidney_dice_avg = np.mean(ct_Lkid_dice_all)
        mri_L_kidney_dice_avg = np.mean(mri_Lkid_dice_all)
        ct_Liver_dice_avg = np.mean(ct_Liver_dice_all)
        mri_Liver_dice_avg = np.mean(mri_Liver_dice_all)

        print("CT case: spleen avg dice: {}, R kidney avg dice: {}, L kidney avg dice : {}, Liver avg dice:{}".format(
            ct_spleen_dice_avg, ct_R_kidney_dice_avg, ct_L_kidney_dice_avg, ct_Liver_dice_avg))
        print("MRI case: spleen avg dice: {}, R kidney avg dice: {}, L kidney avg dice : {}, Liver avg dice:{}".format(
            mri_spleen_dice_avg, mri_R_kidney_dice_avg, mri_L_kidney_dice_avg, mri_Liver_dice_avg))
        ct_all_avg_dice = np.mean([ct_spleen_dice_avg, ct_R_kidney_dice_avg, ct_L_kidney_dice_avg, ct_Liver_dice_avg])
        mri_all_avg_dice = np.mean([mri_spleen_dice_avg, mri_R_kidney_dice_avg, mri_L_kidney_dice_avg, mri_Liver_dice_avg])
        print("CT Overall Mean Dice: {}, MRI Overall Mean Dice: {}".format(ct_all_avg_dice, mri_all_avg_dice))


def main():
    import argparse
    parser = argparse.ArgumentParser(description='medical contest')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--img_size', default=96, type=int)
    parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
    parser.add_argument("--logdir", default="test_250", type=str, help="directory to save the tensorboard logs")
    parser.add_argument('--trained_weights', default=f"./checkpoint/test/best_model1_20.pt", type=str)
    parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
    parser.add_argument("--test_mode", default=1, type=int, help="node rank for distributed training")
    parser.add_argument('--backbone', default='Foundation_model', help='backbone [Foundation_model or VIT3D]')
    parser.add_argument("--workers", default=8, type=int, help="number of workers")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--out_channels", default=16, type=int, help="number of output channels")
    parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
    parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
    parser.add_argument("--RandScaleIntensityd_prob", default=0.3, type=float,
                        help="RandScaleIntensityd aug probability")
    parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float,
                        help="RandShiftIntensityd aug probability")
    parser.add_argument("--amp", default=1, type=int, help="use amp for training")
    parser.add_argument("--save_checkpoint", default=1, type=int, help="save checkpoint during training")
    parser.add_argument('--output_path', type=str, default='./output')
    args = parser.parse_args()
    torch.set_float32_matmul_precision('high')

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    args.device = device

    print("MAIN Argument values:")
    for k, v in vars(args).items():
        print(k, '=>', v)
    print('-----------------')
    args.NUM_CLASS = args.out_channels

    model = Semi_SM_model(img_size=args.img_size,
                    n_class=args.out_channels,
                    backbone=args.backbone
                    )
    model.load_state_dict(torch.load(args.trained_weights)['net'])
    model.to(device)

    train_loader, val_loader, test_loader = get_loader(args)

    validation(model, test_loader, args)

if __name__ == "__main__":
    setup_seed()
    main()