import os
import torch
import random
import numpy as np
from torch.nn import functional as F

from utils import pyutils
from utils.imutils import crf_inference_label
from module import loss_helper


# Classification
def train_cls(train_loader, model, optimizer, max_step, args):
    avg_meter = pyutils.AverageMeter('loss')
    timer = pyutils.Timer("Session started: ")
    loader_iter = iter(train_loader)
    for iteration in range(args.max_iters):
        for _ in range(args.iter_size):
            # Fetch data
            try:
                img_id, img, label = next(loader_iter)
            except:
                loader_iter = iter(train_loader)
                img_id, img, label = next(loader_iter)
            img = img.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            pred = model(img)

            # Classification loss
            loss = F.multilabel_soft_margin_loss(pred, label)
            avg_meter.add({'loss': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step - 1) % 50 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print(
                    'Iter:%5d/%5d' % (iteration, args.max_iters),
                    'Loss:%.4f' % (avg_meter.pop('loss')),
                    'imps:%.1f' % ((iteration + 1) * args.batch_size / timer.get_stage_elapsed()),
                    'Fin:%s' % (timer.str_est_finish()),
                    'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True
                )

            # validate(model, val_data_loader, epoch=ep + 1)
            timer.reset_stage()

    # torch.save(model.module.state_dict(), os.path.join(args.log_folder, 'checkpoint_cls.pth'))


# ROI-level Semantic Contrast
def get_RSC_loss(cam, f_map, cam1, f_map1, label, pseudo_label, pseudo_label1):
    global memobank_fb, memobank1_fb

    # Parameters
    l1 = 0.5

    label = label.cuda()
    n, c, _, _ = f_map.size()

    criterion = loss_helper.SimMinLoss().cuda()

    # Foreground-background Estimation
    # -------------------------------------------------------------------------------------------------
    # scores = F.softmax(cam * label, dim=1)
    # pseudo_label = scores.argmax(dim=1, keepdim=True)
    mask_bg = (pseudo_label == 20).squeeze(1)
    mask_fore = (pseudo_label != 20).squeeze(1)

    # scores1 = F.softmax(cam1 * label, dim=1)
    # pseudo_label1 = scores1.argmax(dim=1, keepdim=True)
    mask_bg_1 = (pseudo_label1 == 20).squeeze(1)
    mask_fore_1 = (pseudo_label1 != 20).squeeze(1)

    f_map = F.normalize(f_map, dim=-1)
    f_map1 = F.normalize(f_map1, dim=-1)

    # Rep
    # ------------------------------------------------------------------------------------------------
    context_bg = f_map * mask_bg.float().detach().unsqueeze(1)
    context_bg_pool = F.normalize(
        (context_bg.reshape(n, c, -1).sum(2) /
         (mask_bg.float().reshape(n, -1).sum(1).unsqueeze(-1) + 1e-5)).unsqueeze(0),
        eps=1e-5, dim=2
    ).squeeze()

    context_fore = f_map * mask_fore.float().detach().unsqueeze(1)
    context_fore_pool = F.normalize(
        (context_fore.reshape(n, c, -1).sum(2) /
         (mask_fore.float().reshape(n, -1).sum(1).unsqueeze(-1) + 1e-5)).unsqueeze(0),
        eps=1e-5, dim=2
    ).squeeze()

    context_bg_1 = f_map1 * mask_bg_1.float().detach().unsqueeze(1)
    context_bg_pool_1 = F.normalize(
        (context_bg_1.reshape(n, c, -1).sum(2) /
         (mask_bg_1.float().reshape(n, -1).sum(1).unsqueeze(-1) + 1e-5)).unsqueeze(0),
        eps=1e-5, dim=2
    ).squeeze()

    context_fore_1 = f_map1 * mask_fore_1.float().detach().unsqueeze(1)
    context_fore_pool_1 = F.normalize(
        (context_fore_1.reshape(n, c, -1).sum(2) /
         (mask_fore_1.float().reshape(n, -1).sum(1).unsqueeze(-1) + 1e-5)).unsqueeze(0),
        eps=1e-5, dim=2
    ).squeeze()

    # Loss calculation
    # ------------------------------------------------------------------------------------------------
    # foreground vs. background
    loss_f2b_intra = (criterion(context_bg_pool, context_fore_pool) +
                      criterion(context_bg_pool_1, context_fore_pool_1)) / 2  # intra

    loss_f2b_cross = (criterion(context_bg_pool, context_fore_pool_1) +
                      criterion(context_bg_pool_1, context_fore_pool)) / 2  # Cross

    # foreground
    loss_f2b_fore = torch.mean(torch.abs(memobank_fb[1].repeat(n, 1).detach() - context_fore_pool))
    loss_f2b_fore1 = torch.mean(
        torch.abs(memobank1_fb[1].repeat(n, 1).detach() - context_fore_pool_1))
    loss_f2b_fore_all = (loss_f2b_fore + loss_f2b_fore1) / 2

    # background
    loss_f2b_back = torch.mean(torch.abs(memobank_fb[0].repeat(n, 1).detach() - context_bg_pool))
    loss_f2b_back1 = torch.mean(torch.abs(memobank1_fb[0].repeat(n, 1).detach() - context_bg_pool_1))
    loss_f2b_back_all = (loss_f2b_back + loss_f2b_back1) / 2

    # EMA Update
    # ------------------------------------------------------------------------------------------------
    with torch.no_grad():
        memobank_fb[0] = l1 * context_bg_pool.mean(0) + (1. - l1) * memobank_fb[0]
        memobank_fb[1] = l1 * context_fore_pool.mean(0) + (1. - l1) * memobank_fb[1]

        memobank1_fb[0] = l1 * context_bg_pool_1.mean(0) + (1. - l1) * memobank1_fb[0]
        memobank1_fb[1] = l1 * context_fore_pool_1.mean(0) + (1. - l1) * memobank1_fb[1]

    return 0.1 * (loss_f2b_intra + loss_f2b_cross + loss_f2b_fore_all + loss_f2b_back_all)


# Class-level Semantic Contrast
def get_CSC_loss(f_map1, cam1, f_map2, cam2, label):
    global memobank1, memobank2
    # Parameters
    # -----------------------------------------------------------------
    l, temp, h = 0.5, 0.1, 16

    # Setup
    # -----------------------------------------------------------------
    fea1, fea2 = f_map1, f_map2

    cam_soft1 = F.softmax(cam1 * label, dim=1)[:, :-1, :, :]
    cam_soft2 = F.softmax(cam2 * label, dim=1)[:, :-1, :, :]

    label = label[:, :-1]

    c_fea1 = fea1.shape[1]
    fea1 = fea1.permute(0, 2, 3, 1).reshape(-1, c_fea1)
    top_values1, top_indices1 = torch.topk(
        cam_soft1.transpose(0, 1).reshape(20, -1), k=128, dim=-1
    )

    c_fea2 = fea2.shape[1]
    fea2 = fea2.permute(0, 2, 3, 1).reshape(-1, c_fea2)
    top_values2, top_indices2 = torch.topk(
        cam_soft2.transpose(0, 1).reshape(20, -1), k=128, dim=-1
    )

    loss_csc = torch.zeros(1).cuda()
    loss_csc_m = torch.zeros(1).cuda()

    index = torch.unique(torch.nonzero(label.squeeze(-1).squeeze(-1))[:, 1])

    # Loss calculation
    # -----------------------------------------------------------------
    for cls in index:
        ######################## View 1 ########################
        # Intra-class
        top_fea1 = fea1[top_indices1[cls]]
        class_rep1 = F.normalize(
            torch.sum(top_values1[cls].unsqueeze(-1) * top_fea1, dim=0) / torch.sum(top_values1[cls], dim=0), dim=-1
        )

        with torch.no_grad():
            prototype = memobank1[cls]
            neg_index = torch.ones(21, dtype=torch.bool)
            neg_index[cls] = False
            negative = memobank1[neg_index]
            rep_all = torch.cat((prototype.unsqueeze(0), negative), dim=0).cuda()

        seg_logits = torch.cosine_similarity(class_rep1.unsqueeze(0), rep_all, dim=1)
        loss_csc = loss_csc + \
                   F.cross_entropy((seg_logits / temp).unsqueeze(0), torch.zeros(1).long().cuda())

        # mutual
        with torch.no_grad():
            prototype_m = memobank2[cls]
            neg_index = torch.ones(21, dtype=torch.bool)
            neg_index[cls] = False
            negative_m = memobank2[neg_index]
            rep_all_m = torch.cat((prototype_m.unsqueeze(0), negative_m), dim=0).cuda()

        seg_logits_m = torch.cosine_similarity(class_rep1.unsqueeze(0), rep_all_m, dim=1)
        loss_csc_m = loss_csc_m + \
                     F.cross_entropy((seg_logits_m / temp).unsqueeze(0), torch.zeros(1).long().cuda())

        ######################## View 2 ########################
        # Intra-class
        top_fea2 = fea2[top_indices2[cls]]
        class_rep2 = F.normalize(
            torch.sum(top_values2[cls].unsqueeze(-1) * top_fea2, dim=0) / torch.sum(top_values2[cls], dim=0), dim=-1
        )

        with torch.no_grad():
            prototype = memobank2[cls]
            neg_index = torch.ones(21, dtype=torch.bool)
            neg_index[cls] = False
            negative = memobank2[neg_index]
            rep_all = torch.cat((prototype.unsqueeze(0), negative), dim=0).cuda()

        seg_logits = torch.cosine_similarity(class_rep2.unsqueeze(0), rep_all, dim=1)
        loss_csc = loss_csc + \
                   F.cross_entropy((seg_logits / temp).unsqueeze(0), torch.zeros(1).long().cuda())

        # mutual
        with torch.no_grad():
            prototype_m = memobank1[cls]
            neg_index = torch.ones(21, dtype=torch.bool)
            neg_index[cls] = False
            negative_m = memobank1[neg_index]
            rep_all_m = torch.cat((prototype_m.unsqueeze(0), negative_m), dim=0).cuda()

        seg_logits_m = torch.cosine_similarity(class_rep2.unsqueeze(0), rep_all_m, dim=1)
        loss_csc_m = loss_csc_m + \
                     F.cross_entropy((seg_logits_m / temp).unsqueeze(0), torch.zeros(1).long().cuda())

        # EMA Update
        with torch.no_grad():
            high_num = h
            # View 1
            weight1 = top_values1[cls][:high_num]
            top_fea1 = fea1[top_indices1[cls]][:high_num, :]
            class_rep_high1 = F.normalize(
                torch.sum(weight1.unsqueeze(-1) * top_fea1, dim=0) / torch.sum(top_values1[cls], dim=0), dim=-1
            )
            memobank1[cls] = l * class_rep_high1 + (1 - l) * memobank1[cls]

            # View 2
            weight2 = top_values2[cls][:high_num]
            top_fea2 = fea2[top_indices2[cls]][:high_num, :]
            class_rep_high2 = F.normalize(
                torch.sum(weight2.unsqueeze(-1) * top_fea2, dim=0) / torch.sum(top_values2[cls], dim=0), dim=-1
            )
            memobank2[cls] = l * class_rep_high2 + (1 - l) * memobank2[cls]

    # Loss calculation
    # -----------------------------------------------------------------
    loss_csc = loss_csc / index.shape[0]
    loss_csc_m = loss_csc_m / index.shape[0]

    return 0.1 * (loss_csc + loss_csc_m) / 4


# Pixel-level Semantic Contrast
def get_PSC_loss(cam1, f_map1, cam2, f_map2, label, pseudo_label1, pseudo_label2):
    global memobank1_p, memobank2_p
    l1 = 0.5

    with torch.no_grad():
        # Generate Pseudo label
        # -------------------------------------------------------------------------------------------------
        # View 1 (Original size)
        # scores1 = F.softmax(cam1 * label, dim=1)
        # n_sc1, c_sc1, h_sc1, w_sc1 = scores1.shape
        # pseudo_label1 = scores1.argmax(dim=1, keepdim=True)
        #
        # # View 2 (Small size)
        # scores2 = F.softmax(cam2 * label, dim=1)
        # n_sc2, c_sc2, h_sc2, w_sc2 = scores2.shape
        # pseudo_label2 = scores2.argmax(dim=1, keepdim=True)

        n_sc1, c_sc1, h_sc1, w_sc1 = cam1.shape
        n_sc2, c_sc2, h_sc2, w_sc2 = cam2.shape

        # Generate prototype
        # -------------------------------------------------------------------------------------------------
        # View 1 (Original size)
        fea1 = f_map1.detach()
        c_fea1 = fea1.shape[1]
        fea1 = fea1.permute(0, 2, 3, 1).reshape(-1, c_fea1)
        top_values, top_indices = torch.topk(
            cam1.transpose(0, 1).reshape(21, -1), k=h_sc1 * w_sc1 // 16, dim=-1
        )
        prototypes1 = torch.zeros(21, 128, device='cuda')  # [21, 128]
        for i in range(21):
            top_fea = fea1[top_indices[i]]
            prototypes1[i] = torch.sum(top_values[i].unsqueeze(-1) * top_fea, dim=0) / torch.sum(top_values[i])
        prototypes1 = F.normalize(prototypes1, dim=-1)

        # View 2 (Small size)
        fea2 = f_map2.detach()
        c_fea2 = fea2.shape[1]
        fea2 = fea2.permute(0, 2, 3, 1).reshape(-1, c_fea2)
        top_values, top_indices = torch.topk(
            cam2.transpose(0, 1).reshape(21, -1), k=h_sc2 * w_sc2 // 16, dim=-1
        )
        prototypes2 = torch.zeros(21, 128, device='cuda')  # [21, 128]
        for i in range(21):
            top_fea = fea2[top_indices[i]].detach()
            prototypes2[i] = torch.sum(top_values[i].unsqueeze(-1) * top_fea, dim=0) / torch.sum(top_values[i])
        prototypes2 = F.normalize(prototypes2, dim=-1)

    # View 1
    n_f, c_f, h_f, w_f = f_map1.shape
    f_map1 = f_map1.permute(0, 2, 3, 1).reshape(n_f * h_f * w_f, c_f)
    f_map1 = F.normalize(f_map1, dim=-1)
    pseudo_label1 = pseudo_label1.reshape(-1)
    positives1 = prototypes2[pseudo_label1]
    negitives1 = prototypes2

    # View 2
    n_f, c_f, h_f, w_f = f_map2.shape
    f_map2 = f_map2.permute(0, 2, 3, 1).reshape(n_f * h_f * w_f, c_f)
    f_map2 = F.normalize(f_map2, dim=-1)
    pseudo_label2 = pseudo_label2.reshape(-1)
    positives2 = prototypes1[pseudo_label2]
    negitives2 = prototypes1

    # Loss calculation
    # -------------------------------------------------------------------------------------------------
    ######################## Cross prototype ########################
    A1 = torch.exp(torch.sum(f_map1 * positives1, dim=-1) / 0.1)
    A2 = torch.sum(torch.exp(torch.matmul(f_map1, negitives1.transpose(0, 1)) / 0.1), dim=-1)
    loss_nce1 = torch.mean(-1 * torch.log(A1 / A2))

    A3 = torch.exp(torch.sum(f_map2 * positives2, dim=-1) / 0.1)
    A4 = torch.sum(torch.exp(torch.matmul(f_map2, negitives2.transpose(0, 1)) / 0.1), dim=-1)
    loss_nce2 = torch.mean(-1 * torch.log(A3 / A4))

    loss_cp = 0.1 * (loss_nce1 + loss_nce2) / 2

    ######################## Cross pseudo-label ########################
    A1_view1 = torch.exp(torch.sum(f_map1 * positives2, dim=-1) / 0.1)
    A2_view1 = torch.sum(torch.exp(torch.matmul(f_map1, negitives2.transpose(0, 1)) / 0.1), dim=-1)
    loss_cross_nce2_1 = torch.mean(-1 * torch.log(A1_view1 / A2_view1))

    A3_view2 = torch.exp(torch.sum(f_map2 * positives1, dim=-1) / 0.1)
    A4_view2 = torch.sum(torch.exp(torch.matmul(f_map2, negitives1.transpose(0, 1)) / 0.1), dim=-1)
    loss_cross_nce2_2 = torch.mean(-1 * torch.log(A3_view2 / A4_view2))

    loss_cross_cl = 0.1 * (loss_cross_nce2_1 + loss_cross_nce2_2) / 2

    ######################## Intra class ########################
    # View 1
    positives1 = prototypes1[pseudo_label1]
    negitives1 = prototypes1

    # View 2
    positives2 = prototypes2[pseudo_label2]
    negitives2 = prototypes2

    A1_intra1 = torch.exp(torch.sum(f_map1 * positives1, dim=-1) / 0.1)
    A2_intra1 = torch.sum(torch.exp(torch.matmul(f_map1, negitives1.transpose(0, 1)) / 0.1), dim=-1)
    loss_intra_nce_1 = torch.mean(-1 * torch.log(A1_intra1 / A2_intra1))

    A3_intra2 = torch.exp(torch.sum(f_map2 * positives2, dim=-1) / 0.1)
    A4_intra2 = torch.sum(torch.exp(torch.matmul(f_map2, negitives2.transpose(0, 1)) / 0.1), dim=-1)
    loss_intra_nce_2 = torch.mean(-1 * torch.log(A3_intra2 / A4_intra2))

    loss_intra_cl = 0.1 * (loss_intra_nce_1 + loss_intra_nce_2) / 2

    # Memory bank
    # -------------------------------------------------------------------------------------------------
    positives1 = memobank1_p[pseudo_label1]
    negitives1 = memobank1_p

    positives2 = memobank2_p[pseudo_label2]
    negitives2 = memobank2_p

    A1 = torch.exp(torch.sum(f_map1 * positives1, dim=-1) / 0.1)
    A2 = torch.sum(torch.exp(torch.matmul(f_map1, negitives1.transpose(0, 1)) / 0.1), dim=-1)
    loss_memo_nce1 = torch.mean(-1 * torch.log(A1 / A2))

    A3 = torch.exp(torch.sum(f_map2 * positives2, dim=-1) / 0.1)
    A4 = torch.sum(torch.exp(torch.matmul(f_map2, negitives2.transpose(0, 1)) / 0.1), dim=-1)
    loss_memo_nce2 = torch.mean(-1 * torch.log(A3 / A4))

    loss_memo = 0.1 * (loss_memo_nce1 + loss_memo_nce2) / 2

    # EMA Update
    with torch.no_grad():
        memobank1_p = l1 * prototypes1 + (1. - l1) * memobank1_p
        memobank2_p = l1 * prototypes2 + (1. - l1) * memobank2_p

    # print('cp:{} | cl:{} | intra:{} | memo:{}'.format(loss_cp, loss_cross_cl, loss_intra_cl, loss_memo))

    return loss_cp + loss_cross_cl + loss_intra_cl + loss_memo


# Build Memory Bank
# View 1
memobank1 = torch.randn(21, 256, device='cuda')
memobank_fb = torch.randn(2, 256, device='cuda')  # 0 - prototype of background 1 - foreground
memobank1_p = torch.randn(21, 128, device='cuda')

# View 2
memobank2 = torch.randn(21, 256, device='cuda')
memobank1_fb = torch.randn(2, 256, device='cuda')  # 0 - prototype of background 1 - foreground
memobank2_p = torch.randn(21, 128, device='cuda')


# The overall loss of Hierarchical Semantic Contrast
def get_hsc_loss(cam1, cam2, f_proj1_1, f_proj1_2, f_proj2_1, f_proj2_2, img1, label, bg_thres=0.05):
    # f_proj1_1, f_proj1_2 -- View 1 - pixel 1_1 / bg 1_2
    # f_proj2_1, f_proj2_2 -- View 2 - pixel 2_1/ bg 2_2

    # Init
    # ------------------------------------------------------------------------------------------------
    # CAM -- Norm
    cam_rv1_down = F.interpolate(cam1, size=(128 // 8, 128 // 8), mode='bilinear', align_corners=True)
    cam_rv2_down = cam2

    # Rep
    # View 1
    f_proj1_1 = F.interpolate(f_proj1_1, size=(128 // 8, 128 // 8), mode='bilinear', align_corners=True)
    f_proj1_2 = F.interpolate(f_proj1_2, size=(128 // 8, 128 // 8), mode='bilinear', align_corners=True)
    f_proj2_1 = f_proj2_1
    f_proj2_2 = f_proj2_2

    # CAM
    with torch.no_grad():
        # Norm -> ~(0,1)
        n1, c1, h1, w1 = cam_rv1_down.shape
        cam_rv1_down = F.relu(cam_rv1_down.detach())
        max1 = torch.max(cam_rv1_down.view(n1, c1, -1), dim=-1)[0].view(n1, c1, 1, 1)
        min1 = torch.min(cam_rv1_down.view(n1, c1, -1), dim=-1)[0].view(n1, c1, 1, 1)
        cam_rv1_down[cam_rv1_down < min1 + 1e-5] = 0.
        norm_cam1 = (cam_rv1_down - min1 - 1e-5) / (max1 - min1 + 1e-5)
        cam_rv1_down = norm_cam1
        cam_rv1_down[:, -1, :, :] = bg_thres

        cam_rv2_down = F.relu(cam_rv2_down.detach())
        n2, c2, h2, w2 = cam_rv2_down.shape
        max2 = torch.max(cam_rv2_down.view(n2, c2, -1), dim=-1)[0].view(n2, c2, 1, 1)
        min2 = torch.min(cam_rv2_down.view(n2, c2, -1), dim=-1)[0].view(n2, c2, 1, 1)
        cam_rv2_down[cam_rv2_down < min2 + 1e-5] = 0.
        norm_cam2 = (cam_rv2_down - min2 - 1e-5) / (max2 - min2 + 1e-5)
        cam_rv2_down = norm_cam2
        cam_rv2_down[:, -1, :, :] = bg_thres

    # Generate pseudo label
    n, c, h, w = cam1.shape

    # bg_score = torch.ones((n, 1), device='cuda')  # 初始的 bg_score 都是1
    # label = torch.cat((bg_score, label), dim=1).unsqueeze(2).unsqueeze(3)  # N * C * H * W

    bg_score = torch.ones((n1, 1)).cuda()
    label = torch.cat((label, bg_score), dim=1).unsqueeze(2).unsqueeze(3)

    with torch.no_grad():
        # Part 0 => Generate pseudo-label
        # ------------------------------------------------------------------------------------------------
        mean = torch.Tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).repeat(n, 1)
        std = torch.Tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).repeat(n, 1)

        img_p = F.interpolate(img1.cpu(), size=(128 // 8, 128 // 8), mode='bilinear', align_corners=True)  # [16, 16]
        img_p = img_p * std.unsqueeze(-1).unsqueeze(-1) + mean.unsqueeze(-1).unsqueeze(-1)

        # View 1
        scores1 = F.softmax(cam_rv1_down * label, dim=1)
        n_sc1, c_sc1, h_sc1, w_sc1 = scores1.shape
        pseudo_label1 = scores1.argmax(dim=1, keepdim=True)
        pseudo_label_crf1 = torch.zeros(n, w_sc1, h_sc1, device='cuda', dtype=torch.int64)
        for i in range(n):
            pseudo_label_crf1[i] = torch.from_numpy(
                crf_inference_label(
                    img_p[i].permute(1, 2, 0).numpy().astype(np.uint8),
                    pseudo_label1[i].squeeze(0).cpu().detach().numpy().astype(np.uint8),
                    n_labels=21, t=10, gt_prob=0.7
                )
            ).cuda()

        # View 2
        scores2 = F.softmax(cam_rv2_down * label, dim=1)
        pseudo_label2 = scores2.argmax(dim=1, keepdim=True)
        pseudo_label_crf2 = torch.zeros(n, w_sc1, h_sc1, device='cuda', dtype=torch.int64)
        for i in range(n):
            pseudo_label_crf2[i] = torch.from_numpy(
                crf_inference_label(
                    img_p[i].permute(1, 2, 0).numpy().astype(np.uint8),
                    pseudo_label2[i].squeeze(0).cpu().detach().numpy().astype(np.uint8),
                    n_labels=21, t=10, gt_prob=0.7
                )
            ).cuda()

    # Loss calculation
    # ------------------------------------------------------------------------------------------------
    # RSC
    loss_rsc = get_RSC_loss(
        cam_rv1_down, f_proj1_2, cam_rv2_down, f_proj2_2, label, pseudo_label_crf1, pseudo_label_crf2
    )

    # CSC
    loss_csc = get_CSC_loss(f_proj1_2, cam_rv1_down, f_proj2_2, cam_rv2_down, label)

    # PSC
    loss_psc = get_PSC_loss(
        cam_rv1_down, f_proj1_1, cam_rv2_down, f_proj2_1, label, pseudo_label_crf1, pseudo_label_crf2
    )

    return loss_rsc, loss_csc, loss_psc


def train_hsc(train_dataloader, model, optimizer, max_step, args):
    avg_meter = pyutils.AverageMeter(
        'loss', 'loss_cls', 'loss_sal', 'loss_er', 'loss_ecr', 'loss_rsc', 'loss_csc', 'loss_psc'
    )

    timer = pyutils.Timer("Session started: ")
    loader_iter = iter(train_dataloader)
    print(args)
    for iteration in range(args.max_iters):
        for _ in range(args.iter_size):
            try:
                img_id, img, saliency, label = next(loader_iter)
            except:
                loader_iter = iter(train_dataloader)
                img_id, img, saliency, label = next(loader_iter)

            img = img.cuda(non_blocking=True)
            saliency = saliency.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            img2 = F.interpolate(img, size=(128, 128), mode='bilinear', align_corners=True)
            saliency2 = F.interpolate(saliency, size=(128, 128), mode='bilinear', align_corners=True)

            pred1, cam1, pred_rv1, cam_rv1, feat1_1, feat1_2 = model(img)
            pred2, cam2, pred_rv2, cam_rv2, feat2_1, feat2_2 = model(img2)

            # Loss (baseline)
            # ------------------------------------------------------------------------------------------------
            loss_cls = F.multilabel_soft_margin_loss(pred1[:, :-1], label)  # Classification loss 1
            loss_sal, fg_map, bg_map, sal_pred = loss_helper.get_eps_loss(
                cam1, saliency, label, args.tau, args.alpha, intermediate=True
            )
            loss_sal_rv, _, _, _ = loss_helper.get_eps_loss(
                cam_rv1, saliency, label, args.tau, args.alpha, intermediate=True
            )

            loss_cls2 = F.multilabel_soft_margin_loss(pred2[:, :-1], label)  # Classification loss 2
            loss_sal2, fg_map2, bg_map2, sal_pred2 = loss_helper.get_eps_loss(
                cam2, saliency2, label, args.tau, args.alpha, intermediate=True
            )
            loss_sal_rv2, _, _, _ = loss_helper.get_eps_loss(
                cam_rv2, saliency2, label, args.tau, args.alpha, intermediate=True
            )

            bg_score = torch.ones((img.shape[0], 1)).cuda()
            label_append_bg = torch.cat((label, bg_score), dim=1).unsqueeze(2).unsqueeze(3)  # (N, 21, 1, 1)
            loss_cls_rv1 = loss_helper.adaptive_min_pooling_loss((cam_rv1 * label_append_bg)[:, :-1, :, :])
            loss_cls_rv2 = loss_helper.adaptive_min_pooling_loss((cam_rv2 * label_append_bg)[:, :-1, :, :])

            loss_er, loss_ecr = loss_helper.get_er_loss(cam1, cam2, cam_rv1, cam_rv2, label_append_bg)

            # Loss (HSC)
            # ------------------------------------------------------------------------------------------------
            loss_rsc, loss_csc, loss_psc = get_hsc_loss(
                cam_rv1, cam_rv2, feat1_1, feat1_2, feat2_1, feat2_2, img2, label, bg_thres=0.10
            )

            # Total Loss
            # ------------------------------------------------------------------------------------------------
            loss_cls = (loss_cls + loss_cls2) / 2. + (loss_cls_rv1 + loss_cls_rv2) / 2.

            loss_sal = (loss_sal + loss_sal2) / 2. + (loss_sal_rv + loss_sal_rv2) / 2.

            loss = loss_cls + loss_sal + loss_er + loss_ecr + (loss_rsc + loss_csc + loss_psc)

            avg_meter.add({
                'loss': loss.item(),
                'loss_cls': loss_cls.item(),
                'loss_sal': loss_sal.item(),
                'loss_er': loss_er.item(),
                'loss_ecr': loss_ecr.item(),
                'loss_rsc': loss_rsc.item(),
                'loss_csc': loss_csc.item(),
                'loss_psc': loss_psc.item()
            })

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step - 1) % 50 == 0:
                timer.update_progress(optimizer.global_step / max_step)
                print(
                    'Iter:%5d/%5d' % (iteration, args.max_iters),
                    'Loss_Cls:%.4f' % (avg_meter.pop('loss_cls')),
                    'Loss_Sal:%.4f' % (avg_meter.pop('loss_sal')),
                    'Loss_ER: %.4f' % (avg_meter.pop('loss_er')),
                    'Loss_ECR:%.4f' % (avg_meter.pop('loss_ecr')),
                    'loss_rsc:%.4f' % (avg_meter.pop('loss_rsc')),
                    'loss_csc:%.4f' % (avg_meter.pop('loss_csc')),
                    'loss_psc:%.4f' % (avg_meter.pop('loss_psc')),
                    'imps:%.1f' % ((iteration + 1) * args.batch_size / timer.get_stage_elapsed()),
                    'Fin:%s' % (timer.str_est_finish()),
                    'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True
                )

            # validate(model, val_data_loader, epoch=ep + 1)
            timer.reset_stage()

    print('finished...')

    torch.save(model.module.state_dict(), os.path.join(args.log_folder, '{}.pth'.format(args.session)))
