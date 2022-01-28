import torch
import numpy as np

from utils import smp_metrics
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import ConfusionMatrix, postprocess, invert_affine, scale_coords, process_batch, ap_per_class, fitness, \
    save_checkpoint
from tqdm.autonotebook import tqdm


@torch.no_grad()

def val(model, optimizer, val_generator, params, opt, writer, epoch, step, best_fitness, best_loss, best_epoch):
    model.eval()
    loss_regression_ls = []
    loss_classification_ls = []
    loss_segmentation_ls = []
    jdict, stats, ap, ap_class = [], [], [], []
    iou_thresholds = torch.linspace(0.5, 0.95, 10).cuda()  # iou vector for mAP@0.5:0.95
    num_thresholds = iou_thresholds.numel()
    nc = 1
    seen = 0
    plots = True
    confusion_matrix = ConfusionMatrix(nc=nc)
    s = ('%15s' + '%11s' * 12) % (
    'Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95', 'mIoU', 'mF1', 'rIoU', 'rF1', 'lIoU', 'lF1')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    iou_ls = [[] for _ in range(3)]
    f1_ls = [[] for _ in range(3)]
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    val_loader = tqdm(val_generator)
    for iter, data in enumerate(val_loader):
        imgs = data['img']
        annot = data['annot']
        seg_annot = data['segmentation']
        filenames = data['filenames']
        shapes = data['shapes']


        if params.num_gpus == 1:
            imgs = imgs.cuda()
            annot = annot.cuda()
            seg_annot = seg_annot.cuda()

        cls_loss, reg_loss, seg_loss, regression, classification, anchors, segmentation = model(imgs, annot,
                                                                                                seg_annot,
                                                                                                obj_list=params.obj_list)
        cls_loss = cls_loss.mean()
        reg_loss = reg_loss.mean()
        seg_loss = seg_loss.mean()

        if opt.cal_map:
            out = postprocess(imgs.detach(),
                              torch.stack([anchors[0]] * imgs.shape[0], 0).detach(), regression.detach(),
                              classification.detach(),
                              regressBoxes, clipBoxes,
                              0.5, 0.3)

            for i in range(annot.size(0)):
                seen += 1
                labels = annot[i]
                labels = labels[labels[:, 4] != -1]

                ou = out[i]
                nl = len(labels)

                pred = np.column_stack([ou['rois'], ou['scores']])
                pred = np.column_stack([pred, ou['class_ids']])
                pred = torch.from_numpy(pred).cuda()

                target_class = labels[:, 4].tolist() if nl else []  # target class

                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, num_thresholds, dtype=torch.bool),
                                      torch.Tensor(), torch.Tensor(), target_class))
                    # print("here")
                    continue

                if nl:
                    pred[:, :4] = scale_coords(imgs[i][1:], pred[:, :4], shapes[i][0], shapes[i][1])
                    labels = scale_coords(imgs[i][1:], labels, shapes[i][0], shapes[i][1])
                    correct = process_batch(pred, labels, iou_thresholds)
                    if plots:
                        confusion_matrix.process_batch(pred, labels)
                else:
                    correct = torch.zeros(pred.shape[0], num_thresholds, dtype=torch.bool)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), target_class))

                # print(stats)

                # Visualization
                # seg_0 = segmentation[i]
                # # print('bbb', seg_0.shape)
                # seg_0 = torch.argmax(seg_0, dim = 0)
                # # print('before', seg_0.shape)
                # seg_0 = seg_0.cpu().numpy()
                #     #.transpose(1, 2, 0)
                # # print(seg_0.shape)
                # anh = np.zeros((384,640,3))
                # anh[seg_0 == 0] = (255,0,0)
                # anh[seg_0 == 1] = (0,255,0)
                # anh[seg_0 == 2] = (0,0,255)
                # anh = np.uint8(anh)
                # cv2.imwrite('segmentation-{}.jpg'.format(filenames[i]),anh)

                for i in range(len(params.seg_list) + 1):
                    # print(segmentation[:,i,...].unsqueeze(1).size())
                    tp_seg, fp_seg, fn_seg, tn_seg = smp_metrics.get_stats(segmentation[:, i, ...].unsqueeze(1).cuda(),
                                                                           seg_annot[:, i, ...].unsqueeze(
                                                                               1).round().long().cuda(),
                                                                           mode='binary', threshold=0.5)

                    iou = smp_metrics.iou_score(tp_seg, fp_seg, fn_seg, tn_seg).mean()
                    # print("I", i , iou)
                    f1 = smp_metrics.f1_score(tp_seg, fp_seg, fn_seg, tn_seg).mean()

                    iou_ls[i].append(iou.detach().cpu().numpy())
                    f1_ls[i].append(f1.detach().cpu().numpy())

        loss = cls_loss + reg_loss + seg_loss
        if loss == 0 or not torch.isfinite(loss):
            continue

        loss_classification_ls.append(cls_loss.item())
        loss_regression_ls.append(reg_loss.item())
        loss_segmentation_ls.append(seg_loss.item())

    cls_loss = np.mean(loss_classification_ls)
    reg_loss = np.mean(loss_regression_ls)
    seg_loss = np.mean(loss_segmentation_ls)
    loss = cls_loss + reg_loss + seg_loss

    print(
        'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Segmentation loss: {:1.5f}. Total loss: {:1.5f}'.format(
            epoch, opt.num_epochs, cls_loss, reg_loss, seg_loss, loss))
    writer.add_scalars('Loss', {'val': loss}, step)
    writer.add_scalars('Regression_loss', {'val': reg_loss}, step)
    writer.add_scalars('Classfication_loss', {'val': cls_loss}, step)
    writer.add_scalars('Segmentation_loss', {'val': seg_loss}, step)

    if opt.cal_map:
        # print(len(iou_ls[0]))
        iou_score = np.mean(iou_ls)
        # print(iou_score)
        f1_score = np.mean(f1_ls)

        for i in range(len(params.seg_list) + 1):
            iou_ls[i] = np.mean(iou_ls[i])
            f1_ls[i] = np.mean(f1_ls[i])

        # Compute statistics
        stats = [np.concatenate(x, 0) for x in zip(*stats)]
        # print(stats[3])

        # Count detected boxes per class
        # boxes_per_class = np.bincount(stats[2].astype(np.int64), minlength=1)

        ap50 = None
        save_dir = 'abc'
        names = {
            0: 'car'
        }
        # Compute metrics
        if len(stats) and stats[0].any():
            p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=1)  # number of targets per class
        else:
            nt = torch.zeros(1)

        # Print results
        print(s)
        pf = '%15s' + '%11i' * 2 + '%11.3g' * 10  # print format
        print(pf % ('all', seen, nt.sum(), mp, mr, map50, map, iou_score, f1_score,
                    iou_ls[1], f1_ls[1], iou_ls[2], f1_ls[2]))

        # Print results per class
        verbose = True
        training = False
        nc = 1
        if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
            for i, c in enumerate(ap_class):
                print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

        # Plots
        if plots:
            confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
            confusion_matrix.tp_fp()

        results = (mp, mr, map50, map, iou_score, f1_score, loss)
        fi = fitness(
            np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95, iou, f1, loss ]

        # if calculating map, save by best fitness
        if fi > best_fitness:
            best_fitness = fi
            ckpt = {'epoch': epoch,
                    'step': step,
                    'best_fitness': best_fitness,
                    'model': model,
                    'optimizer': optimizer.state_dict()}
            print("Saving checkpoint with best fitness", fi[0])
            save_checkpoint(ckpt, opt.saved_path, f'efficientdet-d{opt.compound_coef}_best.pth')
    else:
        # if not calculating map, save by best loss
        if loss + opt.es_min_delta < best_loss:
            best_loss = loss
            best_epoch = epoch

            save_checkpoint(model, opt.saved_path, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')

    # Early stopping
    if epoch - best_epoch > opt.es_patience > 0:
        print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, best_loss))
        writer.close()
        exit(0)

    model.train()
    return best_fitness, best_loss, best_epoch

