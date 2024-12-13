from .image_processing import rgb_to_yuv
import numpy as np
import cv2
from skimage.metrics import structural_similarity
import torch
import json 
import os


def _jaccard(gt, pred):
    """
    
    gt : 
    pred : 
    
    """
    xa, ya, x2a, y2a = gt[0][1], \
                        gt[0][0], \
                        gt[1][1], \
                        gt[1][0]
    xb, yb, x2b, y2b = pred['x_bbox'], pred['y_bbox'], pred['x_bbox'] + pred['width'], pred['y_bbox'] + pred['height']

    # innermost left x
    xi = max(xa, xb)
    # innermost right x
    x2i = min(x2a, x2b)
    # same for y
    yi = max(ya, yb)
    y2i = min(y2a, y2b)

    # calculate areas
    Aa = max(x2a - xa, 0) * max(y2a - ya, 0)
    Ab = max(x2b - xb, 0) * max(y2b - yb, 0)
    Ai = max(x2i - xi, 0) * max(y2i - yi, 0)

    return Ai / (Aa + Ab - Ai)


def precision_recall_curve(gt_folder,pred, iou_threshold=0.5,save=False, filter_key = None):

    gt_json = os.listdir(gt_folder)
    filelist = []
    jaccard = np.array([])
    confidence = []
    TP = np.array([])
    FN = 0
    n_samples = len(gt_json)
    for gt_path in gt_json:
        filename, ext = os.path.splitext(gt_path)
        #TODO when dataset with multiple faces is available, un-hardcode max_faces according to gt schema.
        max_faces = 1
        with open(os.path.join(gt_folder,gt_path),'r') as f:
            gt = json.load(f)['face_bbox_coordinates']['2d_corners']


        #TODO np.vectorize this loop.
        jaccard_list = []
        if filter_key is not None:
            filename = filename + filter_key
        for _, face_pred in pred[filename].items(): #TODO remove leading zeros in detection script. str(int()) needed for the moment.
            if face_pred['confidence'] is not None:
                jaccard_list.append(_jaccard(gt,face_pred))
                filelist.append(filename)
                confidence.append(face_pred['confidence'])
            #TODO rework false negative counting when multiple faces are allowed
            else:
                FN += 1
        jaccard_single_image = np.array(jaccard_list)

        TP_single_image = jaccard_single_image >= iou_threshold
        TP_single_image[max_faces:] = False 

        jaccard = np.concatenate((jaccard,jaccard_single_image),axis=0)
        TP = np.concatenate((TP,TP_single_image),axis=0)
    filelist = np.array(filelist)
    confidence = np.array(confidence)
    sorted_confidence_ind = confidence.argsort()[::-1] #descending 
    TP = TP[sorted_confidence_ind]
    filelist = filelist[sorted_confidence_ind]
    jaccard = jaccard[sorted_confidence_ind]
    precision = np.cumsum(TP) / np.arange(1,len(TP) + 1)
    recall = np.cumsum(TP) / len(TP)
    if save:
        save_path = "precision_recall.png"
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Precision-Recall curve saved at {save_path}")
    return recall, precision, jaccard, TP, FN, filelist, n_samples



def calculate_nme(gt,pred,device, filter_key=None):
    """
    Disclaimer: Afaik, there is no canonical NME implementation. 
    The overall formula of the NME between a set of i-indexed ground_truth landmarks {g_i} predictions landmarks {p_i} is
    \sum_{i=0}^{n} l_2(g_i,p_i)/norm_factor
    
    where norm_factor is (to the best of my knowledge) number of landmarks * distance between ocular landmarks
    
    Reference implementations: 
    
    SPIGA 2nd place SOTA : https://github.com/andresprados/SPIGA/blob/a709c95cc93f8246d7bff4cfb970a2dd0d62ffed/spiga/eval/benchmark/metrics/landmarks.py#L99
    HRNet : https://github.com/HRNet/HRNet-Facial-Landmark-Detection/blob/master/lib/core/evaluation.py#L36"""

    N_LANDMARKS = 5
    filenames = []
    results = []
    
    left_eye_gt = np.array(gt["iris_coordinates"]["2d"][1]).astype(np.float32) /4 #TODO original images are 1024x1024 not 256x256
    right_eye_gt = np.array(gt["iris_coordinates"]["2d"][0]).astype(np.float32) /4
    left_mouth_gt = np.array(gt["landmarks_coordinates"]["2d"][10]).astype(np.float32) /4#4 right mouth #10 left mouth #3 point of nose, 0 eye left
    right_mouth_gt = np.array(gt["landmarks_coordinates"]["2d"][4]).astype(np.float32) /4
    nose_gt = np.array(gt["landmarks_coordinates"]["2d"][3]).astype(np.float32) /4
    interocular_dist = np.linalg.norm(left_eye_gt-right_eye_gt,ord=2)
    norm_factor = torch.tensor(interocular_dist*N_LANDMARKS, device=device)
    print(f"norm factor : {norm_factor}")
    curr_confidence = 0
    results = []
    result = torch.tensor(0., device=device)
    for face_pred, kps_pred in zip(pred.bbox,pred.kps):

        if face_pred[0][-1] is not None and face_pred[0][-1] > curr_confidence:
            curr_confidence = face_pred[-1]

            right_eye_pred = kps_pred[0][0:2] #First Keypoint

            
            left_eye_pred = kps_pred[0][2:4] #Second

            
            nose_pred = kps_pred[0][4:6] #Third
            

            left_mouth_pred = kps_pred[0][8:]  # Fourth 

            right_mouth_pred = kps_pred[0][6:8] #Fifth


            gt_points = torch.stack((
                    torch.tensor(left_eye_gt),
                    torch.tensor(right_eye_gt),
                    torch.tensor(left_mouth_gt),
                    torch.tensor(right_mouth_gt),
                    torch.tensor(nose_gt)
                    ), dim=-1).to(device)
            pred_points = torch.stack((
                    left_eye_pred,
                    right_eye_pred,
                    left_mouth_pred,
                    right_mouth_pred,
                    nose_pred
                    ), dim=-1).to(device)
            print("linalg")
            print(torch.linalg.norm(gt_points-pred_points,axis=0))
            result += torch.sum(torch.linalg.norm(gt_points-pred_points,axis=0)) / norm_factor
            results.append(result) #TODO decide if this is the correct manner of calculating NME when face is wrong
    return result / len(results)


def format_results(filename, experiment_name, filter_key, ground_truth_path):
    ## Face detection Metrics - IoU, Recall, Precision, 'Missed detections'

    results_for_plot = {}
    with open(filename,'r') as f:
            pred = json.load(f)

    recall, precision, jaccard, TP, n_FN, filelist, n_samples = precision_recall_curve(ground_truth_path,pred,filter_key=filter_key)


    print(f"Percentage of 'multiple-face' detections: {(len(TP) + n_FN - n_samples)/n_samples:.2%}")
    print(f"Percentage of missed detections: {n_FN/n_samples:.2%}")


    results_for_plot[experiment_name] = {
        'IoU' : jaccard,
        'multiple_faces' : (len(TP) + n_FN - n_samples)/n_samples,
        'missed_detections' : n_FN/n_samples,
        'recall' : recall,
        'precision' : precision,
    }

    ## Landmark detection Metrics - NME
    filenames_ll_noisy, nmes_ll_noisy = calculate_nme(ground_truth_path,pred,filter_key=filter_key)
    print(f"Count of images with at least one detection: {len(filenames_ll_noisy)}")
    filenames_filter = [int(filename.split("_")[0]) for filename in filenames_ll_noisy]
    filenames_gt_clean = np.array([True if int(filename.split("_")[0]) in filenames_filter else False for filename in filenames_gt ])

    nmes_gt_filtered = nmes_gt[filenames_gt_clean]
    print(f"Reference NME (same set, original images) : {np.mean(nmes_gt_filtered)}")
    print(f"NME (after possible degradation and restauration) : {np.mean(nmes_ll_noisy)}")

    results_for_plot['experience_name'] = experiment_name
    results_for_plot['NME_gt_comparison'] = nmes_gt_filtered
    results_for_plot['NME'] = nmes_ll_noisy
    results_for_plot['n_detected'] = len(filenames_ll_noisy)



def precision_recall_curve(gt_folder,pred, iou_threshold=0.5,save=False, filter_key = None):

    gt_json = os.listdir(gt_folder)
    filelist = []
    jaccard = np.array([])
    confidence = []
    TP = np.array([])
    FN = 0
    n_samples = len(gt_json)
    for gt_path in gt_json:
        filename, ext = os.path.splitext(gt_path)
        #TODO when dataset with multiple faces is available, un-hardcode max_faces according to gt schema.
        max_faces = 1
        with open(os.path.join(gt_folder,gt_path),'r') as f:
            gt = json.load(f)['face_bbox_coordinates']['2d_corners']


        #TODO np.vectorize this loop.
        jaccard_list = []
        if filter_key is not None:
            filename = filename + filter_key
        for _, face_pred in pred[filename].items(): #TODO remove leading zeros in detection script. str(int()) needed for the moment.
            if face_pred['confidence'] is not None:
                jaccard_list.append(_jaccard(gt,face_pred))
                filelist.append(filename)
                confidence.append(face_pred['confidence'])
            #TODO rework false negative counting when multiple faces are allowed
            else:
                FN += 1
        jaccard_single_image = np.array(jaccard_list)

        TP_single_image = jaccard_single_image >= iou_threshold
        TP_single_image[max_faces:] = False 

        jaccard = np.concatenate((jaccard,jaccard_single_image),axis=0)
        TP = np.concatenate((TP,TP_single_image),axis=0)
    filelist = np.array(filelist)
    confidence = np.array(confidence)
    sorted_confidence_ind = confidence.argsort()[::-1] #descending 
    TP = TP[sorted_confidence_ind]
    filelist = filelist[sorted_confidence_ind]
    jaccard = jaccard[sorted_confidence_ind]
    precision = np.cumsum(TP) / np.arange(1,len(TP) + 1)
    recall = np.cumsum(TP) / len(TP)
    if save:
        save_path = "precision_recall.png"
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Precision-Recall curve saved at {save_path}")
    return recall, precision, jaccard, TP, FN, filelist, n_samples