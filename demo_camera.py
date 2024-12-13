import cv2
import torch
from utils.builder import init_detector
import numpy as np
from archs.PMRID import PMRIDd2
from archs._PMRIDu3 import PMRIDu3
from archs.pipelined_yunet import PipelinedYuNet
def crop_square(img, size, interpolation=cv2.INTER_AREA):
    #https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
    h, w = img.shape[:2]
    min_size = np.amin([h,w])

    # Centralize and crop
    crop_img = img[int(h/2-min_size/2):int(h/2+min_size/2), int(w/2-min_size/2):int(w/2+min_size/2)]
    resized = cv2.resize(crop_img, (size, size), interpolation=interpolation)

    return resized


def main():
    model = init_detector("experiment_config/pipelined_yunet_n.py",checkpoint="weights/yunet_n_retrained.pth")
    model.eval()

    ll_enhancer = PMRIDd2()
    denoiser = PMRIDd2()

    ll_enhancer.load_state_dict(torch.load("weights\PMRIDd2_LLE.pth")['params'])
    debug_denoiser = PMRIDd2()
    debug_denoiser.load_state_dict(torch.load("weights\PMRIDd2_denoise.pth")['params'])
    #debug_ll_enhancer = PMRIDd2()
    debug_ll_enhancer = torch.load("weights\pruned90_PMRIDd2_rescaled_LLE.pth",map_location='cuda:0')
    debug_model = init_detector("experiment_config/yunet_n.py",checkpoint="weights/yunet_n_retrained.pth")

    model.set_enhancers([denoiser,ll_enhancer])
    

    #model.load_state_dict(torch.load("weights\PipelinedYuNET.pth")['state_dict'])
    model = torch.load("weights\pruned90_pipelinedYuNET.pth")
    model.backbone = debug_model.backbone
    model.neck = debug_model.neck
    model.bbox_head = debug_model.bbox_head

    ll_enhancer.eval()
    denoiser.eval()
    debug_denoiser.to('cuda')
    denoiser.to('cuda')
    ll_enhancer.to('cuda')
    debug_ll_enhancer.to('cuda')
    cap = cv2.VideoCapture(0)
    dummy_img = np.load("dummy_img.npy")

    if not cap.isOpened():
        return
    
    while True:
        ret, frame = cap.read()
        frame = cv2.cvtColor(crop_square(frame,256), cv2.COLOR_BGR2RGB)



        orig_frame = cv2.cvtColor(frame.copy(),cv2.COLOR_BGR2RGB)
        frame = np.transpose(dummy_img,(1,2,0))
        tensor_frame = torch.Tensor(frame).to('cuda').permute(2,0,1).unsqueeze(0)
        with torch.no_grad():
            denoised_frame = (torch.clip(debug_denoiser(tensor_frame/255)*244,0,255)).squeeze().permute(1,2,0).cpu().numpy().astype(np.uint8)


        ### Baseline : Histogram equalization: colored

        img_yuv = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)

        # equalize the histogram of the Y channel
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

        # convert the YUV image back to RGB format
        frame_equalized = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        tensor_frame_equalized =  torch.Tensor(frame_equalized).to('cuda').permute(2,0,1).unsqueeze(0)


        ### Histogram equalization with denoiser

        img_yuv_denoised = cv2.cvtColor(denoised_frame, cv2.COLOR_RGB2YUV)

        # equalize the histogram of the Y channel
        img_yuv_denoised[:,:,0] = cv2.equalizeHist(img_yuv_denoised[:,:,0])

        # convert the YUV image back to RGB format
        frame_equalized_denoised = cv2.cvtColor(img_yuv_denoised, cv2.COLOR_YUV2BGR)
        tensor_frame_equalized_denoised =  torch.Tensor(frame_equalized_denoised).to('cuda').permute(2,0,1).unsqueeze(0)


        with torch.no_grad():

            ### Preds 1 : original image
            cls_preds, bbox_preds, obj_preds, kps_preds = debug_model(tensor_frame)

            result_list, _,  kps_list = debug_model.bbox_head.get_bboxes(cls_preds, bbox_preds, obj_preds, kps_preds)

            relighted_frame = model.output_enhancers_result(tensor_frame)

            ### Preds 2 : pipelined model, light enhanced and denoised

            cls_preds_cleaned, bbox_preds_cleaned, obj_preds_cleaned, kps_preds_cleaned = model(tensor_frame)
            print(cls_preds_cleaned)

            result_list_cleaned, _,  kps_list_cleaned = model.bbox_head.get_bboxes(cls_preds_cleaned, bbox_preds_cleaned, obj_preds_cleaned, kps_preds_cleaned)

            relighted_frame = (relighted_frame).squeeze().permute(1,2,0).cpu().numpy().astype(np.uint8)

            ### Preds 3 : Histogram equalized
            cls_preds_equalized, bbox_preds_equalized, obj_preds_equalized, kps_preds_equalized = debug_model(tensor_frame_equalized)
            result_list_equalized, _,  kps_list_equalized = debug_model.bbox_head.get_bboxes(cls_preds_equalized, bbox_preds_equalized, obj_preds_equalized, kps_preds_equalized)

            ### Preds 4 : Histogram equalized + denoised
            cls_preds_equalized_denoised, bbox_preds_equalized_denoised, obj_preds_equalized_denoised, kps_preds_equalized_denoised = debug_model(tensor_frame_equalized_denoised)
            result_list_equalized_denoised, _,  kps_list_equalized_denoised = debug_model.bbox_head.get_bboxes(cls_preds_equalized_denoised, bbox_preds_equalized_denoised, obj_preds_equalized_denoised, kps_preds_equalized_denoised)

            try:
                result = result_list[0][0]
                kps = kps_list[0][0].cpu().numpy().astype(np.uint8)

                            
                # Extract rectangle coordinates
                x1, y1, x2, y2 = result[:4].cpu().numpy().astype(np.uint8)
                # Draw the rectangle
                cv2.rectangle(frame, (y1, x1), (y2, x2), (0, 255, 0), 2)

                # Loop through the remaining values in pairs and draw the points
                for i in range(0, len(kps), 2):
                    x, y = kps[i], kps[i + 1]
                    cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
            except: 
                pass

            try:
                result = result_list_cleaned[0][0]
                kps = kps_list_cleaned[0][0].cpu().numpy().astype(np.uint8)

                            
                # Extract rectangle coordinates
                x1, y1, x2, y2 = result[:4].cpu().numpy().astype(np.uint8)
                # Draw the rectangle
                cv2.rectangle(relighted_frame, (y1, x1), (y2, x2), (0, 255, 0), 2)

                # Loop through the remaining values in pairs and draw the points
                for i in range(0, len(kps), 2):
                    x, y = kps[i], kps[i + 1]
                    cv2.circle(relighted_frame, (x, y), 3, (0, 0, 255), -1)
            except: 
                pass


            try:
                result = result_list_equalized[0][0]
                kps = kps_list_equalized[0][0].cpu().numpy().astype(np.uint8)

                            
                # Extract rectangle coordinates
                x1, y1, x2, y2 = result[:4].cpu().numpy().astype(np.uint8)
                # Draw the rectangle
                cv2.rectangle(frame_equalized, (y1, x1), (y2, x2), (0, 255, 0), 2)

                # Loop through the remaining values in pairs and draw the points
                for i in range(0, len(kps), 2):
                    x, y = kps[i], kps[i + 1]
                    cv2.circle(frame_equalized, (x, y), 3, (0, 0, 255), -1)
            except: 
                pass

            try:
                result = result_list_equalized_denoised[0][0]
                kps = kps_list_equalized_denoised[0][0].cpu().numpy().astype(np.uint8)

                            
                # Extract rectangle coordinates
                x1, y1, x2, y2 = result[:4].cpu().numpy().astype(np.uint8)
                # Draw the rectangle
                cv2.rectangle(frame_equalized_denoised, (y1, x1), (y2, x2), (0, 255, 0), 2)

                # Loop through the remaining values in pairs and draw the points
                for i in range(0, len(kps), 2):
                    x, y = kps[i], kps[i + 1]
                    cv2.circle(frame_equalized_denoised, (x, y), 3, (0, 0, 255), -1)
            except: 
                pass
        if not ret:
            break
        if not ret:
            break



        relighted_frame =  cv2.cvtColor(relighted_frame, cv2.COLOR_RGB2BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #cv2.imshow('Original frame', orig_frame)
        cv2.imshow('Relighted frame',relighted_frame)
        #cv2.imshow('Baseline detector',frame)
        #cv2.imshow('Histogram equalized', frame_equalized)
        #cv2.imshow('HE + Denoising', frame_equalized_denoised)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    
if __name__ == "__main__":
    main()