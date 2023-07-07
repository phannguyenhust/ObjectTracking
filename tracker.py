#limit the number of cpus used by high performance libraries
#python tracker.py --source car.mp4 --weights yolov5s.pt --show-vid --save-vid
import os
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import sys
sys.path.insert(0, './yolov5')

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_boxes, 
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
#-----------------------------------------------------------------------------------------------------------------

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

#-----------------------------------------------------------------------------------------------------------------

up_count = 0
down_count = 0
car_count = 0
truck_count = 0
tracker1 = []
tracker2 = []

dir_data = {}
#-----------------------------------------------------------------------------------------------------------------
def detect(opt):
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt,   imgsz, evaluate, half, project, name, exist_ok= \
        opt.output, opt.source, opt.weights, opt.deep_sort_model, opt.show_vid, opt.save_vid,  \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.name, opt.exist_ok
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')
    #-----------------------------------------------------------------------------------------------------------------

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(deep_sort_model,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)
    #-----------------------------------------------------------------------------------------------------------------

    # Initialize
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    #-----------------------------------------------------------------------------------------------------------------

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
#-----------------------------------------------------------------------------------------------------------------
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir
    #-----------------------------------------------------------------------------------------------------------------

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    #-----------------------------------------------------------------------------------------------------------------

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    #-----------------------------------------------------------------------------------------------------------------

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
#-----------------------------------------------------------------------------------------------------------------
    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'
#-----------------------------------------------------------------------------------------------------------------
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    #Điều kiện này kiểm tra xem chương trình phụ trợ PyTorch (pt) có được bật hay không và thiết bị không phải là CPU. Nếu điều kiện là đúng, nó sẽ thực hiện suy luận khởi động trên tenxơ bằng không.
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    #Khởi tạo dt danh sách (thời gian phát hiện) để lưu trữ thời lượng cho các giai đoạn khác nhau và biến seenđể theo dõi số lượng khung hình được xử lý.
    #vòng lặp chính 
    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        # print(f"Image: {img.shape} ")
        # print(f"Image Type: {type(img)} ")
        
        t1 = time_sync()

        #tiền xử lý hình ảnh: thay đổi kích thước hình ảnh, chuyển nó thành một tensor pytorch, chuẩn hóa giá trị pixel
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # print(f"Dim Before {img.shape}\n")
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        # print(f"Dim After unsqueeze {img.shape}\n")

        # Inference
        #sử dụng model để thực hiện phát hiện 
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
        pred = model(img, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS: triệt tiêu không tối đa loại bỏ các box dư thừa và ngưỡng iou
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        dt[2] += time_sync() - t3

        # Process detections: phát hiện và theo dõi bằng deepsort
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            s += '%gx%g ' % img.shape[2:]  # print string

            annotator = Annotator(im0, line_width=2, pil=not ascii)
            w, h = im0.shape[1],im0.shape[0]
            # print(f"W: {w} h {h}")F
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # cv2.line(im0,(0,int(w//2)),(300,w),(0,255,0),2)
                # pass detections to deepsort: đầu ra lưu vào output
                t4 = time_sync()
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4
#----------------------------------------------------------Xong------------------------------------
                # draw boxes for visualization: vẽ trên ảnh gốc im0. draw box từ deepsort_utils_draw
                if len(outputs) > 0: #len đầu ra >0
                    for j, (output, conf) in enumerate(zip(outputs, confs)):

                        bboxes = output[0:4] #lưu trữ tọa độ
                        id = output[4] #lưu trữ id track
                        cls = output[5] #lưu lable của class
                        # # print(f"Img: {im0.shape}\n")
                        _dir =  direction(id,bboxes[1])

                        #count
                        count_obj(bboxes,w,h,id,_dir,int(cls))
                        # print(im0.shape)
                        c = int(cls)  # integer class  #c là số tên class
                        label = f'{id} {names[c]} {conf:.2f}' #tạo label cùng id, name class và confidence score với định dạng là `"{id} {names[c]} {conf:.2f}"`
                        annotator.box_label(bboxes, label, color=colors(c, True)) #vẽ bouding box trên ảnh: sd box_lable để vẽ ảnh, lấy box, string, màu

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))

                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')
                #hàm ghi thông tin kết quả theo dõi
                    
            #nếu không có phát hiện, tăng tuổi của các đối tượng đc theo dõi và k cập nhật trạng thái
            else:
                deepsort.increment_ages()
                LOGGER.info('No detections')
#-------------------------------------------------------------Xong---------------------------------------------------
            # Stream results
            im0 = annotator.result() #lưu = truy xuất hình ảnh vs các hộp giới hạn và nhãn
            global up_count,down_count
            # print(f"Shape: {im0.shape}")

            # Green Line01
            cv2.line(im0, (0, h-300), (w, h-300), (0,255,0), thickness=2)
            # Blue Line02
            cv2.line(im0, (0, h-150), (w, h-150), (255,0,0), thickness=2)
            thickness = 2

            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1

            #Objects 
            cv2.putText(im0, "Cars: "+str(car_count),(10,140),font,2,(0,0,255),3,cv2.LINE_AA) 
            cv2.putText(im0, "Trucks: "+str(truck_count),(10,210),font,2,(0,0,255),3,cv2.LINE_AA)
            cv2.putText(im0, "Up: "+str(up_count),(10,70),font,2,(0,0,255),3,cv2.LINE_AA) 
            cv2.putText(im0, "Down: "+str(down_count),(900,70),font,2,(0,0,255),3,cv2.LINE_AA) 
            if show_vid:
                show_vid = check_imshow()

            # Save results (image with detections)
            # w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            # h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # output_file = 'output.mp4'
            # fps = vid_cap.get(cv2.CAP_PROP_FPS)
            # vid_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (1000,700))
            # vid_writer.write(im0)
            if True:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
                vid_writer.write(im0)
        # print(im0.shape)        
#-----------------------------------------------------------------------------------------------------------------
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
        per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        print('Results saved to %s' % save_path)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)


#-----------------------------------------------------------------------------------------------------------------

def count_obj(box,w,h,id,direct,cls):
    global up_count,down_count,tracker1, tracker2, car_count, truck_count
    cx, cy = (int(box[0]+(box[2]-box[0])/2) , int(box[1]+(box[3]-box[1])/2))

    # For South
    # if cy<= int(h//2):
    #     return
    if direct=="South":
        if cy > (h - 300):
            if id not in tracker1:
                print(f"\nID: {id}, H: {h} South\n")
                down_count +=1
                tracker1.append(id)

                if cls==2:
                    car_count+=1
                elif cls==7:
                    truck_count+=1
            
    elif direct=="North":
        if cy < (h - 150):
            if id not in tracker2:
                print(f"\nID: {id}, H: {h} North\n")
                up_count +=1
                tracker2.append(id)
                
                if cls==2:
                    car_count+=1
                elif cls==7:
                    truck_count+=1


#-----------------------------------------------------------------------------------------------------------------
        

def direction(id,y):
    global dir_data

    if id not in dir_data:
        dir_data[id] = y
    else:
        diff = dir_data[id] -y

        if diff<0:
            return "South"
        else:
            return "North"
    

#-----------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default='videos/Truck.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[480], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.35, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    with torch.no_grad():
        detect(opt)
#-----------------------------------------------------------------------------------------------------------------