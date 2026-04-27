
#### MPS으로 학습

epoch 당 18s ~ 30s 걸림

```bash
python scripts/run_active_training.py seed \
        --images data/solar/images/RGB \
        --device mps \
        --seed-labels ./workspace/labels_seed_r2 \
        --output ./workspace/round_2
Seed 이미지 100장 (수동 라벨 완료)
  train: 80장
  val: 20장
New https://pypi.org/project/ultralytics/8.4.41 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.4.37 🚀 Python-3.11.1 torch-2.9.1 MPS (Apple M4 Pro)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=4, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=workspace/round_2/dataset/data.yaml, degrees=5.0, deterministic=True, device=mps, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=50, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.5, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=1280, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=models/yolo11n.pt, momentum=0.937, mosaic=0.5, multi_scale=0.0, name=weights, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=15, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=workspace/round_2, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/round_2/weights, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.3, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.05, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=3

                   from  n    params  module                                       arguments                     
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                
  2                  -1  1      6640  ultralytics.nn.modules.block.C3k2            [32, 64, 1, False, 0.25]      
  3                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
  4                  -1  1     26080  ultralytics.nn.modules.block.C3k2            [64, 128, 1, False, 0.25]     
  5                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
  6                  -1  1     87040  ultralytics.nn.modules.block.C3k2            [128, 128, 1, True]           
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              
  8                  -1  1    346112  ultralytics.nn.modules.block.C3k2            [256, 256, 1, True]           
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]                 
 10                  -1  1    249728  ultralytics.nn.modules.block.C2PSA           [256, 256, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1    111296  ultralytics.nn.modules.block.C3k2            [384, 128, 1, False]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1     32096  ultralytics.nn.modules.block.C3k2            [256, 64, 1, False]           
 17                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1     86720  ultralytics.nn.modules.block.C3k2            [192, 128, 1, False]          
 20                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1    378880  ultralytics.nn.modules.block.C3k2            [384, 256, 1, True]           
 23        [16, 19, 22]  1    431257  ultralytics.nn.modules.head.Detect           [3, 16, None, [64, 128, 256]] 
YOLO11n summary: 182 layers, 2,590,425 parameters, 2,590,409 gradients, 6.4 GFLOPs

Transferred 448/499 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 5075.5±167.3 MB/s, size: 11795.0 KB)
train: Scanning /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/round_2/dataset/labels/train.cache... 80 images, 1 backgrounds, 8 corrupt: 100% ━━━━━━━━━━━━ 80/80 16.0Mit/s 0.0s
train: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/round_2/dataset/images/train/DJI_20251217130317_0029_Z.JPG: ignoring corrupt image/label: Label class 2 exceeds dataset class count 2. Possible class labels are 0-1
train: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/round_2/dataset/images/train/DJI_20251217130325_0032_Z.JPG: ignoring corrupt image/label: Label class 2 exceeds dataset class count 2. Possible class labels are 0-1
train: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/round_2/dataset/images/train/DJI_20251217130328_0033_Z.JPG: ignoring corrupt image/label: Label class 2 exceeds dataset class count 2. Possible class labels are 0-1
train: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/round_2/dataset/images/train/DJI_20251217130403_0046_Z.JPG: ignoring corrupt image/label: Label class 2 exceeds dataset class count 2. Possible class labels are 0-1
train: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/round_2/dataset/images/train/DJI_20251217130445_0062_Z.JPG: ignoring corrupt image/label: Label class 2 exceeds dataset class count 2. Possible class labels are 0-1
train: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/round_2/dataset/images/train/DJI_20251217130528_0078_Z.JPG: ignoring corrupt image/label: Label class 2 exceeds dataset class count 2. Possible class labels are 0-1
train: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/round_2/dataset/images/train/DJI_20251217130547_0085_Z.JPG: ignoring corrupt image/label: Label class 2 exceeds dataset class count 2. Possible class labels are 0-1
train: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/round_2/dataset/images/train/DJI_20251217130605_0092_Z.JPG: ignoring corrupt image/label: Label class 2 exceeds dataset class count 2. Possible class labels are 0-1
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 5209.8±352.5 MB/s, size: 12132.5 KB)
val: Scanning /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/round_2/dataset/labels/val.cache... 20 images, 0 backgrounds, 4 corrupt: 100% ━━━━━━━━━━━━ 20/20 10.5Mit/s 0.0s
val: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/round_2/dataset/images/val/DJI_20251217130331_0034_Z.JPG: ignoring corrupt image/label: Label class 2 exceeds dataset class count 2. Possible class labels are 0-1
val: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/round_2/dataset/images/val/DJI_20251217130515_0073_Z.JPG: ignoring corrupt image/label: Label class 2 exceeds dataset class count 2. Possible class labels are 0-1
val: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/round_2/dataset/images/val/DJI_20251217130558_0089_Z.JPG: ignoring corrupt image/label: Label class 2 exceeds dataset class count 2. Possible class labels are 0-1
val: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/round_2/dataset/images/val/DJI_20251217130608_0093_Z.JPG: ignoring corrupt image/label: Label class 2 exceeds dataset class count 2. Possible class labels are 0-1
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001429, momentum=0.9) with parameter groups 81 weight(decay=0.0), 88 weight(decay=0.0005), 87 bias(decay=0.0)
Plotting labels to /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/round_2/weights/labels.jpg... 
Image sizes 1280 train, 1280 val
Using 0 dataloader workers
Logging results to /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/round_2/weights
Starting training for 50 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/50      5.39G      1.173      3.172      1.241         91       1280: 100% ━━━━━━━━━━━━ 18/18 1.2s/it 22.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 2.0s/it 4.0s
                   all         16        163      0.168      0.972      0.577      0.374

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       2/50      5.31G     0.8668      2.475      1.036         44       1280: 100% ━━━━━━━━━━━━ 18/18 1.0s/it 18.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.5s/it 3.0s
                   all         16        163      0.072      0.993      0.661      0.344
```

#### CPU으로 학습

epoch 당 45s ~ 1:10 걸림

```bash
python scripts/run_active_training.py seed \
        --images data/solar/images/RGB \
        --device cpu \
        --seed-labels ./workspace/labels_seed_r2 \
        --output ./workspace/round_2
Seed 이미지 100장 (수동 라벨 완료)
  train: 80장
  val: 20장
New https://pypi.org/project/ultralytics/8.4.41 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.4.37 🚀 Python-3.11.1 torch-2.9.1 CPU (Apple M4 Pro)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=4, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=workspace/round_2/dataset/data.yaml, degrees=5.0, deterministic=True, device=cpu, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=50, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.5, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=1280, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=models/yolo11n.pt, momentum=0.937, mosaic=0.5, multi_scale=0.0, name=weights6, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=15, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=workspace/round_2, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/round_2/weights6, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.3, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.05, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=3

                   from  n    params  module                                       arguments                     
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                
  2                  -1  1      6640  ultralytics.nn.modules.block.C3k2            [32, 64, 1, False, 0.25]      
  3                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
  4                  -1  1     26080  ultralytics.nn.modules.block.C3k2            [64, 128, 1, False, 0.25]     
  5                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
  6                  -1  1     87040  ultralytics.nn.modules.block.C3k2            [128, 128, 1, True]           
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              
  8                  -1  1    346112  ultralytics.nn.modules.block.C3k2            [256, 256, 1, True]           
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]                 
 10                  -1  1    249728  ultralytics.nn.modules.block.C2PSA           [256, 256, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1    111296  ultralytics.nn.modules.block.C3k2            [384, 128, 1, False]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1     32096  ultralytics.nn.modules.block.C3k2            [256, 64, 1, False]           
 17                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1     86720  ultralytics.nn.modules.block.C3k2            [192, 128, 1, False]          
 20                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1    378880  ultralytics.nn.modules.block.C3k2            [384, 256, 1, True]           
 23        [16, 19, 22]  1    431257  ultralytics.nn.modules.head.Detect           [3, 16, None, [64, 128, 256]] 
YOLO11n summary: 182 layers, 2,590,425 parameters, 2,590,409 gradients, 6.4 GFLOPs

Transferred 448/499 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 12818.4±3891.4 MB/s, size: 11795.0 KB)
train: Scanning /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/round_2/dataset/labels/train.cache... 80 images, 1 backgrounds, 8 corrupt: 100% ━━━━━━━━━━━━ 80/80 11.6Mit/s 0.0s
train: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/round_2/dataset/images/train/DJI_20251217130317_0029_Z.JPG: ignoring corrupt image/label: Label class 2 exceeds dataset class count 2. Possible class labels are 0-1
train: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/round_2/dataset/images/train/DJI_20251217130325_0032_Z.JPG: ignoring corrupt image/label: Label class 2 exceeds dataset class count 2. Possible class labels are 0-1
train: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/round_2/dataset/images/train/DJI_20251217130328_0033_Z.JPG: ignoring corrupt image/label: Label class 2 exceeds dataset class count 2. Possible class labels are 0-1
train: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/round_2/dataset/images/train/DJI_20251217130403_0046_Z.JPG: ignoring corrupt image/label: Label class 2 exceeds dataset class count 2. Possible class labels are 0-1
train: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/round_2/dataset/images/train/DJI_20251217130445_0062_Z.JPG: ignoring corrupt image/label: Label class 2 exceeds dataset class count 2. Possible class labels are 0-1
train: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/round_2/dataset/images/train/DJI_20251217130528_0078_Z.JPG: ignoring corrupt image/label: Label class 2 exceeds dataset class count 2. Possible class labels are 0-1
train: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/round_2/dataset/images/train/DJI_20251217130547_0085_Z.JPG: ignoring corrupt image/label: Label class 2 exceeds dataset class count 2. Possible class labels are 0-1
train: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/round_2/dataset/images/train/DJI_20251217130605_0092_Z.JPG: ignoring corrupt image/label: Label class 2 exceeds dataset class count 2. Possible class labels are 0-1
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 13273.6±3396.8 MB/s, size: 12132.5 KB)
val: Scanning /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/round_2/dataset/labels/val.cache... 20 images, 0 backgrounds, 4 corrupt: 100% ━━━━━━━━━━━━ 20/20 12.0Mit/s 0.0s
val: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/round_2/dataset/images/val/DJI_20251217130331_0034_Z.JPG: ignoring corrupt image/label: Label class 2 exceeds dataset class count 2. Possible class labels are 0-1
val: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/round_2/dataset/images/val/DJI_20251217130515_0073_Z.JPG: ignoring corrupt image/label: Label class 2 exceeds dataset class count 2. Possible class labels are 0-1
val: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/round_2/dataset/images/val/DJI_20251217130558_0089_Z.JPG: ignoring corrupt image/label: Label class 2 exceeds dataset class count 2. Possible class labels are 0-1
val: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/round_2/dataset/images/val/DJI_20251217130608_0093_Z.JPG: ignoring corrupt image/label: Label class 2 exceeds dataset class count 2. Possible class labels are 0-1
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001429, momentum=0.9) with parameter groups 81 weight(decay=0.0), 88 weight(decay=0.0005), 87 bias(decay=0.0)
Plotting labels to /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/round_2/weights6/labels.jpg... 
Image sizes 1280 train, 1280 val
Using 0 dataloader workers
Logging results to /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/round_2/weights6
Starting training for 50 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/50         0G      1.173      3.172      1.241         91       1280: 100% ━━━━━━━━━━━━ 18/18 2.9s/it 52.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 2.4s/it 4.8s
                   all         16        163      0.171      0.972      0.578      0.375

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       2/50         0G     0.8656      2.473      1.035         44       1280: 100% ━━━━━━━━━━━━ 18/18 2.9s/it 52.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 2.5s/it 5.0s
                   all         16        163     0.0747      0.993      0.664       0.35
```