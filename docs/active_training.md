#### SAM2으로 초기 Seed 라벨 생성

```bash
python scripts/run_training.py \
        --images data/solar/images/RGB \
        --work-dir ./workspace \
        --strategy sam2 \
        --classes solar_panel \
        --steps auto_label
2026-04-27 14:12:53,517 [INFO] __main__: ============================================================
2026-04-27 14:12:53,517 [INFO] __main__: STEP 1: Auto-labeling (strategy=sam2)
2026-04-27 14:12:53,517 [INFO] __main__: ============================================================
2026-04-27 14:12:53,725 [INFO] solar_thermal.dataset.auto_label: 대상 이미지 50장, strategy=sam2
2026-04-27 14:13:24,053 [INFO] solar_thermal.dataset.auto_label: [1/50] DJI_20251217130200_0001_Z.JPG → 2 boxes (30s)
2026-04-27 14:13:57,188 [INFO] solar_thermal.dataset.auto_label: [2/50] DJI_20251217130204_0002_Z.JPG → 4 boxes (63s)
2026-04-27 14:14:31,495 [INFO] solar_thermal.dataset.auto_label: [3/50] DJI_20251217130206_0003_Z.JPG → 4 boxes (97s)
2026-04-27 14:15:06,832 [INFO] solar_thermal.dataset.auto_label: [4/50] DJI_20251217130209_0004_Z.JPG → 6 boxes (133s)
```

```bash
python scripts/run_active_training.py seed \
        --images data/solar/images/RGB \
        --seed-labels ./workspace/labels_s10 \
        --device mps \
        --output ./workspace/round_s10
Seed 이미지 10장 (수동 라벨 완료)
  train: 8장
  val: 2장
Ultralytics 8.4.41 🚀 Python-3.11.1 torch-2.11.0 MPS (Apple M4 Pro)
engine/trainer: agnostic_nms=False, amp=False, angle=1.0, augment=False, auto_augment=randaugment, batch=4, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=workspace/round_s10/dataset/data.yaml, degrees=5.0, deterministic=True, device=mps, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=50, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.5, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=1280, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=models/yolo11n.pt, momentum=0.937, mosaic=0.5, multi_scale=0.0, name=weights-2, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=15, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=workspace/round_s10, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/round_s10/weights-2, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.3, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.05, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
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
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 4045.5±901.0 MB/s, size: 12361.7 KB)
train: Scanning /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/round_s10/dataset/labels/train... 8 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 8/8 490.0it/s 0.0s
train: New cache created: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/round_s10/dataset/labels/train.cache
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 4344.0±649.2 MB/s, size: 11018.7 KB)
val: Scanning /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/round_s10/dataset/labels/val... 2 images, 1 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 2/2 3.5Kit/s 0.0s
val: New cache created: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/round_s10/dataset/labels/val.cache
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001429, momentum=0.9) with parameter groups 81 weight(decay=0.0), 88 weight(decay=0.0005), 87 bias(decay=0.0)
Plotting labels to /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/round_s10/weights-2/labels.jpg... 
Image sizes 1280 train, 1280 val
Using 0 dataloader workers
Logging results to /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/round_s10/weights-2
Starting training for 50 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/50      4.25G      1.564      3.711      1.472         45       1280: 100% ━━━━━━━━━━━━ 2/2 1.8s/it 3.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.3s/it 1.3s
                   all          2          3    0.00598          1     0.0242    0.00634

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       2/50      5.28G      1.232      3.559      1.223         35       1280: 100% ━━━━━━━━━━━━ 2/2 1.2it/s 1.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 3.8it/s 0.3s
                   all          2          3    0.00513          1      0.113     0.0252

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       3/50      5.26G      1.066      3.398      1.137         30       1280: 100% ━━━━━━━━━━━━ 2/2 1.6it/s 1.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 3.8it/s 0.3s
                   all          2          3    0.00505          1       0.23      0.161

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       4/50       4.3G     0.9027      3.445      1.052         27       1280: 100% ━━━━━━━━━━━━ 2/2 1.7it/s 1.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 4.0it/s 0.3s
                   all          2          3    0.00504          1      0.288      0.227

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       5/50      4.34G     0.8236      3.183      1.016         26       1280: 100% ━━━━━━━━━━━━ 2/2 1.6it/s 1.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 3.6it/s 0.3s
                   all          2          3    0.00504          1      0.863      0.658

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       6/50      5.29G     0.8214      2.935       1.09         29       1280: 100% ━━━━━━━━━━━━ 2/2 1.5it/s 1.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 3.6it/s 0.3s
                   all          2          3    0.00504          1      0.863      0.658

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       7/50      5.29G     0.7595      2.961     0.9939         27       1280: 100% ━━━━━━━━━━━━ 2/2 1.9it/s 1.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 3.3it/s 0.3s
                   all          2          3    0.00503          1      0.995      0.785

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       8/50      5.26G     0.6639       2.75     0.9168         43       1280: 100% ━━━━━━━━━━━━ 2/2 1.4it/s 1.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 3.5it/s 0.3s
                   all          2          3    0.00504          1      0.995      0.863

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       9/50      5.26G     0.6224      2.515     0.9031         41       1280: 100% ━━━━━━━━━━━━ 2/2 1.5it/s 1.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 3.7it/s 0.3s
                   all          2          3    0.00504          1      0.995      0.863

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      10/50      5.26G     0.8926      2.544      1.041         29       1280: 100% ━━━━━━━━━━━━ 2/2 1.5it/s 1.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 3.2it/s 0.3s
                   all          2          3    0.00504          1      0.995      0.857

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      11/50      5.26G     0.6999      2.326     0.9751         30       1280: 100% ━━━━━━━━━━━━ 2/2 1.8it/s 1.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 3.7it/s 0.3s
                   all          2          3    0.00504          1      0.995      0.857

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      12/50      5.26G     0.8137      2.485      1.008         29       1280: 100% ━━━━━━━━━━━━ 2/2 1.7it/s 1.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 3.3it/s 0.3s
                   all          2          3    0.00504          1      0.995      0.912

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      13/50      4.31G       0.91      2.507      1.094         32       1280: 100% ━━━━━━━━━━━━ 2/2 1.5it/s 1.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 3.9it/s 0.3s
                   all          2          3    0.00504          1      0.995      0.912

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      14/50      5.28G     0.6618      2.169     0.9609         20       1280: 100% ━━━━━━━━━━━━ 2/2 1.5it/s 1.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 3.9it/s 0.3s
                   all          2          3    0.00504          1      0.995      0.912

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      15/50      4.31G     0.7208      2.186      1.005         44       1280: 100% ━━━━━━━━━━━━ 2/2 1.6it/s 1.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 3.4it/s 0.3s
                   all          2          3    0.00506          1      0.995      0.807

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      16/50       4.3G     0.7804      1.904      1.021         30       1280: 100% ━━━━━━━━━━━━ 2/2 1.6it/s 1.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 3.9it/s 0.3s
                   all          2          3    0.00506          1      0.995      0.807

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      17/50      4.28G     0.6264      2.308     0.9447         22       1280: 100% ━━━━━━━━━━━━ 2/2 1.6it/s 1.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 4.0it/s 0.3s
                   all          2          3    0.00506          1      0.995      0.807

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      18/50      4.29G     0.7922      1.992      1.035         25       1280: 100% ━━━━━━━━━━━━ 2/2 1.8it/s 1.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 3.3it/s 0.3s
                   all          2          3    0.00508          1      0.995      0.813

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      19/50      4.29G     0.6558      1.831     0.9203         25       1280: 100% ━━━━━━━━━━━━ 2/2 2.0it/s 1.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 4.0it/s 0.3s
                   all          2          3    0.00508          1      0.995      0.813

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      20/50      5.26G     0.7849      1.874       0.96         42       1280: 100% ━━━━━━━━━━━━ 2/2 1.3it/s 1.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 3.7it/s 0.3s
                   all          2          3    0.00508          1      0.995      0.813

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      21/50       5.3G     0.7541      1.773     0.9476         36       1280: 100% ━━━━━━━━━━━━ 2/2 1.6it/s 1.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 3.4it/s 0.3s
                   all          2          3    0.00515          1      0.995       0.84

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      22/50      5.29G      0.703      1.795     0.9212         27       1280: 100% ━━━━━━━━━━━━ 2/2 1.8it/s 1.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 3.7it/s 0.3s
                   all          2          3    0.00515          1      0.995       0.84

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      23/50      5.26G     0.7481       1.84     0.9875         37       1280: 100% ━━━━━━━━━━━━ 2/2 1.5it/s 1.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 3.7it/s 0.3s
                   all          2          3    0.00515          1      0.995       0.84

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      24/50       5.3G     0.6978      1.762      0.945         36       1280: 100% ━━━━━━━━━━━━ 2/2 1.7it/s 1.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 3.9it/s 0.3s
                   all          2          3    0.00515          1      0.995       0.84

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      25/50      4.29G     0.7079      1.792     0.9597         21       1280: 100% ━━━━━━━━━━━━ 2/2 1.9it/s 1.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 3.2it/s 0.3s
                   all          2          3     0.0053          1      0.995      0.852

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      26/50      4.33G     0.7343      1.927     0.9748         30       1280: 100% ━━━━━━━━━━━━ 2/2 1.6it/s 1.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 4.0it/s 0.2s
                   all          2          3     0.0053          1      0.995      0.852

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      27/50      4.29G     0.6886      1.751      1.031         30       1280: 100% ━━━━━━━━━━━━ 2/2 1.6it/s 1.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 4.0it/s 0.2s
                   all          2          3     0.0053          1      0.995      0.852
EarlyStopping: Training stopped early as no improvement observed in last 15 epochs. Best results observed at epoch 12, best model saved as best.pt.
To update EarlyStopping(patience=15) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

27 epochs completed in 0.017 hours.
Optimizer stripped from /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/round_s10/weights-2/weights/last.pt, 5.6MB
Optimizer stripped from /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/round_s10/weights-2/weights/best.pt, 5.6MB

Validating /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/round_s10/weights-2/weights/best.pt...
Ultralytics 8.4.41 🚀 Python-3.11.1 torch-2.11.0 MPS (Apple M4 Pro)
YOLO11n summary (fused): 101 layers, 2,582,737 parameters, 0 gradients, 6.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 2.4it/s 0.4s
                   all          2          3    0.00505          1      0.995      0.912
             pv_module          1          3    0.00505          1      0.995      0.912
Speed: 1.1ms preprocess, 74.9ms inference, 0.0ms loss, 32.1ms postprocess per image
Results saved to /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/round_s10/weights-2
Elapsed: 0:01:10
```

```bash
python scripts/run_active_training.py seed \
        --images data/solar/images/RGB \
        --seed-labels ./workspace/labels_s10 \
        --device cpu \                
        --output ./workspace/round_s10
Seed 이미지 10장 (수동 라벨 완료)
  train: 8장
  val: 2장
Ultralytics 8.4.41 🚀 Python-3.11.1 torch-2.11.0 CPU (Apple M4 Pro)
engine/trainer: agnostic_nms=False, amp=False, angle=1.0, augment=False, auto_augment=randaugment, batch=4, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=workspace/round_s10/dataset/data.yaml, degrees=5.0, deterministic=True, device=cpu, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=50, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.5, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=1280, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=models/yolo11n.pt, momentum=0.937, mosaic=0.5, multi_scale=0.0, name=weights-4, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=15, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=workspace/round_s10, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/round_s10/weights-4, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.3, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.05, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
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
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 13401.3±3262.5 MB/s, size: 12361.7 KB)
train: Scanning /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/round_s10/dataset/labels/train.cache... 8 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 8/8 2.8Mit/s 0.0s
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 13156.3±1998.0 MB/s, size: 11018.7 KB)
val: Scanning /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/round_s10/dataset/labels/val.cache... 2 images, 1 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 2/2 1.0Mit/s 0.0s
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001429, momentum=0.9) with parameter groups 81 weight(decay=0.0), 88 weight(decay=0.0005), 87 bias(decay=0.0)
Plotting labels to /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/round_s10/weights-4/labels.jpg... 
Image sizes 1280 train, 1280 val
Using 0 dataloader workers
Logging results to /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/round_s10/weights-4
Starting training for 50 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/50         0G      1.564      3.711      1.472         45       1280: 100% ━━━━━━━━━━━━ 2/2 2.6s/it 5.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.7it/s 0.6s
                   all          2          3    0.00599          1     0.0242    0.00635

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       2/50         0G      1.232      3.559      1.223         35       1280: 100% ━━━━━━━━━━━━ 2/2 2.3s/it 4.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.6it/s 0.6s
                   all          2          3    0.00513          1      0.113     0.0252

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       3/50         0G      1.066      3.398      1.137         30       1280: 100% ━━━━━━━━━━━━ 2/2 2.3s/it 4.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.6it/s 0.6s
                   all          2          3    0.00505          1       0.23      0.161

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       4/50         0G     0.9027      3.445      1.052         27       1280: 100% ━━━━━━━━━━━━ 2/2 2.3s/it 4.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.6it/s 0.6s
                   all          2          3    0.00504          1      0.288      0.227

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       5/50         0G     0.8236      3.183      1.016         26       1280: 100% ━━━━━━━━━━━━ 2/2 2.5s/it 5.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.6it/s 0.6s
                   all          2          3    0.00504          1      0.863      0.658

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       6/50         0G     0.8215      2.935       1.09         29       1280: 100% ━━━━━━━━━━━━ 2/2 2.4s/it 4.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.6it/s 0.6s
                   all          2          3    0.00504          1      0.863      0.658

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       7/50         0G     0.7596      2.961     0.9939         27       1280: 100% ━━━━━━━━━━━━ 2/2 2.4s/it 4.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.6it/s 0.6s
                   all          2          3    0.00503          1      0.995      0.785

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       8/50         0G     0.6639       2.75     0.9167         43       1280: 100% ━━━━━━━━━━━━ 2/2 2.4s/it 4.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.6it/s 0.6s
                   all          2          3    0.00504          1      0.995      0.863

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       9/50         0G     0.6225      2.515     0.9043         41       1280: 100% ━━━━━━━━━━━━ 2/2 2.3s/it 4.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.6it/s 0.6s
                   all          2          3    0.00504          1      0.995      0.863

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      10/50         0G     0.8926      2.544      1.041         29       1280: 100% ━━━━━━━━━━━━ 2/2 2.3s/it 4.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.6it/s 0.6s
                   all          2          3    0.00504          1      0.995      0.857

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      11/50         0G     0.7002      2.326     0.9765         30       1280: 100% ━━━━━━━━━━━━ 2/2 2.4s/it 4.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.6it/s 0.6s
                   all          2          3    0.00504          1      0.995      0.857

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      12/50         0G     0.8136      2.485      1.008         29       1280: 100% ━━━━━━━━━━━━ 2/2 2.4s/it 4.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.5it/s 0.6s
                   all          2          3    0.00504          1      0.995      0.912

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      13/50         0G     0.9099      2.508      1.094         32       1280: 100% ━━━━━━━━━━━━ 2/2 2.5s/it 5.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.5it/s 0.6s
                   all          2          3    0.00504          1      0.995      0.912

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      14/50         0G      0.661      2.169     0.9606         20       1280: 100% ━━━━━━━━━━━━ 2/2 2.4s/it 4.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.6it/s 0.6s
                   all          2          3    0.00504          1      0.995      0.912

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      15/50         0G     0.7215      2.186      1.005         44       1280: 100% ━━━━━━━━━━━━ 2/2 2.3s/it 4.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.6it/s 0.6s
                   all          2          3    0.00506          1      0.995      0.807

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      16/50         0G     0.7802      1.902      1.021         30       1280: 100% ━━━━━━━━━━━━ 2/2 2.4s/it 4.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.5it/s 0.7s
                   all          2          3    0.00506          1      0.995      0.807

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      17/50         0G     0.6248      2.305     0.9441         22       1280: 100% ━━━━━━━━━━━━ 2/2 2.4s/it 4.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.5it/s 0.7s
                   all          2          3    0.00506          1      0.995      0.807

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      18/50         0G     0.7925      1.993      1.035         25       1280: 100% ━━━━━━━━━━━━ 2/2 2.3s/it 4.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.5it/s 0.7s
                   all          2          3    0.00508          1      0.995      0.813

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      19/50         0G     0.6552      1.833     0.9196         25       1280: 100% ━━━━━━━━━━━━ 2/2 2.3s/it 4.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.6it/s 0.6s
                   all          2          3    0.00508          1      0.995      0.813

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      20/50         0G     0.7871      1.875     0.9604         42       1280: 100% ━━━━━━━━━━━━ 2/2 2.5s/it 4.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.5it/s 0.7s
                   all          2          3    0.00508          1      0.995      0.813

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      21/50         0G     0.7536      1.773     0.9469         36       1280: 100% ━━━━━━━━━━━━ 2/2 2.5s/it 4.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.5it/s 0.7s
                   all          2          3    0.00515          1      0.995       0.84

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      22/50         0G     0.7024      1.799     0.9213         27       1280: 100% ━━━━━━━━━━━━ 2/2 2.4s/it 4.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.5it/s 0.7s
                   all          2          3    0.00515          1      0.995       0.84

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      23/50         0G     0.7475       1.84     0.9884         37       1280: 100% ━━━━━━━━━━━━ 2/2 2.4s/it 4.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.5it/s 0.7s
                   all          2          3    0.00515          1      0.995       0.84

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      24/50         0G     0.6971      1.762     0.9448         36       1280: 100% ━━━━━━━━━━━━ 2/2 2.4s/it 4.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.5it/s 0.6s
                   all          2          3    0.00515          1      0.995       0.84

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      25/50         0G     0.7071      1.792     0.9595         21       1280: 100% ━━━━━━━━━━━━ 2/2 2.4s/it 4.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.5it/s 0.7s
                   all          2          3    0.00529          1      0.995      0.852

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      26/50         0G     0.7368       1.92     0.9741         30       1280: 100% ━━━━━━━━━━━━ 2/2 2.3s/it 4.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.5it/s 0.7s
                   all          2          3    0.00529          1      0.995      0.852

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      27/50         0G     0.6877      1.752      1.032         30       1280: 100% ━━━━━━━━━━━━ 2/2 2.5s/it 4.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.5it/s 0.7s
                   all          2          3    0.00529          1      0.995      0.852
EarlyStopping: Training stopped early as no improvement observed in last 15 epochs. Best results observed at epoch 12, best model saved as best.pt.
To update EarlyStopping(patience=15) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

27 epochs completed in 0.042 hours.
Optimizer stripped from /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/round_s10/weights-4/weights/last.pt, 5.6MB
Optimizer stripped from /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/round_s10/weights-4/weights/best.pt, 5.6MB

Validating /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/round_s10/weights-4/weights/best.pt...
Ultralytics 8.4.41 🚀 Python-3.11.1 torch-2.11.0 CPU (Apple M4 Pro)
YOLO11n summary (fused): 101 layers, 2,582,737 parameters, 0 gradients, 6.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.7it/s 0.6s
                   all          2          3    0.00505          1      0.995      0.912
             pv_module          1          3    0.00505          1      0.995      0.912
Speed: 1.1ms preprocess, 98.8ms inference, 0.0ms loss, 117.9ms postprocess per image
Results saved to /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/round_s10/weights-4
Elapsed: 0:02:40
```

```bash
python scripts/run_active_training.py seed \
        --images data/solar/images/RGB \
        --seed-labels ./workspace/labels_s20 \
        --device mps \
        --output ./workspace/round_s20
Seed 이미지 20장 (수동 라벨 완료)
  train: 16장
  val: 4장

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/50      5.26G      1.247      3.714      1.299         30       1280: 100% ━━━━━━━━━━━━ 4/4 1.5s/it 5.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.3s/it 1.3s
                   all          4         27     0.0671      0.842      0.273      0.169

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       2/50      4.32G     0.8555      3.456      1.027         25       1280: 100% ━━━━━━━━━━━━ 4/4 1.9it/s 2.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.8it/s 0.6s
                   all          4         27      0.188      0.967      0.344      0.209

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       3/50      4.35G     0.7248      3.085     0.9423         28       1280: 100% ━━━━━━━━━━━━ 4/4 1.7it/s 2.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.7it/s 0.6s
                   all          4         27      0.183          1      0.466      0.321

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      45/50      4.27G     0.6425      1.461     0.9096         15       1280: 100% ━━━━━━━━━━━━ 4/4 1.8it/s 2.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.8it/s 0.6s
                   all          4         27      0.992      0.971      0.995       0.84
EarlyStopping: Training stopped early as no improvement observed in last 15 epochs. Best results observed at epoch 30, best model saved as best.pt.
To update EarlyStopping(patience=15) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

45 epochs completed in 0.046 hours.
Optimizer stripped from /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/round_s20/weights/weights/last.pt, 5.6MB
Optimizer stripped from /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/round_s20/weights/weights/best.pt, 5.6MB

Validating /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/round_s20/weights/weights/best.pt...
Ultralytics 8.4.41 🚀 Python-3.11.1 torch-2.11.0 MPS (Apple M4 Pro)
YOLO11n summary (fused): 101 layers, 2,582,737 parameters, 0 gradients, 6.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.2s/it 1.2s
                   all          4         27      0.944      0.945      0.995      0.854
             pv_string          3         12      0.887          1      0.995      0.872
             pv_module          3         15          1      0.891      0.995      0.836
Speed: 0.7ms preprocess, 52.2ms inference, 0.0ms loss, 89.7ms postprocess per image
Results saved to /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/round_s20/weights
Elapsed: 0:02:56
```

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
전체 35m ~ 40m 걸림

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

#### Round 4: 100장 CPU 학습

100장을 MPS으로 학습할 때 오류가 발생해서, CPU으로 학습

```bash
python scripts/run_active_training.py seed \   
        --images data/solar/images/RGB \
        --seed-labels ./workspace/labels_r4_100 \                              
        --device cpu \   
        --output ./workspace/round_4_100
Seed 이미지 100장 (수동 라벨 완료)
  train: 80장
  val: 20장
New https://pypi.org/project/ultralytics/8.4.42 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.4.41 🚀 Python-3.11.1 torch-2.11.0 CPU (Apple M4 Pro)
engine/trainer: agnostic_nms=False, amp=False, angle=1.0, augment=False, auto_augment=randaugment, batch=4, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=workspace/round_4_100/dataset/data.yaml, degrees=5.0, deterministic=True, device=cpu, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=50, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.5, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=1280, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=models/yolo11n.pt, momentum=0.937, mosaic=0.5, multi_scale=0.0, name=weights-3, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=15, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=workspace/round_4_100, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/round_4_100/weights-3, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.3, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.05, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
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
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 5157.2±297.0 MB/s, size: 11795.0 KB)
train: Scanning /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/round_4_100/dataset/labels/train... 80 images, 1 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 80/80 2.1Kit/s 0.0s
train: New cache created: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/round_4_100/dataset/labels/train.cache
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 4947.0±432.5 MB/s, size: 12132.5 KB)
val: Scanning /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/round_4_100/dataset/labels/val... 20 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 20/20 3.7Kit/s 0.0s
val: New cache created: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/round_4_100/dataset/labels/val.cache
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001429, momentum=0.9) with parameter groups 81 weight(decay=0.0), 88 weight(decay=0.0005), 87 bias(decay=0.0)
Plotting labels to /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/round_4_100/weights-3/labels.jpg... 
Image sizes 1280 train, 1280 val
Using 0 dataloader workers
Logging results to /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/round_4_100/weights-3
Starting training for 50 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/50         0G      1.172      3.026      1.232         80       1280: 100% ━━━━━━━━━━━━ 20/20 3.0s/it 1:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0s/it 6.0s
                   all         20        203       0.18      0.989      0.533      0.329

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       2/50         0G     0.9228      2.056      1.037         41       1280: 100% ━━━━━━━━━━━━ 20/20 2.9s/it 58.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.1s/it 6.2s
                   all         20        203     0.0996      0.996      0.607      0.469

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       3/50         0G     0.7901      1.604      1.011         56       1280: 100% ━━━━━━━━━━━━ 20/20 3.0s/it 1:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.9s/it 5.8s
                   all         20        203      0.959      0.746      0.913      0.734

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       4/50         0G     0.8153      1.509      1.046         29       1280: 100% ━━━━━━━━━━━━ 20/20 3.1s/it 1:02
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.9s/it 5.8s
                   all         20        203      0.899      0.762      0.867      0.542


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      31/50         0G     0.5987     0.8526     0.8985         77       1280: 100% ━━━━━━━━━━━━ 20/20 2.7s/it 53.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.4s/it 4.3s
                   all         20        203      0.967      0.898      0.966      0.801

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      32/50         0G     0.6531     0.8109     0.9045         37       1280: 100% ━━━━━━━━━━━━ 20/20 2.7s/it 54.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.4s/it 4.2s
                   all         20        203      0.945      0.919      0.973      0.759
EarlyStopping: Training stopped early as no improvement observed in last 15 epochs. Best results observed at epoch 17, best model saved as best.pt.
To update EarlyStopping(patience=15) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

32 epochs completed in 0.577 hours.
Optimizer stripped from /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/round_4_100/weights-3/weights/last.pt, 5.6MB
Optimizer stripped from /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/round_4_100/weights-3/weights/best.pt, 5.6MB

Validating /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/round_4_100/weights-3/weights/best.pt...
Ultralytics 8.4.41 🚀 Python-3.11.1 torch-2.11.0 CPU (Apple M4 Pro)
YOLO11n summary (fused): 101 layers, 2,582,737 parameters, 0 gradients, 6.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.3s/it 4.0s
                   all         20        203      0.865      0.972      0.978      0.826
             pv_string         19        128      0.947      0.971      0.991      0.823
             pv_module         14         75      0.784      0.973      0.966      0.829
Speed: 2.5ms preprocess, 92.2ms inference, 0.0ms loss, 11.8ms postprocess per image
Results saved to /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/round_4_100/weights-3
Elapsed: 0:34:54
```

```bash
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s20 \
    --model models/yolo11s.pt \
    --device mps \
    --output ./workspace/train_s20_s
Seed 이미지 20장 (수동 라벨 완료)
  train: 16장
  val: 4장
Ultralytics 8.4.42 🚀 Python-3.11.1 torch-2.11.0 MPS (Apple M4 Pro)
engine/trainer: agnostic_nms=False, amp=False, angle=1.0, augment=False, auto_augment=randaugment, batch=4, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=workspace/train_s20_s/dataset/data.yaml, degrees=5.0, deterministic=True, device=mps, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=50, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.5, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=1280, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=models/yolo11s.pt, momentum=0.937, mosaic=0.5, multi_scale=0.0, name=weights, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=15, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=workspace/train_s20_s, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/train_s20_s/weights, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.3, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.05, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=3

                   from  n    params  module                                       arguments                     
  0                  -1  1       928  ultralytics.nn.modules.conv.Conv             [3, 32, 3, 2]                 
  1                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                
  2                  -1  1     26080  ultralytics.nn.modules.block.C3k2            [64, 128, 1, False, 0.25]     
  3                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
  4                  -1  1    103360  ultralytics.nn.modules.block.C3k2            [128, 256, 1, False, 0.25]    
  5                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
  6                  -1  1    346112  ultralytics.nn.modules.block.C3k2            [256, 256, 1, True]           
  7                  -1  1   1180672  ultralytics.nn.modules.conv.Conv             [256, 512, 3, 2]              
  8                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
 10                  -1  1    990976  ultralytics.nn.modules.block.C2PSA           [512, 512, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1    443776  ultralytics.nn.modules.block.C3k2            [768, 256, 1, False]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1    127680  ultralytics.nn.modules.block.C3k2            [512, 128, 1, False]          
 17                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1    345472  ultralytics.nn.modules.block.C3k2            [384, 256, 1, False]          
 20                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1   1511424  ultralytics.nn.modules.block.C3k2            [768, 512, 1, True]           
 23        [16, 19, 22]  1    820569  ultralytics.nn.modules.head.Detect           [3, 16, None, [128, 256, 512]]
YOLO11s summary: 182 layers, 9,428,953 parameters, 9,428,937 gradients, 21.6 GFLOPs

Transferred 493/499 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 9047.0±2987.5 MB/s, size: 11854.7 KB)
train: Scanning /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/train_s20_s/dataset/labels/train... 16 images, 1 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 16/16 1.1Kit/s 0.0s
train: New cache created: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/train_s20_s/dataset/labels/train.cache
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 6921.1±1245.3 MB/s, size: 12761.2 KB)
val: Scanning /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/train_s20_s/dataset/labels/val... 4 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 4/4 3.7Kit/s 0.0s
val: New cache created: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/train_s20_s/dataset/labels/val.cache
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001429, momentum=0.9) with parameter groups 81 weight(decay=0.0), 88 weight(decay=0.0005), 87 bias(decay=0.0)
Plotting labels to /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/train_s20_s/weights/labels.jpg... 
Image sizes 1280 train, 1280 val
Using 0 dataloader workers
Logging results to /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/train_s20_s/weights
Starting training for 50 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/50      8.38G      1.582      5.379      1.566         30       1280: 100% ━━━━━━━━━━━━ 4/4 1.8s/it 7.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.4s/it 1.4s
                   all          4         27      0.938      0.458      0.598       0.25

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       2/50      8.42G     0.8057      3.116      1.077         25       1280: 100% ━━━━━━━━━━━━ 4/4 1.2it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.2it/s 0.8s
                   all          4         27      0.769      0.671      0.842      0.623

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       3/50       8.4G     0.7164      2.454     0.9892         28       1280: 100% ━━━━━━━━━━━━ 4/4 1.1it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.2it/s 0.9s
                   all          4         27      0.662      0.575      0.593      0.441

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       4/50       8.4G     0.8205      2.072     0.9942         32       1280: 100% ━━━━━━━━━━━━ 4/4 1.0it/s 3.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.1it/s 0.9s
                   all          4         27      0.943      0.987      0.995      0.761

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       5/50       8.4G     0.7689      1.793     0.9928         33       1280: 100% ━━━━━━━━━━━━ 4/4 1.0it/s 4.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.2it/s 0.8s
                   all          4         27      0.972        0.8      0.932      0.718

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       6/50       8.4G      0.718      1.617     0.9483         24       1280: 100% ━━━━━━━━━━━━ 4/4 1.1it/s 3.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.1it/s 0.9s
                   all          4         27      0.921      0.949      0.989      0.859

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       7/50      8.39G     0.8164      1.522     0.9771         27       1280: 100% ━━━━━━━━━━━━ 4/4 1.1it/s 3.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.2it/s 0.8s
                   all          4         27      0.921      0.949      0.989      0.859

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       8/50       8.4G     0.7617      1.495     0.9709         52       1280: 100% ━━━━━━━━━━━━ 4/4 1.1it/s 3.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.2it/s 0.9s
                   all          4         27      0.915       0.98      0.995      0.806

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       9/50       8.4G     0.7108      1.507     0.9367         27       1280: 100% ━━━━━━━━━━━━ 4/4 1.1it/s 3.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.1it/s 0.9s
                   all          4         27      0.918      0.938      0.991      0.715

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      10/50      8.47G     0.7604      1.364     0.9394         31       1280: 100% ━━━━━━━━━━━━ 4/4 1.1it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.6it/s 0.6s
                   all          4         27      0.918      0.938      0.991      0.715

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      11/50      8.43G     0.8133       1.49     0.9346         32       1280: 100% ━━━━━━━━━━━━ 4/4 1.2it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.2it/s 0.8s
                   all          4         27      0.487      0.869       0.81      0.654

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      12/50       8.4G     0.7795      1.533     0.9565         31       1280: 100% ━━━━━━━━━━━━ 4/4 1.1it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.5it/s 0.7s
                   all          4         27      0.487      0.869       0.81      0.654

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      13/50      8.43G     0.7466      1.351      1.001         38       1280: 100% ━━━━━━━━━━━━ 4/4 1.1it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.1it/s 0.9s
                   all          4         27      0.956      0.907      0.995      0.791

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      14/50      8.44G     0.6951      1.204     0.9079         30       1280: 100% ━━━━━━━━━━━━ 4/4 1.2it/s 3.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.5it/s 0.7s
                   all          4         27      0.956      0.907      0.995      0.791

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      15/50      8.46G      0.794      1.209     0.9261         47       1280: 100% ━━━━━━━━━━━━ 4/4 1.0it/s 4.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.2it/s 0.9s
                   all          4         27      0.935          1      0.991      0.852

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      16/50      8.43G     0.6948      1.088      0.925         29       1280: 100% ━━━━━━━━━━━━ 4/4 1.1it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.5it/s 0.7s
                   all          4         27      0.935          1      0.991      0.852

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      17/50       8.4G     0.7923      1.034     0.9751         45       1280: 100% ━━━━━━━━━━━━ 4/4 1.1it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.6it/s 0.6s
                   all          4         27      0.935          1      0.991      0.852

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      18/50      8.44G     0.7932      1.055     0.9463         30       1280: 100% ━━━━━━━━━━━━ 4/4 1.1it/s 3.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.2it/s 0.9s
                   all          4         27      0.975          1      0.995      0.733

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      19/50      8.42G     0.8411      1.174     0.9633         30       1280: 100% ━━━━━━━━━━━━ 4/4 1.1it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.6it/s 0.6s
                   all          4         27      0.975          1      0.995      0.733

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      20/50       8.4G      0.876      1.091      1.006         43       1280: 100% ━━━━━━━━━━━━ 4/4 1.1it/s 3.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.6it/s 0.6s
                   all          4         27      0.975          1      0.995      0.733

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      21/50      8.39G     0.7133      1.165     0.9531         33       1280: 100% ━━━━━━━━━━━━ 4/4 1.1it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.6it/s 0.6s
                   all          4         27      0.975          1      0.995      0.733
EarlyStopping: Training stopped early as no improvement observed in last 15 epochs. Best results observed at epoch 6, best model saved as best.pt.
To update EarlyStopping(patience=15) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

21 epochs completed in 0.031 hours.
Optimizer stripped from /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/train_s20_s/weights/weights/last.pt, 19.3MB
Optimizer stripped from /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/train_s20_s/weights/weights/best.pt, 19.3MB

Validating /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/train_s20_s/weights/weights/best.pt...
Ultralytics 8.4.42 🚀 Python-3.11.1 torch-2.11.0 MPS (Apple M4 Pro)
YOLO11s summary (fused): 101 layers, 9,413,961 parameters, 0 gradients, 21.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.2s/it 1.2s
                   all          4         27      0.921      0.949      0.989      0.859
             pv_string          3         12      0.842          1      0.995      0.885
             pv_module          3         15          1      0.898      0.982      0.832
Speed: 0.7ms preprocess, 61.2ms inference, 0.0ms loss, 87.2ms postprocess per image
Results saved to /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/train_s20_s/weights
Elapsed: 0:02:04
```

```bash
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s20 \
    --model models/yolo11m.pt \
    --device mps \
    --output ./workspace/train_s20_m
Seed 이미지 20장 (수동 라벨 완료)
  train: 16장
  val: 4장
Ultralytics 8.4.42 🚀 Python-3.11.1 torch-2.11.0 MPS (Apple M4 Pro)
engine/trainer: agnostic_nms=False, amp=False, angle=1.0, augment=False, auto_augment=randaugment, batch=4, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=workspace/train_s20_m/dataset/data.yaml, degrees=5.0, deterministic=True, device=mps, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=50, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.5, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=1280, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=models/yolo11m.pt, momentum=0.937, mosaic=0.5, multi_scale=0.0, name=weights, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=15, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=workspace/train_s20_m, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/train_s20_m/weights, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.3, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.05, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=3

                   from  n    params  module                                       arguments                     
  0                  -1  1      1856  ultralytics.nn.modules.conv.Conv             [3, 64, 3, 2]                 
  1                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  2                  -1  1    111872  ultralytics.nn.modules.block.C3k2            [128, 256, 1, True, 0.25]     
  3                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
  4                  -1  1    444928  ultralytics.nn.modules.block.C3k2            [256, 512, 1, True, 0.25]     
  5                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  6                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  7                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  8                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
 10                  -1  1    990976  ultralytics.nn.modules.block.C2PSA           [512, 512, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1    542720  ultralytics.nn.modules.block.C3k2            [1024, 256, 1, True]          
 17                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1   1511424  ultralytics.nn.modules.block.C3k2            [768, 512, 1, True]           
 20                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 23        [16, 19, 22]  1   1413337  ultralytics.nn.modules.head.Detect           [3, 16, None, [256, 512, 512]]
YOLO11m summary: 232 layers, 20,055,321 parameters, 20,055,305 gradients, 68.2 GFLOPs

Transferred 643/649 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 9549.6±1965.1 MB/s, size: 11854.7 KB)
train: Scanning /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/train_s20_m/dataset/labels/train... 16 images, 1 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 16/16 1.1Kit/s 0.0s
train: New cache created: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/train_s20_m/dataset/labels/train.cache
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 9268.5±1755.8 MB/s, size: 12761.2 KB)
val: Scanning /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/train_s20_m/dataset/labels/val... 4 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 4/4 3.4Kit/s 0.0s
val: New cache created: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/train_s20_m/dataset/labels/val.cache
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001429, momentum=0.9) with parameter groups 106 weight(decay=0.0), 113 weight(decay=0.0005), 112 bias(decay=0.0)
Plotting labels to /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/train_s20_m/weights/labels.jpg... 
Image sizes 1280 train, 1280 val
Using 0 dataloader workers
Logging results to /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/train_s20_m/weights
Starting training for 50 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/50      16.8G      1.263      4.075      1.402         30       1280: 100% ━━━━━━━━━━━━ 4/4 4.4s/it 17.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 2.0s/it 2.0s
                   all          4         27      0.344      0.544      0.697      0.554

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       2/50      16.8G     0.6074      1.893     0.9321         25       1280: 100% ━━━━━━━━━━━━ 4/4 2.7s/it 10.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.3s/it 1.3s
                   all          4         27      0.821          1      0.995      0.792

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       3/50      16.4G     0.7187       1.62     0.9749         28       1280: 100% ━━━━━━━━━━━━ 4/4 13.0s/it 52.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 2.4s/it 2.4s
                   all          4         27      0.797          1      0.995      0.818

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       4/50      16.4G     0.8546      1.824      1.033         32       1280: 100% ━━━━━━━━━━━━ 4/4 32.9s/it 2:12
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.7s/it 1.7s
                   all          4         27      0.875        0.7      0.853      0.638

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       5/50      16.4G     0.8554      1.403      1.073         33       1280: 100% ━━━━━━━━━━━━ 4/4 9.4s/it 37.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.5s/it 1.5s
                   all          4         27      0.618      0.884      0.749      0.492

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       6/50      16.8G      1.064      1.397      1.103         24       1280: 100% ━━━━━━━━━━━━ 4/4 3.7s/it 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.5s/it 1.5s
                   all          4         27      0.612      0.912      0.818      0.636

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       7/50      16.8G     0.9619      1.296      1.018         27       1280: 100% ━━━━━━━━━━━━ 4/4 2.9s/it 11.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.2s/it 1.2s
                   all          4         27      0.612      0.912      0.818      0.636

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       8/50      16.4G     0.8953      1.421      1.067         52       1280: 100% ━━━━━━━━━━━━ 4/4 85.8s/it 5:43
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 3.5s/it 3.5s
                   all          4         27      0.747          1      0.995      0.802

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       9/50      16.8G     0.8206      1.455      1.006         27       1280: 100% ━━━━━━━━━━━━ 4/4 67.1s/it 4:28
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 6.3s/it 6.3s
                   all          4         27      0.529      0.862       0.79      0.589

      poch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      10/50      16.8G     0.8344      1.289     0.9709         31       1280: 100% ━━━━━━━━━━━━ 4/4 118.8s/it 7:55
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 2.3s/it 2.3s
                   all          4         27      0.529      0.862       0.79      0.589

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      11/50      16.4G       1.01      1.499      1.028         32       1280: 100% ━━━━━━━━━━━━ 4/4 13.6s/it 54.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.9s/it 1.9s
                   all          4         27      0.797          1       0.98        0.7

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      12/50      16.8G     0.9056      1.368      1.051         31       1280: 100% ━━━━━━━━━━━━ 4/4 66.0s/it 4:24
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 2.1s/it 2.1s
                   all          4         27      0.797          1       0.98        0.7

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      13/50      16.4G     0.8408      1.243      1.099         38       1280: 100% ━━━━━━━━━━━━ 4/4 27.3s/it 1:49
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 2.2s/it 2.2s
                   all          4         27      0.847          1      0.995      0.757

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      14/50      16.8G     0.7731       1.13       0.98         30       1280: 100% ━━━━━━━━━━━━ 4/4 4.7s/it 18.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.5s/it 1.5s
                   all          4         27      0.847          1      0.995      0.757

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      15/50      16.8G     0.8514      1.134      1.028         47       1280: 100% ━━━━━━━━━━━━ 4/4 9.7s/it 38.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.3s/it 1.3s
                   all          4         27      0.779      0.842      0.827      0.498

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      16/50      16.8G     0.8067      1.529      1.012         29       1280: 100% ━━━━━━━━━━━━ 4/4 2.4s/it 9.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.3s/it 1.3s
                   all          4         27      0.779      0.842      0.827      0.498

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      17/50      16.8G     0.8453      1.404       1.04         45       1280: 100% ━━━━━━━━━━━━ 4/4 2.3s/it 9.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.2s/it 1.2s
                   all          4         27      0.779      0.842      0.827      0.498

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      18/50      16.8G     0.9067      1.439      1.023         30       1280: 100% ━━━━━━━━━━━━ 4/4 2.4s/it 9.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.2s/it 1.2s
                   all          4         27      0.767      0.808      0.892      0.661
EarlyStopping: Training stopped early as no improvement observed in last 15 epochs. Best results observed at epoch 3, best model saved as best.pt.
To update EarlyStopping(patience=15) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

18 epochs completed in 0.567 hours.
Optimizer stripped from /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/train_s20_m/weights/weights/last.pt, 40.6MB
Optimizer stripped from /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/train_s20_m/weights/weights/best.pt, 40.6MB

Validating /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/train_s20_m/weights/weights/best.pt...
Ultralytics 8.4.42 🚀 Python-3.11.1 torch-2.11.0 MPS (Apple M4 Pro)
YOLO11m summary (fused): 126 layers, 20,032,345 parameters, 0 gradients, 67.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.2s/it 1.2s
                   all          4         27      0.797          1      0.995      0.818
             pv_string          3         12       0.62          1      0.995      0.803
             pv_module          3         15      0.974          1      0.995      0.833
Speed: 0.5ms preprocess, 76.8ms inference, 0.0ms loss, 60.9ms postprocess per image
Results saved to /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/train_s20_m/weights
Elapsed: 0:34:14
```

```bash
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s50_s \
    --model models/yolo11s.pt \
    --device mps \
    --output ./workspace/train_s50_s
Seed 이미지 50장 (수동 라벨 완료)
  train: 40장
  val: 10장
Ultralytics 8.4.42 🚀 Python-3.11.1 torch-2.11.0 MPS (Apple M4 Pro)
engine/trainer: agnostic_nms=False, amp=False, angle=1.0, augment=False, auto_augment=randaugment, batch=4, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=workspace/train_s50_s/dataset/data.yaml, degrees=5.0, deterministic=True, device=mps, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=50, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.5, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=1280, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=models/yolo11s.pt, momentum=0.937, mosaic=0.5, multi_scale=0.0, name=weights, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=15, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=workspace/train_s50_s, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/train_s50_s/weights, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.3, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.05, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=3

                   from  n    params  module                                       arguments                     
  0                  -1  1       928  ultralytics.nn.modules.conv.Conv             [3, 32, 3, 2]                 
  1                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                
  2                  -1  1     26080  ultralytics.nn.modules.block.C3k2            [64, 128, 1, False, 0.25]     
  3                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
  4                  -1  1    103360  ultralytics.nn.modules.block.C3k2            [128, 256, 1, False, 0.25]    
  5                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
  6                  -1  1    346112  ultralytics.nn.modules.block.C3k2            [256, 256, 1, True]           
  7                  -1  1   1180672  ultralytics.nn.modules.conv.Conv             [256, 512, 3, 2]              
  8                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
 10                  -1  1    990976  ultralytics.nn.modules.block.C2PSA           [512, 512, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1    443776  ultralytics.nn.modules.block.C3k2            [768, 256, 1, False]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1    127680  ultralytics.nn.modules.block.C3k2            [512, 128, 1, False]          
 17                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1    345472  ultralytics.nn.modules.block.C3k2            [384, 256, 1, False]          
 20                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1   1511424  ultralytics.nn.modules.block.C3k2            [768, 512, 1, True]           
 23        [16, 19, 22]  1    820569  ultralytics.nn.modules.head.Detect           [3, 16, None, [128, 256, 512]]
YOLO11s summary: 182 layers, 9,428,953 parameters, 9,428,937 gradients, 21.6 GFLOPs

Transferred 493/499 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 3944.0±1242.6 MB/s, size: 12414.0 KB)
train: Scanning /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/train_s50_s/dataset/labels/train... 40 images, 1 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 40/40 1.7Kit/s 0.0s
train: New cache created: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/train_s50_s/dataset/labels/train.cache
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 4087.3±725.9 MB/s, size: 12748.2 KB)
val: Scanning /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/train_s50_s/dataset/labels/val... 10 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 10/10 2.3Kit/s 0.0s
val: New cache created: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/train_s50_s/dataset/labels/val.cache
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001429, momentum=0.9) with parameter groups 81 weight(decay=0.0), 88 weight(decay=0.0005), 87 bias(decay=0.0)
Plotting labels to /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/train_s50_s/weights/labels.jpg... 
Image sizes 1280 train, 1280 val
Using 0 dataloader workers
Logging results to /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/train_s50_s/weights
Starting training for 50 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/50      8.39G      1.148      3.595      1.262         44       1280: 100% ━━━━━━━━━━━━ 10/10 1.9s/it 19.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 2.2s/it 4.5s
                   all         10         96      0.902      0.381      0.471      0.309

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       2/50       8.4G     0.8328      1.892      1.003         38       1280: 100% ━━━━━━━━━━━━ 10/10 1.4s/it 13.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.2s/it 2.4s
                   all         10         96      0.379      0.702      0.633      0.471

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       3/50       8.4G     0.8457      1.562      1.033         30       1280: 100% ━━━━━━━━━━━━ 10/10 1.3s/it 12.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.5s/it 3.0s
                   all         10         96      0.534      0.536      0.592      0.321

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       4/50       8.4G      0.825      1.437     0.9643         43       1280: 100% ━━━━━━━━━━━━ 10/10 1.3s/it 13.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.3s/it 2.5s
                   all         10         96      0.892      0.556      0.586      0.346

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       5/50      8.43G     0.9096       1.69       1.04         50       1280: 100% ━━━━━━━━━━━━ 10/10 1.2s/it 12.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.3s/it 2.5s
                   all         10         96      0.742      0.483       0.54      0.444

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       6/50      8.43G     0.9434      1.598      1.049         73       1280: 100% ━━━━━━━━━━━━ 10/10 1.2s/it 12.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.2s/it 2.4s
                   all         10         96      0.775      0.509        0.5      0.316

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       7/50       8.4G     0.8539      1.227      1.032         38       1280: 100% ━━━━━━━━━━━━ 10/10 1.5s/it 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.2s/it 2.5s
                   all         10         96      0.775      0.509        0.5      0.316

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       8/50       8.4G      0.816      1.061     0.9738         55       1280: 100% ━━━━━━━━━━━━ 10/10 1.3s/it 13.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.5s/it 2.9s
                   all         10         96      0.455      0.728      0.686      0.432

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       9/50      8.52G     0.7872      1.124     0.9703         71       1280: 100% ━━━━━━━━━━━━ 10/10 1.3s/it 12.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.3s/it 2.5s
                   all         10         96      0.945      0.504      0.632      0.449

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      10/50       8.4G     0.8633      1.071     0.9839         36       1280: 100% ━━━━━━━━━━━━ 10/10 1.7s/it 17.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.2it/s 1.7s
                   all         10         96      0.945      0.504      0.632      0.449

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      11/50       8.5G     0.8333      1.085      1.006         70       1280: 100% ━━━━━━━━━━━━ 10/10 1.5s/it 14.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.5s/it 3.1s
                   all         10         96      0.968      0.516      0.611      0.447

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      12/50      8.51G     0.9083      1.037      1.058         52       1280: 100% ━━━━━━━━━━━━ 10/10 1.3s/it 12.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.4s/it 2.7s
                   all         10         96      0.997      0.499      0.614      0.421

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      13/50       8.4G     0.8491     0.9138      1.031         53       1280: 100% ━━━━━━━━━━━━ 10/10 1.6s/it 16.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.1it/s 1.8s
                   all         10         96      0.997      0.499      0.614      0.421

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      14/50       8.4G     0.8138     0.8927      1.013         56       1280: 100% ━━━━━━━━━━━━ 10/10 1.7s/it 16.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 2.0s/it 4.1s
                   all         10         96      0.634      0.444      0.614      0.405

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      15/50       8.4G     0.8358        0.9      1.022         34       1280: 100% ━━━━━━━━━━━━ 10/10 1.5s/it 15.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.3s/it 2.6s
                   all         10         96      0.711      0.594      0.642      0.451

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      16/50       8.5G     0.8437     0.9075      1.027         62       1280: 100% ━━━━━━━━━━━━ 10/10 1.4s/it 13.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.2it/s 1.7s
                   all         10         96      0.711      0.594      0.642      0.451

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      17/50       8.4G     0.8332     0.9103      1.025         39       1280: 100% ━━━━━━━━━━━━ 10/10 1.4s/it 13.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.5s/it 3.1s
                   all         10         96      0.506      0.604      0.584      0.403
EarlyStopping: Training stopped early as no improvement observed in last 15 epochs. Best results observed at epoch 2, best model saved as best.pt.
To update EarlyStopping(patience=15) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

17 epochs completed in 0.086 hours.
Optimizer stripped from /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/train_s50_s/weights/weights/last.pt, 19.3MB
Optimizer stripped from /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/train_s50_s/weights/weights/best.pt, 19.3MB

Validating /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/train_s50_s/weights/weights/best.pt...
Ultralytics 8.4.42 🚀 Python-3.11.1 torch-2.11.0 MPS (Apple M4 Pro)
YOLO11s summary (fused): 101 layers, 9,413,961 parameters, 0 gradients, 21.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 3.1s/it 6.2s
                   all         10         96      0.378      0.702      0.634       0.47
             pv_string          8         43       0.21          1      0.988      0.717
             pv_module          8         40      0.676      0.875      0.809      0.623
                 other          4         13      0.247      0.231      0.104     0.0699
Speed: 4.1ms preprocess, 179.9ms inference, 0.0ms loss, 184.4ms postprocess per image
Results saved to /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/train_s50_s/weights
Elapsed: 0:05:31
```

```bash
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s50_m \
    --model models/yolo11m.pt \
    --device cuda \
    --output ./workspace/train_s50_m

Seed 이미지 50장 (수동 라벨 완료)
  train: 40장
  val: 10장
Ultralytics 8.4.42 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
engine/trainer: agnostic_nms=False, amp=False, angle=1.0, augment=False, auto_augment=randaugment, batch=4, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=workspace/train_s50_m/dataset/data.yaml, degrees=5.0, deterministic=True, device=0, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=50, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.5, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=1280, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=models/yolo11m.pt, momentum=0.937, mosaic=0.5, multi_scale=0.0, name=weights-2, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=15, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=workspace/train_s50_m, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/home/sjkim/solar-thermal/runs/detect/workspace/train_s50_m/weights-2, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.3, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.05, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=3

                   from  n    params  module                                       arguments                     
  0                  -1  1      1856  ultralytics.nn.modules.conv.Conv             [3, 64, 3, 2]                 
  1                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  2                  -1  1    111872  ultralytics.nn.modules.block.C3k2            [128, 256, 1, True, 0.25]     
  3                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
  4                  -1  1    444928  ultralytics.nn.modules.block.C3k2            [256, 512, 1, True, 0.25]     
  5                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  6                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  7                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  8                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
 10                  -1  1    990976  ultralytics.nn.modules.block.C2PSA           [512, 512, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1    542720  ultralytics.nn.modules.block.C3k2            [1024, 256, 1, True]          
 17                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1   1511424  ultralytics.nn.modules.block.C3k2            [768, 512, 1, True]           
 20                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 23        [16, 19, 22]  1   1413337  ultralytics.nn.modules.head.Detect           [3, 16, None, [256, 512, 512]]
YOLO11m summary: 232 layers, 20,055,321 parameters, 20,055,305 gradients, 68.2 GFLOPs

Transferred 643/649 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 3715.3±1204.5 MB/s, size: 12414.0 KB)
train: Scanning /home/sjkim/solar-thermal/workspace/train_s50_m/dataset/labels/train.cache... 40 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 40/40 7.3Mit/s 0.0s
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2116.6±1353.9 MB/s, size: 12870.9 KB)
val: Scanning /home/sjkim/solar-thermal/workspace/train_s50_m/dataset/labels/val.cache... 10 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 10/10 1.4Mit/s 0.0s
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001429, momentum=0.9) with parameter groups 106 weight(decay=0.0), 113 weight(decay=0.0005), 112 bias(decay=0.0)
Plotting labels to /home/sjkim/solar-thermal/runs/detect/workspace/train_s50_m/weights-2/labels.jpg... 
Image sizes 1280 train, 1280 val
Using 8 dataloader workers
Logging results to /home/sjkim/solar-thermal/runs/detect/workspace/train_s50_m/weights-2
Starting training for 50 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/50      16.4G      1.614      2.664      1.646         74       1280: 100% ━━━━━━━━━━━━ 10/10 1.5s/it 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.5it/s 1.3s
                   all         10        114      0.869      0.556      0.523      0.349

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       2/50      16.5G      1.297      1.847      1.402         46       1280: 100% ━━━━━━━━━━━━ 10/10 1.5it/s 6.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.2it/s 0.5s
                   all         10        114      0.834       0.57      0.585      0.445

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       3/50      16.5G      1.416      2.029      1.429         60       1280: 100% ━━━━━━━━━━━━ 10/10 1.5it/s 6.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.2it/s 0.5s
                   all         10        114      0.671      0.533      0.396      0.294

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       4/50      16.6G      1.264       1.66      1.324         79       1280: 100% ━━━━━━━━━━━━ 10/10 1.5it/s 6.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.3it/s 0.5s
                   all         10        114      0.789      0.614      0.544      0.349

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       5/50      16.5G      1.281      1.712      1.341         57       1280: 100% ━━━━━━━━━━━━ 10/10 1.5it/s 6.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.4it/s 0.5s
                   all         10        114      0.111      0.557      0.112     0.0742

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       6/50      16.5G      1.412      1.667      1.427         58       1280: 100% ━━━━━━━━━━━━ 10/10 1.5it/s 6.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.4it/s 0.5s
                   all         10        114      0.105      0.408       0.14     0.0812

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       7/50      16.5G      1.653      1.881       1.48         72       1280: 100% ━━━━━━━━━━━━ 10/10 1.5it/s 6.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.4it/s 0.5s
                   all         10        114      0.105      0.408       0.14     0.0812

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       8/50      16.6G      1.648      1.847      1.575         80       1280: 100% ━━━━━━━━━━━━ 10/10 1.5it/s 6.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.5it/s 0.4s
                   all         10        114      0.129       0.38       0.08     0.0388

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       9/50      16.5G      1.546      1.769      1.469         60       1280: 100% ━━━━━━━━━━━━ 10/10 1.5it/s 6.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.5it/s 0.4s
                   all         10        114       0.81      0.276       0.35      0.178

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      10/50      16.6G      1.563      2.058      1.505         52       1280: 100% ━━━━━━━━━━━━ 10/10 1.5it/s 6.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.5it/s 0.4s
                   all         10        114       0.81      0.276       0.35      0.178

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      11/50      16.6G      1.494      2.059      1.479         57       1280: 100% ━━━━━━━━━━━━ 10/10 1.5it/s 6.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.4it/s 0.5s
                   all         10        114      0.446       0.59       0.37      0.188

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      12/50      16.6G      1.545      2.165      1.545         71       1280: 100% ━━━━━━━━━━━━ 10/10 1.5it/s 6.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.6it/s 0.4s
                   all         10        114      0.771      0.579      0.494      0.326

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      13/50      16.6G      1.482      1.784      1.424         62       1280: 100% ━━━━━━━━━━━━ 10/10 1.5it/s 6.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.5it/s 0.4s
                   all         10        114      0.771      0.579      0.494      0.326

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      14/50      16.6G      1.585      1.873      1.535         55       1280: 100% ━━━━━━━━━━━━ 10/10 1.5it/s 6.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.5it/s 0.4s
                   all         10        114       0.91      0.361      0.381      0.208

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      15/50      16.6G      1.721      2.162      1.608         40       1280: 100% ━━━━━━━━━━━━ 10/10 1.4it/s 6.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.5it/s 0.4s
                   all         10        114      0.367      0.328       0.34      0.218

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      16/50      16.5G      1.376      1.605      1.407         66       1280: 100% ━━━━━━━━━━━━ 10/10 1.5it/s 6.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.5it/s 0.4s
                   all         10        114      0.367      0.328       0.34      0.218

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      17/50      16.6G       1.33      1.589      1.401         68       1280: 100% ━━━━━━━━━━━━ 10/10 1.4it/s 6.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.5it/s 0.4s
                   all         10        114       0.75      0.539      0.533      0.359
EarlyStopping: Training stopped early as no improvement observed in last 15 epochs. Best results observed at epoch 2, best model saved as best.pt.
To update EarlyStopping(patience=15) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

17 epochs completed in 0.045 hours.
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s50_m/weights-2/weights/last.pt, 40.6MB
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s50_m/weights-2/weights/best.pt, 40.6MB

Validating /home/sjkim/solar-thermal/runs/detect/workspace/train_s50_m/weights-2/weights/best.pt...
Ultralytics 8.4.42 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
YOLO11m summary (fused): 126 layers, 20,032,345 parameters, 0 gradients, 67.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.2it/s 0.5s
                   all         10        114      0.834       0.57      0.585      0.445
             pv_string          9         47      0.935      0.916      0.965      0.734
             pv_module          7         29      0.566      0.793      0.776      0.598
                 other          9         38          1          0     0.0132    0.00322
Speed: 0.6ms preprocess, 31.9ms inference, 0.0ms loss, 11.3ms postprocess per image
Results saved to /home/sjkim/solar-thermal/runs/detect/workspace/train_s50_m/weights-2
Elapsed: 0:02:58
```

#### 

```bash
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s50_m_2 \
    --model models/yolo11m.pt \
    --device mps \
    --output ./workspace/train_s50_m_2
Seed 이미지 50장 (수동 라벨 완료)
  train: 40장
  val: 10장
New https://pypi.org/project/ultralytics/8.4.43 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.4.42 🚀 Python-3.11.1 torch-2.11.0 MPS (Apple M4 Pro)
engine/trainer: agnostic_nms=False, amp=False, angle=1.0, augment=False, auto_augment=randaugment, batch=4, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=workspace/train_s50_m_2/dataset/data.yaml, degrees=5.0, deterministic=True, device=mps, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=50, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.5, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=1280, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=models/yolo11m.pt, momentum=0.937, mosaic=0.5, multi_scale=0.0, name=weights-2, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=15, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=workspace/train_s50_m_2, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/train_s50_m_2/weights-2, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.3, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.05, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=3

                   from  n    params  module                                       arguments                     
  0                  -1  1      1856  ultralytics.nn.modules.conv.Conv             [3, 64, 3, 2]                 
  1                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  2                  -1  1    111872  ultralytics.nn.modules.block.C3k2            [128, 256, 1, True, 0.25]     
  3                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
  4                  -1  1    444928  ultralytics.nn.modules.block.C3k2            [256, 512, 1, True, 0.25]     
  5                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  6                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  7                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  8                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
 10                  -1  1    990976  ultralytics.nn.modules.block.C2PSA           [512, 512, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1    542720  ultralytics.nn.modules.block.C3k2            [1024, 256, 1, True]          
 17                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1   1511424  ultralytics.nn.modules.block.C3k2            [768, 512, 1, True]           
 20                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 23        [16, 19, 22]  1   1413337  ultralytics.nn.modules.head.Detect           [3, 16, None, [256, 512, 512]]
YOLO11m summary: 232 layers, 20,055,321 parameters, 20,055,305 gradients, 68.2 GFLOPs

Transferred 643/649 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 9738.4±2159.9 MB/s, size: 12414.0 KB)
train: Scanning /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/train_s50_m_2/dataset/labels/train.cache... 40 images, 1 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 40/40 9.9Mit/s 0.0s
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 8215.7±1724.3 MB/s, size: 12748.2 KB)
val: Scanning /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/train_s50_m_2/dataset/labels/val.cache... 10 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 10/10 8.4Mit/s 0.0s
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001429, momentum=0.9) with parameter groups 106 weight(decay=0.0), 113 weight(decay=0.0005), 112 bias(decay=0.0)
Plotting labels to /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/train_s50_m_2/weights-2/labels.jpg... 
Image sizes 1280 train, 1280 val
Using 0 dataloader workers
Logging results to /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/train_s50_m_2/weights-2
Starting training for 50 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/50      16.4G      1.086      2.795      1.259         37       1280: 100% ━━━━━━━━━━━━ 10/10 145.1s/it 24:11
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 3.9s/it 7.8s
                   all         10         81      0.704      0.813      0.817      0.648

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       2/50      16.4G     0.9661      1.571      1.053         33       1280: 100% ━━━━━━━━━━━━ 10/10 249.4s/it 41:34
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 2.5s/it 5.0s
                   all         10         81      0.455      0.875      0.675      0.422

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       3/50      16.4G     0.9324      1.436      1.077         29       1280: 100% ━━━━━━━━━━━━ 10/10 159.2s/it 26:32
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 5.3s/it 10.6s
                   all         10         81      0.626      0.886      0.797      0.581

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       4/50      16.4G     0.7712      1.164       1.01         36       1280: 100% ━━━━━━━━━━━━ 10/10 33.2s/it 5:32
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 2.9s/it 5.9s
                   all         10         81      0.455      0.794      0.631      0.448

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       5/50      16.4G       1.07       1.31      1.214         45       1280: 100% ━━━━━━━━━━━━ 10/10 185.0s/it 30:50
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 9.7s/it 19.4s
                   all         10         81      0.389      0.575       0.51      0.347

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       6/50      16.4G      1.041      1.198      1.181         61       1280: 100% ━━━━━━━━━━━━ 10/10 270.5s/it 45:05
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 10.9s/it 21.9s
                   all         10         81      0.288      0.163      0.118     0.0923

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       7/50      16.4G      1.408       1.72      1.448         32       1280: 100% ━━━━━━━━━━━━ 10/10 112.6s/it 18:46
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 2.6s/it 5.2s
                   all         10         81      0.288      0.163      0.118     0.0923

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       8/50      16.4G       1.01      1.374      1.179         53       1280: 100% ━━━━━━━━━━━━ 10/10 58.6s/it 9:46
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 5.0s/it 9.9s
                   all         10         81      0.758      0.763       0.82      0.507

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       9/50      16.4G     0.9211      1.544      1.102         67       1280: 100% ━━━━━━━━━━━━ 10/10 34.6s/it 5:46
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 2.6s/it 5.2s
                   all         10         81      0.724        0.5      0.589      0.363

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      10/50      16.8G     0.9836      1.662      1.113         31       1280: 100% ━━━━━━━━━━━━ 10/10 28.1s/it 4:41
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 2.5s/it 5.0s
                   all         10         81      0.724        0.5      0.589      0.363

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      11/50      16.4G      1.213      1.854      1.196         57       1280: 100% ━━━━━━━━━━━━ 10/10 74.5s/it 12:25
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 3.4s/it 6.8s
                   all         10         81      0.203      0.852      0.397      0.178

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      12/50      16.4G       1.31      1.958      1.259         40       1280: 100% ━━━━━━━━━━━━ 10/10 118.4s/it 19:44
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 3.2s/it 6.3s
                   all         10         81      0.136      0.838      0.267      0.118

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      13/50      16.4G      1.225      1.819      1.166         48       1280: 100% ━━━━━━━━━━━━ 10/10 114.5s/it 19:05
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 3.1s/it 6.2s
                   all         10         81      0.136      0.838      0.267      0.118

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      14/50      16.4G      1.333      1.697      1.273         53       1280: 100% ━━━━━━━━━━━━ 10/10 114.1s/it 19:01
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 3.8s/it 7.5s
                   all         10         81      0.775      0.476      0.288     0.0766

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      15/50      16.4G       1.36      1.587      1.335         31       1280: 100% ━━━━━━━━━━━━ 10/10 95.3s/it 15:53
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 5.1s/it 10.2s
                   all         10         81       0.96       0.28      0.614      0.348

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      16/50      16.4G      1.173      1.379      1.193         59       1280: 100% ━━━━━━━━━━━━ 10/10 38.2s/it 6:22
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 5.8s/it 11.7s
                   all         10         81       0.96       0.28      0.614      0.348
EarlyStopping: Training stopped early as no improvement observed in last 15 epochs. Best results observed at epoch 1, best model saved as best.pt.
To update EarlyStopping(patience=15) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

16 epochs completed in 5.190 hours.
Optimizer stripped from /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/train_s50_m_2/weights-2/weights/last.pt, 40.6MB
Optimizer stripped from /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/train_s50_m_2/weights-2/weights/best.pt, 40.6MB

Validating /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/train_s50_m_2/weights-2/weights/best.pt...
Ultralytics 8.4.42 🚀 Python-3.11.1 torch-2.11.0 MPS (Apple M4 Pro)
YOLO11m summary (fused): 126 layers, 20,032,345 parameters, 0 gradients, 67.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 2.8s/it 5.5s
                   all         10         81      0.703      0.813      0.818      0.649
             pv_string          8         41      0.742      0.976      0.955      0.718
             pv_module          9         40      0.665       0.65       0.68      0.579
Speed: 3.0ms preprocess, 189.2ms inference, 0.0ms loss, 113.1ms postprocess per image
Results saved to /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/train_s50_m_2/weights-2
Elapsed: 5:11:45
```

```bash
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s50_m_2 \
    --model models/yolo11m.pt \
    --device cuda \
    --output ./workspace/train_s50_m_2
Seed 이미지 50장 (수동 라벨 완료)
  train: 40장
  val: 10장
New https://pypi.org/project/ultralytics/8.4.43 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.4.42 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
engine/trainer: agnostic_nms=False, amp=False, angle=1.0, augment=False, auto_augment=randaugment, batch=4, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=workspace/train_s50_m_2/dataset/data.yaml, degrees=5.0, deterministic=True, device=0, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=50, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.5, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=1280, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=models/yolo11m.pt, momentum=0.937, mosaic=0.5, multi_scale=0.0, name=weights, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=15, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=workspace/train_s50_m_2, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/home/sjkim/solar-thermal/runs/detect/workspace/train_s50_m_2/weights, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.3, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.05, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=3

                   from  n    params  module                                       arguments                     
  0                  -1  1      1856  ultralytics.nn.modules.conv.Conv             [3, 64, 3, 2]                 
  1                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  2                  -1  1    111872  ultralytics.nn.modules.block.C3k2            [128, 256, 1, True, 0.25]     
  3                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
  4                  -1  1    444928  ultralytics.nn.modules.block.C3k2            [256, 512, 1, True, 0.25]     
  5                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  6                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  7                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  8                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
 10                  -1  1    990976  ultralytics.nn.modules.block.C2PSA           [512, 512, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1    542720  ultralytics.nn.modules.block.C3k2            [1024, 256, 1, True]          
 17                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1   1511424  ultralytics.nn.modules.block.C3k2            [768, 512, 1, True]           
 20                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 23        [16, 19, 22]  1   1413337  ultralytics.nn.modules.head.Detect           [3, 16, None, [256, 512, 512]]
YOLO11m summary: 232 layers, 20,055,321 parameters, 20,055,305 gradients, 68.2 GFLOPs

Transferred 643/649 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2374.7±1257.9 MB/s, size: 12414.0 KB)
train: Scanning /home/sjkim/solar-thermal/workspace/train_s50_m_2/dataset/labels/train... 40 images, 1 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 40/40 957.0it/s 0.0s
train: New cache created: /home/sjkim/solar-thermal/workspace/train_s50_m_2/dataset/labels/train.cache
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2729.8±1313.9 MB/s, size: 12870.9 KB)
val: Scanning /home/sjkim/solar-thermal/workspace/train_s50_m_2/dataset/labels/val... 10 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 10/10 828.0it/s 0.0s
val: New cache created: /home/sjkim/solar-thermal/workspace/train_s50_m_2/dataset/labels/val.cache
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001429, momentum=0.9) with parameter groups 106 weight(decay=0.0), 113 weight(decay=0.0005), 112 bias(decay=0.0)
Plotting labels to /home/sjkim/solar-thermal/runs/detect/workspace/train_s50_m_2/weights/labels.jpg... 
Image sizes 1280 train, 1280 val
Using 8 dataloader workers
Logging results to /home/sjkim/solar-thermal/runs/detect/workspace/train_s50_m_2/weights
Starting training for 50 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/50      16.3G      1.212      2.585      1.364         46       1280: 100% ━━━━━━━━━━━━ 10/10 1.5s/it 15.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 3.5it/s 0.6s
                   all         10         78      0.683       0.79       0.76      0.534

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       2/50      16.5G     0.8335      1.579      1.034         27       1280: 100% ━━━━━━━━━━━━ 10/10 1.5it/s 6.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.3it/s 0.5s
                   all         10         78      0.751      0.925      0.918      0.672

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       3/50      16.5G      1.048      1.593      1.105         38       1280: 100% ━━━━━━━━━━━━ 10/10 1.5it/s 6.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.3it/s 0.5s
                   all         10         78      0.458      0.683      0.563      0.444

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       4/50      16.6G     0.9344      1.403      1.048         48       1280: 100% ━━━━━━━━━━━━ 10/10 1.5it/s 6.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.4it/s 0.5s
                   all         10         78      0.523      0.869      0.727      0.528

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       5/50      16.5G     0.9632      1.338      1.073         38       1280: 100% ━━━━━━━━━━━━ 10/10 1.5it/s 6.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.5it/s 0.4s
                   all         10         78      0.353      0.861      0.396      0.186

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       6/50      16.5G      1.141       1.36      1.232         37       1280: 100% ━━━━━━━━━━━━ 10/10 1.5it/s 6.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.5it/s 0.4s
                   all         10         78      0.356      0.791      0.419      0.207

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       7/50      16.5G      1.081      1.337      1.151         53       1280: 100% ━━━━━━━━━━━━ 10/10 1.5it/s 6.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.5it/s 0.4s
                   all         10         78      0.356      0.791      0.419      0.207

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       8/50      16.6G      1.224      1.535      1.177         52       1280: 100% ━━━━━━━━━━━━ 10/10 1.5it/s 6.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.6it/s 0.4s
                   all         10         78      0.302      0.769      0.343      0.161

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       9/50      16.5G       1.18      1.273      1.138         53       1280: 100% ━━━━━━━━━━━━ 10/10 1.5it/s 6.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.4it/s 0.4s
                   all         10         78      0.314      0.765      0.408      0.247

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      10/50      16.6G     0.9931      1.161      1.114         34       1280: 100% ━━━━━━━━━━━━ 10/10 1.5it/s 6.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.5it/s 0.4s
                   all         10         78      0.314      0.765      0.408      0.247

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      11/50      16.6G      1.221      1.104      1.194         46       1280: 100% ━━━━━━━━━━━━ 10/10 1.5it/s 6.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.5it/s 0.4s
                   all         10         78      0.556      0.871      0.616      0.249

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      12/50      16.5G      1.213      1.189      1.228         42       1280: 100% ━━━━━━━━━━━━ 10/10 1.5it/s 6.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.5it/s 0.4s
                   all         10         78      0.395      0.754      0.441      0.248

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      13/50      16.6G      1.235      1.146      1.197         35       1280: 100% ━━━━━━━━━━━━ 10/10 1.5it/s 6.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.5it/s 0.4s
                   all         10         78      0.395      0.754      0.441      0.248

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      14/50      16.5G      1.232      1.191      1.306         37       1280: 100% ━━━━━━━━━━━━ 10/10 1.5it/s 6.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.5it/s 0.4s
                   all         10         78      0.139      0.828       0.19      0.118

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      15/50      16.5G      1.276      1.295      1.277         26       1280: 100% ━━━━━━━━━━━━ 10/10 1.5it/s 6.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.6it/s 0.4s
                   all         10         78       0.17       0.78      0.515      0.252

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      16/50      16.5G      1.068      1.003      1.095         40       1280: 100% ━━━━━━━━━━━━ 10/10 1.5it/s 6.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.5it/s 0.4s
                   all         10         78       0.17       0.78      0.515      0.252

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      17/50      16.5G      1.061       1.06      1.237         47       1280: 100% ━━━━━━━━━━━━ 10/10 1.5it/s 6.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.4it/s 0.5s
                   all         10         78      0.139      0.632      0.285      0.134
EarlyStopping: Training stopped early as no improvement observed in last 15 epochs. Best results observed at epoch 2, best model saved as best.pt.
To update EarlyStopping(patience=15) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

17 epochs completed in 0.045 hours.
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s50_m_2/weights/weights/last.pt, 40.6MB
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s50_m_2/weights/weights/best.pt, 40.6MB

Validating /home/sjkim/solar-thermal/runs/detect/workspace/train_s50_m_2/weights/weights/best.pt...
Ultralytics 8.4.42 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
YOLO11m summary (fused): 126 layers, 20,032,345 parameters, 0 gradients, 67.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.2it/s 0.5s
                   all         10         78      0.751      0.925      0.917      0.671
             pv_string          9         47      0.879      0.979      0.974      0.657
             pv_module          7         31      0.623      0.871       0.86      0.685
Speed: 0.6ms preprocess, 32.0ms inference, 0.0ms loss, 10.8ms postprocess per image
Results saved to /home/sjkim/solar-thermal/runs/detect/workspace/train_s50_m_2/weights
Elapsed: 0:02:57
```

```bash
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s50_m_3 \
    --model models/yolo11m.pt \
    --device cuda \
    --amp True \
    --output ./workspace/train_s50_m_3
Seed 이미지 50장 (수동 라벨 완료)
  train: 40장
  val: 10장
New https://pypi.org/project/ultralytics/8.4.43 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.4.42 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=4, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=workspace/train_s50_m_3/dataset/data.yaml, degrees=5.0, deterministic=True, device=0, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=50, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.5, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=1280, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=models/yolo11m.pt, momentum=0.937, mosaic=0.5, multi_scale=0.0, name=weights-2, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=15, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=workspace/train_s50_m_3, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/home/sjkim/solar-thermal/runs/detect/workspace/train_s50_m_3/weights-2, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.3, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.05, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=3

                   from  n    params  module                                       arguments                     
  0                  -1  1      1856  ultralytics.nn.modules.conv.Conv             [3, 64, 3, 2]                 
  1                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  2                  -1  1    111872  ultralytics.nn.modules.block.C3k2            [128, 256, 1, True, 0.25]     
  3                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
  4                  -1  1    444928  ultralytics.nn.modules.block.C3k2            [256, 512, 1, True, 0.25]     
  5                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  6                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  7                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  8                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
 10                  -1  1    990976  ultralytics.nn.modules.block.C2PSA           [512, 512, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1    542720  ultralytics.nn.modules.block.C3k2            [1024, 256, 1, True]          
 17                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1   1511424  ultralytics.nn.modules.block.C3k2            [768, 512, 1, True]           
 20                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 23        [16, 19, 22]  1   1413337  ultralytics.nn.modules.head.Detect           [3, 16, None, [256, 512, 512]]
YOLO11m summary: 232 layers, 20,055,321 parameters, 20,055,305 gradients, 68.2 GFLOPs

Transferred 643/649 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 100% ━━━━━━━━━━━━ 5.3MB 66.1MB/s 0.1s
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2017.1±1094.5 MB/s, size: 12414.0 KB)
train: Scanning /home/sjkim/solar-thermal/workspace/train_s50_m_3/dataset/labels/train... 40 images, 1 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 40/40 923.5it/s 0.0s
train: New cache created: /home/sjkim/solar-thermal/workspace/train_s50_m_3/dataset/labels/train.cache
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2117.2±893.8 MB/s, size: 12870.9 KB)
val: Scanning /home/sjkim/solar-thermal/workspace/train_s50_m_3/dataset/labels/val... 10 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 10/10 770.3it/s 0.0s
val: New cache created: /home/sjkim/solar-thermal/workspace/train_s50_m_3/dataset/labels/val.cache
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001429, momentum=0.9) with parameter groups 106 weight(decay=0.0), 113 weight(decay=0.0005), 112 bias(decay=0.0)
Plotting labels to /home/sjkim/solar-thermal/runs/detect/workspace/train_s50_m_3/weights-2/labels.jpg... 
Image sizes 1280 train, 1280 val
Using 8 dataloader workers
Logging results to /home/sjkim/solar-thermal/runs/detect/workspace/train_s50_m_3/weights-2
Starting training for 50 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/50      8.09G      1.946      3.765      1.911         63       1280: 100% ━━━━━━━━━━━━ 10/10 1.6s/it 16.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.3s/it 2.6s
                   all         10        106      0.829      0.395      0.445      0.176

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       2/50      8.57G      1.301      2.096      1.357         39       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.4it/s 0.3s
                   all         10        106      0.663      0.502      0.376      0.248

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       3/50      8.57G      1.264       1.88      1.314         54       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.3it/s 0.3s
                   all         10        106      0.672      0.617      0.433       0.32

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       4/50       8.6G       1.27      1.602      1.321         60       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.1it/s 0.3s
                   all         10        106      0.636      0.475      0.418      0.248

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       5/50       8.6G      1.275      1.684      1.332         51       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.9it/s 0.3s
                   all         10        106      0.633      0.531      0.466      0.337

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       6/50       8.6G       1.29      1.744      1.357         53       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.1it/s 0.2s
                   all         10        106      0.529      0.418      0.211      0.143

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       7/50       8.6G      1.352      1.659       1.36         61       1280: 100% ━━━━━━━━━━━━ 10/10 2.9it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.9it/s 0.3s
                   all         10        106      0.529      0.418      0.211      0.143

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       8/50       8.6G      1.367      1.718      1.377         72       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.9it/s 0.3s
                   all         10        106      0.432      0.551      0.493      0.249

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       9/50       8.6G      1.407      1.632      1.357         59       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.7it/s 0.3s
                   all         10        106      0.292      0.541      0.249      0.211

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      10/50       8.6G      1.291      1.671      1.318         46       1280: 100% ━━━━━━━━━━━━ 10/10 2.9it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.0it/s 0.3s
                   all         10        106      0.292      0.541      0.249      0.211

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      11/50       8.6G      1.216       1.52      1.274         55       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.6it/s 0.3s
                   all         10        106      0.693      0.563      0.505      0.363

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      12/50       8.6G      1.363      1.653      1.396         63       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.2it/s 0.2s
                   all         10        106      0.528      0.557      0.334      0.159

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      13/50       8.6G      1.716      1.778      1.497         43       1280: 100% ━━━━━━━━━━━━ 10/10 2.9it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.3it/s 0.2s
                   all         10        106      0.528      0.557      0.334      0.159

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      14/50       8.6G      1.469      1.785      1.484         49       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.1it/s 0.2s
                   all         10        106      0.445      0.609      0.579      0.372

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      15/50       8.6G      1.439      2.031      1.442         35       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.2it/s 0.2s
                   all         10        106      0.326      0.469       0.39      0.265

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      16/50       8.6G      1.191      1.474      1.274         53       1280: 100% ━━━━━━━━━━━━ 10/10 2.9it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.2it/s 0.2s
                   all         10        106      0.326      0.469       0.39      0.265

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      17/50       8.6G      1.219      1.462       1.32         53       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.3it/s 0.2s
                   all         10        106      0.834      0.541      0.528      0.383

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      18/50       8.6G      1.245      1.359      1.292         51       1280: 100% ━━━━━━━━━━━━ 10/10 2.9it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.4it/s 0.2s
                   all         10        106      0.834      0.541      0.528      0.383

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      19/50       8.6G      1.276      1.394      1.328         64       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.3it/s 0.2s
                   all         10        106      0.947      0.552      0.601      0.469

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      20/50       8.6G      1.203      1.256      1.282         36       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.2it/s 0.2s
                   all         10        106      0.772      0.573      0.616      0.469

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      21/50       8.6G      1.244      1.316      1.293         68       1280: 100% ━━━━━━━━━━━━ 10/10 2.9it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.2it/s 0.2s
                   all         10        106      0.772      0.573      0.616      0.469

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      22/50       8.6G      1.123      1.222      1.213         78       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.1it/s 0.2s
                   all         10        106      0.665      0.645      0.637      0.452

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      23/50       8.6G       1.14      1.167      1.212         48       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.4it/s 0.2s
                   all         10        106      0.641      0.642       0.65      0.514

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      24/50       8.6G      1.089      1.133      1.172         59       1280: 100% ━━━━━━━━━━━━ 10/10 2.9it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.2it/s 0.2s
                   all         10        106      0.641      0.642       0.65      0.514

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      25/50       8.6G      1.061      1.113      1.168         70       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.1it/s 0.2s
                   all         10        106      0.785      0.586      0.648      0.491

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      26/50       8.6G     0.9993      1.139      1.151         31       1280: 100% ━━━━━━━━━━━━ 10/10 2.9it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.1it/s 0.2s
                   all         10        106      0.785      0.586      0.648      0.491

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      27/50       8.6G      1.031      1.138      1.168         59       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.4it/s 0.2s
                   all         10        106      0.726      0.612      0.651      0.521

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      28/50       8.6G      1.141      1.218      1.255         42       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.3it/s 0.2s
                   all         10        106      0.749       0.63      0.674      0.451

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      29/50       8.6G      1.058      1.066      1.156         52       1280: 100% ━━━━━━━━━━━━ 10/10 2.9it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.4it/s 0.2s
                   all         10        106      0.749       0.63      0.674      0.451

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      30/50       8.6G      1.061      1.082      1.186         40       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.4it/s 0.2s
                   all         10        106      0.721      0.624      0.662      0.508

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      31/50       8.6G      1.011      1.122      1.188         56       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.4it/s 0.2s
                   all         10        106      0.722      0.625      0.663      0.519

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      32/50       8.6G     0.9572      1.035      1.137         76       1280: 100% ━━━━━━━━━━━━ 10/10 2.9it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.3it/s 0.2s
                   all         10        106      0.722      0.625      0.663      0.519

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      33/50       8.6G     0.9398     0.9619      1.126         58       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.3it/s 0.2s
                   all         10        106      0.673      0.663      0.666      0.542

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      34/50       8.6G      0.967      1.053      1.154         77       1280: 100% ━━━━━━━━━━━━ 10/10 2.9it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.4it/s 0.2s
                   all         10        106      0.673      0.663      0.666      0.542

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      35/50       8.6G      0.957      1.014      1.125         60       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.3it/s 0.2s
                   all         10        106      0.688      0.664      0.671      0.494

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      36/50       8.6G     0.9499       1.02      1.151         49       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.4it/s 0.2s
                   all         10        106      0.719      0.675      0.672      0.526

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      37/50       8.6G     0.9603      1.061      1.185         63       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.3it/s 0.2s
                   all         10        106      0.719      0.675      0.672      0.526

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      38/50       8.6G      0.976      1.003      1.126         45       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.3it/s 0.2s
                   all         10        106      0.704      0.687      0.704      0.552

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      39/50       8.6G      1.025       1.01      1.185         71       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.4it/s 0.2s
                   all         10        106      0.704      0.699      0.701      0.523

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      40/50       8.6G     0.9717     0.9858      1.164         71       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.2it/s 0.2s
                   all         10        106      0.704      0.699      0.701      0.523

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      41/50       8.6G     0.8599      0.869      1.087         38       1280: 100% ━━━━━━━━━━━━ 10/10 1.7it/s 5.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.5it/s 0.3s
                   all         10        106      0.732      0.687      0.692        0.5

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      42/50       8.6G     0.8669     0.8331      1.071         50       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.4it/s 0.2s
                   all         10        106      0.732      0.687      0.692        0.5

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      43/50       8.6G     0.8024     0.8054      1.044         42       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.4it/s 0.2s
                   all         10        106      0.703      0.687      0.688      0.527

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      44/50       8.6G     0.8657     0.8167      1.062         41       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.0it/s 0.3s
                   all         10        106      0.694       0.72      0.684      0.541

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      45/50       8.6G     0.8471     0.8601      1.089         36       1280: 100% ━━━━━━━━━━━━ 10/10 2.9it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.4it/s 0.2s
                   all         10        106      0.694       0.72      0.684      0.541

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      46/50       8.6G     0.7389     0.7814      1.019         40       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.2it/s 0.2s
                   all         10        106      0.678      0.687      0.675      0.524

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      47/50       8.6G     0.8045     0.7585       1.04         38       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.3it/s 0.2s
                   all         10        106      0.696      0.693      0.676      0.516

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      48/50       8.6G     0.7699     0.7869      1.032         34       1280: 100% ━━━━━━━━━━━━ 10/10 2.9it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.3it/s 0.2s
                   all         10        106      0.696      0.693      0.676      0.516

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      49/50       8.6G     0.7823     0.7792      1.049         34       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.3it/s 0.2s
                   all         10        106      0.693      0.705       0.68      0.525

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      50/50       8.6G     0.7863     0.8037      1.047         49       1280: 100% ━━━━━━━━━━━━ 10/10 2.9it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.3it/s 0.2s
                   all         10        106      0.693      0.705       0.68      0.525

50 epochs completed in 0.075 hours.
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s50_m_3/weights-2/weights/last.pt, 40.6MB
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s50_m_3/weights-2/weights/best.pt, 40.6MB

Validating /home/sjkim/solar-thermal/runs/detect/workspace/train_s50_m_3/weights-2/weights/best.pt...
Ultralytics 8.4.42 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
YOLO11m summary (fused): 126 layers, 20,032,345 parameters, 0 gradients, 67.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.1it/s 0.2s
                   all         10        106      0.705      0.687      0.704      0.553
             pv_string          9         47      0.881      0.979       0.97      0.751
             pv_module          7         31      0.711      0.903      0.892      0.785
                 other          9         28      0.523      0.179       0.25      0.122
Speed: 0.5ms preprocess, 16.5ms inference, 0.0ms loss, 3.8ms postprocess per image
Results saved to /home/sjkim/solar-thermal/runs/detect/workspace/train_s50_m_3/weights-2
Elapsed: 0:04:50
```

```bash
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s50_m_3 \
    --model models/yolo11m.pt \
    --epochs 100 \
    --device cuda \
    --amp True \
    --output ./workspace/train_s50_m_3
Seed 이미지 50장 (수동 라벨 완료)
  train: 40장
  val: 10장
New https://pypi.org/project/ultralytics/8.4.43 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.4.42 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=4, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=workspace/train_s50_m_3/dataset/data.yaml, degrees=5.0, deterministic=True, device=0, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=100, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.5, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=1280, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=models/yolo11m.pt, momentum=0.937, mosaic=0.5, multi_scale=0.0, name=weights, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=15, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=workspace/train_s50_m_3, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/home/sjkim/solar-thermal/runs/detect/workspace/train_s50_m_3/weights, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.3, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.05, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=3

                   from  n    params  module                                       arguments                     
  0                  -1  1      1856  ultralytics.nn.modules.conv.Conv             [3, 64, 3, 2]                 
  1                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  2                  -1  1    111872  ultralytics.nn.modules.block.C3k2            [128, 256, 1, True, 0.25]     
  3                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
  4                  -1  1    444928  ultralytics.nn.modules.block.C3k2            [256, 512, 1, True, 0.25]     
  5                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  6                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  7                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  8                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
 10                  -1  1    990976  ultralytics.nn.modules.block.C2PSA           [512, 512, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1    542720  ultralytics.nn.modules.block.C3k2            [1024, 256, 1, True]          
 17                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1   1511424  ultralytics.nn.modules.block.C3k2            [768, 512, 1, True]           
 20                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 23        [16, 19, 22]  1   1413337  ultralytics.nn.modules.head.Detect           [3, 16, None, [256, 512, 512]]
YOLO11m summary: 232 layers, 20,055,321 parameters, 20,055,305 gradients, 68.2 GFLOPs

Transferred 643/649 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2247.7±1578.6 MB/s, size: 12414.0 KB)
train: Scanning /home/sjkim/solar-thermal/workspace/train_s50_m_3/dataset/labels/train... 40 images, 1 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 40/40 1.0Kit/s 0.0s
train: New cache created: /home/sjkim/solar-thermal/workspace/train_s50_m_3/dataset/labels/train.cache
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2655.9±1333.1 MB/s, size: 12870.9 KB)
val: Scanning /home/sjkim/solar-thermal/workspace/train_s50_m_3/dataset/labels/val... 10 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 10/10 621.2it/s 0.0s
val: New cache created: /home/sjkim/solar-thermal/workspace/train_s50_m_3/dataset/labels/val.cache
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001429, momentum=0.9) with parameter groups 106 weight(decay=0.0), 113 weight(decay=0.0005), 112 bias(decay=0.0)
Plotting labels to /home/sjkim/solar-thermal/runs/detect/workspace/train_s50_m_3/weights/labels.jpg... 
Image sizes 1280 train, 1280 val
Using 8 dataloader workers
Logging results to /home/sjkim/solar-thermal/runs/detect/workspace/train_s50_m_3/weights
Starting training for 100 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/100      8.09G      1.946      3.765      1.911         63       1280: 100% ━━━━━━━━━━━━ 10/10 1.5s/it 14.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.3s/it 2.5s
                   all         10        106      0.829      0.395      0.445      0.176

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/100      8.57G      1.301       2.11      1.356         39       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.2it/s 0.3s
                   all         10        106      0.284      0.463      0.305      0.175

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      3/100      8.57G      1.375      1.983       1.34         54       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.1it/s 0.3s
                   all         10        106      0.609      0.595       0.39      0.241

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      4/100       8.6G      1.252      1.667      1.299         60       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.1it/s 0.3s
                   all         10        106      0.712      0.545      0.514      0.319

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      5/100       8.6G      1.298      1.779      1.346         51       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.9it/s 0.3s
                   all         10        106      0.548      0.237      0.298      0.199

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      6/100       8.6G      1.661      2.238      1.607         53       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.0it/s 0.3s
                   all         10        106      0.157      0.235      0.137     0.0889

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      7/100       8.6G      1.454      2.462      1.458         61       1280: 100% ━━━━━━━━━━━━ 10/10 2.9it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.8it/s 0.3s
                   all         10        106      0.157      0.235      0.137     0.0889

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      8/100       8.6G      1.647      2.258      1.595         72       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.8it/s 0.3s
                   all         10        106      0.398      0.177     0.0794     0.0426

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      9/100       8.6G       1.45      1.787      1.407         59       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.9it/s 0.3s
                   all         10        106      0.467      0.531      0.139      0.081

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     10/100       8.6G      1.332      1.567      1.395         46       1280: 100% ━━━━━━━━━━━━ 10/10 2.9it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.0it/s 0.2s
                   all         10        106      0.467      0.531      0.139      0.081

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     11/100       8.6G      1.478      1.716      1.507         55       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.8it/s 0.3s
                   all         10        106     0.0091      0.271    0.00421    0.00193

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     12/100       8.6G      1.648      1.913      1.663         63       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.9it/s 0.3s
                   all         10        106      0.159      0.374      0.231      0.108

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     13/100       8.6G      1.634      1.922      1.436         43       1280: 100% ━━━━━━━━━━━━ 10/10 2.9it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.0it/s 0.3s
                   all         10        106      0.159      0.374      0.231      0.108

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     14/100       8.6G      1.527      1.875      1.505         49       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.6it/s 0.3s
                   all         10        106     0.0228      0.459     0.0213    0.00833

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     15/100       8.6G      1.526      2.111       1.52         35       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.2it/s 0.3s
                   all         10        106    0.00092     0.0238     0.0001   1.29e-05

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     16/100       8.6G      1.447      1.992      1.421         53       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.3it/s 0.3s
                   all         10        106    0.00092     0.0238     0.0001   1.29e-05

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     17/100       8.6G      1.435      1.785      1.425         53       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.7it/s 0.3s
                   all         10        106    0.00372      0.107    0.00178   0.000219

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     18/100       8.6G      1.327      1.595       1.33         51       1280: 100% ━━━━━━━━━━━━ 10/10 2.9it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.7it/s 0.3s
                   all         10        106    0.00372      0.107    0.00178   0.000219

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     19/100       8.6G      1.456      1.572      1.428         64       1280: 100% ━━━━━━━━━━━━ 10/10 2.8it/s 3.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.9it/s 0.3s
                   all         10        106          0          0          0          0
EarlyStopping: Training stopped early as no improvement observed in last 15 epochs. Best results observed at epoch 4, best model saved as best.pt.
To update EarlyStopping(patience=15) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

19 epochs completed in 0.030 hours.
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s50_m_3/weights/weights/last.pt, 40.6MB
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s50_m_3/weights/weights/best.pt, 40.6MB

Validating /home/sjkim/solar-thermal/runs/detect/workspace/train_s50_m_3/weights/weights/best.pt...
Ultralytics 8.4.42 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
YOLO11m summary (fused): 126 layers, 20,032,345 parameters, 0 gradients, 67.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.9it/s 0.3s
                   all         10        106      0.711      0.545      0.523      0.322
             pv_string          9         47      0.778      0.894      0.881      0.514
             pv_module          7         31      0.356      0.742       0.68      0.447
                 other          9         28          1          0    0.00816    0.00319
Speed: 0.5ms preprocess, 16.4ms inference, 0.0ms loss, 7.8ms postprocess per image
Results saved to /home/sjkim/solar-thermal/runs/detect/workspace/train_s50_m_3/weights
Elapsed: 0:02:04
```

```bash
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s50_m_4 \
    --model models/yolo11m.pt \
    --epochs 100 \
    --device cuda \
    --amp True \
    --output ./workspace/train_s50_m_4
Seed 이미지 52장 (수동 라벨 완료)
  train: 42장
  val: 10장
Ultralytics 8.4.43 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=4, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=workspace/train_s50_m_4/dataset/data.yaml, degrees=5.0, deterministic=True, device=0, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=100, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.5, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=1280, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=models/yolo11m.pt, momentum=0.937, mosaic=0.5, multi_scale=0.0, name=weights, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=15, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=workspace/train_s50_m_4, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/home/sjkim/solar-thermal/runs/detect/workspace/train_s50_m_4/weights, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.3, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.05, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=3

                   from  n    params  module                                       arguments                     
  0                  -1  1      1856  ultralytics.nn.modules.conv.Conv             [3, 64, 3, 2]                 
  1                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  2                  -1  1    111872  ultralytics.nn.modules.block.C3k2            [128, 256, 1, True, 0.25]     
  3                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
  4                  -1  1    444928  ultralytics.nn.modules.block.C3k2            [256, 512, 1, True, 0.25]     
  5                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  6                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  7                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  8                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
 10                  -1  1    990976  ultralytics.nn.modules.block.C2PSA           [512, 512, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1    542720  ultralytics.nn.modules.block.C3k2            [1024, 256, 1, True]          
 17                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1   1511424  ultralytics.nn.modules.block.C3k2            [768, 512, 1, True]           
 20                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 23        [16, 19, 22]  1   1413337  ultralytics.nn.modules.head.Detect           [3, 16, None, [256, 512, 512]]
YOLO11m summary: 232 layers, 20,055,321 parameters, 20,055,305 gradients, 68.2 GFLOPs

Transferred 643/649 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 100% ━━━━━━━━━━━━ 5.3MB 63.8MB/s 0.1s
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2194.0±1376.1 MB/s, size: 12421.7 KB)
train: Scanning /home/sjkim/solar-thermal/workspace/train_s50_m_4/dataset/labels/train... 42 images, 3 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 42/42 973.0it/s 0.0s
train: New cache created: /home/sjkim/solar-thermal/workspace/train_s50_m_4/dataset/labels/train.cache
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2804.8±1301.5 MB/s, size: 12922.0 KB)
val: Scanning /home/sjkim/solar-thermal/workspace/train_s50_m_4/dataset/labels/val... 10 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 10/10 787.4it/s 0.0s
val: New cache created: /home/sjkim/solar-thermal/workspace/train_s50_m_4/dataset/labels/val.cache
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001429, momentum=0.9) with parameter groups 106 weight(decay=0.0), 113 weight(decay=0.0005), 112 bias(decay=0.0)
Plotting labels to /home/sjkim/solar-thermal/runs/detect/workspace/train_s50_m_4/weights/labels.jpg... 
Image sizes 1280 train, 1280 val
Using 8 dataloader workers
Logging results to /home/sjkim/solar-thermal/runs/detect/workspace/train_s50_m_4/weights
Starting training for 100 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/100      8.09G      1.994      3.806       1.91         17       1280: 100% ━━━━━━━━━━━━ 11/11 1.6s/it 17.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.3s/it 2.6s
                   all         10        114       0.83      0.404       0.38      0.223

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/100       8.6G      1.302      2.416      1.355         39       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.3it/s 0.3s
                   all         10        114      0.332      0.509      0.388      0.182

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      3/100       8.6G      1.342      1.931      1.366         16       1280: 100% ━━━━━━━━━━━━ 11/11 3.0it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.1it/s 0.3s
                   all         10        114      0.664      0.205      0.308      0.164

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      4/100       8.6G      1.229      1.752      1.295         34       1280: 100% ━━━━━━━━━━━━ 11/11 3.0it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.6it/s 0.3s
                   all         10        114      0.656      0.565      0.505      0.323

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      5/100       8.6G       1.33      1.803      1.366         17       1280: 100% ━━━━━━━━━━━━ 11/11 3.0it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.1it/s 0.3s
                   all         10        114     0.0303      0.155     0.0316     0.0113

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      6/100       8.6G      1.413      2.213      1.386         14       1280: 100% ━━━━━━━━━━━━ 11/11 3.0it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.5it/s 0.3s
                   all         10        114     0.0769      0.376     0.0985      0.051

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      7/100       8.6G      1.493      1.988      1.447         29       1280: 100% ━━━━━━━━━━━━ 11/11 3.0it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.3it/s 0.3s
                   all         10        114    0.00686      0.127    0.00297    0.00121

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      8/100       8.6G       1.64      2.245       1.54         28       1280: 100% ━━━━━━━━━━━━ 11/11 3.0it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 5.9it/s 0.3s
                   all         10        114    0.00178     0.0108   0.000583   5.83e-05

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      9/100       8.6G      1.606      2.675      1.555         26       1280: 100% ━━━━━━━━━━━━ 11/11 3.0it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 5.9it/s 0.3s
                   all         10        114    0.00178     0.0108   0.000583   5.83e-05

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     10/100       8.6G      1.647      2.355      1.662         19       1280: 100% ━━━━━━━━━━━━ 11/11 3.0it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 6.7it/s 0.3s
                   all         10        114    0.00114      0.031   0.000309   5.01e-05

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     11/100       8.6G      1.519      2.115      1.593         42       1280: 100% ━━━━━━━━━━━━ 11/11 3.0it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.0it/s 0.2s
                   all         10        114     0.0017     0.0853    0.00051   9.04e-05

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     12/100       8.6G      1.503      1.883       1.53         13       1280: 100% ━━━━━━━━━━━━ 11/11 3.0it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.3it/s 0.2s
                   all         10        114     0.0017     0.0853    0.00051   9.04e-05

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     13/100       8.6G      1.672      2.066      1.624         11       1280: 100% ━━━━━━━━━━━━ 11/11 3.0it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.1it/s 0.2s
                   all         10        114      0.004      0.109    0.00142   0.000252

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     14/100       8.6G      1.588      1.902        1.6         35       1280: 100% ━━━━━━━━━━━━ 11/11 3.0it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.0it/s 0.3s
                   all         10        114     0.0264      0.414     0.0195    0.00868

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     15/100       8.6G      1.625      1.774      1.557         18       1280: 100% ━━━━━━━━━━━━ 11/11 3.0it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.9it/s 0.3s
                   all         10        114     0.0264      0.414     0.0195    0.00868

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     16/100       8.6G      1.403      1.556      1.443         27       1280: 100% ━━━━━━━━━━━━ 11/11 3.0it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.9it/s 0.3s
                   all         10        114      0.183      0.511      0.191      0.111

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     17/100       8.6G      1.357      1.648      1.348         20       1280: 100% ━━━━━━━━━━━━ 11/11 3.0it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.2it/s 0.2s
                   all         10        114       0.71       0.58      0.521      0.337

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     18/100       8.6G      1.405      1.719      1.366         16       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.9it/s 0.3s
                   all         10        114       0.62      0.627      0.512      0.327

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     19/100       8.6G      1.413      1.473      1.415         28       1280: 100% ━━━━━━━━━━━━ 11/11 3.0it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.0it/s 0.3s
                   all         10        114       0.62      0.627      0.512      0.327

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     20/100       8.6G      1.368      1.385      1.369         28       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.5it/s 0.3s
                   all         10        114      0.786      0.627      0.603      0.354

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     21/100       8.6G      1.394      1.434      1.368         46       1280: 100% ━━━━━━━━━━━━ 11/11 3.0it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.9it/s 0.3s
                   all         10        114      0.873      0.601      0.599       0.45

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     22/100       8.6G      1.275      1.318      1.384         19       1280: 100% ━━━━━━━━━━━━ 11/11 3.0it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.8it/s 0.3s
                   all         10        114      0.873      0.601      0.599       0.45

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     23/100       8.6G      1.246        1.3        1.3         40       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.0it/s 0.3s
                   all         10        114      0.888      0.586      0.608      0.421

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     24/100       8.6G      1.215      1.237       1.28         19       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.8it/s 0.3s
                   all         10        114      0.963      0.541      0.628      0.389

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     25/100       8.6G      1.197      1.366      1.278         12       1280: 100% ━━━━━━━━━━━━ 11/11 3.0it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.0it/s 0.3s
                   all         10        114      0.963      0.541      0.628      0.389

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     26/100       8.6G      1.137      1.269      1.262         27       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.0it/s 0.3s
                   all         10        114       0.91      0.571      0.629      0.474

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     27/100       8.6G      1.203      1.271       1.31         32       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.8it/s 0.3s
                   all         10        114      0.921      0.594      0.601      0.452

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     28/100       8.6G      1.239      1.426      1.322         17       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.1it/s 0.2s
                   all         10        114      0.921      0.594      0.601      0.452

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     29/100       8.6G      1.143      1.265      1.278         20       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.1it/s 0.2s
                   all         10        114      0.633      0.621      0.627       0.44

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     30/100       8.6G      1.181      1.411      1.265         12       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.2it/s 0.2s
                   all         10        114      0.615      0.604      0.626      0.438

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     31/100       8.6G       1.09      1.193      1.193         26       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.1it/s 0.2s
                   all         10        114      0.615      0.604      0.626      0.438

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     32/100       8.6G      1.063      1.103      1.194         21       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.2it/s 0.2s
                   all         10        114        0.9      0.612      0.622      0.473

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     33/100       8.6G      1.057      1.161      1.184         22       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.9it/s 0.3s
                   all         10        114      0.958      0.511      0.587      0.426

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     34/100       8.6G       1.07      1.142      1.225         27       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.9it/s 0.3s
                   all         10        114      0.937      0.587      0.628      0.445

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     35/100       8.6G       1.05      1.132      1.209         43       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.1it/s 0.2s
                   all         10        114      0.937      0.587      0.628      0.445

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     36/100       8.6G      1.132      1.157      1.207         34       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.1it/s 0.2s
                   all         10        114      0.943      0.586      0.642       0.44

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     37/100       8.6G      1.207      1.236      1.278         30       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.0it/s 0.3s
                   all         10        114      0.621      0.648      0.637      0.425

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     38/100       8.6G      1.108      1.115       1.18         51       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.0it/s 0.2s
                   all         10        114      0.621      0.648      0.637      0.425

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     39/100       8.6G      1.059      1.113       1.17         29       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.2it/s 0.2s
                   all         10        114      0.952      0.586      0.635      0.448

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     40/100       8.6G      1.045       1.05      1.165         25       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.0it/s 0.2s
                   all         10        114      0.661      0.633      0.638      0.505

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     41/100       8.6G      1.105      1.111      1.232         40       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.1it/s 0.2s
                   all         10        114      0.661      0.633      0.638      0.505

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     42/100       8.6G      1.046     0.9854      1.126         22       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.0it/s 0.3s
                   all         10        114      0.693      0.648      0.642      0.486

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     43/100       8.6G      1.027      1.028      1.155         38       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.0it/s 0.3s
                   all         10        114      0.976      0.569      0.644      0.489

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     44/100       8.6G     0.9761      1.009      1.146         30       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.9it/s 0.3s
                   all         10        114      0.976      0.569      0.644      0.489

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     45/100       8.6G      1.007     0.9883      1.153         21       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.8it/s 0.3s
                   all         10        114      0.957       0.59      0.654      0.434

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     46/100       8.6G      1.059      1.089      1.221         17       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.3it/s 0.2s
                   all         10        114       0.96      0.572      0.653      0.477

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     47/100       8.6G     0.9863     0.9859      1.135         24       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.1it/s 0.2s
                   all         10        114       0.96      0.572      0.653      0.477

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     48/100       8.6G      1.053      1.038      1.164         29       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.2it/s 0.2s
                   all         10        114      0.627      0.633       0.65      0.491

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     49/100       8.6G      1.036     0.9994      1.169         22       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.1it/s 0.2s
                   all         10        114      0.707       0.62      0.637       0.49

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     50/100       8.6G     0.9979      1.083      1.165         16       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.0it/s 0.3s
                   all         10        114      0.609      0.611       0.63      0.485

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     51/100       8.6G     0.9883      1.017      1.162         24       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.3it/s 0.2s
                   all         10        114      0.609      0.611       0.63      0.485

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     52/100       8.6G      1.061      1.036      1.202         29       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.0it/s 0.3s
                   all         10        114      0.588      0.619      0.634      0.438

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     53/100       8.6G      1.004     0.9977      1.124         39       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.2it/s 0.2s
                   all         10        114      0.936      0.582      0.633      0.449

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     54/100       8.6G     0.9506     0.9229      1.112         21       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.1it/s 0.2s
                   all         10        114      0.936      0.582      0.633      0.449

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     55/100       8.6G      1.023     0.9641      1.171         24       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.1it/s 0.2s
                   all         10        114      0.953      0.574      0.644      0.518

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     56/100       8.6G     0.9796     0.9934       1.13         30       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.3it/s 0.2s
                   all         10        114      0.939      0.588      0.643      0.529

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     57/100       8.6G     0.9547     0.9371      1.111         28       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.1it/s 0.2s
                   all         10        114      0.939      0.588      0.643      0.529

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     58/100       8.6G      1.014      1.002      1.163         29       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.1it/s 0.2s
                   all         10        114      0.975      0.583       0.65      0.478

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     59/100       8.6G     0.9726     0.9275      1.122         23       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.2it/s 0.2s
                   all         10        114      0.683      0.605      0.648      0.443

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     60/100       8.6G     0.9802       0.95      1.143         46       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.2it/s 0.2s
                   all         10        114      0.683      0.605      0.648      0.443

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     61/100       8.6G     0.9991     0.9685       1.15         28       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.2it/s 0.2s
                   all         10        114      0.624      0.605      0.643      0.477

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     62/100       8.6G      1.025      1.052      1.181          7       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.2it/s 0.2s
                   all         10        114      0.592      0.564      0.614      0.514

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     63/100       8.6G     0.9546          1      1.128         22       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.0it/s 0.2s
                   all         10        114      0.592      0.564      0.614      0.514

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     64/100       8.6G      1.007     0.9383      1.169         22       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.2it/s 0.2s
                   all         10        114      0.547      0.602      0.624      0.528

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     65/100       8.6G     0.8571     0.8548      1.077         26       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.2it/s 0.2s
                   all         10        114      0.565      0.616      0.641      0.494

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     66/100       8.6G      0.934     0.8984      1.102         14       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.2it/s 0.2s
                   all         10        114      0.598      0.643      0.641      0.433

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     67/100       8.6G     0.9837     0.8854      1.092         22       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.0it/s 0.2s
                   all         10        114      0.598      0.643      0.641      0.433

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     68/100       8.6G     0.9121      1.023      1.117          6       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.2it/s 0.2s
                   all         10        114      0.611      0.637      0.643      0.435

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     69/100       8.6G      0.942     0.8942      1.091         51       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.3it/s 0.2s
                   all         10        114      0.594      0.649      0.644      0.496

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     70/100       8.6G      1.027      1.012      1.186         35       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.3it/s 0.2s
                   all         10        114      0.594      0.649      0.644      0.496

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     71/100       8.6G     0.8926     0.9343      1.087         18       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.3it/s 0.2s
                   all         10        114      0.612      0.622      0.658      0.534

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     72/100       8.6G      0.906     0.9199      1.109         27       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.0it/s 0.3s
                   all         10        114      0.555      0.652      0.641      0.523

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     73/100       8.6G      1.016      1.017      1.148         13       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.2it/s 0.2s
                   all         10        114      0.555      0.652      0.641      0.523

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     74/100       8.6G     0.9166     0.8988       1.13         25       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.2it/s 0.2s
                   all         10        114       0.57      0.644      0.636      0.481

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     75/100       8.6G     0.9281     0.8939      1.077         40       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.2it/s 0.2s
                   all         10        114      0.598      0.633      0.644      0.475

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     76/100       8.6G      0.876       0.88      1.059         38       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.2it/s 0.2s
                   all         10        114      0.598      0.633      0.644      0.475

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     77/100       8.6G     0.8713     0.8886      1.097         22       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.1it/s 0.2s
                   all         10        114      0.585      0.623      0.647      0.502

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     78/100       8.6G     0.9374     0.9549      1.125         25       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.8it/s 0.3s
                   all         10        114      0.703      0.619      0.665      0.524

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     79/100       8.6G     0.8871     0.9373      1.127         22       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.1it/s 0.2s
                   all         10        114      0.703      0.619      0.665      0.524

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     80/100       8.6G     0.8911     0.8676      1.108         35       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.2it/s 0.2s
                   all         10        114      0.689      0.628      0.674      0.519

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     81/100       8.6G     0.7684     0.7717      1.056         17       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.0it/s 0.2s
                   all         10        114      0.704      0.621      0.673      0.479

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     82/100       8.6G     0.9129     0.8382      1.103         34       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.0it/s 0.3s
                   all         10        114      0.751      0.621      0.673      0.458

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     83/100       8.6G      0.915     0.8983      1.089         20       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.1it/s 0.2s
                   all         10        114      0.751      0.621      0.673      0.458

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     84/100       8.6G     0.8437     0.8509      1.086         22       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.1it/s 0.2s
                   all         10        114      0.702      0.659      0.678      0.504

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     85/100       8.6G     0.8032     0.8547       1.05         24       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.3it/s 0.2s
                   all         10        114      0.689       0.66      0.674      0.523

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     86/100       8.6G     0.8527     0.8499      1.062         48       1280: 100% ━━━━━━━━━━━━ 11/11 2.9it/s 3.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.9it/s 0.3s
                   all         10        114      0.689       0.66      0.674      0.523
EarlyStopping: Training stopped early as no improvement observed in last 15 epochs. Best results observed at epoch 71, best model saved as best.pt.
To update EarlyStopping(patience=15) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

86 epochs completed in 0.128 hours.
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s50_m_4/weights/weights/last.pt, 40.6MB
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s50_m_4/weights/weights/best.pt, 40.6MB

Validating /home/sjkim/solar-thermal/runs/detect/workspace/train_s50_m_4/weights/weights/best.pt...
Ultralytics 8.4.43 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
YOLO11m summary (fused): 126 layers, 20,032,345 parameters, 0 gradients, 67.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 8.1it/s 0.2s
                   all         10        114      0.612      0.622      0.658      0.533
             pv_string          9         40      0.886      0.975       0.97      0.773
             pv_module          8         43       0.95      0.893       0.95      0.805
                 other          9         31          0          0      0.053     0.0223
Speed: 0.5ms preprocess, 16.8ms inference, 0.0ms loss, 1.1ms postprocess per image
Results saved to /home/sjkim/solar-thermal/runs/detect/workspace/train_s50_m_4/weights
Elapsed: 0:08:02
```

```bash
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s100_m \
    --model models/yolo11m.pt \
    --epochs 100 \
    --device cuda \
    --amp True \
    --output ./workspace/train_s100_m
Seed 이미지 122장 (수동 라벨 완료)
  train: 98장
  val: 24장
New https://pypi.org/project/ultralytics/8.4.45 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.4.43 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=4, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=workspace/train_s100_m/dataset/data.yaml, degrees=5.0, deterministic=True, device=0, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=100, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.5, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=1280, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=models/yolo11m.pt, momentum=0.937, mosaic=0.5, multi_scale=0.0, name=weights, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=15, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=workspace/train_s100_m, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/home/sjkim/solar-thermal/runs/detect/workspace/train_s100_m/weights, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.3, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.05, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=3

                   from  n    params  module                                       arguments                     
  0                  -1  1      1856  ultralytics.nn.modules.conv.Conv             [3, 64, 3, 2]                 
  1                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  2                  -1  1    111872  ultralytics.nn.modules.block.C3k2            [128, 256, 1, True, 0.25]     
  3                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
  4                  -1  1    444928  ultralytics.nn.modules.block.C3k2            [256, 512, 1, True, 0.25]     
  5                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  6                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  7                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  8                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
 10                  -1  1    990976  ultralytics.nn.modules.block.C2PSA           [512, 512, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1    542720  ultralytics.nn.modules.block.C3k2            [1024, 256, 1, True]          
 17                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1   1511424  ultralytics.nn.modules.block.C3k2            [768, 512, 1, True]           
 20                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 23        [16, 19, 22]  1   1413337  ultralytics.nn.modules.head.Detect           [3, 16, None, [256, 512, 512]]
YOLO11m summary: 232 layers, 20,055,321 parameters, 20,055,305 gradients, 68.2 GFLOPs

Transferred 643/649 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1492.7±578.9 MB/s, size: 8339.4 KB)
train: Scanning /home/sjkim/solar-thermal/workspace/train_s100_m/dataset/labels/train... 114 images, 21 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 114/114 1.3Kit/s 0.1s
train: New cache created: /home/sjkim/solar-thermal/workspace/train_s100_m/dataset/labels/train.cache
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1441.8±742.7 MB/s, size: 12578.1 KB)
val: Scanning /home/sjkim/solar-thermal/workspace/train_s100_m/dataset/labels/val... 38 images, 2 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 38/38 894.8it/s 0.0s
val: New cache created: /home/sjkim/solar-thermal/workspace/train_s100_m/dataset/labels/val.cache
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001429, momentum=0.9) with parameter groups 106 weight(decay=0.0), 113 weight(decay=0.0005), 112 bias(decay=0.0)
Plotting labels to /home/sjkim/solar-thermal/runs/detect/workspace/train_s100_m/weights/labels.jpg... 
Image sizes 1280 train, 1280 val
Using 8 dataloader workers
Logging results to /home/sjkim/solar-thermal/runs/detect/workspace/train_s100_m/weights
Starting training for 100 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/100      8.11G      1.568      2.999      1.591         20       1280: 100% ━━━━━━━━━━━━ 29/29 1.0it/s 28.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 1.4it/s 3.5s
                   all         38        387       0.51      0.522      0.464      0.268

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/100      8.58G       1.34      1.982      1.324         20       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 4.8it/s 1.0s
                   all         38        387      0.633      0.284      0.284      0.155

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      3/100       8.6G      1.478      2.023      1.526         37       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 4.6it/s 1.1s
                   all         38        387     0.0336      0.216     0.0341     0.0118

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      4/100       8.6G      1.635      1.861      1.483         22       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 4.1it/s 1.2s
                   all         38        387    0.00699      0.183    0.00219   0.000389

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      5/100       8.6G       1.46      1.861      1.404          8       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 4.5it/s 1.1s
                   all         38        387     0.0197    0.00617    0.00068   0.000233

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      6/100       8.6G      1.464      1.714      1.416         27       1280: 100% ━━━━━━━━━━━━ 29/29 2.8it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 4.9it/s 1.0s
                   all         38        387    0.00593     0.0807   0.000719   0.000133

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      7/100       8.6G      1.382       1.63      1.318         27       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 4.8it/s 1.0s
                   all         38        387        0.4        0.5      0.496      0.279

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      8/100       8.6G      1.252      1.379      1.254         43       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.0it/s 1.0s
                   all         38        387      0.409      0.699      0.447      0.278

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      9/100       8.6G      1.086      1.178      1.166         29       1280: 100% ━━━━━━━━━━━━ 29/29 2.8it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 4.9it/s 1.0s
                   all         38        387      0.522      0.596      0.512      0.366

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     10/100       8.6G      1.059      1.125      1.191         15       1280: 100% ━━━━━━━━━━━━ 29/29 2.8it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.0it/s 1.0s
                   all         38        387      0.757      0.781      0.861      0.589

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     11/100       8.6G      1.076       1.04      1.177         29       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.1it/s 1.0s
                   all         38        387      0.917       0.78      0.899      0.566

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     12/100       8.6G      1.002      1.005      1.104         18       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.0it/s 1.0s
                   all         38        387      0.756      0.684      0.774      0.574

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     13/100       8.6G      1.005     0.9275      1.119         26       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.1it/s 1.0s
                   all         38        387      0.945      0.631       0.79       0.48

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     14/100       8.6G       0.94     0.8985      1.083         46       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.871      0.848      0.908      0.676

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     15/100       8.6G     0.9604     0.9256      1.099         15       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.1it/s 1.0s
                   all         38        387      0.857      0.835      0.892      0.616

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     16/100       8.6G      0.913     0.8174       1.06         21       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.0it/s 1.0s
                   all         38        387      0.878       0.75      0.857      0.597

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     17/100       8.6G     0.8763     0.8146       1.04         29       1280: 100% ━━━━━━━━━━━━ 29/29 2.8it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.1it/s 1.0s
                   all         38        387      0.875      0.806      0.911      0.676

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     18/100       8.6G     0.8866     0.8211      1.041         38       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 4.9it/s 1.0s
                   all         38        387      0.805      0.771      0.846      0.624

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     19/100       8.6G     0.9066     0.8243      1.062         25       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.1it/s 1.0s
                   all         38        387      0.908      0.718      0.899      0.625

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     20/100       8.6G     0.9116     0.8361      1.046          5       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.939      0.722      0.903      0.657

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     21/100       8.6G     0.8826     0.7469      1.023         24       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.905      0.746      0.899      0.692

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     22/100       8.6G     0.8243     0.7425      1.005         43       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.933       0.81      0.925       0.55

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     23/100       8.6G     0.8797     0.7926      1.039         34       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387       0.92      0.837      0.924      0.697

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     24/100       8.6G     0.8337     0.7095       1.01         18       1280: 100% ━━━━━━━━━━━━ 29/29 2.8it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.962      0.796      0.921      0.591

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     25/100       8.6G     0.8034     0.6953     0.9891         12       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.1it/s 1.0s
                   all         38        387      0.832      0.885      0.942      0.734

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     26/100       8.6G     0.8008     0.6979     0.9878         24       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.892      0.858      0.945      0.695

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     27/100       8.6G     0.7836      0.715     0.9986          4       1280: 100% ━━━━━━━━━━━━ 29/29 2.8it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.1it/s 1.0s
                   all         38        387      0.937      0.802      0.927      0.726

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     28/100       8.6G     0.8043     0.6906          1         33       1280: 100% ━━━━━━━━━━━━ 29/29 2.8it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.1it/s 1.0s
                   all         38        387      0.909      0.743      0.887      0.705

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     29/100       8.6G     0.8564     0.7453      1.048          6       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.0it/s 1.0s
                   all         38        387       0.92      0.808      0.935      0.648

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     30/100       8.6G     0.7902     0.6842      1.012         22       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.926      0.816      0.925      0.695

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     31/100       8.6G     0.7349     0.6878     0.9814         26       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.1it/s 1.0s
                   all         38        387      0.908      0.891       0.94      0.686

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     32/100       8.6G     0.7742     0.6738     0.9861         39       1280: 100% ━━━━━━━━━━━━ 29/29 2.8it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.1it/s 1.0s
                   all         38        387       0.92      0.811      0.928      0.651

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     33/100       8.6G     0.7338      0.663     0.9851         15       1280: 100% ━━━━━━━━━━━━ 29/29 2.8it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.924      0.853      0.949      0.729

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     34/100       8.6G     0.7909      0.666     0.9819         42       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.843       0.84      0.944       0.75

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     35/100       8.6G     0.7835      0.722      1.015         14       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.886      0.848       0.95      0.612

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     36/100       8.6G     0.8432     0.6845      1.008         26       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.909      0.868      0.945      0.614

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     37/100       8.6G     0.7793     0.6634     0.9978         16       1280: 100% ━━━━━━━━━━━━ 29/29 2.8it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.0it/s 1.0s
                   all         38        387      0.961      0.833      0.942      0.762

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     38/100       8.6G     0.7231     0.6636     0.9734         43       1280: 100% ━━━━━━━━━━━━ 29/29 2.8it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.965      0.807      0.915      0.653

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     39/100       8.6G     0.7647     0.6411     0.9785         35       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387       0.94      0.821      0.919        0.7

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     40/100       8.6G     0.7193     0.6517      0.986         12       1280: 100% ━━━━━━━━━━━━ 29/29 2.8it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387       0.93      0.837      0.924      0.736

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     41/100       8.6G     0.6821     0.6028     0.9629         22       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.3it/s 0.9s
                   all         38        387      0.961      0.825      0.924       0.73

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     42/100       8.6G     0.7272     0.6418     0.9886         18       1280: 100% ━━━━━━━━━━━━ 29/29 2.8it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.921      0.828      0.939      0.735

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     43/100       8.6G     0.6963     0.6164     0.9756         39       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.1it/s 1.0s
                   all         38        387      0.945      0.766      0.937      0.723

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     44/100       8.6G     0.6614     0.6135      0.955         22       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.896      0.859      0.915      0.754

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     45/100       8.6G      0.684     0.6298     0.9775         12       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.1it/s 1.0s
                   all         38        387       0.92      0.836      0.917       0.74

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     46/100       8.6G     0.6765     0.5811     0.9587         40       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.821      0.864      0.955      0.754

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     47/100       8.6G     0.6747     0.6287     0.9487         19       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.792      0.891      0.952       0.72

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     48/100       8.6G     0.6672       0.65     0.9553          7       1280: 100% ━━━━━━━━━━━━ 29/29 2.8it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.939      0.885      0.969      0.783

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     49/100       8.6G     0.6733     0.6394     0.9635         18       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.963      0.841      0.945      0.725

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     50/100       8.6G     0.6764      0.658     0.9535         14       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.1it/s 1.0s
                   all         38        387      0.834       0.89      0.934      0.791

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     51/100       8.6G      0.692     0.5895     0.9705         18       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.1it/s 1.0s
                   all         38        387       0.81      0.879      0.942      0.721

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     52/100       8.6G     0.6744     0.6314      0.948          8       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.947      0.763      0.942      0.714

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     53/100       8.6G     0.6385     0.5436     0.9261         11       1280: 100% ━━━━━━━━━━━━ 29/29 2.8it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.921      0.835      0.941      0.762

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     54/100       8.6G     0.6501     0.5947     0.9511         22       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.956      0.858      0.942      0.776

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     55/100       8.6G      0.631     0.5763     0.9346         42       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.3it/s 1.0s
                   all         38        387      0.952      0.867      0.951      0.741

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     56/100       8.6G     0.6459     0.5826     0.9502         55       1280: 100% ━━━━━━━━━━━━ 29/29 2.8it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.965      0.844       0.95      0.794

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     57/100       8.6G     0.6397     0.5873     0.9443         11       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387       0.97      0.835      0.956      0.756

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     58/100       8.6G       0.62     0.5795     0.9429         28       1280: 100% ━━━━━━━━━━━━ 29/29 2.8it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.816      0.928      0.941      0.748

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     59/100       8.6G     0.6435     0.5858     0.9591         30       1280: 100% ━━━━━━━━━━━━ 29/29 2.8it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.1it/s 1.0s
                   all         38        387      0.904      0.879      0.923       0.72

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     60/100       8.6G     0.6543     0.6069     0.9584         47       1280: 100% ━━━━━━━━━━━━ 29/29 2.8it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.1it/s 1.0s
                   all         38        387      0.933      0.858       0.92      0.769

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     61/100       8.6G     0.5981     0.5286     0.9205         35       1280: 100% ━━━━━━━━━━━━ 29/29 2.8it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.871      0.885      0.938      0.769

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     62/100       8.6G     0.5835     0.5721     0.9347         33       1280: 100% ━━━━━━━━━━━━ 29/29 2.8it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.3it/s 0.9s
                   all         38        387       0.88      0.885      0.932      0.762

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     63/100       8.6G     0.5842     0.5386     0.9205         23       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.1it/s 1.0s
                   all         38        387      0.903      0.866      0.931      0.779

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     64/100       8.6G     0.5851     0.5115     0.9178         31       1280: 100% ━━━━━━━━━━━━ 29/29 2.8it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.913      0.854      0.933      0.775

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     65/100       8.6G     0.6068      0.557     0.9303         13       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.1it/s 1.0s
                   all         38        387      0.936      0.847      0.932      0.766

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     66/100       8.6G     0.6072     0.5461     0.9347         16       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.905      0.844      0.933      0.765

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     67/100       8.6G     0.6114     0.5639     0.9267         16       1280: 100% ━━━━━━━━━━━━ 29/29 2.8it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.1it/s 1.0s
                   all         38        387      0.907      0.852      0.933      0.741

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     68/100       8.6G      0.594     0.5613     0.9192         41       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.1it/s 1.0s
                   all         38        387       0.94      0.857      0.939      0.757

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     69/100       8.6G     0.5741     0.5429     0.9215         21       1280: 100% ━━━━━━━━━━━━ 29/29 2.8it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387       0.88      0.864       0.94      0.781

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     70/100       8.6G     0.5744     0.5552      0.925         32       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.877      0.912      0.948      0.798

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     71/100       8.6G      0.604     0.5682     0.9481          7       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.851      0.877      0.942      0.753

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     72/100       8.6G     0.5848     0.5592     0.9283         20       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.1it/s 1.0s
                   all         38        387      0.858      0.854      0.932      0.793

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     73/100       8.6G     0.5636     0.5216     0.9054         11       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.879      0.852      0.934      0.796

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     74/100       8.6G     0.5449     0.5012     0.9072         19       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.1it/s 1.0s
                   all         38        387      0.882      0.888       0.94      0.776

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     75/100       8.6G     0.5509     0.5064     0.9146         16       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.921       0.89      0.943        0.8

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     76/100       8.6G     0.5786     0.5157     0.9165         26       1280: 100% ━━━━━━━━━━━━ 29/29 2.8it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.1it/s 1.0s
                   all         38        387      0.908      0.866      0.939      0.786

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     77/100       8.6G     0.5545     0.5017      0.906         22       1280: 100% ━━━━━━━━━━━━ 29/29 2.8it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.915      0.873      0.937      0.789

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     78/100       8.6G     0.5496     0.4963     0.9059         15       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.3it/s 0.9s
                   all         38        387      0.918      0.872      0.938      0.782

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     79/100       8.6G     0.5443     0.4965     0.9007         15       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.1it/s 1.0s
                   all         38        387      0.934      0.871      0.936      0.797

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     80/100       8.6G     0.5482     0.5052     0.9255         29       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.918      0.872      0.943      0.768

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     81/100       8.6G     0.5547     0.5125     0.9107         29       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.894      0.893      0.947      0.801

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     82/100       8.6G     0.5628     0.5009     0.9249         27       1280: 100% ━━━━━━━━━━━━ 29/29 2.8it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 4.9it/s 1.0s
                   all         38        387      0.893      0.911      0.942      0.796

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     83/100       8.6G     0.5593     0.5278     0.9203         21       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.3it/s 1.0s
                   all         38        387      0.909      0.877      0.933      0.778

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     84/100       8.6G     0.5506     0.5141     0.9078         36       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.3it/s 1.0s
                   all         38        387      0.886      0.868      0.927      0.798

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     85/100       8.6G     0.5316     0.4869     0.9131         13       1280: 100% ━━━━━━━━━━━━ 29/29 2.8it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.3it/s 1.0s
                   all         38        387      0.894      0.885      0.928      0.806

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     86/100       8.6G     0.5141     0.4703     0.9017         30       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.903       0.88      0.931       0.77

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     87/100       8.6G     0.5467     0.4845     0.9092         28       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.927      0.872      0.936      0.788

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     88/100       8.6G     0.5309     0.4933     0.9085          6       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387       0.95      0.867      0.945      0.786

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     89/100       8.6G     0.5667     0.5153     0.9186         20       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.926       0.89      0.944      0.789

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     90/100       8.6G     0.5072     0.4717     0.8984         23       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.917      0.904      0.941       0.81
Closing dataloader mosaic

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     91/100       8.6G     0.4749     0.4069     0.8879         11       1280: 100% ━━━━━━━━━━━━ 29/29 2.3it/s 12.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.1it/s 1.0s
                   all         38        387      0.919      0.911       0.94      0.793

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     92/100       8.6G     0.4472     0.3767      0.841         15       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.1it/s 1.0s
                   all         38        387      0.912      0.896      0.937        0.8

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     93/100       8.6G     0.4237     0.5407     0.8238          0       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.1it/s 1.0s
                   all         38        387      0.911      0.911      0.942      0.808

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     94/100       8.6G     0.4638     0.3804     0.8706         15       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.3it/s 0.9s
                   all         38        387      0.939      0.906      0.945      0.801

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     95/100       8.6G      0.461     0.3913     0.8574         29       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.1it/s 1.0s
                   all         38        387      0.947      0.895       0.95      0.803

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     96/100       8.6G     0.4459     0.9938     0.8386          9       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.949      0.899       0.95      0.815

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     97/100       8.6G     0.4655     0.3939     0.8583         30       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.947      0.889      0.953      0.821

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     98/100       8.6G     0.4675     0.4444     0.8739         31       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.3it/s 1.0s
                   all         38        387      0.959      0.888      0.952      0.822

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     99/100       8.6G      0.446     0.3874     0.8586          6       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.959      0.876      0.947      0.814

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    100/100       8.6G     0.4332     0.3674     0.8468         25       1280: 100% ━━━━━━━━━━━━ 29/29 2.9it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 5.2it/s 1.0s
                   all         38        387      0.962       0.87      0.944      0.809

100 epochs completed in 0.356 hours.
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s100_m/weights/weights/last.pt, 40.7MB
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s100_m/weights/weights/best.pt, 40.7MB

Validating /home/sjkim/solar-thermal/runs/detect/workspace/train_s100_m/weights/weights/best.pt...
Ultralytics 8.4.43 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
YOLO11m summary (fused): 126 layers, 20,032,345 parameters, 0 gradients, 67.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 3.1it/s 1.6s
                   all         38        387      0.959      0.888      0.952      0.822
             pv_string         32        208      0.956          1      0.995      0.899
             pv_module         31        162      0.993       0.91      0.993      0.835
                 other         12         17      0.928      0.754      0.868      0.731
Speed: 0.5ms preprocess, 18.3ms inference, 0.0ms loss, 8.7ms postprocess per image
Results saved to /home/sjkim/solar-thermal/runs/detect/workspace/train_s100_m/weights
Elapsed: 0:21:41
```

```bash
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s100_m \
    --model models/yolo11m.pt \
    --epochs 200 \
    --device cuda \
    --amp True \
    --output ./workspace/train_s100_m_2
Seed 이미지 122장 (수동 라벨 완료)
  train: 98장
  val: 24장
New https://pypi.org/project/ultralytics/8.4.45 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.4.43 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=4, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=workspace/train_s100_m_2/dataset/data.yaml, degrees=5.0, deterministic=True, device=0, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=200, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.5, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=1280, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=models/yolo11m.pt, momentum=0.937, mosaic=0.5, multi_scale=0.0, name=weights, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=15, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=workspace/train_s100_m_2, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/home/sjkim/solar-thermal/runs/detect/workspace/train_s100_m_2/weights, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.3, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.05, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=3

                   from  n    params  module                                       arguments                     
  0                  -1  1      1856  ultralytics.nn.modules.conv.Conv             [3, 64, 3, 2]                 
  1                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  2                  -1  1    111872  ultralytics.nn.modules.block.C3k2            [128, 256, 1, True, 0.25]     
  3                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
  4                  -1  1    444928  ultralytics.nn.modules.block.C3k2            [256, 512, 1, True, 0.25]     
  5                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  6                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  7                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  8                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
 10                  -1  1    990976  ultralytics.nn.modules.block.C2PSA           [512, 512, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1    542720  ultralytics.nn.modules.block.C3k2            [1024, 256, 1, True]          
 17                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1   1511424  ultralytics.nn.modules.block.C3k2            [768, 512, 1, True]           
 20                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 23        [16, 19, 22]  1   1413337  ultralytics.nn.modules.head.Detect           [3, 16, None, [256, 512, 512]]
YOLO11m summary: 232 layers, 20,055,321 parameters, 20,055,305 gradients, 68.2 GFLOPs

Transferred 643/649 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2391.0±1666.3 MB/s, size: 13465.5 KB)
train: Scanning /home/sjkim/solar-thermal/workspace/train_s100_m_2/dataset/labels/train... 98 images, 21 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 98/98 1.3Kit/s 0.1s
train: New cache created: /home/sjkim/solar-thermal/workspace/train_s100_m_2/dataset/labels/train.cache
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2103.3±1274.0 MB/s, size: 11615.0 KB)
val: Scanning /home/sjkim/solar-thermal/workspace/train_s100_m_2/dataset/labels/val... 24 images, 2 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 24/24 942.9it/s 0.0s
val: New cache created: /home/sjkim/solar-thermal/workspace/train_s100_m_2/dataset/labels/val.cache
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001429, momentum=0.9) with parameter groups 106 weight(decay=0.0), 113 weight(decay=0.0005), 112 bias(decay=0.0)
Plotting labels to /home/sjkim/solar-thermal/runs/detect/workspace/train_s100_m_2/weights/labels.jpg... 
Image sizes 1280 train, 1280 val
Using 8 dataloader workers
Logging results to /home/sjkim/solar-thermal/runs/detect/workspace/train_s100_m_2/weights
Starting training for 200 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/200      8.13G      1.572      3.316        1.6         26       1280: 100% ━━━━━━━━━━━━ 25/25 1.1s/it 27.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.1it/s 2.8s
                   all         24        231      0.339      0.472       0.38      0.184

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/200      9.67G      1.299      1.997      1.279         34       1280: 100% ━━━━━━━━━━━━ 25/25 2.8it/s 8.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.9it/s 0.8s
                   all         24        231     0.0198      0.414     0.0199    0.00493

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      3/200      9.67G      1.284       2.01      1.332         23       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.2it/s 0.7s
                   all         24        231     0.0536      0.509     0.0408     0.0191

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      4/200      9.67G      1.532      2.068      1.593         31       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.0it/s 0.7s
                   all         24        231      0.119       0.19     0.0954     0.0343

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      5/200      9.67G      1.591      2.187      1.722         13       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.6it/s 1.2s
                   all         24        231   0.000439     0.0213   5.08e-05   1.58e-05

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      6/200      9.67G      1.315      2.285      1.369         11       1280: 100% ━━━━━━━━━━━━ 25/25 2.8it/s 8.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.1it/s 0.7s
                   all         24        231   0.000347     0.0358   4.06e-05   4.28e-06

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      7/200      9.67G      1.397      2.033      1.344         22       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.3it/s 0.7s
                   all         24        231    0.00399      0.039    0.00155   0.000246

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      8/200      9.67G      1.428      1.672      1.363         11       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.3it/s 0.7s
                   all         24        231      0.367      0.138     0.0131    0.00653

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      9/200      9.67G      1.314      1.393      1.276         35       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.3it/s 0.7s
                   all         24        231     0.0083      0.161     0.0015   0.000524

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     10/200      9.67G      1.308        1.4      1.281         21       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.5it/s 0.7s
                   all         24        231      0.141      0.106     0.0678     0.0273

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     11/200      9.67G      1.214      1.419      1.225         26       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.7s
                   all         24        231      0.487      0.175     0.0636      0.039

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     12/200      9.67G      1.141      1.157      1.195         24       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.7s
                   all         24        231      0.708      0.383      0.405      0.217

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     13/200      9.67G      1.016      1.212      1.141         21       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.7it/s 0.6s
                   all         24        231      0.711      0.472      0.467      0.294

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     14/200      9.67G      1.068      1.218       1.14         37       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.7s
                   all         24        231      0.437      0.394      0.399      0.281

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     15/200      9.67G     0.9446      1.029      1.079         13       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.6s
                   all         24        231      0.782      0.625      0.736      0.454

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     16/200      9.67G     0.9974      1.118       1.09          6       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.5it/s 0.7s
                   all         24        231      0.862      0.697      0.758      0.563

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     17/200      9.67G      1.034        1.1      1.114         46       1280: 100% ━━━━━━━━━━━━ 25/25 2.8it/s 8.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.6s
                   all         24        231      0.706      0.767      0.795      0.498

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     18/200      9.67G     0.9575      2.939      1.043          0       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.7s
                   all         24        231        0.9      0.689      0.824      0.552

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     19/200      9.67G     0.9882     0.9237      1.087         22       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.7s
                   all         24        231      0.904      0.803      0.868       0.66

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     20/200      9.67G     0.9348     0.9177      1.074         31       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.6s
                   all         24        231      0.886       0.77      0.854      0.604

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     21/200      9.67G       1.01     0.9567      1.077         19       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.7s
                   all         24        231      0.794      0.651      0.785       0.49

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     22/200      9.67G      0.974     0.9059      1.077         33       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.7it/s 0.6s
                   all         24        231      0.796      0.343      0.396      0.266

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     23/200      9.67G     0.8904     0.8332      1.033         16       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.5it/s 0.7s
                   all         24        231      0.382      0.346      0.358      0.194

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     24/200      9.67G     0.8972     0.7731      1.026         11       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.7it/s 0.6s
                   all         24        231      0.831      0.806      0.897      0.686

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     25/200      9.67G     0.9492     0.8268      1.058         15       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.7it/s 0.6s
                   all         24        231      0.831      0.822      0.891      0.662

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     26/200      9.67G     0.9367     0.8671      1.066          9       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.7it/s 0.6s
                   all         24        231      0.811      0.638      0.692       0.43

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     27/200      9.67G     0.9206      0.828       1.02         34       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.7s
                   all         24        231       0.62      0.446      0.497      0.334

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     28/200      9.67G     0.8998     0.8531      1.058         33       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.7s
                   all         24        231      0.848      0.719       0.83      0.565

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     29/200      9.67G     0.9328     0.8155      1.049         14       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.7it/s 0.6s
                   all         24        231       0.84      0.761      0.848      0.597

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     30/200      9.67G     0.8918     0.7686      1.052         16       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.5it/s 0.7s
                   all         24        231      0.921      0.765      0.873      0.631

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     31/200      9.67G     0.8296      3.666      0.963         16       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.7it/s 0.6s
                   all         24        231      0.912      0.797      0.898      0.593

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     32/200      9.67G     0.8873     0.7932      1.045         20       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.7it/s 0.6s
                   all         24        231      0.948      0.765      0.901      0.564

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     33/200      9.67G     0.8623     0.7535      1.034         21       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.7it/s 0.6s
                   all         24        231       0.95      0.786      0.891      0.644

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     34/200      9.67G      0.805     0.7117      1.007         23       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.6s
                   all         24        231      0.882      0.784      0.877      0.607

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     35/200      9.67G     0.8159      0.792      1.032         10       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.6s
                   all         24        231      0.894      0.712      0.846      0.585

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     36/200      9.67G     0.8069     0.7566      1.002         45       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.5it/s 0.7s
                   all         24        231      0.916      0.761      0.851      0.632

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     37/200      9.67G      0.796     0.7118      1.003         49       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.6s
                   all         24        231      0.847      0.832      0.912      0.674

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     38/200      9.67G     0.7616     0.7008     0.9892         27       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.7it/s 0.6s
                   all         24        231      0.891      0.773      0.864      0.636

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     39/200      9.67G      0.845     0.7631       1.02         25       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.7s
                   all         24        231      0.949      0.724      0.867      0.559
EarlyStopping: Training stopped early as no improvement observed in last 15 epochs. Best results observed at epoch 24, best model saved as best.pt.
To update EarlyStopping(patience=15) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

39 epochs completed in 0.121 hours.
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s100_m_2/weights/weights/last.pt, 40.7MB
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s100_m_2/weights/weights/best.pt, 40.7MB

Validating /home/sjkim/solar-thermal/runs/detect/workspace/train_s100_m_2/weights/weights/best.pt...
Ultralytics 8.4.43 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
YOLO11m summary (fused): 126 layers, 20,032,345 parameters, 0 gradients, 67.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.5it/s 0.7s
                   all         24        231      0.829      0.826      0.893       0.68
             pv_string         18        125      0.837      0.976      0.983      0.831
             pv_module         17         94      0.845      0.809      0.875      0.669
                 other          8         12      0.806      0.694      0.821       0.54
Speed: 0.6ms preprocess, 21.1ms inference, 0.0ms loss, 2.4ms postprocess per image
Results saved to /home/sjkim/solar-thermal/runs/detect/workspace/train_s100_m_2/weights
Elapsed: 0:07:32
```

```bash
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s200_m \
    --model models/yolo11m.pt \
    --epochs 400 \
    --device cuda \
    --amp True \
    --output ./workspace/train_s200_m
Seed 이미지 222장 (수동 라벨 완료)
  train: 178장
  val: 44장
New https://pypi.org/project/ultralytics/8.4.45 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.4.43 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=4, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=workspace/train_s200_m/dataset/data.yaml, degrees=5.0, deterministic=True, device=0, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=400, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.5, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=1280, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=models/yolo11m.pt, momentum=0.937, mosaic=0.5, multi_scale=0.0, name=weights, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=15, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=workspace/train_s200_m, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m/weights, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.3, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.05, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=3

                   from  n    params  module                                       arguments                     
  0                  -1  1      1856  ultralytics.nn.modules.conv.Conv             [3, 64, 3, 2]                 
  1                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  2                  -1  1    111872  ultralytics.nn.modules.block.C3k2            [128, 256, 1, True, 0.25]     
  3                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
  4                  -1  1    444928  ultralytics.nn.modules.block.C3k2            [256, 512, 1, True, 0.25]     
  5                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  6                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  7                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  8                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
 10                  -1  1    990976  ultralytics.nn.modules.block.C2PSA           [512, 512, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1    542720  ultralytics.nn.modules.block.C3k2            [1024, 256, 1, True]          
 17                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1   1511424  ultralytics.nn.modules.block.C3k2            [768, 512, 1, True]           
 20                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 23        [16, 19, 22]  1   1413337  ultralytics.nn.modules.head.Detect           [3, 16, None, [256, 512, 512]]
YOLO11m summary: 232 layers, 20,055,321 parameters, 20,055,305 gradients, 68.2 GFLOPs

Transferred 643/649 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1602.0±67.7 MB/s, size: 11641.4 KB)
train: Scanning /home/sjkim/solar-thermal/workspace/train_s200_m/dataset/labels/train... 178 images, 19 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 178/178 1.2Kit/s 0.1s
train: New cache created: /home/sjkim/solar-thermal/workspace/train_s200_m/dataset/labels/train.cache
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2658.6±1458.8 MB/s, size: 11244.6 KB)
val: Scanning /home/sjkim/solar-thermal/workspace/train_s200_m/dataset/labels/val... 44 images, 3 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 44/44 1.1Kit/s 0.0s
val: New cache created: /home/sjkim/solar-thermal/workspace/train_s200_m/dataset/labels/val.cache
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001429, momentum=0.9) with parameter groups 106 weight(decay=0.0), 113 weight(decay=0.0005), 112 bias(decay=0.0)
Plotting labels to /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m/weights/labels.jpg... 
Image sizes 1280 train, 1280 val
Using 8 dataloader workers
Logging results to /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m/weights
Starting training for 400 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/400      8.15G       1.34      2.006      1.411         43       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 34.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 1.6it/s 3.8s
                   all         44        514      0.749      0.426      0.593      0.404

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/400      8.64G      1.145      1.374      1.194         29       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.2it/s 1.2s
                   all         44        514       0.64        0.8      0.752       0.45

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      3/400      8.64G      1.117      1.194      1.205         65       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.2it/s 1.2s
                   all         44        514      0.128      0.163      0.135     0.0664

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      4/400      8.64G       1.59       1.54      1.506         37       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.2it/s 1.2s
                   all         44        514      0.252      0.356      0.263      0.113

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      5/400      8.64G      1.171      1.117      1.227         58       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 16.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 4.7it/s 1.3s
                   all         44        514      0.334     0.0279   4.35e-05   1.64e-05

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      6/400      8.64G       1.12      1.003      1.196         48       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 4.8it/s 1.3s
                   all         44        514      0.134      0.141      0.044     0.0302

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      7/400      8.64G      1.037     0.8941      1.155         37       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s
                   all         44        514      0.766     0.0404     0.0481     0.0276

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      8/400      8.64G     0.9965     0.8537      1.107         31       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s
                   all         44        514      0.682      0.567      0.618      0.382

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      9/400      8.64G     0.9837     0.8362      1.109         32       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.3it/s 1.1s
                   all         44        514      0.772       0.52      0.581      0.396

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     10/400      8.64G     0.9311     0.8204      1.082         43       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s
                   all         44        514      0.823       0.85      0.852      0.568

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     11/400      8.64G      0.944     0.8275      1.096         24       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.3it/s 1.1s
                   all         44        514       0.94      0.576      0.625      0.463

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     12/400      8.64G     0.9208     0.7265      1.061         13       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.3it/s 1.1s
                   all         44        514      0.824      0.777      0.875      0.682

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     13/400      8.64G      0.911     0.6983      1.059         66       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s
                   all         44        514      0.825      0.845      0.882      0.655

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     14/400      8.64G     0.9217     0.6943      1.053         41       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s
                   all         44        514      0.891      0.836      0.916      0.681

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     15/400      8.64G     0.8999     0.6711      1.048         41       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s
                   all         44        514      0.884      0.845      0.918      0.702

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     16/400      8.64G      0.881     0.6427      1.057         29       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s
                   all         44        514      0.831      0.842      0.909      0.666

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     17/400      8.64G     0.8301     0.6492      1.016         20       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.3it/s 1.1s
                   all         44        514      0.852       0.88       0.93      0.725

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     18/400      8.64G      0.864     0.6848       1.04         38       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s
                   all         44        514      0.836       0.78       0.86      0.611

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     19/400      8.64G     0.8255     0.6436      1.031         29       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s
                   all         44        514      0.844      0.825      0.873       0.57

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     20/400      8.64G     0.8058     0.6277      1.007          9       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s
                   all         44        514      0.858      0.821      0.868      0.643

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     21/400      8.64G     0.8209     0.6336      1.028         41       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.2it/s 1.1s
                   all         44        514      0.912      0.813      0.861       0.63

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     22/400      8.64G     0.8023     0.6304      1.022         34       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s
                   all         44        514      0.883      0.827      0.861      0.589

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     23/400      8.64G     0.7901     0.6156      1.008         47       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s
                   all         44        514      0.882      0.823      0.875      0.593

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     24/400      8.64G     0.8015     0.6198      1.004         37       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s
                   all         44        514      0.857      0.822      0.853      0.586

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     25/400      8.64G     0.7672     0.5943     0.9904         42       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s
                   all         44        514      0.865       0.84      0.865      0.648

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     26/400      8.64G     0.8056     0.5905      1.018         40       1280: 100% ━━━━━━━━━━━━ 45/45 2.9it/s 15.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s
                   all         44        514      0.889      0.747      0.862      0.589

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     27/400      8.64G     0.7692     0.6001      1.007         19       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s
                   all         44        514      0.894      0.839      0.893      0.698

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     28/400      8.64G     0.7707     0.5895      1.002         19       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.3it/s 1.1s
                   all         44        514       0.91      0.836      0.896      0.715

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     29/400      8.64G     0.8043     0.5705      1.016         30       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s
                   all         44        514      0.918      0.847      0.894      0.613

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     30/400      8.64G      0.776     0.5729     0.9955         21       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s
                   all         44        514      0.843      0.817      0.877      0.676

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     31/400      8.64G      0.808     0.5885      1.016         43       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.3it/s 1.1s
                   all         44        514      0.874      0.801      0.868      0.687

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     32/400      8.64G     0.7607     0.5941     0.9996         52       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s
                   all         44        514      0.806      0.833      0.871       0.65
EarlyStopping: Training stopped early as no improvement observed in last 15 epochs. Best results observed at epoch 17, best model saved as best.pt.
To update EarlyStopping(patience=15) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

32 epochs completed in 0.170 hours.
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m/weights/weights/last.pt, 40.7MB
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m/weights/weights/best.pt, 40.7MB

Validating /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m/weights/weights/best.pt...
Ultralytics 8.4.43 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
YOLO11m summary (fused): 126 layers, 20,032,345 parameters, 0 gradients, 67.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        514      0.852       0.88       0.93      0.725
             pv_string         35        292       0.95      0.997      0.992      0.805
             pv_module         36        215      0.983      0.788      0.953      0.726
                 other          6          7      0.623      0.857      0.846      0.646
Speed: 0.9ms preprocess, 19.5ms inference, 0.0ms loss, 4.1ms postprocess per image
Results saved to /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m/weights
Elapsed: 0:10:30
```

```bash
ython scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s200_m_2 \
    --model models/yolo11m.pt \
    --epochs 400 \
    --device cuda \
    --amp True \
    --output ./workspace/train_s200_m_2
Seed 이미지 222장 (수동 라벨 완료)
  train: 178장
  val: 44장
New https://pypi.org/project/ultralytics/8.4.45 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.4.43 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=4, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=workspace/train_s200_m_2/dataset/data.yaml, degrees=5.0, deterministic=True, device=0, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=400, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.5, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=1280, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=models/yolo11m.pt, momentum=0.937, mosaic=0.5, multi_scale=0.0, name=weights, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=15, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=workspace/train_s200_m_2, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m_2/weights, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.3, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.05, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=3

                   from  n    params  module                                       arguments                     
  0                  -1  1      1856  ultralytics.nn.modules.conv.Conv             [3, 64, 3, 2]                 
  1                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  2                  -1  1    111872  ultralytics.nn.modules.block.C3k2            [128, 256, 1, True, 0.25]     
  3                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
  4                  -1  1    444928  ultralytics.nn.modules.block.C3k2            [256, 512, 1, True, 0.25]     
  5                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  6                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  7                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  8                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
 10                  -1  1    990976  ultralytics.nn.modules.block.C2PSA           [512, 512, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1    542720  ultralytics.nn.modules.block.C3k2            [1024, 256, 1, True]          
 17                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1   1511424  ultralytics.nn.modules.block.C3k2            [768, 512, 1, True]           
 20                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 23        [16, 19, 22]  1   1413337  ultralytics.nn.modules.head.Detect           [3, 16, None, [256, 512, 512]]
YOLO11m summary: 232 layers, 20,055,321 parameters, 20,055,305 gradients, 68.2 GFLOPs

Transferred 643/649 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1617.2±81.3 MB/s, size: 11641.4 KB)
train: Scanning /home/sjkim/solar-thermal/workspace/train_s200_m_2/dataset/labels/train... 178 images, 19 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 178/178 1.2Kit/s 0.1s
train: New cache created: /home/sjkim/solar-thermal/workspace/train_s200_m_2/dataset/labels/train.cache
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2619.2±1434.0 MB/s, size: 11244.6 KB)
val: Scanning /home/sjkim/solar-thermal/workspace/train_s200_m_2/dataset/labels/val... 44 images, 3 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 44/44 1.0Kit/s 0.0s
val: New cache created: /home/sjkim/solar-thermal/workspace/train_s200_m_2/dataset/labels/val.cache
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001429, momentum=0.9) with parameter groups 106 weight(decay=0.0), 113 weight(decay=0.0005), 112 bias(decay=0.0)
Plotting labels to /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m_2/weights/labels.jpg... 
Image sizes 1280 train, 1280 val
Using 8 dataloader workers
Logging results to /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m_2/weights
Starting training for 400 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/400      8.15G      1.394      2.073      1.437         43       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 34.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 1.7it/s 3.5s
                   all         44        514      0.656      0.471      0.364      0.239

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/400      8.64G      1.154      1.364      1.204         29       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.0it/s 1.2s
                   all         44        514       0.79      0.521      0.589      0.359

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      3/400      8.64G      1.356      1.373      1.431         65       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.1it/s 1.2s
                   all         44        514      0.321      0.458      0.341      0.228

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      4/400      8.64G      1.369      1.651      1.352         37       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.2it/s 1.2s
                   all         44        514       0.54      0.468      0.489      0.257

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      5/400      8.64G      1.287      1.419      1.274         58       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.2it/s 1.2s
                   all         44        514      0.228      0.408       0.29      0.151

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      6/400      8.64G      1.136      1.114      1.166         48       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.2it/s 1.1s
                   all         44        514      0.479      0.451      0.173      0.115

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      7/400      8.64G      1.018      1.003      1.121         37       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.3it/s 1.1s
                   all         44        514       0.76      0.735      0.816      0.597

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      8/400      8.64G     0.9775      0.862      1.074         31       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s
                   all         44        514      0.824      0.799      0.848      0.576

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      9/400      8.64G      1.028     0.8461      1.144         32       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s
                   all         44        514      0.798      0.841      0.857      0.649

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     10/400      8.64G     0.9861     0.8402      1.093         43       1280: 100% ━━━━━━━━━━━━ 45/45 2.9it/s 15.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s
                   all         44        514      0.891      0.804      0.848      0.533

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     11/400      8.64G     0.9726     0.8444      1.103         24       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.2it/s 1.2s
                   all         44        514      0.926      0.582      0.746      0.392

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     12/400      8.64G     0.8583     0.7157      1.022         13       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s
                   all         44        514      0.856      0.811      0.853      0.582

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     13/400      8.64G     0.9046     0.7216      1.037         66       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s
                   all         44        514      0.829      0.322      0.317      0.238

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     14/400      8.64G     0.9164      0.701      1.053         41       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.3it/s 1.1s
                   all         44        514      0.877      0.866      0.919      0.568

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     15/400      8.64G     0.9637       0.71      1.072         41       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s
                   all         44        514      0.774       0.83      0.873      0.674

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     16/400      8.64G     0.9369     0.6923      1.075         29       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s
                   all         44        514      0.911      0.908       0.92      0.601

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     17/400      8.64G     0.8753     0.6754       1.03         20       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.3it/s 1.1s
                   all         44        514      0.905      0.915      0.926       0.71

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     18/400      8.64G        0.9     0.6761      1.048         38       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.2it/s 1.1s
                   all         44        514       0.86      0.866      0.933      0.703

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     19/400      8.64G     0.8599     0.6524       1.04         29       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s
                   all         44        514      0.906      0.859      0.883      0.613

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     20/400      8.64G     0.8163     0.6072      1.012          9       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.2it/s 1.1s
                   all         44        514      0.876      0.862      0.922      0.689

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     21/400      8.64G     0.8476     0.6302       1.02         41       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.3it/s 1.1s
                   all         44        514      0.914      0.884      0.909      0.614

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     22/400      8.64G     0.8421     0.6287       1.03         34       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s
                   all         44        514       0.83      0.832      0.871      0.568

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     23/400      8.64G     0.8072     0.5851       1.01         47       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s
                   all         44        514       0.84       0.85      0.909      0.713

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     24/400      8.64G     0.8233     0.5923       1.01         37       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s
                   all         44        514      0.921      0.867      0.939      0.701

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     25/400      8.64G     0.7819     0.5561     0.9917         42       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s
                   all         44        514      0.885      0.922      0.944      0.684

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     26/400      8.64G     0.8197       0.56      1.016         40       1280: 100% ━━━━━━━━━━━━ 45/45 2.9it/s 15.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s
                   all         44        514      0.909      0.875      0.933       0.61

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     27/400      8.64G     0.8063      0.583      1.017         19       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s
                   all         44        514      0.894      0.839      0.907      0.673

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     28/400      8.64G     0.7961     0.5561      1.006         19       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s
                   all         44        514      0.918      0.871      0.915      0.737

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     29/400      8.64G     0.8276     0.5499      1.022         30       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s
                   all         44        514      0.935      0.859      0.879      0.617

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     30/400      8.64G     0.8121     0.5656      1.014         21       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s
                   all         44        514      0.903       0.81      0.869      0.661

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     31/400      8.64G     0.8363      0.575      1.022         43       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s
                   all         44        514       0.92      0.871      0.859      0.594

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     32/400      8.64G     0.7801     0.5417     0.9939         52       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s
                   all         44        514       0.92      0.881      0.868      0.622

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     33/400      8.64G     0.8047     0.5408      1.014         46       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.2it/s 1.1s
                   all         44        514      0.917      0.842      0.875      0.608

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     34/400      8.64G     0.8404     0.5612      1.014         18       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s
                   all         44        514      0.916      0.863      0.873      0.699

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     35/400      8.64G     0.7845     0.5329     0.9943         44       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 16.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.3it/s 1.1s
                   all         44        514      0.881      0.869      0.884      0.651

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     36/400      8.64G     0.7764      0.545      1.017         41       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s
                   all         44        514      0.889      0.887      0.884      0.709

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     37/400      8.64G     0.7703     0.5249     0.9962         24       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.3it/s 1.1s
                   all         44        514       0.91      0.881      0.882      0.716

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     38/400      8.64G     0.8407      0.563      1.035         16       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s
                   all         44        514      0.927      0.878       0.87      0.598

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     39/400      8.64G     0.8015     0.5337      1.014         43       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s
                   all         44        514      0.834      0.873      0.822      0.592

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     40/400      8.64G     0.8088     0.5161      1.014         34       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.6it/s 1.1s
                   all         44        514      0.852      0.772      0.882      0.697

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     41/400      8.64G      0.784     0.5233     0.9983         39       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s
                   all         44        514      0.893      0.871      0.914      0.766

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     42/400      8.64G     0.7631     0.4998      0.987         21       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.3it/s 1.1s
                   all         44        514      0.936      0.879      0.913      0.658

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     43/400      8.64G     0.7714     0.4908     0.9855         49       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s
                   all         44        514      0.783      0.881      0.781      0.535

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     44/400      8.64G     0.7957     0.5246      1.008         18       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.3it/s 1.1s
                   all         44        514      0.918      0.868      0.935      0.601

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     45/400      8.64G     0.7648     0.4982     0.9828         48       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.3it/s 1.1s
                   all         44        514        0.9      0.868      0.939      0.769

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     46/400      8.64G     0.7933     0.5169      0.992         31       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s
                   all         44        514      0.924      0.864      0.895      0.649

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     47/400      8.64G     0.7543     0.4915     0.9745         59       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s
                   all         44        514      0.921      0.832      0.887      0.683

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     48/400      8.64G     0.7275     0.4947     0.9753         37       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s
                   all         44        514      0.932      0.887      0.903      0.636

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     49/400      8.64G     0.7505     0.5016     0.9803         41       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.3it/s 1.1s
                   all         44        514      0.894      0.796      0.926      0.694

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     50/400      8.64G     0.7423     0.4978     0.9771         56       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s
                   all         44        514       0.94      0.854       0.93      0.772

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     51/400      8.64G     0.7399     0.5004     0.9737         47       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.3it/s 1.1s
                   all         44        514      0.889       0.88      0.892      0.556

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     52/400      8.64G     0.7617     0.4935     0.9921         26       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s
                   all         44        514      0.919      0.884       0.92      0.681

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     53/400      8.64G     0.8326     0.5372      1.016         39       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s
                   all         44        514      0.914      0.868      0.911      0.762

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     54/400      8.64G     0.7887     0.5153      1.007         57       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s
                   all         44        514      0.923      0.878      0.891       0.67

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     55/400      8.64G     0.7458     0.4731     0.9854         41       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s
                   all         44        514       0.93      0.876      0.919      0.712

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     56/400      8.64G     0.7317     0.4751     0.9779         33       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s
                   all         44        514       0.89      0.881      0.901      0.682

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     57/400      8.64G     0.7493     0.4704     0.9809         48       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s
                   all         44        514      0.888      0.907      0.924      0.756

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     58/400      8.64G     0.7154     0.4556     0.9705         46       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s
                   all         44        514      0.929       0.88      0.895      0.575

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     59/400      8.64G     0.7248     0.4675     0.9776         77       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s
                   all         44        514      0.924      0.879      0.886      0.717

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     60/400      8.64G     0.7267     0.4804      0.974         54       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.2it/s 1.1s
                   all         44        514      0.815      0.857      0.919      0.708

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     61/400      8.64G     0.7031     0.4525     0.9686         30       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.3it/s 1.1s
                   all         44        514      0.922      0.879      0.924      0.738

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     62/400      8.64G        0.7     0.4435     0.9613         34       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s
                   all         44        514      0.924      0.869      0.931      0.792

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     63/400      8.64G     0.6884     0.4356     0.9675          8       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s
                   all         44        514      0.868      0.881       0.91      0.726

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     64/400      8.64G     0.6741     0.4245     0.9589         28       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s
                   all         44        514      0.927       0.88      0.878      0.658

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     65/400      8.64G      0.706     0.4674     0.9747         22       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s
                   all         44        514      0.907      0.877      0.908      0.662

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     66/400      8.64G     0.7132      0.452     0.9645         24       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s
                   all         44        514       0.97      0.872      0.914      0.675

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     67/400      8.64G     0.7021     0.4444     0.9702         25       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s
                   all         44        514      0.943       0.86      0.913      0.758

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     68/400      8.64G     0.7492     0.4643     0.9749         30       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s
                   all         44        514      0.914      0.876      0.915      0.675

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     69/400      8.64G     0.7201     0.4431     0.9705         24       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.3it/s 1.1s
                   all         44        514      0.914      0.881      0.911      0.686

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     70/400      8.64G     0.6973      0.447     0.9675         61       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s
                   all         44        514      0.894      0.862      0.922       0.74

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     71/400      8.64G     0.7074     0.4324     0.9568         58       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s
                   all         44        514      0.917      0.867      0.916      0.747

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     72/400      8.64G     0.7356     0.4831      0.982         27       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s
                   all         44        514      0.908      0.872      0.903      0.719

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     73/400      8.64G     0.7126     0.4479     0.9575         40       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s
                   all         44        514       0.93      0.877       0.89      0.692

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     74/400      8.64G     0.6817     0.4416     0.9631         14       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s
                   all         44        514      0.927      0.875      0.889      0.674

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     75/400      8.64G     0.6897     0.4283     0.9449         38       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.3it/s 1.1s
                   all         44        514      0.909      0.872      0.874      0.709

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     76/400      8.64G     0.6731     0.4295     0.9515         32       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s
                   all         44        514       0.92      0.883      0.898      0.742

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     77/400      8.64G     0.6774     0.4313     0.9573         49       1280: 100% ━━━━━━━━━━━━ 45/45 2.8it/s 15.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.3it/s 1.1s
                   all         44        514      0.931      0.883      0.914      0.687
EarlyStopping: Training stopped early as no improvement observed in last 15 epochs. Best results observed at epoch 62, best model saved as best.pt.
To update EarlyStopping(patience=15) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

77 epochs completed in 0.400 hours.
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m_2/weights/weights/last.pt, 40.7MB
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m_2/weights/weights/best.pt, 40.7MB

Validating /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m_2/weights/weights/best.pt...
Ultralytics 8.4.43 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
YOLO11m summary (fused): 126 layers, 20,032,345 parameters, 0 gradients, 67.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.9it/s 2.1s
                   all         44        514      0.924       0.87       0.93      0.792
             pv_string         31        167      0.994       0.96      0.994      0.881
             pv_module         41        340      0.988      0.934       0.99      0.819
                 other          6          7      0.791      0.714      0.806      0.677
Speed: 1.1ms preprocess, 18.5ms inference, 0.0ms loss, 17.4ms postprocess per image
Results saved to /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m_2/weights
Elapsed: 0:24:19
```

```bash
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s200_m \
    --model models/yolo11m.pt \
    --epochs 400 \
    --batch 8 \
    --device cuda \
    --amp True \
    --output ./workspace/train_s200_m_b8
Seed 이미지 222장 (수동 라벨 완료)
  train: 178장
  val: 44장
New https://pypi.org/project/ultralytics/8.4.45 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.4.43 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=8, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=workspace/train_s200_m_b8/dataset/data.yaml, degrees=5.0, deterministic=True, device=0, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=400, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.5, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=1280, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=models/yolo11m.pt, momentum=0.937, mosaic=0.5, multi_scale=0.0, name=weights, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=15, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=workspace/train_s200_m_b8, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m_b8/weights, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.3, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.05, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=3

                   from  n    params  module                                       arguments                     
  0                  -1  1      1856  ultralytics.nn.modules.conv.Conv             [3, 64, 3, 2]                 
  1                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  2                  -1  1    111872  ultralytics.nn.modules.block.C3k2            [128, 256, 1, True, 0.25]     
  3                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
  4                  -1  1    444928  ultralytics.nn.modules.block.C3k2            [256, 512, 1, True, 0.25]     
  5                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  6                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  7                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  8                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
 10                  -1  1    990976  ultralytics.nn.modules.block.C2PSA           [512, 512, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1    542720  ultralytics.nn.modules.block.C3k2            [1024, 256, 1, True]          
 17                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1   1511424  ultralytics.nn.modules.block.C3k2            [768, 512, 1, True]           
 20                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 23        [16, 19, 22]  1   1413337  ultralytics.nn.modules.head.Detect           [3, 16, None, [256, 512, 512]]
YOLO11m summary: 232 layers, 20,055,321 parameters, 20,055,305 gradients, 68.2 GFLOPs

Transferred 643/649 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1575.9±82.3 MB/s, size: 11641.4 KB)
train: Scanning /home/sjkim/solar-thermal/workspace/train_s200_m_b8/dataset/labels/train... 178 images, 19 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 178/178 1.2Kit/s 0.1s
train: New cache created: /home/sjkim/solar-thermal/workspace/train_s200_m_b8/dataset/labels/train.cache
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1764.0±1459.4 MB/s, size: 11244.6 KB)
val: Scanning /home/sjkim/solar-thermal/workspace/train_s200_m_b8/dataset/labels/val... 44 images, 3 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 44/44 998.7it/s 0.0s
val: New cache created: /home/sjkim/solar-thermal/workspace/train_s200_m_b8/dataset/labels/val.cache
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001429, momentum=0.9) with parameter groups 106 weight(decay=0.0), 113 weight(decay=0.0005), 112 bias(decay=0.0)
Plotting labels to /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m_b8/weights/labels.jpg... 
Image sizes 1280 train, 1280 val
Using 8 dataloader workers
Logging results to /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m_b8/weights
Starting training for 400 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/400      15.8G      1.514      2.346      1.575         39       1280: 100% ━━━━━━━━━━━━ 23/23 2.0s/it 47.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.0s/it 3.1s
                   all         44        514       0.62      0.752      0.674      0.458

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/400      15.9G     0.9274      1.082      1.096         29       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.4it/s 1.2s
                   all         44        514      0.591      0.359      0.364      0.216

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      3/400        16G      1.037      0.966      1.153         37       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.3it/s 1.3s
                   all         44        514        0.4      0.399       0.34      0.182

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      4/400        16G     0.9896     0.8877      1.118         37       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        514      0.172      0.599      0.148      0.106

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      5/400        16G      1.096     0.8448      1.211         48       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.184      0.225     0.0954     0.0621

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      6/400      15.8G      1.071     0.9145      1.163         38       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.4it/s 1.2s
                   all         44        514       0.02     0.0123    0.00161   0.000602

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      7/400        16G      0.981     0.8325      1.112         16       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.745      0.659      0.681      0.478

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      8/400        16G     0.9079       0.75      1.084         31       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.546      0.622      0.556      0.365

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      9/400        16G     0.9435     0.7268      1.082         26       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.723      0.606      0.685      0.458

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     10/400      15.9G     0.9149     0.6955      1.078         46       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.718      0.655       0.75      0.487

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     11/400        16G      0.877     0.6851       1.06         24       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.727      0.878      0.818      0.617

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     12/400      15.9G     0.8612     0.6766      1.041          5       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514       0.91      0.711      0.822      0.587

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     13/400      15.9G     0.8571     0.6617      1.037         28       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514       0.87       0.78      0.815      0.619

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     14/400        16G     0.8538     0.6864       1.03         22       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514       0.67      0.829      0.809      0.473

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     15/400      15.9G     0.8196     0.6372       1.02         36       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.721      0.859      0.794      0.602

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     16/400        16G     0.7685     0.5915      1.007         29       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.815      0.489      0.535      0.375

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     17/400        16G     0.7752     0.6145     0.9915         49       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.686     0.0647     0.0666     0.0585

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     18/400        16G     0.8128     0.6129       1.02         18       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.859      0.311      0.406      0.299

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     19/400        16G       0.75     0.6172      1.005         18       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.791      0.524      0.597      0.451

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     20/400      15.9G     0.7184     0.5765     0.9889          9       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.873        0.8       0.89      0.628

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     21/400      15.9G     0.7276     0.5806     0.9835         74       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.787      0.835      0.888      0.686

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     22/400        16G     0.7219     0.5866     0.9846         53       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.864      0.807      0.905      0.664

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     23/400        16G     0.7506     0.5902     0.9956         46       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.876      0.851      0.862       0.54

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     24/400      15.9G     0.7246     0.5595     0.9862         28       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.909      0.862      0.935       0.65

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     25/400        16G     0.7229     0.5382     0.9806         32       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.898      0.843      0.914      0.729

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     26/400      15.9G     0.7217     0.5283     0.9844         35       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.895      0.845      0.948      0.717

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     27/400        16G     0.7074     0.5574     0.9899         23       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.857      0.859      0.911      0.697

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     28/400      15.9G      0.698     0.5273     0.9758         27       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514        0.9      0.864      0.919      0.699

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     29/400      15.9G       0.76     0.5558      1.001         30       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.738      0.877      0.857      0.676

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     30/400      15.9G     0.7062     0.5301     0.9749         16       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514       0.79       0.91      0.875      0.682

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     31/400        16G      0.687      0.565     0.9683         74       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514       0.82      0.905      0.918      0.755

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     32/400      15.9G     0.7148     0.5283     0.9705         41       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.928      0.879      0.938      0.634

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     33/400      15.9G     0.7175     0.5356     0.9729         42       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.894      0.865      0.927       0.71

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     34/400      15.9G     0.6872     0.5257     0.9715         19       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.895      0.818      0.904      0.709

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     35/400        16G     0.7052     0.5173     0.9618         44       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.898      0.847       0.91      0.718

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     36/400      15.9G     0.6743     0.5147     0.9692         46       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.856      0.869      0.921      0.706

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     37/400      15.9G     0.6487     0.4893     0.9488         23       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.888      0.846      0.926      0.738

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     38/400      15.9G     0.6758     0.5204     0.9691         47       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.913      0.894      0.935      0.712

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     39/400      15.9G     0.6334     0.4958     0.9597         40       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.903      0.854      0.894      0.742

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     40/400        16G      0.629     0.5045     0.9549         27       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.863      0.838      0.852      0.716

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     41/400        16G     0.6385     0.4772     0.9524         20       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.882      0.846      0.894      0.739

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     42/400      15.9G     0.6449     0.5551     0.9489          2       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.893      0.918      0.921      0.759

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     43/400      15.9G     0.6572     0.5104     0.9513         46       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.819      0.871      0.886      0.724

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     44/400        16G     0.6637     0.4954     0.9637         18       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.877      0.811      0.919       0.71

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     45/400      15.9G     0.7121     0.5174     0.9768         40       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.831      0.894      0.898      0.742

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     46/400      15.9G     0.6615     0.5018     0.9566         35       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.911       0.91      0.929       0.73

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     47/400      15.9G     0.6662     0.5027     0.9535         30       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.898      0.861      0.924       0.77

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     48/400        16G     0.6606     0.4872     0.9553         38       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514       0.87      0.833      0.906      0.686

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     49/400      15.9G     0.6332     0.4796     0.9431         49       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.908      0.836      0.875      0.723

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     50/400        16G     0.6262      0.457     0.9408         44       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514       0.89      0.845      0.894      0.758

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     51/400      15.9G     0.6196     0.4881     0.9558         26       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.897      0.861      0.909      0.738

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     52/400        16G     0.6299     0.4914     0.9436         27       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514       0.83      0.873       0.91      0.726

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     53/400        16G     0.6697     0.4957     0.9637         29       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.898      0.864       0.92      0.638

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     54/400      15.9G     0.6073     0.4638      0.936         47       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.915      0.892       0.94      0.753

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     55/400        16G     0.6047     0.4632     0.9417         32       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.6it/s 1.2s
                   all         44        514      0.888      0.854      0.911      0.746

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     56/400        16G     0.6473     0.4794     0.9469         29       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.911       0.86      0.944      0.785

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     57/400        16G     0.6568     0.4643     0.9563         30       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514       0.89      0.865       0.93      0.732

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     58/400        16G     0.6205      0.451     0.9451         51       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.903      0.844      0.934       0.78

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     59/400      15.9G     0.6039     0.4323      0.935         40       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.905      0.846      0.917       0.77

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     60/400        16G     0.5769     0.4637      0.928         43       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.916      0.863      0.921      0.782

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     61/400        16G     0.5799      0.443     0.9305         27       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.887      0.861      0.917      0.777

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     62/400      15.9G     0.5968      0.465     0.9358         42       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.897      0.836      0.906      0.771

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     63/400        16G     0.5616       0.44     0.9238          8       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.881      0.846      0.904      0.775

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     64/400        16G     0.5512     0.4274     0.9244         22       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.922      0.856      0.902      0.756

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     65/400      15.9G     0.5595     0.4339     0.9178         25       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.919      0.849      0.896       0.74

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     66/400      15.9G     0.6075     0.4696     0.9315         12       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.919       0.85      0.886      0.713

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     67/400      15.9G     0.6034     0.4379     0.9273         25       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.897      0.845      0.904      0.757

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     68/400      15.9G      0.579     0.4349     0.9212         35       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.855      0.904      0.924       0.78

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     69/400      15.9G     0.5716     0.4481      0.921         51       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514       0.93      0.904      0.954      0.826

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     70/400      15.9G     0.5836     0.4345      0.931         39       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.939      0.857       0.95      0.813

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     71/400        16G     0.5904     0.4266     0.9342         61       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.927        0.9      0.944      0.806

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     72/400        16G     0.5531      0.418     0.9073         26       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.909      0.838      0.942      0.761

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     73/400      15.9G     0.5533     0.4337     0.9172         57       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514       0.88        0.9      0.937      0.777

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     74/400        16G     0.5669     0.4193     0.9211         38       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.883      0.878        0.9      0.745

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     75/400        16G     0.5617     0.4187     0.9161         42       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.881      0.903      0.945      0.777

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     76/400        16G     0.5727     0.4392     0.9243         30       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.859      0.917      0.937      0.778

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     77/400        16G     0.6043     0.4416     0.9342         40       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.916      0.865      0.935      0.773

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     78/400      15.9G      0.606      0.438     0.9303         32       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.914      0.872      0.938      0.795

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     79/400        16G     0.5609     0.4236     0.9195         18       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.891      0.913      0.939      0.778

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     80/400      15.9G     0.5884     0.4241     0.9311         41       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.853      0.917      0.903      0.769

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     81/400      15.9G     0.5997     0.4201     0.9301         17       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.851      0.916      0.902      0.767

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     82/400      15.9G     0.5707     0.4157     0.9179         34       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.6it/s 1.2s
                   all         44        514      0.869      0.846       0.88      0.721

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     83/400      15.9G     0.5535     0.4201     0.9172         40       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.881      0.851      0.888      0.764

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     84/400      15.9G     0.5704     0.4458     0.9251         30       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.894      0.846      0.904      0.761
EarlyStopping: Training stopped early as no improvement observed in last 15 epochs. Best results observed at epoch 69, best model saved as best.pt.
To update EarlyStopping(patience=15) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

84 epochs completed in 0.465 hours.
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m_b8/weights/weights/last.pt, 40.7MB
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m_b8/weights/weights/best.pt, 40.7MB

Validating /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m_b8/weights/weights/best.pt...
Ultralytics 8.4.43 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
YOLO11m summary (fused): 126 layers, 20,032,345 parameters, 0 gradients, 67.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514       0.93      0.904      0.954      0.826
             pv_string         35        292      0.983      0.989      0.994      0.902
             pv_module         36        215      0.984      0.866      0.981      0.798
                 other          6          7      0.824      0.857      0.887      0.778
Speed: 0.5ms preprocess, 21.5ms inference, 0.0ms loss, 2.2ms postprocess per image
Results saved to /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m_b8/weights
Elapsed: 0:28:15
```

```bash
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s200_m_2 \
    --model models/yolo11m.pt \
    --epochs 400 \
    --batch 8 \
    --device cuda \
    --amp True \
    --output ./workspace/train_s200_m_2_b8
Seed 이미지 222장 (수동 라벨 완료)
  train: 178장
  val: 44장
New https://pypi.org/project/ultralytics/8.4.45 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.4.43 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=8, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=workspace/train_s200_m_2_b8/dataset/data.yaml, degrees=5.0, deterministic=True, device=0, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=400, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.5, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=1280, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=models/yolo11m.pt, momentum=0.937, mosaic=0.5, multi_scale=0.0, name=weights, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=15, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=workspace/train_s200_m_2_b8, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m_2_b8/weights, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.3, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.05, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=3

                   from  n    params  module                                       arguments                     
  0                  -1  1      1856  ultralytics.nn.modules.conv.Conv             [3, 64, 3, 2]                 
  1                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  2                  -1  1    111872  ultralytics.nn.modules.block.C3k2            [128, 256, 1, True, 0.25]     
  3                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
  4                  -1  1    444928  ultralytics.nn.modules.block.C3k2            [256, 512, 1, True, 0.25]     
  5                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  6                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  7                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  8                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
 10                  -1  1    990976  ultralytics.nn.modules.block.C2PSA           [512, 512, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1    542720  ultralytics.nn.modules.block.C3k2            [1024, 256, 1, True]          
 17                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1   1511424  ultralytics.nn.modules.block.C3k2            [768, 512, 1, True]           
 20                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 23        [16, 19, 22]  1   1413337  ultralytics.nn.modules.head.Detect           [3, 16, None, [256, 512, 512]]
YOLO11m summary: 232 layers, 20,055,321 parameters, 20,055,305 gradients, 68.2 GFLOPs

Transferred 643/649 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1557.9±51.3 MB/s, size: 11641.4 KB)
train: Scanning /home/sjkim/solar-thermal/workspace/train_s200_m_2_b8/dataset/labels/train... 178 images, 19 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 178/178 1.2Kit/s 0.1s
train: New cache created: /home/sjkim/solar-thermal/workspace/train_s200_m_2_b8/dataset/labels/train.cache
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2656.1±1489.9 MB/s, size: 11244.6 KB)
val: Scanning /home/sjkim/solar-thermal/workspace/train_s200_m_2_b8/dataset/labels/val... 44 images, 3 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 44/44 1.0Kit/s 0.0s
val: New cache created: /home/sjkim/solar-thermal/workspace/train_s200_m_2_b8/dataset/labels/val.cache
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001429, momentum=0.9) with parameter groups 106 weight(decay=0.0), 113 weight(decay=0.0005), 112 bias(decay=0.0)
Plotting labels to /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m_2_b8/weights/labels.jpg... 
Image sizes 1280 train, 1280 val
Using 8 dataloader workers
Logging results to /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m_2_b8/weights
Starting training for 400 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/400      15.8G      1.525      2.424      1.597         39       1280: 100% ━━━━━━━━━━━━ 23/23 2.0s/it 46.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.0s/it 3.1s
                   all         44        514      0.659      0.618      0.517      0.388

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/400      15.9G     0.9438      1.193       1.09         29       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.4it/s 1.2s
                   all         44        514      0.477      0.428       0.36      0.232

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      3/400        16G      1.039      1.014       1.16         37       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.598      0.387      0.374      0.217

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      4/400        16G      1.016     0.9207      1.125         37       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.1it/s 1.4s
                   all         44        514     0.0579       0.28     0.0353     0.0196

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      5/400      15.9G      1.009     0.9579      1.139         48       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.576      0.177      0.146     0.0836

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      6/400      15.8G     0.9436     0.9483      1.105         38       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.649      0.626       0.56      0.333

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      7/400        16G      1.016     0.8421      1.129         16       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514       0.74      0.383      0.567      0.344

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      8/400        16G     0.9572     0.7716      1.097         31       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.339      0.526      0.401      0.256

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      9/400        16G     0.9448     0.7144      1.106         26       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.309      0.433      0.289      0.186

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     10/400      15.9G     0.9039     0.6901      1.076         46       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.681       0.75      0.672      0.522

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     11/400        16G     0.9058     0.6802      1.054         24       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514       0.87      0.719      0.875      0.672

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     12/400      15.9G     0.8649     0.7066      1.032          5       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.752        0.7      0.742      0.547

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     13/400      15.9G     0.9129     0.7642      1.067         28       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.198      0.256      0.162     0.0912

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     14/400        16G     0.8529     0.6641      1.043         22       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.417      0.386      0.364      0.217

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     15/400      15.9G     0.9113     0.6669      1.061         36       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.782      0.732      0.784      0.584

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     16/400        16G     0.8641     0.6039       1.04         29       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.889       0.81      0.842      0.501

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     17/400        16G      0.846     0.6064      1.024         49       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.872      0.838      0.863      0.683

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     18/400      16.1G     0.8891     0.6218      1.051         18       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.873      0.821      0.846      0.624

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     19/400        16G     0.8219      0.596      1.034         18       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.878      0.794      0.858       0.65

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     20/400      15.9G     0.8256     0.5882      1.027          9       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.847      0.881      0.888        0.7

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     21/400      15.9G     0.8318     0.5553      1.032         74       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.796      0.734      0.804      0.518

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     22/400        16G     0.7957     0.5523      1.013         53       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.842      0.801      0.839      0.551

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     23/400        16G     0.7755     0.5589       1.01         46       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.808      0.769      0.823      0.633

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     24/400      15.9G     0.7822     0.5375      1.003         28       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.891      0.808      0.845      0.519

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     25/400        16G     0.7567     0.5311     0.9928         32       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.899       0.84      0.923      0.721

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     26/400      15.9G     0.7898     0.5705      1.007         35       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.856      0.838      0.899      0.718

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     27/400        16G     0.7743     0.5509      1.014         23       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.862      0.884      0.888      0.573

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     28/400      15.9G     0.7725     0.5056      1.001         27       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.908      0.876      0.902      0.658

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     29/400      15.9G     0.8077     0.5174      1.019         30       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.885      0.806      0.874      0.715

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     30/400      15.9G      0.766     0.4974      0.991         16       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.896      0.802      0.904      0.632

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     31/400        16G     0.7851     0.5105     0.9995         74       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.839      0.758      0.848      0.677

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     32/400      15.9G     0.7819     0.5009      1.002         41       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514       0.91      0.874      0.915      0.652

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     33/400      15.9G     0.7524     0.5026     0.9869         42       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.874      0.837      0.889      0.658

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     34/400      15.9G     0.7326     0.4917     0.9974         19       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.886        0.8      0.856      0.628

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     35/400        16G      0.751     0.4777     0.9764         44       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.856      0.858      0.865      0.695

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     36/400      15.9G     0.7288     0.4692     0.9888         46       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.896       0.84      0.864      0.586

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     37/400      15.9G     0.6979     0.4642     0.9662         23       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.884      0.852      0.904      0.727

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     38/400      15.9G      0.757      0.486     0.9968         47       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.837      0.885      0.878      0.585

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     39/400      15.9G     0.7284     0.4455     0.9928         40       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514       0.88      0.894      0.902      0.697

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     40/400      15.9G     0.7047     0.4561     0.9751         27       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.913      0.827      0.871       0.66

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     41/400        16G     0.6992     0.4458     0.9725         20       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.933      0.928      0.921      0.715

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     42/400        16G     0.7073      0.585      0.974          2       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514       0.92      0.878       0.91       0.65

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     43/400      15.9G     0.7149     0.5008       0.97         46       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.888      0.835      0.884      0.668

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     44/400        16G     0.6949     0.4917     0.9775         18       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.884      0.869      0.874      0.533

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     45/400      15.9G     0.7449     0.5087     0.9939         40       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.918       0.89      0.887      0.736

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     46/400      15.9G     0.7098     0.4709     0.9784         35       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514       0.92      0.892      0.879      0.676

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     47/400      15.9G     0.7204     0.4737     0.9726         30       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.931      0.875      0.902      0.747

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     48/400        16G     0.7343     0.4621     0.9814         38       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.6it/s 1.2s
                   all         44        514      0.924      0.883      0.917      0.702

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     49/400      15.9G     0.7349     0.4808     0.9778         49       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.904      0.823      0.846      0.709

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     50/400        16G     0.7152     0.4642     0.9706         44       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514       0.93      0.886      0.899      0.733

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     51/400      15.9G     0.7018     0.4508     0.9863         26       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.878      0.856      0.896      0.648

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     52/400        16G     0.7156     0.4443     0.9752         27       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.929      0.879       0.89      0.758

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     53/400        16G     0.7412     0.4665     0.9932         29       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514       0.91      0.865      0.871      0.704

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     54/400      15.9G     0.6795     0.4307     0.9662         47       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.918      0.834      0.878      0.724

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     55/400        16G     0.6661     0.4241     0.9629         32       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.6it/s 1.2s
                   all         44        514      0.918       0.84      0.889      0.669

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     56/400        16G     0.7003     0.4475     0.9652         29       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.882      0.877        0.9      0.751

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     57/400        16G     0.7219     0.4523     0.9793         30       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.937      0.864      0.893      0.727

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     58/400      15.9G     0.6675     0.4425     0.9657         51       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.925      0.881      0.883      0.576

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     59/400      15.9G     0.6917     0.4254     0.9706         40       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.926      0.876      0.884      0.749

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     60/400        16G     0.6466     0.4282     0.9497         43       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.925      0.877      0.878      0.706

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     61/400        16G     0.6386     0.4169     0.9531         27       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.912      0.835      0.873      0.699

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     62/400      15.9G      0.637     0.4174       0.95         42       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.906      0.822      0.886      0.724

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     63/400        16G     0.6432     0.4084     0.9524          8       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.844      0.894      0.878      0.671

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     64/400        16G     0.6358     0.4001     0.9539         22       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.892      0.812      0.874      0.724

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     65/400      15.9G      0.638      0.401     0.9517         25       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.904      0.817      0.854      0.677

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     66/400      15.9G     0.6815     0.4355     0.9641         12       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514       0.86      0.817      0.842      0.714

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     67/400      15.9G     0.6398      0.398     0.9454         25       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.869      0.816      0.849      0.688
EarlyStopping: Training stopped early as no improvement observed in last 15 epochs. Best results observed at epoch 52, best model saved as best.pt.
To update EarlyStopping(patience=15) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

67 epochs completed in 0.373 hours.
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m_2_b8/weights/weights/last.pt, 40.7MB
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m_2_b8/weights/weights/best.pt, 40.7MB

Validating /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m_2_b8/weights/weights/best.pt...
Ultralytics 8.4.43 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
YOLO11m summary (fused): 126 layers, 20,032,345 parameters, 0 gradients, 67.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        514      0.929      0.879      0.889      0.758
             pv_string         31        167      0.988      0.983      0.993      0.866
             pv_module         41        340      0.991       0.94      0.993      0.811
                 other          6          7      0.808      0.714      0.682      0.598
Speed: 0.5ms preprocess, 21.3ms inference, 0.0ms loss, 2.2ms postprocess per image
Results saved to /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m_2_b8/weights
Elapsed: 0:22:42
```

Defect (class 3) 라벨 추가, class 3 인식 안됨. GPU L4에서 학습

```bash
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s100_m_d \
    --model models/yolo11m.pt \
    --epochs 200 \
    --device cuda \
    --amp True \
    --output ./workspace/train_s100_m_d
Seed 이미지 122장 (수동 라벨 완료)
  train: 98장
  val: 24장
New https://pypi.org/project/ultralytics/8.4.47 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.4.43 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=4, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=workspace/train_s100_m_d/dataset/data.yaml, degrees=5.0, deterministic=True, device=0, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=200, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.5, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=1280, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=models/yolo11m.pt, momentum=0.937, mosaic=0.5, multi_scale=0.0, name=weights-2, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=15, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=workspace/train_s100_m_d, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/home/sjkim/solar-thermal/runs/detect/workspace/train_s100_m_d/weights-2, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.3, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.05, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=4

                   from  n    params  module                                       arguments                     
  0                  -1  1      1856  ultralytics.nn.modules.conv.Conv             [3, 64, 3, 2]                 
  1                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  2                  -1  1    111872  ultralytics.nn.modules.block.C3k2            [128, 256, 1, True, 0.25]     
  3                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
  4                  -1  1    444928  ultralytics.nn.modules.block.C3k2            [256, 512, 1, True, 0.25]     
  5                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  6                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  7                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  8                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
 10                  -1  1    990976  ultralytics.nn.modules.block.C2PSA           [512, 512, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1    542720  ultralytics.nn.modules.block.C3k2            [1024, 256, 1, True]          
 17                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1   1511424  ultralytics.nn.modules.block.C3k2            [768, 512, 1, True]           
 20                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 23        [16, 19, 22]  1   1414108  ultralytics.nn.modules.head.Detect           [4, 16, None, [256, 512, 512]]
YOLO11m summary: 232 layers, 20,056,092 parameters, 20,056,076 gradients, 68.2 GFLOPs

Transferred 643/649 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2367.4±1621.2 MB/s, size: 13465.5 KB)
train: Scanning /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/labels/train.cache... 98 images, 21 backgrounds, 36 corrupt: 100% ━━━━━━━━━━━━ 98/98 14.7Mit/s 0.0s
train: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/train/DJI_20251217130214_0006_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
train: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/train/DJI_20251217130217_0007_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
train: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/train/DJI_20251217130220_0008_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
train: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/train/DJI_20251217130223_0009_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
train: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/train/DJI_20251217130225_0010_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
train: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/train/DJI_20251217130228_0011_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
train: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/train/DJI_20251217130231_0012_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
train: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/train/DJI_20251217130236_0014_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
train: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/train/DJI_20251217130239_0015_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
train: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/train/DJI_20251217130242_0016_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
train: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/train/DJI_20251217130306_0025_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
train: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/train/DJI_20251217130308_0026_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
train: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/train/DJI_20251217130311_0027_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
train: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/train/DJI_20251217130314_0028_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
train: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/train/DJI_20251217130317_0029_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
train: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/train/DJI_20251217130319_0030_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
train: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/train/DJI_20251217130325_0032_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
train: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/train/DJI_20251217130328_0033_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
train: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/train/DJI_20251217130331_0034_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
train: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/train/DJI_20251217130333_0035_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
train: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/train/DJI_20251217130413_0050_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
train: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/train/DJI_20251217130416_0051_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
train: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/train/DJI_20251217130419_0052_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
train: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/train/DJI_20251217130424_0054_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
train: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/train/DJI_20251217130456_0066_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
train: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/train/DJI_20251217130458_0067_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
train: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/train/DJI_20251217130506_0070_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
train: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/train/DJI_20251217130550_0086_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
train: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/train/DJI_20251217130553_0087_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
train: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/train/DJI_20251217130558_0089_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
train: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/train/DJI_20251217130600_0090_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
train: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/train/DJI_20251217130602_0091_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
train: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/train/DJI_20251217130605_0092_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
train: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/train/DJI_20251217130608_0093_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
train: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/train/DJI_20251217130610_0094_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
train: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/train/DJI_20251217130613_0095_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2623.4±723.7 MB/s, size: 11615.0 KB)
val: Scanning /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/labels/val.cache... 24 images, 2 backgrounds, 7 corrupt: 100% ━━━━━━━━━━━━ 24/24 3.4Mit/s 0.0s
val: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/val/DJI_20251217130233_0013_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
val: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/val/DJI_20251217130322_0031_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
val: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/val/DJI_20251217130422_0053_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
val: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/val/DJI_20251217130501_0068_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
val: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/val/DJI_20251217130504_0069_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
val: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/val/DJI_20251217130547_0085_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
val: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/images/val/DJI_20251217130555_0088_Z.JPG: ignoring corrupt image/label: Label class 3 exceeds dataset class count 3. Possible class labels are 0-2
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.00125, momentum=0.9) with parameter groups 106 weight(decay=0.0), 113 weight(decay=0.0005), 112 bias(decay=0.0)
Plotting labels to /home/sjkim/solar-thermal/runs/detect/workspace/train_s100_m_d/weights-2/labels.jpg... 
Image sizes 1280 train, 1280 val
Using 8 dataloader workers
Logging results to /home/sjkim/solar-thermal/runs/detect/workspace/train_s100_m_d/weights-2
Starting training for 200 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/200       8.1G      1.869      4.512       1.91         17       1280: 100% ━━━━━━━━━━━━ 16/16 1.4s/it 22.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.1s/it 3.4s
                   all         17        155      0.234      0.456      0.279     0.0996

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/200      8.57G      1.334      2.631      1.355         14       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 6.4it/s 0.5s
                   all         17        155      0.507      0.593      0.311       0.18

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      3/200       8.6G      1.186      2.297      1.278         37       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 6.5it/s 0.5s
                   all         17        155      0.223      0.433      0.188     0.0544

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      4/200       8.6G      1.238      2.591      1.357         13       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 6.7it/s 0.4s
                   all         17        155      0.853      0.381      0.464      0.257

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      5/200       8.6G       1.24      2.202      1.353         22       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 6.7it/s 0.5s
                   all         17        155      0.768      0.514      0.498      0.359

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      6/200       8.6G      1.145      1.962      1.287         34       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 6.6it/s 0.5s
                   all         17        155      0.728      0.468      0.472      0.271

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      7/200       8.6G      1.232      1.807      1.276         33       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 6.9it/s 0.4s
                   all         17        155      0.444      0.349      0.139     0.0728

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      8/200       8.6G       1.28      1.712      1.298         13       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 6.8it/s 0.4s
                   all         17        155     0.0157      0.217    0.00531    0.00198

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      9/200       8.6G      1.323      2.157      1.503          1       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.2it/s 0.4s
                   all         17        155     0.0194      0.107    0.00404    0.00171

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     10/200       8.6G      1.396      1.691      1.336         34       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 6.9it/s 0.4s
                   all         17        155     0.0238      0.186      0.016    0.00553

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     11/200       8.6G      1.316      1.781      1.317         10       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.0it/s 0.4s
                   all         17        155      0.433      0.548       0.47      0.246

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     12/200       8.6G      1.168      4.696      1.179          0       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 5.6it/s 0.5s
                   all         17        155      0.381      0.433      0.417      0.212

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     13/200       8.6G      1.203      1.338      1.274         17       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.0it/s 0.4s
                   all         17        155       0.75       0.52      0.504      0.313

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     14/200       8.6G       1.18      1.364      1.211         18       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 6.3it/s 0.5s
                   all         17        155      0.556      0.389      0.317      0.172

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     15/200       8.6G      1.177      1.207      1.195         23       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 6.3it/s 0.5s
                   all         17        155      0.535      0.461      0.219      0.143

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     16/200       8.6G      1.144       1.26      1.215         30       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 6.5it/s 0.5s
                   all         17        155        0.8      0.507      0.501      0.297

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     17/200       8.6G      1.121      1.237      1.215         29       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 6.3it/s 0.5s
                   all         17        155      0.721      0.466      0.479       0.36

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     18/200       8.6G      1.063      1.127      1.182         14       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 6.6it/s 0.5s
                   all         17        155      0.856      0.453      0.496      0.315

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     19/200       8.6G        1.1       1.23      1.188         54       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 6.8it/s 0.4s
                   all         17        155      0.825      0.466      0.504      0.277

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     20/200       8.6G     0.9309      1.021      1.099         26       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.0it/s 0.4s
                   all         17        155      0.927       0.52      0.557      0.353

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     21/200       8.6G     0.9677     0.9491      1.098         18       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.0it/s 0.4s
                   all         17        155       0.94      0.522       0.75      0.539

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     22/200       8.6G     0.9551     0.8998      1.111         11       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.0it/s 0.4s
                   all         17        155      0.679      0.713      0.738      0.524

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     23/200       8.6G     0.9003     0.9372        1.1          7       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 6.9it/s 0.4s
                   all         17        155       0.82      0.601      0.765       0.51

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     24/200       8.6G     0.9588     0.9111      1.101         38       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.2it/s 0.4s
                   all         17        155      0.622      0.723      0.764      0.531

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     25/200       8.6G     0.8997      1.747      1.038          0       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.3it/s 0.4s
                   all         17        155      0.758      0.837      0.802      0.488

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     26/200       8.6G     0.9788     0.8536      1.083         15       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.2it/s 0.4s
                   all         17        155      0.866       0.79      0.834      0.522

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     27/200       8.6G      1.002     0.8626      1.122         26       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.0it/s 0.4s
                   all         17        155      0.772      0.701       0.79      0.584

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     28/200       8.6G     0.8686     0.7918      1.037         24       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.2it/s 0.4s
                   all         17        155      0.829      0.792      0.836      0.601

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     29/200       8.6G     0.8823     0.9179      1.041         31       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.0it/s 0.4s
                   all         17        155      0.702      0.739      0.765      0.493

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     30/200       8.6G     0.9385     0.9369      1.055          2       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 6.8it/s 0.4s
                   all         17        155      0.667      0.748      0.652      0.376

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     31/200       8.6G     0.8946     0.8117      1.051         33       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.3it/s 0.4s
                   all         17        155      0.913      0.723      0.766      0.476

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     32/200       8.6G     0.8287      0.767      1.013         35       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.2it/s 0.4s
                   all         17        155      0.877      0.688      0.806       0.55

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     33/200       8.6G     0.8222     0.7784      1.005         24       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.1it/s 0.4s
                   all         17        155      0.804       0.79      0.853      0.564

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     34/200       8.6G      0.827     0.7374      1.029         12       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.1it/s 0.4s
                   all         17        155      0.864      0.732      0.835      0.538

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     35/200       8.6G      0.865      1.265     0.9778          0       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.0it/s 0.4s
                   all         17        155      0.678      0.937      0.824      0.483

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     36/200       8.6G     0.9484     0.8602      1.052         29       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.3it/s 0.4s
                   all         17        155      0.857      0.807      0.827      0.489

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     37/200       8.6G     0.8873     0.8348      1.043         28       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.0it/s 0.4s
                   all         17        155      0.841      0.734      0.803      0.548

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     38/200       8.6G      1.025     0.8773      1.109         27       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.3it/s 0.4s
                   all         17        155      0.952      0.745      0.838      0.642

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     39/200       8.6G     0.9481     0.8011      1.063         12       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.2it/s 0.4s
                   all         17        155      0.845      0.719      0.844      0.643

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     40/200       8.6G     0.9588     0.9576      1.107         10       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.2it/s 0.4s
                   all         17        155      0.758      0.732      0.799      0.541

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     41/200       8.6G      1.002     0.8433      1.065         15       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.0it/s 0.4s
                   all         17        155      0.912      0.626      0.778      0.474

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     42/200       8.6G     0.8707     0.8645      1.033         28       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.2it/s 0.4s
                   all         17        155      0.661      0.761      0.745      0.476

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     43/200       8.6G      0.833     0.7509      1.033         20       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.2it/s 0.4s
                   all         17        155      0.865      0.778       0.86      0.611

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     44/200       8.6G      0.802       0.74     0.9832         22       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.3it/s 0.4s
                   all         17        155      0.831      0.741      0.829      0.612

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     45/200       8.6G     0.8433     0.7184      1.024         14       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.1it/s 0.4s
                   all         17        155      0.869      0.717      0.904      0.639

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     46/200       8.6G     0.8171      0.703     0.9988         20       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.3it/s 0.4s
                   all         17        155      0.597      0.864       0.75      0.474

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     47/200       8.6G      0.864     0.7342      1.038         10       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.3it/s 0.4s
                   all         17        155      0.783      0.688      0.799      0.536

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     48/200       8.6G     0.9058     0.7222      1.035         12       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 6.9it/s 0.4s
                   all         17        155      0.813      0.749      0.847      0.605

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     49/200       8.6G     0.8262     0.7527      1.012          7       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.1it/s 0.4s
                   all         17        155      0.876      0.737       0.85      0.582

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     50/200       8.6G     0.8296     0.7309      1.024         37       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.3it/s 0.4s
                   all         17        155       0.91      0.732      0.883      0.638

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     51/200       8.6G     0.8184      0.667      1.005         19       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.4it/s 0.4s
                   all         17        155      0.912      0.746      0.894      0.672

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     52/200       8.6G     0.8627     0.7109      1.033         19       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.4it/s 0.4s
                   all         17        155       0.91      0.811      0.895       0.64

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     53/200       8.6G     0.7531     0.6466     0.9856         20       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.2it/s 0.4s
                   all         17        155      0.889      0.855      0.903      0.586

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     54/200       8.6G      0.787     0.6508      0.986          9       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.3it/s 0.4s
                   all         17        155      0.876      0.833      0.913      0.619

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     55/200       8.6G     0.7515     0.6547     0.9921         23       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.2it/s 0.4s
                   all         17        155      0.854      0.909      0.902       0.62

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     56/200       8.6G     0.8155     0.7847      1.021          3       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.3it/s 0.4s
                   all         17        155       0.88      0.862      0.926      0.572

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     57/200       8.6G     0.8468     0.7007       1.01         27       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.3it/s 0.4s
                   all         17        155      0.805      0.885      0.877      0.609

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     58/200       8.6G     0.8981      1.119      1.114          7       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.2it/s 0.4s
                   all         17        155      0.823      0.852      0.885      0.659

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     59/200       8.6G     0.8524      0.815      1.012         18       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.3it/s 0.4s
                   all         17        155      0.864      0.877      0.917      0.724

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     60/200       8.6G     0.8592     0.7016       1.01         20       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.3it/s 0.4s
                   all         17        155      0.873      0.894      0.905      0.696

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     61/200       8.6G     0.7824     0.6991     0.9837         43       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.2it/s 0.4s
                   all         17        155      0.877      0.877      0.897      0.608

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     62/200       8.6G     0.8267     0.7449      1.036         25       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.1it/s 0.4s
                   all         17        155      0.813      0.838      0.866      0.589

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     63/200       8.6G     0.8365     0.7234      1.004         11       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.3it/s 0.4s
                   all         17        155      0.875      0.766      0.876      0.634

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     64/200       8.6G     0.7271      1.409     0.9375          6       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.2it/s 0.4s
                   all         17        155      0.839      0.833      0.873      0.631

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     65/200       8.6G     0.8055     0.6389     0.9883          9       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.1it/s 0.4s
                   all         17        155      0.828      0.891      0.918      0.727

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     66/200       8.6G     0.8952     0.7139      1.022         35       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.3it/s 0.4s
                   all         17        155      0.973      0.766       0.91      0.742

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     67/200       8.6G     0.8017     0.6954      1.004          6       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.1it/s 0.4s
                   all         17        155      0.939      0.858      0.936      0.717

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     68/200       8.6G     0.7146     0.6377     0.9788         43       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.3it/s 0.4s
                   all         17        155      0.916      0.809      0.903      0.584

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     69/200       8.6G     0.8539     0.7341      1.032         18       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.0it/s 0.4s
                   all         17        155       0.91      0.849      0.926      0.603

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     70/200       8.6G     0.7649     0.6854     0.9776         32       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.3it/s 0.4s
                   all         17        155      0.938       0.87       0.94       0.68

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     71/200       8.6G     0.7724     0.6814     0.9792         29       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.3it/s 0.4s
                   all         17        155      0.934      0.852      0.945      0.695

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     72/200       8.6G     0.7129      1.111     0.9274         12       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.1it/s 0.4s
                   all         17        155      0.884      0.899      0.957      0.718

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     73/200       8.6G     0.7629     0.7112     0.9895         40       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.4it/s 0.4s
                   all         17        155      0.952      0.814      0.933      0.724

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     74/200       8.6G     0.8024     0.7487      1.011         14       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.3it/s 0.4s
                   all         17        155      0.874      0.879      0.908      0.715

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     75/200       8.6G     0.7804     0.6797     0.9875         19       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.2it/s 0.4s
                   all         17        155      0.917      0.858      0.907      0.669

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     76/200       8.6G     0.7427     0.6907     0.9846         26       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.4it/s 0.4s
                   all         17        155      0.903      0.882      0.913      0.673

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     77/200       8.6G     0.7738     0.7066       1.03         12       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.4it/s 0.4s
                   all         17        155      0.952      0.815      0.884      0.624

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     78/200       8.6G     0.7221     0.6325     0.9555         29       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.3it/s 0.4s
                   all         17        155      0.938      0.821      0.864      0.611

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     79/200       8.6G     0.7592     0.6285     0.9925          8       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.1it/s 0.4s
                   all         17        155      0.959        0.8      0.897      0.667

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     80/200       8.6G     0.7399      0.648     0.9781         17       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.2it/s 0.4s
                   all         17        155      0.886      0.825      0.909      0.681

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     81/200       8.6G     0.7187     0.6721     0.9608         30       1280: 100% ━━━━━━━━━━━━ 16/16 2.9it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.4it/s 0.4s
                   all         17        155      0.878      0.839      0.904      0.631
EarlyStopping: Training stopped early as no improvement observed in last 15 epochs. Best results observed at epoch 66, best model saved as best.pt.
To update EarlyStopping(patience=15) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

81 epochs completed in 0.170 hours.
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s100_m_d/weights-2/weights/last.pt, 40.6MB
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s100_m_d/weights-2/weights/best.pt, 40.6MB

Validating /home/sjkim/solar-thermal/runs/detect/workspace/train_s100_m_d/weights-2/weights/best.pt...
Ultralytics 8.4.43 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
YOLO11m summary (fused): 126 layers, 20,033,116 parameters, 0 gradients, 67.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.4it/s 0.4s
                   all         17        155      0.973      0.766      0.909      0.743
             pv_string         12         80       0.92      0.875       0.96      0.864
             pv_module         13         68          1       0.72      0.899      0.718
                 other          4          7          1      0.704      0.869      0.646
Speed: 0.5ms preprocess, 17.7ms inference, 0.0ms loss, 2.5ms postprocess per image
Results saved to /home/sjkim/solar-thermal/runs/detect/workspace/train_s100_m_d/weights-2
Elapsed: 0:10:34
```

Defect (class 3) 라벨 추가, class 3 인식됨. GPU L4에서 학습

```bash
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s100_m_d \
    --model models/yolo11m.pt \
    --epochs 200 \
    --device cuda \
    --amp True \
    --output ./workspace/train_s100_m_d
Seed 이미지 122장 (수동 라벨 완료)
  train: 98장
  val: 24장
New https://pypi.org/project/ultralytics/8.4.47 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.4.43 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=4, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=workspace/train_s100_m_d/dataset/data.yaml, degrees=5.0, deterministic=True, device=0, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=200, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.5, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=1280, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=models/yolo11m.pt, momentum=0.937, mosaic=0.5, multi_scale=0.0, name=weights, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=15, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=workspace/train_s100_m_d, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/home/sjkim/solar-thermal/runs/detect/workspace/train_s100_m_d/weights, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.3, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.05, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=4

                   from  n    params  module                                       arguments                     
  0                  -1  1      1856  ultralytics.nn.modules.conv.Conv             [3, 64, 3, 2]                 
  1                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  2                  -1  1    111872  ultralytics.nn.modules.block.C3k2            [128, 256, 1, True, 0.25]     
  3                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
  4                  -1  1    444928  ultralytics.nn.modules.block.C3k2            [256, 512, 1, True, 0.25]     
  5                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  6                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  7                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  8                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
 10                  -1  1    990976  ultralytics.nn.modules.block.C2PSA           [512, 512, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1    542720  ultralytics.nn.modules.block.C3k2            [1024, 256, 1, True]          
 17                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1   1511424  ultralytics.nn.modules.block.C3k2            [768, 512, 1, True]           
 20                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 23        [16, 19, 22]  1   1414108  ultralytics.nn.modules.head.Detect           [4, 16, None, [256, 512, 512]]
YOLO11m summary: 232 layers, 20,056,092 parameters, 20,056,076 gradients, 68.2 GFLOPs

Transferred 643/649 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2212.4±1324.6 MB/s, size: 13465.5 KB)
train: Scanning /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/labels/train... 98 images, 21 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 98/98 1.2Kit/s 0.1s
train: New cache created: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/labels/train.cache
val: Fast image access ✅ (ping: 0.6±1.3 ms, read: 2355.4±1271.6 MB/s, size: 11615.0 KB)
val: Scanning /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/labels/val... 24 images, 2 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 24/24 929.8it/s 0.0s
val: New cache created: /home/sjkim/solar-thermal/workspace/train_s100_m_d/dataset/labels/val.cache
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.00125, momentum=0.9) with parameter groups 106 weight(decay=0.0), 113 weight(decay=0.0005), 112 bias(decay=0.0)
Plotting labels to /home/sjkim/solar-thermal/runs/detect/workspace/train_s100_m_d/weights/labels.jpg... 
Image sizes 1280 train, 1280 val
Using 8 dataloader workers
Logging results to /home/sjkim/solar-thermal/runs/detect/workspace/train_s100_m_d/weights
Starting training for 200 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/200       8.1G      1.762      3.479      1.725         28       1280: 100% ━━━━━━━━━━━━ 25/25 1.1s/it 28.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.1it/s 2.8s
                   all         24        241      0.853      0.418      0.439      0.297

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/200      9.64G      1.172      2.043      1.205         37       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.4it/s 0.7s
                   all         24        241      0.516      0.478      0.377      0.192

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      3/200      9.64G       1.26      1.792       1.28         25       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.1it/s 0.7s
                   all         24        241      0.509       0.31      0.316      0.195

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      4/200      9.64G       1.28      2.034      1.253         31       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.5it/s 0.8s
                   all         24        241     0.0535       0.34     0.0247      0.014

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      5/200      9.65G      1.184       1.76      1.234         13       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.0it/s 0.8s
                   all         24        241     0.0115      0.212    0.00675    0.00357

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      6/200      9.65G      1.204      1.835      1.258         13       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.2it/s 0.7s
                   all         24        241      0.179      0.126      0.045     0.0236

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      7/200      9.65G      1.418      1.846      1.489         22       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.0it/s 0.8s
                   all         24        241     0.0206      0.109    0.00946    0.00328

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      8/200      9.65G      1.507      1.986      1.493         13       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.0it/s 0.7s
                   all         24        241      0.273      0.225     0.0945     0.0571

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      9/200      9.65G      1.275      1.459      1.338         37       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.5it/s 0.7s
                   all         24        241      0.434       0.43        0.3      0.188

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     10/200      9.65G       1.14      1.317      1.196         21       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.5it/s 0.7s
                   all         24        241      0.659      0.473      0.436      0.303

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     11/200      9.65G      1.167      1.411      1.201         30       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.5it/s 0.7s
                   all         24        241      0.818      0.416      0.517      0.373

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     12/200      9.65G      1.005      1.129      1.143         24       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.5it/s 0.7s
                   all         24        241      0.865      0.512      0.536      0.373

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     13/200      9.65G      1.035      1.132      1.156         26       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.7s
                   all         24        241      0.787      0.465      0.568      0.429

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     14/200      9.65G       1.07      1.254      1.136         38       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.7s
                   all         24        241      0.838      0.471       0.55       0.39

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     15/200      9.65G     0.9905       1.08      1.076         13       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.6s
                   all         24        241      0.829      0.515      0.583      0.403

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     16/200      9.65G     0.9596      1.101      1.064          7       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.7it/s 0.6s
                   all         24        241       0.87      0.524      0.657      0.435

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     17/200      9.65G     0.9573      1.085      1.063         50       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.7it/s 0.6s
                   all         24        241      0.872      0.429      0.597      0.408

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     18/200      9.65G     0.8657      2.962       1.01          0       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.7s
                   all         24        241      0.911      0.488      0.611      0.467

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     19/200      9.65G     0.9176     0.9105      1.048         29       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.6s
                   all         24        241      0.946      0.504      0.623      0.454

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     20/200      9.65G     0.8812       0.89      1.044         32       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.5it/s 0.7s
                   all         24        241      0.646      0.607      0.651      0.432

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     21/200      9.65G     0.9279     0.8929      1.038         23       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.6s
                   all         24        241      0.671      0.667      0.642      0.482

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     22/200      9.65G     0.9034     0.8894      1.042         33       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.7s
                   all         24        241       0.74      0.582      0.641      0.423

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     23/200      9.65G     0.8953     0.8283      1.022         16       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.6s
                   all         24        241      0.873      0.532      0.662      0.455

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     24/200      9.65G     0.8743     0.7882      1.022         12       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.7it/s 0.6s
                   all         24        241      0.643      0.764      0.713      0.512

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     25/200      9.65G     0.9047     0.8371      1.039         16       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.7it/s 0.6s
                   all         24        241      0.789       0.64      0.688      0.464

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     26/200      9.65G     0.9211     0.8877      1.065          9       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.7it/s 0.6s
                   all         24        241      0.807      0.621      0.714      0.459

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     27/200      9.65G     0.8617     0.8074      1.013         37       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.6s
                   all         24        241      0.826      0.665      0.724      0.507

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     28/200      9.65G      0.841      0.817      1.023         36       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.7s
                   all         24        241      0.876      0.708      0.771       0.53

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     29/200      9.65G     0.8609     0.7971      1.015         16       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.7it/s 0.6s
                   all         24        241      0.949      0.678      0.773      0.551

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     30/200      9.65G     0.8619     0.7888      1.022         16       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.6s
                   all         24        241      0.874      0.687      0.749      0.522

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     31/200      9.65G     0.7961       2.44     0.9465         22       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.7it/s 0.6s
                   all         24        241      0.905      0.633      0.752      0.543

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     32/200      9.65G     0.8391      0.769      1.027         21       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.7it/s 0.6s
                   all         24        241      0.783      0.732       0.76      0.536

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     33/200      9.65G     0.8334     0.7461      1.025         23       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.7it/s 0.6s
                   all         24        241      0.982      0.647      0.772      0.537

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     34/200      9.65G     0.7992     0.7023      1.006         23       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.6s
                   all         24        241       0.86      0.641      0.782      0.537

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     35/200      9.65G     0.8076     0.8086      1.029         12       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.6s
                   all         24        241        0.8      0.743      0.751      0.579

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     36/200      9.65G     0.8172     0.7541      1.004         46       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.7it/s 0.6s
                   all         24        241      0.789      0.777      0.785      0.596

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     37/200      9.65G     0.8006     0.7025     0.9985         49       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.7it/s 0.6s
                   all         24        241      0.938      0.742      0.808       0.63

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     38/200      9.65G     0.7504      0.714     0.9899         30       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.7it/s 0.6s
                   all         24        241      0.882      0.737      0.808      0.612

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     39/200      9.65G     0.7653     0.7551     0.9797         25       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.7it/s 0.6s
                   all         24        241      0.855      0.772      0.798      0.602

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     40/200      9.65G     0.8122     0.7408      1.014         13       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.7it/s 0.6s
                   all         24        241       0.87      0.682      0.777      0.499

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     41/200      9.65G     0.8223     0.6993      1.004         21       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.7it/s 0.6s
                   all         24        241       0.77      0.812      0.782      0.469

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     42/200      9.65G     0.8303     0.9172      1.023         19       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.7s
                   all         24        241      0.819      0.608      0.708      0.538

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     43/200      9.65G     0.8028     0.7594     0.9976         20       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.5it/s 0.7s
                   all         24        241      0.846      0.636      0.745      0.541

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     44/200      9.65G     0.8538     0.7847      1.014         26       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.7it/s 0.6s
                   all         24        241      0.813      0.647      0.757      0.508

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     45/200      9.65G     0.7764     0.7065     0.9589         13       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.7s
                   all         24        241      0.917      0.642      0.743      0.561

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     46/200      9.65G      0.818     0.7334      1.017         27       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.6s
                   all         24        241      0.949      0.676      0.789      0.562

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     47/200      9.65G     0.8344     0.7305      1.007         16       1280: 100% ━━━━━━━━━━━━ 25/25 2.8it/s 8.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.7it/s 0.6s
                   all         24        241      0.925      0.644      0.778      0.511

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     48/200      9.65G     0.7599     0.6577     0.9741         26       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.6s
                   all         24        241      0.871       0.62      0.757      0.541

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     49/200      9.65G     0.7289     0.6841     0.9726         27       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.7s
                   all         24        241      0.781      0.762      0.758      0.537

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     50/200      9.65G      0.726     0.6303     0.9741         14       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.6s
                   all         24        241      0.806      0.686      0.737       0.56

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     51/200      9.65G     0.7111     0.6952     0.9715         24       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.7it/s 0.6s
                   all         24        241      0.932      0.632      0.768      0.597

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     52/200      9.65G     0.7112     0.6718     0.9521         29       1280: 100% ━━━━━━━━━━━━ 25/25 2.9it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.5it/s 0.7s
                   all         24        241      0.931      0.692      0.765      0.582
EarlyStopping: Training stopped early as no improvement observed in last 15 epochs. Best results observed at epoch 37, best model saved as best.pt.
To update EarlyStopping(patience=15) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

52 epochs completed in 0.159 hours.
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s100_m_d/weights/weights/last.pt, 40.7MB
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s100_m_d/weights/weights/best.pt, 40.7MB

Validating /home/sjkim/solar-thermal/runs/detect/workspace/train_s100_m_d/weights/weights/best.pt...
Ultralytics 8.4.43 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
YOLO11m summary (fused): 126 layers, 20,033,116 parameters, 0 gradients, 67.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 4.5it/s 0.7s
                   all         24        241      0.938      0.742      0.808       0.63
             pv_string         18        125      0.899          1       0.99      0.862
             pv_module         17         94      0.955      0.819      0.956      0.739
                 other          8         12      0.912       0.75       0.88      0.649
                defect          7         10      0.988        0.4      0.406      0.269
Speed: 0.6ms preprocess, 21.2ms inference, 0.0ms loss, 2.2ms postprocess per image
Results saved to /home/sjkim/solar-thermal/runs/detect/workspace/train_s100_m_d/weights
Elapsed: 0:09:52
```

MPS에서 yolo11m 으로 학습했을 때, 시간이 너무 오래 걸림.<br/>
샘플 100개, pv string 및 pv panel, other, defect 지정, nagative 추가, m 모델 사용, MacOS에서 학습 (device=mps)

```bash
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s100_m_d \
    --model models/yolo11m.pt \
    --epochs 200 \
    --device mps \
    --output ./workspace/train_s100_m_d
Seed 이미지 122장 (수동 라벨 완료)
  train: 98장
  val: 24장
yaml path workspace/train_s100_m_d/dataset/data.yaml
New https://pypi.org/project/ultralytics/8.4.47 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.4.43 🚀 Python-3.11.1 torch-2.11.0 MPS (Apple M4 Pro)
engine/trainer: agnostic_nms=False, amp=False, angle=1.0, augment=False, auto_augment=randaugment, batch=4, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=workspace/train_s100_m_d/dataset/data.yaml, degrees=5.0, deterministic=True, device=mps, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=200, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.5, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=1280, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=models/yolo11m.pt, momentum=0.937, mosaic=0.5, multi_scale=0.0, name=weights-4, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=15, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=workspace/train_s100_m_d, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/train_s100_m_d/weights-4, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.3, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.05, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=4

                   from  n    params  module                                       arguments                     
  0                  -1  1      1856  ultralytics.nn.modules.conv.Conv             [3, 64, 3, 2]                 
  1                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  2                  -1  1    111872  ultralytics.nn.modules.block.C3k2            [128, 256, 1, True, 0.25]     
  3                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
  4                  -1  1    444928  ultralytics.nn.modules.block.C3k2            [256, 512, 1, True, 0.25]     
  5                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  6                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  7                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  8                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
 10                  -1  1    990976  ultralytics.nn.modules.block.C2PSA           [512, 512, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1    542720  ultralytics.nn.modules.block.C3k2            [1024, 256, 1, True]          
 17                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1   1511424  ultralytics.nn.modules.block.C3k2            [768, 512, 1, True]           
 20                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 23        [16, 19, 22]  1   1414108  ultralytics.nn.modules.head.Detect           [4, 16, None, [256, 512, 512]]
YOLO11m summary: 232 layers, 20,056,092 parameters, 20,056,076 gradients, 68.2 GFLOPs

Transferred 643/649 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 4917.3±337.5 MB/s, size: 12952.8 KB)
train: Scanning /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/train_s100_m_d/dataset/labels/train.cache... 98 images, 20 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 98/98 19.6Mit/s 0.0s
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 8237.9±1011.1 MB/s, size: 9767.3 KB)
val: Scanning /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/train_s100_m_d/dataset/labels/val.cache... 24 images, 3 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 24/24 11.2Mit/s 0.0s
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.00125, momentum=0.9) with parameter groups 106 weight(decay=0.0), 113 weight(decay=0.0005), 112 bias(decay=0.0)
Plotting labels to /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/train_s100_m_d/weights-4/labels.jpg... 
Image sizes 1280 train, 1280 val
Using 0 dataloader workers
Logging results to /Users/seongjungkim/Development/sayouzone/solar-thermal/runs/detect/workspace/train_s100_m_d/weights-4
Starting training for 200 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/200      16.4G      1.404       2.39      1.389         28       1280: 100% ━━━━━━━━━━━━ 25/25 217.4s/it 1:30:35
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 9.2s/it 27.6s
                   all         24        239       0.53      0.497      0.427      0.258

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/200      16.4G     0.9937      1.919      1.106         22       1280: 100% ━━━━━━━━━━━━ 25/25 187.2s/it 1:17:60
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 5.6s/it 16.9s
                   all         24        239      0.581       0.53      0.566      0.373

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      3/200      16.4G      1.057      1.884      1.182         20       1280: 100% ━━━━━━━━━━━━ 25/25 288.4s/it 2:00:09
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.3s/it 25.0s
                   all         24        239      0.368      0.211      0.207      0.135
```

샘플 200개, pv string 및 pv panel, other, anomaly 지정, m 모델 사용 (nagative 없음) (스트링 조건을 명확히, 배치 8)<br>
GCE g2-standard-8 (vCPU 8개, 메모리 32GB), NVIDIA L4 1개

```bash
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s200_m_d \
    --model models/yolo11m.pt \
    --epochs 400 \
    --batch 8 \
    --device cuda \
    --amp True \
    --output ./workspace/train_s200_m_d
Seed 이미지 200장 (수동 라벨 완료)
  train: 160장
  val: 40장
New https://pypi.org/project/ultralytics/8.4.48 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.4.43 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=8, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=workspace/train_s200_m_d/dataset/data.yaml, degrees=5.0, deterministic=True, device=0, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=400, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.5, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=1280, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=models/yolo11m.pt, momentum=0.937, mosaic=0.5, multi_scale=0.0, name=weights, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=15, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=workspace/train_s200_m_d, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m_d/weights, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.3, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.05, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=4

                   from  n    params  module                                       arguments                     
  0                  -1  1      1856  ultralytics.nn.modules.conv.Conv             [3, 64, 3, 2]                 
  1                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  2                  -1  1    111872  ultralytics.nn.modules.block.C3k2            [128, 256, 1, True, 0.25]     
  3                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
  4                  -1  1    444928  ultralytics.nn.modules.block.C3k2            [256, 512, 1, True, 0.25]     
  5                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  6                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  7                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  8                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
 10                  -1  1    990976  ultralytics.nn.modules.block.C2PSA           [512, 512, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1    542720  ultralytics.nn.modules.block.C3k2            [1024, 256, 1, True]          
 17                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1   1511424  ultralytics.nn.modules.block.C3k2            [768, 512, 1, True]           
 20                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 23        [16, 19, 22]  1   1414108  ultralytics.nn.modules.head.Detect           [4, 16, None, [256, 512, 512]]
YOLO11m summary: 232 layers, 20,056,092 parameters, 20,056,076 gradients, 68.2 GFLOPs

Transferred 643/649 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1575.2±65.6 MB/s, size: 11560.7 KB)
train: Scanning /home/sjkim/solar-thermal/workspace/train_s200_m_d/dataset/labels/train... 160 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 160/160 1.1Kit/s 0.1s
train: New cache created: /home/sjkim/solar-thermal/workspace/train_s200_m_d/dataset/labels/train.cache
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 3569.2±1118.4 MB/s, size: 11306.2 KB)
val: Scanning /home/sjkim/solar-thermal/workspace/train_s200_m_d/dataset/labels/val... 40 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 40/40 817.1it/s 0.0s
val: /home/sjkim/solar-thermal/workspace/train_s200_m_d/dataset/images/val/DJI_20251217130705_0114_Z.JPG: 1 duplicate labels removed
val: New cache created: /home/sjkim/solar-thermal/workspace/train_s200_m_d/dataset/labels/val.cache
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.00125, momentum=0.9) with parameter groups 106 weight(decay=0.0), 113 weight(decay=0.0005), 112 bias(decay=0.0)
Plotting labels to /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m_d/weights/labels.jpg... 
Image sizes 1280 train, 1280 val
Using 8 dataloader workers
Logging results to /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m_d/weights
Starting training for 400 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/400      15.7G       1.61      2.394      1.664        156       1280: 100% ━━━━━━━━━━━━ 20/20 2.1s/it 41.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.1s/it 3.2s
                   all         40        607      0.765      0.346       0.35      0.203

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/400      15.9G      1.006      1.201      1.119        154       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.8it/s 1.1s
                   all         40        607      0.595      0.518      0.552      0.336

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      3/400        16G      1.046      1.085      1.142        138       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.517      0.456      0.411      0.276

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      4/400      15.9G      1.007     0.9765      1.126        129       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.9it/s 1.0s
                   all         40        607      0.414      0.502      0.446      0.219

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      5/400      15.9G      1.026     0.9217      1.152        121       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.9it/s 1.0s
                   all         40        607      0.647       0.52      0.483      0.291

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      6/400      15.9G     0.9766     0.9593      1.118        143       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.9it/s 1.0s
                   all         40        607      0.726      0.461      0.487      0.328

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      7/400      15.9G     0.9735     0.9113      1.096        115       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.9it/s 1.0s
                   all         40        607       0.22      0.347      0.223      0.168

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      8/400        16G     0.9333     0.8499      1.094        146       1280: 100% ━━━━━━━━━━━━ 20/20 1.4it/s 14.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.857      0.498      0.555      0.339

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      9/400      15.9G     0.9015     0.8193      1.066        160       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.9it/s 1.0s
                   all         40        607      0.753      0.502      0.548      0.416

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     10/400      15.9G     0.8893     0.7438      1.046        150       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.585      0.385      0.381      0.251

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     11/400        16G       0.83     0.7348      1.025        206       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.9it/s 1.0s
                   all         40        607      0.671      0.312      0.358      0.271

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     12/400      15.9G     0.8476     0.7135      1.028        146       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.872      0.521      0.591      0.428

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     13/400        16G     0.7888     0.7254      1.014        172       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.9it/s 1.0s
                   all         40        607      0.719      0.164      0.127     0.0888

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     14/400        16G      0.815     0.7382      1.034        146       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.852      0.347      0.419      0.308

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     15/400        16G     0.7892     0.7464      1.007        163       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.9it/s 1.0s
                   all         40        607      0.813      0.504      0.582      0.415

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     16/400        16G     0.8104     0.7212      1.021        104       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.9it/s 1.0s
                   all         40        607      0.607      0.509      0.479      0.341

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     17/400        16G     0.7579     0.6453     0.9936        132       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.9it/s 1.0s
                   all         40        607      0.835      0.525      0.557      0.436

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     18/400      15.9G     0.7905     0.6685      1.006        106       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.9it/s 1.0s
                   all         40        607      0.583      0.587       0.56      0.441

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     19/400        16G     0.8074     0.6708      1.008        208       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.515      0.426      0.436      0.327

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     20/400        16G     0.7506     0.6388     0.9897        135       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.618      0.676      0.646      0.482

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     21/400        16G     0.7432     0.6146     0.9929        126       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.785      0.579       0.62      0.455

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     22/400      15.9G     0.7527     0.5959      1.006        162       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.828      0.633      0.637      0.486

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     23/400        16G     0.7857     0.6379      1.001        223       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.623      0.662       0.65      0.501

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     24/400      15.9G     0.8122      0.687      1.005        174       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607        0.7      0.658      0.639      0.501

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     25/400      15.9G     0.7676       0.68      1.013        165       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.9it/s 1.0s
                   all         40        607      0.705      0.661      0.664      0.493

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     26/400        16G     0.7027     0.6066     0.9674        182       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.687      0.649      0.632      0.515

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     27/400        16G     0.7244     0.6092     0.9795        130       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607       0.66      0.617      0.634      0.442

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     28/400        16G     0.7483     0.6111     0.9814        107       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.668      0.615      0.617      0.509

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     29/400        16G     0.7287     0.6004      0.986        126       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.777      0.625       0.63       0.48

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     30/400        16G     0.7183      0.585     0.9855        185       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.795      0.652       0.68      0.529

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     31/400      15.9G     0.6904     0.5685     0.9799        149       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.876      0.627      0.646      0.536

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     32/400        16G     0.6621     0.5745     0.9612        153       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.652      0.654      0.619      0.516

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     33/400        16G     0.6499     0.5435     0.9514        146       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.553       0.61       0.62      0.473

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     34/400      15.9G     0.6954      0.565     0.9762        159       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.569      0.716      0.672      0.537

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     35/400      15.9G     0.7359     0.5662     0.9901        162       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607       0.76      0.584      0.665      0.536

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     36/400      15.9G     0.6954     0.5672     0.9741        121       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.617      0.666      0.667      0.545

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     37/400        16G     0.6765     0.5563     0.9718        155       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.684      0.744      0.696      0.563

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     38/400      15.9G     0.6859     0.5234     0.9604        232       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.888      0.609      0.684      0.557

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     39/400        16G     0.6719     0.5569     0.9635        161       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.752      0.709      0.732      0.583

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     40/400      15.9G     0.6833     0.5563     0.9612        170       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.9it/s 1.0s
                   all         40        607      0.744      0.688      0.728      0.579

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     41/400      15.9G     0.6407     0.5176     0.9527        130       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.9it/s 1.0s
                   all         40        607      0.665      0.724      0.685      0.558

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     42/400        16G     0.6755      0.551     0.9711        206       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607       0.66      0.735      0.677      0.564

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     43/400        16G     0.6619     0.5154      0.967         91       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.663      0.729       0.71      0.541

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     44/400        16G     0.6802      0.518     0.9645        210       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.745      0.695      0.714      0.576

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     45/400      15.9G      0.672     0.5185     0.9514        132       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.662      0.761      0.693      0.572

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     46/400      15.9G     0.6233     0.4766     0.9403        145       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.684      0.697      0.677      0.557

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     47/400        16G     0.6442     0.4818     0.9418        136       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.569      0.727      0.677      0.517

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     48/400      15.9G      0.623     0.4865     0.9339        165       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607       0.66        0.8      0.755      0.579

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     49/400        16G     0.6609     0.5092     0.9682        163       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.708      0.826      0.756      0.611

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     50/400      15.9G     0.6519     0.5104     0.9547        110       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.686      0.846      0.775      0.595

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     51/400      15.9G     0.6494     0.4884     0.9483        184       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.739      0.736      0.726      0.615

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     52/400        16G     0.6548     0.4918     0.9538        165       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.732      0.764      0.763      0.593

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     53/400      15.9G     0.6307     0.4832     0.9422        207       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.778      0.717      0.774      0.601

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     54/400        16G     0.6255     0.4939     0.9456        126       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.746      0.792      0.801      0.631

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     55/400      15.9G     0.6231     0.4906     0.9423        187       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.693      0.803      0.755      0.632

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     56/400      15.9G     0.6102     0.4942     0.9357        163       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.716      0.726      0.738      0.623

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     57/400      15.9G     0.6337     0.4874     0.9513        145       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607       0.87      0.636      0.709      0.595

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     58/400        16G     0.6392     0.4827     0.9491        172       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.777      0.668      0.713      0.556

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     59/400      15.9G      0.633     0.5234     0.9442        119       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.752       0.74      0.755      0.619

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     60/400      15.9G     0.6287     0.4939     0.9429        168       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.688      0.833      0.772      0.635

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     61/400      15.9G     0.6166     0.4755      0.939        157       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.698      0.748      0.748      0.584

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     62/400      15.9G     0.6062     0.4813     0.9327        123       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607       0.76      0.731      0.754      0.594

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     63/400        16G     0.6212     0.4679     0.9463        187       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.737      0.794      0.795      0.621

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     64/400        16G     0.6105     0.4852     0.9462        144       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.714      0.829      0.804      0.644

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     65/400        16G     0.6209     0.4863     0.9467        153       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.711      0.796      0.794      0.655

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     66/400      15.9G     0.6142     0.4788     0.9394        166       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.673      0.804      0.797      0.642

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     67/400      15.9G     0.6088     0.4885     0.9374        115       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.851      0.668      0.762      0.628

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     68/400        16G     0.5799     0.4645     0.9248        174       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.795      0.677       0.75      0.622

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     69/400        16G     0.5979      0.476     0.9446        136       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.733      0.765      0.745      0.629

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     70/400      15.9G     0.5642     0.4487     0.9297        150       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.665      0.799      0.725      0.596

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     71/400        16G     0.5622     0.4389     0.9291        108       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.643      0.761      0.709      0.575

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     72/400        16G      0.582     0.4477     0.9288        171       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.775      0.771      0.808      0.617

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     73/400      15.9G     0.5638     0.4496     0.9329        145       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.747      0.789      0.787      0.642

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     74/400        16G     0.5817     0.4559     0.9148        152       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.777        0.8      0.822      0.665

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     75/400      15.9G     0.5755      0.432     0.9247        158       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.746      0.812      0.798      0.669

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     76/400      15.9G     0.5952     0.4494     0.9328        141       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.796      0.764      0.799      0.645

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     77/400        16G      0.576     0.4335       0.92        151       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.673      0.828      0.798      0.659

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     78/400        16G     0.5691     0.4374     0.9241        159       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.728      0.801      0.819       0.65

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     79/400        16G     0.5745     0.4218     0.9266        153       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.9it/s 1.0s
                   all         40        607      0.781      0.729      0.806      0.662

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     80/400        16G     0.5933     0.4582     0.9298        132       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607       0.83       0.75       0.81      0.668

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     81/400        16G     0.5793     0.4597     0.9287        189       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.765      0.747      0.815      0.663

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     82/400        16G     0.5719     0.4438     0.9296        205       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.836      0.701      0.821      0.649

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     83/400      15.9G     0.5605      0.444     0.9163        138       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.765      0.851      0.826      0.669

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     84/400      15.9G     0.5574     0.4406     0.9165        128       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.759      0.837      0.819      0.685

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     85/400        16G     0.5443      0.434     0.9104        109       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.774      0.779      0.793      0.662

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     86/400      15.9G     0.5527     0.4232     0.9197        112       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.9it/s 1.0s
                   all         40        607      0.771      0.784      0.804      0.647

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     87/400      15.9G     0.5887     0.4539     0.9257        150       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.727      0.783      0.738      0.621

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     88/400        16G     0.5771     0.4179     0.9162        179       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.711      0.757      0.755      0.632

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     89/400        16G     0.5553     0.4249     0.9158        135       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.775      0.879      0.831      0.686

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     90/400        16G     0.5834     0.4407     0.9223        157       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.769      0.867      0.832      0.685

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     91/400        16G     0.5611     0.4268     0.9204        145       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.752      0.818      0.827      0.694

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     92/400        16G     0.5515     0.4148     0.9074        226       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.806      0.802      0.824      0.682

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     93/400        16G     0.5481     0.3908     0.9066        171       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.824      0.773      0.821      0.669

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     94/400      15.9G     0.5245     0.4041     0.9049        130       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.786      0.849      0.814      0.679

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     95/400        16G     0.5496     0.4223     0.9148        128       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.765      0.823      0.824      0.674

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     96/400        16G     0.5758     0.4192     0.9159        126       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607       0.77      0.732      0.801      0.671

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     97/400        16G     0.5731     0.4202       0.92        171       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.732      0.894      0.821      0.655

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     98/400      15.9G     0.5516     0.4199     0.9128        159       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.724      0.912      0.822      0.668

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     99/400        16G     0.5216     0.3966     0.8972        141       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607       0.83      0.695      0.751      0.637

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    100/400        16G     0.5484     0.3952     0.9038        121       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.797      0.757      0.818      0.646

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    101/400      15.9G     0.5513     0.4097      0.911        179       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607       0.83      0.782      0.841      0.673

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    102/400        16G     0.5266     0.4073     0.9042        145       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.715      0.919      0.829      0.702

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    103/400        16G     0.5313     0.4041     0.9059        146       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.839      0.796       0.87      0.713

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    104/400        16G      0.534     0.4112     0.9045        172       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.815      0.749       0.83      0.701

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    105/400        16G     0.5506     0.4014     0.9135        148       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.745      0.778      0.804      0.689

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    106/400        16G     0.5747     0.4369     0.9141        121       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.749       0.79      0.788       0.65

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    107/400      15.9G     0.5681     0.4486     0.9228        115       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.824      0.747      0.821      0.668

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    108/400        16G     0.5232     0.3995     0.9085        112       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.762      0.818       0.83      0.673

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    109/400        16G     0.5336     0.4089     0.9165         62       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.731      0.854      0.812      0.681

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    110/400        16G     0.5181     0.3963      0.902        167       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.751      0.898      0.823      0.687

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    111/400      15.9G     0.5197     0.3996     0.8997        105       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.799      0.845      0.825      0.693

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    112/400      15.9G     0.5304     0.4212     0.8979        154       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.815      0.878      0.827      0.701

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    113/400        16G     0.5399     0.4254     0.9075        135       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.807      0.858      0.869       0.71

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    114/400        16G     0.5399     0.4071     0.9062        175       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.809      0.844      0.847      0.685

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    115/400        16G     0.5357      0.403     0.9061        130       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.718      0.832      0.794      0.674

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    116/400        16G     0.5253     0.4105     0.9068        136       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.736      0.845      0.807      0.685

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    117/400      15.9G     0.5277     0.3957     0.9018        162       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.809      0.826      0.809      0.674

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    118/400        16G     0.5271     0.3877     0.9032        186       1280: 100% ━━━━━━━━━━━━ 20/20 1.3it/s 14.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 3.0it/s 1.0s
                   all         40        607      0.763      0.858      0.808      0.678
EarlyStopping: Training stopped early as no improvement observed in last 15 epochs. Best results observed at epoch 103, best model saved as best.pt.
To update EarlyStopping(patience=15) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

118 epochs completed in 0.591 hours.
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m_d/weights/weights/last.pt, 40.6MB
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m_d/weights/weights/best.pt, 40.6MB

Validating /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m_d/weights/weights/best.pt...
Ultralytics 8.4.43 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
YOLO11m summary (fused): 126 layers, 20,033,116 parameters, 0 gradients, 67.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.7it/s 1.1s
                   all         40        607      0.837      0.797       0.87      0.712
             pv_string         37        317      0.969      0.991      0.995      0.926
             pv_module         38        259      0.988      0.941      0.991      0.834
                 other          9         11      0.724      0.455      0.688      0.483
                defect         13         20      0.667      0.803      0.806      0.607
Speed: 0.5ms preprocess, 19.7ms inference, 0.0ms loss, 3.0ms postprocess per image
Results saved to /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m_d/weights
Elapsed: 0:35:54
```

샘플 200개, pv string 및 pv panel, other, anomaly 지정, nagative 추가, m 모델 사용 (스트링 조건을 명확히, 배치 8)<br>
GCE g2-standard-8 (vCPU 8개, 메모리 32GB), NVIDIA L4 1개

```bash
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s200_m_d \
    --model models/yolo11m.pt \
    --epochs 400 \
    --batch 8 \
    --device cuda \
    --amp True \
    --output ./workspace/train_s200_m_d
Seed 이미지 222장 (수동 라벨 완료)
  train: 178장
  val: 44장
New https://pypi.org/project/ultralytics/8.4.48 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.4.43 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=8, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=workspace/train_s200_m_d/dataset/data.yaml, degrees=5.0, deterministic=True, device=0, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=400, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.5, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=1280, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=models/yolo11m.pt, momentum=0.937, mosaic=0.5, multi_scale=0.0, name=weights, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=15, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=workspace/train_s200_m_d, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m_d/weights, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.3, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.05, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=4

                   from  n    params  module                                       arguments                     
  0                  -1  1      1856  ultralytics.nn.modules.conv.Conv             [3, 64, 3, 2]                 
  1                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  2                  -1  1    111872  ultralytics.nn.modules.block.C3k2            [128, 256, 1, True, 0.25]     
  3                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
  4                  -1  1    444928  ultralytics.nn.modules.block.C3k2            [256, 512, 1, True, 0.25]     
  5                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  6                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  7                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  8                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
 10                  -1  1    990976  ultralytics.nn.modules.block.C2PSA           [512, 512, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1    542720  ultralytics.nn.modules.block.C3k2            [1024, 256, 1, True]          
 17                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1   1511424  ultralytics.nn.modules.block.C3k2            [768, 512, 1, True]           
 20                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 23        [16, 19, 22]  1   1414108  ultralytics.nn.modules.head.Detect           [4, 16, None, [256, 512, 512]]
YOLO11m summary: 232 layers, 20,056,092 parameters, 20,056,076 gradients, 68.2 GFLOPs

Transferred 643/649 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1613.3±61.4 MB/s, size: 11641.4 KB)
train: Scanning /home/sjkim/solar-thermal/workspace/train_s200_m_d/dataset/labels/train... 178 images, 19 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 178/178 1.2Kit/s 0.1s
train: New cache created: /home/sjkim/solar-thermal/workspace/train_s200_m_d/dataset/labels/train.cache
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2945.8±1570.2 MB/s, size: 11244.6 KB)
val: Scanning /home/sjkim/solar-thermal/workspace/train_s200_m_d/dataset/labels/val... 44 images, 3 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 44/44 1.0Kit/s 0.0s
val: /home/sjkim/solar-thermal/workspace/train_s200_m_d/dataset/images/val/DJI_20251217130705_0114_Z.JPG: 1 duplicate labels removed
val: New cache created: /home/sjkim/solar-thermal/workspace/train_s200_m_d/dataset/labels/val.cache
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.00125, momentum=0.9) with parameter groups 106 weight(decay=0.0), 113 weight(decay=0.0005), 112 bias(decay=0.0)
Plotting labels to /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m_d/weights/labels.jpg... 
Image sizes 1280 train, 1280 val
Using 8 dataloader workers
Logging results to /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m_d/weights
Starting training for 400 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/400      15.7G      1.529      2.448      1.599         39       1280: 100% ━━━━━━━━━━━━ 23/23 2.0s/it 47.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.0s/it 3.1s
                   all         44        546      0.609       0.42      0.571      0.437

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/400      15.9G     0.9711      1.172      1.113         30       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.3it/s 1.3s
                   all         44        546      0.471      0.201      0.222      0.115

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      3/400      15.9G      1.012      1.013      1.142         38       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.4it/s 1.3s
                   all         44        546      0.529      0.369      0.377      0.266

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      4/400        16G      1.055      1.068       1.16         42       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.4it/s 1.2s
                   all         44        546      0.293      0.422      0.287      0.206

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      5/400        16G      1.127      1.032      1.173         49       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.638      0.538      0.566      0.369

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      6/400      15.8G       1.01     0.9504      1.132         40       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.4it/s 1.2s
                   all         44        546      0.807       0.39      0.402      0.244

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      7/400      15.9G     0.9946     0.9285      1.125         19       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.3it/s 1.3s
                   all         44        546      0.855      0.518      0.569      0.369

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      8/400        16G     0.9253     0.8856      1.088         35       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.4it/s 1.2s
                   all         44        546      0.563      0.542      0.442      0.265

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      9/400        16G     0.9549      0.813      1.086         26       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.843      0.524      0.575      0.401

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     10/400        16G     0.8978     0.7843      1.078         46       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.4it/s 1.2s
                   all         44        546      0.899      0.638      0.652      0.508

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     11/400        16G     0.8583      0.748      1.055         24       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.814       0.51      0.606      0.447

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     12/400      15.9G     0.8426     0.7607       1.03          7       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.848      0.298      0.367       0.28

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     13/400      15.9G     0.8973     0.7342      1.056         29       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.827      0.502      0.576      0.391

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     14/400      15.9G     0.8653     0.7106      1.042         22       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.4it/s 1.2s
                   all         44        546      0.873      0.585      0.625      0.435

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     15/400      15.9G     0.8452     0.7202      1.039         43       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546       0.52      0.228      0.152      0.107

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     16/400        16G     0.7976     0.6953      1.021         29       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.4it/s 1.2s
                   all         44        546      0.763      0.365      0.427      0.282

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     17/400        16G     0.7892     0.6781      1.001         49       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.827       0.61      0.646      0.475

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     18/400      15.9G     0.7875     0.6517      1.006         18       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.865      0.612      0.672       0.52

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     19/400        16G     0.7586     0.6721      1.011         20       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.834      0.458      0.564      0.422

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     20/400        16G     0.7693     0.6521      1.008          9       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.925      0.578      0.685      0.562

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     21/400      15.9G     0.7686     0.6553      1.002         81       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546       0.82      0.395      0.433      0.318

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     22/400      15.9G     0.7445     0.6477     0.9953         53       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.925      0.626       0.67      0.483

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     23/400        16G     0.8146     0.6421      1.021         46       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.571      0.686      0.661      0.498

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     24/400      15.9G     0.7404     0.6159     0.9873         28       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.648      0.599      0.658      0.494

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     25/400        16G     0.7381     0.6277     0.9801         33       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.692      0.567      0.581      0.425

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     26/400      15.9G     0.7478     0.6243     0.9881         35       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.911      0.626      0.698      0.547

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     27/400      15.9G      0.709     0.6117     0.9912         25       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.737      0.689      0.718      0.568

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     28/400      15.9G     0.7167     0.5909     0.9882         29       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.938      0.676      0.734      0.562

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     29/400        16G     0.7142     0.5784     0.9822         32       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.936      0.645      0.707      0.525

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     30/400      15.9G     0.7222      0.583     0.9833         18       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.706      0.701      0.713      0.551

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     31/400        16G     0.8427     0.6568      1.024         80       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.813      0.683      0.754      0.567

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     32/400        16G     0.8292     0.6254      1.016         41       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.725      0.763      0.771      0.612

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     33/400      15.9G     0.6967     0.5789     0.9747         42       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.652      0.687      0.712      0.531

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     34/400      15.9G     0.7025     0.5623     0.9785         19       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.769       0.61      0.692      0.533

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     35/400        16G     0.7281     0.5861      0.974         44       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.721      0.625      0.642      0.501

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     36/400        16G     0.7071     0.5813     0.9813         47       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.661      0.631      0.689      0.494

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     37/400        16G     0.6886     0.5636      0.966         26       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.912       0.63      0.691      0.571

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     38/400      15.9G     0.7303     0.5687     0.9878         47       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.906      0.615      0.673       0.49

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     39/400        16G     0.6825     0.5467     0.9765         40       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.776      0.716      0.725      0.592

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     40/400        16G     0.6559     0.5508     0.9605         28       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.664      0.774       0.73      0.595

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     41/400        16G     0.6534     0.5116     0.9534         20       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.727      0.718       0.74      0.548

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     42/400        16G     0.6932     0.6257     0.9627          2       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.731      0.686       0.72      0.589

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     43/400      15.9G     0.6729     0.5708      0.955         46       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.786      0.664      0.756      0.609

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     44/400      16.1G     0.6642      0.538     0.9624         21       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.778      0.647      0.747      0.544

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     45/400      15.9G     0.6958     0.5391     0.9701         41       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.717      0.739      0.738      0.616

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     46/400      15.9G     0.6614     0.5302     0.9556         35       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.766      0.702      0.734      0.572

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     47/400      15.9G     0.6847     0.5405     0.9603         30       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.738      0.682       0.72      0.576

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     48/400        16G     0.6798     0.5328     0.9633         39       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546       0.83      0.678      0.769      0.601

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     49/400        16G     0.6786     0.5142      0.956         49       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.664      0.779       0.78      0.639

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     50/400        16G     0.6746     0.5053     0.9573         44       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.841      0.673      0.788      0.637

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     51/400      15.9G     0.6779     0.5411     0.9734         28       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.817      0.715      0.781       0.57

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     52/400        16G     0.6668     0.5166     0.9548         27       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.4it/s 1.2s
                   all         44        546      0.705      0.695      0.752      0.616

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     53/400        16G     0.6632     0.5069     0.9643         33       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.911      0.671      0.772      0.639

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     54/400      15.9G     0.6471     0.4912      0.945         48       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.681      0.788      0.777      0.652

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     55/400        16G      0.621     0.4961     0.9402         33       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.733      0.762      0.798       0.64

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     56/400        16G     0.6635     0.5256     0.9569         34       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.751      0.721      0.762      0.612

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     57/400        16G     0.7082     0.5142     0.9716         30       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.775      0.741      0.781      0.604

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     58/400        16G     0.6703     0.5035     0.9571         51       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.843      0.719       0.77      0.605

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     59/400      15.9G     0.6372     0.4829     0.9495         40       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.724      0.784      0.792      0.623

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     60/400        16G     0.6364     0.5197     0.9427         46       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.828      0.759      0.804      0.597

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     61/400        16G     0.6579     0.4917     0.9571         31       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.794      0.751      0.812      0.677

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     62/400      15.9G     0.6452     0.5133      0.954         44       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.771      0.772      0.796      0.624

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     63/400        16G      0.595     0.4802      0.934          8       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.764       0.74      0.759      0.616

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     64/400      15.9G     0.6188     0.4719     0.9487         26       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.726      0.771      0.739      0.616

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     65/400      15.9G     0.6039     0.4732      0.942         27       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.764       0.78      0.766      0.625

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     66/400      15.9G     0.6661      0.519     0.9541         12       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.674      0.749      0.742      0.606

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     67/400      15.9G     0.6133     0.4712     0.9339         25       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.859      0.689      0.762      0.624

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     68/400      15.9G     0.6543     0.4826     0.9466         35       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.804      0.697      0.762      0.617

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     69/400        16G     0.6299     0.4951     0.9412         55       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.805      0.825      0.834      0.667

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     70/400      15.9G     0.6646     0.4711      0.954         42       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.779      0.815      0.835       0.65

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     71/400        16G     0.6509     0.4625     0.9496         61       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.813      0.791      0.859      0.663

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     72/400      15.9G     0.6015     0.4555     0.9201         28       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.752      0.844      0.842      0.684

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     73/400        16G     0.5966     0.4596     0.9239         60       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.797      0.787      0.819      0.672

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     74/400        16G     0.5928     0.4453     0.9279         38       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546       0.82      0.773      0.818      0.662

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     75/400      15.9G     0.6007     0.4719     0.9229         43       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.842      0.748      0.826      0.652

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     76/400        16G     0.6234     0.4714     0.9384         30       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.773      0.849      0.843       0.68

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     77/400        16G      0.629     0.4606     0.9348         40       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.869      0.794      0.851       0.66

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     78/400      15.9G     0.6207     0.4727     0.9382         32       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.868      0.778      0.843      0.676

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     79/400      15.9G     0.5942     0.4535     0.9316         19       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.835      0.753      0.847      0.676

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     80/400      15.9G     0.6297     0.4567     0.9414         43       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546       0.77      0.846      0.862      0.701

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     81/400        16G     0.6514     0.4608      0.948         19       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.747      0.894      0.844       0.67

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     82/400      15.9G      0.629     0.4593     0.9393         34       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.762      0.835       0.84      0.682

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     83/400        16G     0.6022     0.4515     0.9305         40       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.793      0.906      0.855      0.702

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     84/400      15.9G      0.638      0.471     0.9365         35       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.801      0.841      0.856      0.693

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     85/400        16G      0.589     0.4296     0.9281         37       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.808      0.817      0.834      0.669

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     86/400      15.9G     0.6057     0.4408     0.9329         29       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.829      0.765      0.815      0.669

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     87/400        16G     0.6076     0.4663     0.9345         18       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.783      0.845      0.832      0.679

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     88/400        16G     0.6018     0.4757     0.9302         33       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546       0.68      0.835      0.827      0.615

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     89/400        16G     0.5987     0.4238     0.9252         36       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.687      0.848      0.811      0.655

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     90/400        16G     0.5897     0.5339     0.9272          6       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.752      0.862      0.829      0.647

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     91/400      15.9G     0.6102      0.443     0.9315         23       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.775      0.879      0.855      0.675

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     92/400      15.9G     0.6007     0.4348     0.9256         27       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.752      0.848      0.829      0.652

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     93/400      15.9G      0.612     0.4688     0.9319         23       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546       0.71      0.829      0.825      0.688

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     94/400        16G     0.5869     0.4577     0.9313         23       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.712       0.73      0.769      0.639

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     95/400      15.9G     0.5937     0.4433     0.9276         70       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.812       0.75      0.817      0.656

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     96/400      15.9G     0.5956     0.4438     0.9222         34       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.775       0.86      0.852      0.699

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     97/400        16G      0.613     0.4912     0.9308         19       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546      0.673      0.824      0.825      0.681

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     98/400        16G     0.6125     0.4856     0.9429         51       1280: 100% ━━━━━━━━━━━━ 23/23 1.4it/s 16.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         44        546       0.81      0.778      0.838      0.674
EarlyStopping: Training stopped early as no improvement observed in last 15 epochs. Best results observed at epoch 83, best model saved as best.pt.
To update EarlyStopping(patience=15) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

98 epochs completed in 0.545 hours.
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m_d/weights/weights/last.pt, 40.7MB
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m_d/weights/weights/best.pt, 40.7MB

Validating /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m_d/weights/weights/best.pt...
Ultralytics 8.4.43 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
YOLO11m summary (fused): 126 layers, 20,033,116 parameters, 0 gradients, 67.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.4it/s 1.2s
                   all         44        546      0.793      0.905      0.855      0.703
             pv_string         35        292      0.945      0.983      0.993      0.868
             pv_module         36        215      0.922      0.949      0.984      0.805
                 other          6          7      0.635      0.994      0.802       0.66
                defect         15         32      0.669      0.696      0.642       0.48
Speed: 0.5ms preprocess, 21.6ms inference, 0.0ms loss, 2.2ms postprocess per image
Results saved to /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_m_d/weights
Elapsed: 0:33:02
```

샘플 200개, pv string 및 pv panel, other, anomaly 지정, nagative 추가, m 모델 사용 (배치 4)<br>
GCE g2-standard-8 (vCPU 8개, 메모리 32GB), NVIDIA L4 1개<br>
WARNING ⚠️ CUDA out of memory with batch=8. Reducing to batch=4 and retrying (1/3).

```bash
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s200_x_d \
    --model models/yolo11x.pt \
    --epochs 400 \
    --batch 8 \
    --device cuda \
    --amp True \
    --output ./workspace/train_s200_x_d
Seed 이미지 222장 (수동 라벨 완료)
  train: 178장
  val: 44장
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11x.pt to 'models/yolo11x.pt': 100% ━━━━━━━━━━━━ 109.3MB 142.7MB/s 0.8s
New https://pypi.org/project/ultralytics/8.4.48 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.4.43 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=8, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=workspace/train_s200_x_d/dataset/data.yaml, degrees=5.0, deterministic=True, device=0, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=400, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.5, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=1280, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=models/yolo11x.pt, momentum=0.937, mosaic=0.5, multi_scale=0.0, name=weights, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=15, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=workspace/train_s200_x_d, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/home/sjkim/solar-thermal/runs/detect/workspace/train_s200_x_d/weights, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.3, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.05, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=4

                   from  n    params  module                                       arguments                     
  0                  -1  1      2784  ultralytics.nn.modules.conv.Conv             [3, 96, 3, 2]                 
  1                  -1  1    166272  ultralytics.nn.modules.conv.Conv             [96, 192, 3, 2]               
  2                  -1  2    389760  ultralytics.nn.modules.block.C3k2            [192, 384, 2, True, 0.25]     
  3                  -1  1   1327872  ultralytics.nn.modules.conv.Conv             [384, 384, 3, 2]              
  4                  -1  2   1553664  ultralytics.nn.modules.block.C3k2            [384, 768, 2, True, 0.25]     
  5                  -1  1   5309952  ultralytics.nn.modules.conv.Conv             [768, 768, 3, 2]              
  6                  -1  2   5022720  ultralytics.nn.modules.block.C3k2            [768, 768, 2, True]           
  7                  -1  1   5309952  ultralytics.nn.modules.conv.Conv             [768, 768, 3, 2]              
  8                  -1  2   5022720  ultralytics.nn.modules.block.C3k2            [768, 768, 2, True]           
  9                  -1  1   1476864  ultralytics.nn.modules.block.SPPF            [768, 768, 5]                 
 10                  -1  2   3264768  ultralytics.nn.modules.block.C2PSA           [768, 768, 2]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  2   5612544  ultralytics.nn.modules.block.C3k2            [1536, 768, 2, True]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  2   1700352  ultralytics.nn.modules.block.C3k2            [1536, 384, 2, True]          
 17                  -1  1   1327872  ultralytics.nn.modules.conv.Conv             [384, 384, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  2   5317632  ultralytics.nn.modules.block.C3k2            [1152, 768, 2, True]          
 20                  -1  1   5309952  ultralytics.nn.modules.conv.Conv             [768, 768, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  2   5612544  ultralytics.nn.modules.block.C3k2            [1536, 768, 2, True]          
 23        [16, 19, 22]  1   3150172  ultralytics.nn.modules.head.Detect           [4, 16, None, [384, 768, 768]]
YOLO11x summary: 358 layers, 56,878,396 parameters, 56,878,380 gradients, 195.5 GFLOPs

Transferred 1009/1015 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2792.4±1592.1 MB/s, size: 11641.4 KB)
train: Scanning /home/sjkim/solar-thermal/workspace/train_s200_x_d/dataset/labels/train... 178 images, 19 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 178/178 1.2Kit/s 0.1s
train: New cache created: /home/sjkim/solar-thermal/workspace/train_s200_x_d/dataset/labels/train.cache
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 4026.8±247.9 MB/s, size: 11244.6 KB)
val: Scanning /home/sjkim/solar-thermal/workspace/train_s200_x_d/dataset/labels/val... 44 images, 3 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 44/44 1.0Kit/s 0.0s
val: /home/sjkim/solar-thermal/workspace/train_s200_x_d/dataset/images/val/DJI_20251217130705_0114_Z.JPG: 1 duplicate labels removed
val: New cache created: /home/sjkim/solar-thermal/workspace/train_s200_x_d/dataset/labels/val.cache
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.00125, momentum=0.9) with parameter groups 167 weight(decay=0.0), 174 weight(decay=0.0005), 173 bias(decay=0.0)
Plotting labels to /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_x_d/weights/labels.jpg... 
Image sizes 1280 train, 1280 val
Using 8 dataloader workers
Logging results to /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_x_d/weights
Starting training for 400 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
WARNING ⚠️ CUDA out of memory with batch=8. Reducing to batch=4 and retrying (1/3).
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1539.6±1525.5 MB/s, size: 11856.5 KB)
train: Scanning /home/sjkim/solar-thermal/workspace/train_s200_x_d/dataset/labels/train.cache... 178 images, 19 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 178/178 43.9Mit/s 0.0s
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2247.3±1712.2 MB/s, size: 12757.9 KB)
val: Scanning /home/sjkim/solar-thermal/workspace/train_s200_x_d/dataset/labels/val.cache... 44 images, 3 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 44/44 10.9Mit/s 0.0s
val: /home/sjkim/solar-thermal/workspace/train_s200_x_d/dataset/images/val/DJI_20251217130705_0114_Z.JPG: 1 duplicate labels removed
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.00125, momentum=0.9) with parameter groups 167 weight(decay=0.0), 174 weight(decay=0.0005), 173 bias(decay=0.0)
: 0% ──────────── 0/23  9.9s

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/400      9.21G      1.393      2.189      1.497         43       1280: 100% ━━━━━━━━━━━━ 45/45 1.5s/it 1:06
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.3s/it 4.0s
                   all         44        546      0.401      0.183     0.0998     0.0523

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/400      15.9G      1.395      1.709      1.457         31       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.0s/it 3.1s
                   all         44        546    0.00567     0.0114   5.27e-05   2.79e-05

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      3/400      15.9G      1.595      1.719      1.549         66       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.2it/s 2.5s
                   all         44        546       0.34      0.205     0.0437     0.0181

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      4/400      15.9G      1.786      2.408      1.809         42       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.2it/s 2.5s
                   all         44        546      0.161      0.328     0.0971      0.026

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      5/400        16G      1.483      1.738      1.541         62       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.3it/s 2.3s
                   all         44        546   0.000346    0.00771   1.21e-05   2.16e-06

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      6/400      15.9G      1.365      1.544      1.488         48       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.3it/s 2.3s
                   all         44        546          0          0          0          0

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      7/400        16G      1.254      1.338      1.415         39       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.3it/s 2.3s
                   all         44        546      0.134     0.0408     0.0255      0.013

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      8/400      15.9G      1.139      1.185      1.254         35       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.2it/s 2.4s
                   all         44        546      0.313      0.306      0.287      0.191

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      9/400        16G      1.209      1.151      1.291         32       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.2it/s 2.4s
                   all         44        546      0.692      0.377      0.458       0.33

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     10/400      15.7G      1.104      1.076      1.211         43       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.3it/s 2.4s
                   all         44        546      0.749      0.518       0.55      0.353

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     11/400      15.9G      1.078      1.049      1.217         24       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.2it/s 2.4s
                   all         44        546      0.722      0.555      0.501      0.346

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     12/400      15.9G      0.951     0.9311      1.142         15       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.3it/s 2.4s
                   all         44        546      0.759      0.564      0.548       0.36

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     13/400      15.9G     0.9701     0.8713      1.132         67       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.3it/s 2.4s
                   all         44        546      0.898      0.617      0.621      0.459

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     14/400      15.9G     0.9854      0.881      1.142         43       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.3it/s 2.4s
                   all         44        546      0.927      0.614      0.642      0.386

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     15/400      15.9G     0.9729     0.8589      1.127         46       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.3it/s 2.4s
                   all         44        546      0.909      0.618      0.668      0.522

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     16/400      15.9G     0.9906     0.8751       1.15         29       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.3it/s 2.4s
                   all         44        546       0.86      0.608      0.637      0.441

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     17/400      15.9G     0.9304     0.8508      1.114         20       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.3it/s 2.4s
                   all         44        546      0.883      0.618      0.647      0.502

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     18/400        16G     0.9256     0.8253      1.109         38       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.3it/s 2.4s
                   all         44        546      0.798      0.633      0.577      0.441

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     19/400      15.9G     0.8983     0.8209      1.112         31       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.3it/s 2.4s
                   all         44        546      0.896      0.618      0.636      0.422

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     20/400      15.9G     0.8567     0.7777      1.079          9       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.3it/s 2.4s
                   all         44        546      0.881      0.608      0.661      0.494

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     21/400        16G     0.8947     0.8039      1.097         43       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.3it/s 2.4s
                   all         44        546      0.917      0.651      0.677      0.475

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     22/400        16G     0.8857     0.8254      1.108         34       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.3it/s 2.4s
                   all         44        546      0.907      0.615      0.653      0.385

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     23/400      15.9G     0.8579     0.7814      1.095         51       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.3it/s 2.4s
                   all         44        546      0.843      0.596       0.62      0.466

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     24/400      15.9G     0.8515     0.7533      1.055         37       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.3it/s 2.4s
                   all         44        546      0.869      0.617       0.67      0.427

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     25/400      15.9G     0.8317     0.7363      1.062         43       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.3it/s 2.4s
                   all         44        546      0.666      0.444      0.315      0.215

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     26/400      15.7G      0.842     0.7156       1.07         40       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.3it/s 2.4s
                   all         44        546      0.905      0.596      0.654      0.463

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     27/400      15.9G     0.8382     0.7337      1.081         19       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.3it/s 2.4s
                   all         44        546      0.911       0.62      0.661      0.504

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     28/400      15.9G     0.8062     0.7226      1.058         21       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.3it/s 2.4s
                   all         44        546      0.928      0.629      0.656      0.459

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     29/400        16G     0.8298     0.6924      1.057         32       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.3it/s 2.4s
                   all         44        546      0.935      0.616      0.658      0.515

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     30/400        16G      0.814      0.726      1.055         21       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.2it/s 2.4s
                   all         44        546      0.932      0.586      0.636      0.451
EarlyStopping: Training stopped early as no improvement observed in last 15 epochs. Best results observed at epoch 15, best model saved as best.pt.
To update EarlyStopping(patience=15) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

30 epochs completed in 0.340 hours.
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_x_d/weights/weights/last.pt, 114.5MB
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_x_d/weights/weights/best.pt, 114.5MB

Validating /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_x_d/weights/weights/best.pt...
Ultralytics 8.4.43 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
YOLO11x summary (fused): 191 layers, 56,831,644 parameters, 0 gradients, 194.4 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.3it/s 2.4s
                   all         44        546      0.909      0.618      0.668      0.522
             pv_string         35        292      0.925      0.997      0.994      0.766
             pv_module         36        215      0.927      0.762      0.934      0.711
                 other          6          7      0.783      0.714      0.738      0.606
                defect         15         32          1          0    0.00566    0.00323
Speed: 0.5ms preprocess, 47.5ms inference, 0.0ms loss, 2.5ms postprocess per image
Results saved to /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_x_d/weights
Elapsed: 0:20:50
```

샘플 200개, pv string 및 pv panel, other, anomaly 지정, nagative 추가, m 모델 사용 (배치 4)<br>
GCE g2-standard-8 (vCPU 8개, 메모리 32GB), NVIDIA L4 1개<br>

```bash
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s200_x_d \
    --model models/yolo11x.pt \
    --epochs 400 \
    --batch 4 \
    --device cuda \
    --amp True \
    --output ./workspace/train_s200_x_d
Seed 이미지 222장 (수동 라벨 완료)
  train: 178장
  val: 44장
New https://pypi.org/project/ultralytics/8.4.48 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.4.43 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=4, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=workspace/train_s200_x_d/dataset/data.yaml, degrees=5.0, deterministic=True, device=0, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=400, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.5, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=1280, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=models/yolo11x.pt, momentum=0.937, mosaic=0.5, multi_scale=0.0, name=weights, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=15, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=workspace/train_s200_x_d, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/home/sjkim/solar-thermal/runs/detect/workspace/train_s200_x_d/weights, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.3, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.05, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=4

                   from  n    params  module                                       arguments                     
  0                  -1  1      2784  ultralytics.nn.modules.conv.Conv             [3, 96, 3, 2]                 
  1                  -1  1    166272  ultralytics.nn.modules.conv.Conv             [96, 192, 3, 2]               
  2                  -1  2    389760  ultralytics.nn.modules.block.C3k2            [192, 384, 2, True, 0.25]     
  3                  -1  1   1327872  ultralytics.nn.modules.conv.Conv             [384, 384, 3, 2]              
  4                  -1  2   1553664  ultralytics.nn.modules.block.C3k2            [384, 768, 2, True, 0.25]     
  5                  -1  1   5309952  ultralytics.nn.modules.conv.Conv             [768, 768, 3, 2]              
  6                  -1  2   5022720  ultralytics.nn.modules.block.C3k2            [768, 768, 2, True]           
  7                  -1  1   5309952  ultralytics.nn.modules.conv.Conv             [768, 768, 3, 2]              
  8                  -1  2   5022720  ultralytics.nn.modules.block.C3k2            [768, 768, 2, True]           
  9                  -1  1   1476864  ultralytics.nn.modules.block.SPPF            [768, 768, 5]                 
 10                  -1  2   3264768  ultralytics.nn.modules.block.C2PSA           [768, 768, 2]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  2   5612544  ultralytics.nn.modules.block.C3k2            [1536, 768, 2, True]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  2   1700352  ultralytics.nn.modules.block.C3k2            [1536, 384, 2, True]          
 17                  -1  1   1327872  ultralytics.nn.modules.conv.Conv             [384, 384, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  2   5317632  ultralytics.nn.modules.block.C3k2            [1152, 768, 2, True]          
 20                  -1  1   5309952  ultralytics.nn.modules.conv.Conv             [768, 768, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  2   5612544  ultralytics.nn.modules.block.C3k2            [1536, 768, 2, True]          
 23        [16, 19, 22]  1   3150172  ultralytics.nn.modules.head.Detect           [4, 16, None, [384, 768, 768]]
YOLO11x summary: 358 layers, 56,878,396 parameters, 56,878,380 gradients, 195.5 GFLOPs

Transferred 1009/1015 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2872.5±1586.7 MB/s, size: 11641.4 KB)
train: Scanning /home/sjkim/solar-thermal/workspace/train_s200_x_d/dataset/labels/train... 178 images, 19 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 178/178 1.2Kit/s 0.1s
train: New cache created: /home/sjkim/solar-thermal/workspace/train_s200_x_d/dataset/labels/train.cache
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2474.5±984.3 MB/s, size: 11244.6 KB)
val: Scanning /home/sjkim/solar-thermal/workspace/train_s200_x_d/dataset/labels/val... 44 images, 3 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 44/44 956.1it/s 0.0s
val: /home/sjkim/solar-thermal/workspace/train_s200_x_d/dataset/images/val/DJI_20251217130705_0114_Z.JPG: 1 duplicate labels removed
val: New cache created: /home/sjkim/solar-thermal/workspace/train_s200_x_d/dataset/labels/val.cache
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.00125, momentum=0.9) with parameter groups 167 weight(decay=0.0), 174 weight(decay=0.0005), 173 bias(decay=0.0)
Plotting labels to /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_x_d/weights/labels.jpg... 
Image sizes 1280 train, 1280 val
Using 8 dataloader workers
Logging results to /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_x_d/weights
Starting training for 400 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/400      15.3G      1.393      2.189      1.497         43       1280: 100% ━━━━━━━━━━━━ 45/45 1.1s/it 51.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 1.4it/s 4.2s
                   all         44        546      0.407       0.19      0.102     0.0536

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/400        16G      1.395      1.709      1.457         31       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.1it/s 2.9s
                   all         44        546     0.0057     0.0114   5.68e-05   2.85e-05

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      3/400        16G      1.595      1.719      1.549         66       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.5it/s 2.4s
                   all         44        546      0.343      0.265     0.0583     0.0239

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      4/400        16G      1.786      2.408      1.809         42       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.6it/s 2.3s
                   all         44        546      0.157      0.336      0.105     0.0288

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      5/400        16G      1.483      1.738      1.541         62       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.8it/s 2.2s
                   all         44        546   0.000384    0.00856   1.35e-05   2.23e-06

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      6/400        16G      1.365      1.544      1.488         48       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.9it/s 2.1s
                   all         44        546          0          0          0          0

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      7/400        16G      1.254      1.338      1.415         39       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.8it/s 2.1s
                   all         44        546      0.134     0.0407     0.0247     0.0127

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      8/400      15.9G      1.139      1.185      1.254         35       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.3s
                   all         44        546      0.317      0.305      0.286      0.189

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      9/400        16G      1.209      1.151      1.291         32       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.3s
                   all         44        546      0.941      0.382      0.459      0.331

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     10/400      15.8G      1.104      1.076      1.211         43       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        546      0.747      0.518       0.55      0.353

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     11/400        16G      1.078      1.049      1.217         24       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.3s
                   all         44        546      0.722      0.556      0.501      0.347

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     12/400        16G      0.951     0.9311      1.142         15       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        546      0.758      0.564      0.548      0.361

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     13/400        16G     0.9701     0.8713      1.132         67       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        546      0.898      0.617      0.615      0.455

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     14/400        16G     0.9854      0.881      1.142         43       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        546      0.927      0.614      0.642      0.383

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     15/400        16G     0.9729     0.8589      1.127         46       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        546       0.91      0.618      0.668      0.521

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     16/400        16G     0.9906     0.8751       1.15         29       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        546       0.86      0.607      0.637      0.442

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     17/400        16G     0.9304     0.8508      1.114         20       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        546      0.883      0.618      0.645      0.501

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     18/400        16G     0.9256     0.8253      1.109         38       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        546      0.793      0.633      0.572      0.438

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     19/400        16G     0.8983     0.8209      1.112         31       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        546      0.896      0.618      0.636      0.423

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     20/400        16G     0.8567     0.7777      1.079          9       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        546       0.88      0.609      0.661      0.495

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     21/400        16G     0.8947     0.8039      1.097         43       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        546      0.917      0.651      0.677      0.475

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     22/400        16G     0.8857     0.8254      1.108         34       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        546      0.907      0.614      0.653      0.385

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     23/400        16G     0.8579     0.7814      1.095         51       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        546      0.845      0.596      0.619      0.466

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     24/400        16G     0.8515     0.7533      1.055         37       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        546       0.87      0.615       0.67      0.427

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     25/400        16G     0.8317     0.7363      1.062         43       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        546      0.666      0.444      0.315      0.215

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     26/400      15.8G      0.842     0.7156       1.07         40       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        546      0.904      0.597      0.653      0.461

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     27/400        16G     0.8382     0.7337      1.081         19       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        546       0.91      0.621      0.661      0.504

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     28/400        16G     0.8062     0.7226      1.058         21       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        546      0.928      0.629      0.656      0.459

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     29/400        16G     0.8298     0.6924      1.057         32       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        546      0.935      0.616      0.658      0.515

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     30/400        16G      0.814      0.726      1.055         21       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        546      0.932      0.586      0.636      0.451
EarlyStopping: Training stopped early as no improvement observed in last 15 epochs. Best results observed at epoch 15, best model saved as best.pt.
To update EarlyStopping(patience=15) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

30 epochs completed in 0.333 hours.
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_x_d/weights/weights/last.pt, 114.5MB
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_x_d/weights/weights/best.pt, 114.5MB

Validating /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_x_d/weights/weights/best.pt...
Ultralytics 8.4.43 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
YOLO11x summary (fused): 191 layers, 56,831,644 parameters, 0 gradients, 194.4 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.3it/s 2.6s
                   all         44        546       0.91      0.618      0.668      0.521
             pv_string         35        292      0.925      0.997      0.994      0.766
             pv_module         36        215      0.932      0.762      0.935      0.709
                 other          6          7      0.783      0.714      0.737      0.606
                defect         15         32          1          0    0.00563     0.0032
Speed: 0.6ms preprocess, 53.4ms inference, 0.0ms loss, 1.3ms postprocess per image
Results saved to /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_x_d/weights
Elapsed: 0:20:22
```

샘플 200개, pv string 및 pv panel, other, anomaly 지정, nagative 추가, x 모델 사용 (배치 4)<br>
anomaly 데이터 추가 후 재학습 <br>
labels_s200_l_2_d = 완전한 pv panel만 pv string으로 저정<br>
GCE g2-standard-8 (vCPU 8개, 메모리 32GB), NVIDIA L4 1개

```bash
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s200_x_d \
    --model models/yolo11x.pt \
    --epochs 400 \
    --batch 4 \
    --device cuda \
    --amp True \
    --output ./workspace/train_s200_x_d
Seed 이미지 222장 (수동 라벨 완료)
  train: 178장
  val: 44장
New https://pypi.org/project/ultralytics/8.4.48 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.4.43 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=4, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=workspace/train_s200_x_d/dataset/data.yaml, degrees=5.0, deterministic=True, device=0, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=400, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.5, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=1280, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=models/yolo11x.pt, momentum=0.937, mosaic=0.5, multi_scale=0.0, name=weights, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=15, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=workspace/train_s200_x_d, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/home/sjkim/solar-thermal/runs/detect/workspace/train_s200_x_d/weights, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.3, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.05, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=4

                   from  n    params  module                                       arguments                     
  0                  -1  1      2784  ultralytics.nn.modules.conv.Conv             [3, 96, 3, 2]                 
  1                  -1  1    166272  ultralytics.nn.modules.conv.Conv             [96, 192, 3, 2]               
  2                  -1  2    389760  ultralytics.nn.modules.block.C3k2            [192, 384, 2, True, 0.25]     
  3                  -1  1   1327872  ultralytics.nn.modules.conv.Conv             [384, 384, 3, 2]              
  4                  -1  2   1553664  ultralytics.nn.modules.block.C3k2            [384, 768, 2, True, 0.25]     
  5                  -1  1   5309952  ultralytics.nn.modules.conv.Conv             [768, 768, 3, 2]              
  6                  -1  2   5022720  ultralytics.nn.modules.block.C3k2            [768, 768, 2, True]           
  7                  -1  1   5309952  ultralytics.nn.modules.conv.Conv             [768, 768, 3, 2]              
  8                  -1  2   5022720  ultralytics.nn.modules.block.C3k2            [768, 768, 2, True]           
  9                  -1  1   1476864  ultralytics.nn.modules.block.SPPF            [768, 768, 5]                 
 10                  -1  2   3264768  ultralytics.nn.modules.block.C2PSA           [768, 768, 2]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  2   5612544  ultralytics.nn.modules.block.C3k2            [1536, 768, 2, True]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  2   1700352  ultralytics.nn.modules.block.C3k2            [1536, 384, 2, True]          
 17                  -1  1   1327872  ultralytics.nn.modules.conv.Conv             [384, 384, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  2   5317632  ultralytics.nn.modules.block.C3k2            [1152, 768, 2, True]          
 20                  -1  1   5309952  ultralytics.nn.modules.conv.Conv             [768, 768, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  2   5612544  ultralytics.nn.modules.block.C3k2            [1536, 768, 2, True]          
 23        [16, 19, 22]  1   3150172  ultralytics.nn.modules.head.Detect           [4, 16, None, [384, 768, 768]]
YOLO11x summary: 358 layers, 56,878,396 parameters, 56,878,380 gradients, 195.5 GFLOPs

Transferred 1009/1015 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2655.0±1399.1 MB/s, size: 11641.4 KB)
train: Scanning /home/sjkim/solar-thermal/workspace/train_s200_x_d/dataset/labels/train... 178 images, 19 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 178/178 1.2Kit/s 0.2s
train: New cache created: /home/sjkim/solar-thermal/workspace/train_s200_x_d/dataset/labels/train.cache
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 3456.8±679.9 MB/s, size: 11244.6 KB)
val: Scanning /home/sjkim/solar-thermal/workspace/train_s200_x_d/dataset/labels/val... 44 images, 3 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 44/44 999.7it/s 0.0s
val: /home/sjkim/solar-thermal/workspace/train_s200_x_d/dataset/images/val/DJI_20251217130705_0114_Z.JPG: 1 duplicate labels removed
val: New cache created: /home/sjkim/solar-thermal/workspace/train_s200_x_d/dataset/labels/val.cache
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.00125, momentum=0.9) with parameter groups 167 weight(decay=0.0), 174 weight(decay=0.0005), 173 bias(decay=0.0)
Plotting labels to /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_x_d/weights/labels.jpg... 
Image sizes 1280 train, 1280 val
Using 8 dataloader workers
Logging results to /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_x_d/weights
Starting training for 400 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/400      15.3G      1.447      2.253       1.51         44       1280: 100% ━━━━━━━━━━━━ 45/45 1.1s/it 51.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 1.3it/s 4.5s
                   all         44        560      0.738      0.366      0.268      0.173

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/400        16G      1.314      1.803      1.414         31       1280: 100% ━━━━━━━━━━━━ 45/45 1.4it/s 33.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.2it/s 2.8s
                   all         44        560      0.052      0.212     0.0378      0.018

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      3/400        16G      1.722      1.963      1.851         68       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.8it/s 2.1s
                   all         44        560          0          0          0          0

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      4/400        16G      1.911      2.259      2.085         42       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 1.8it/s 3.4s
                   all         44        560          0          0          0          0

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      5/400        16G      1.543      1.876      1.611         62       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.0it/s 3.0s
                   all         44        560    0.00631      0.059   0.000578   0.000183

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      6/400        16G       1.35      1.634      1.472         48       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.3it/s 2.6s
                   all         44        560     0.0327      0.109     0.0114     0.0062

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      7/400        16G      1.327       1.33      1.458         40       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.6it/s 2.3s
                   all         44        560      0.221      0.482      0.131      0.083

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      8/400        16G       1.24       1.24      1.314         35       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        560      0.739      0.516      0.548      0.329

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      9/400        16G      1.214      1.149      1.322         32       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        560      0.875      0.583      0.612      0.414

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     10/400      15.8G      1.107      1.113      1.244         44       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        560      0.726      0.538      0.545      0.344


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     61/400        16G     0.6808     0.5614      1.005         32       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        560      0.922      0.755      0.862      0.683

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     62/400        16G      0.678      0.565     0.9981         35       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        560      0.794      0.807      0.862      0.687

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     63/400        16G     0.6746     0.5675          1          9       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        560      0.893      0.789      0.868      0.708

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     64/400        16G     0.6625     0.5333     0.9918         28       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        560      0.907      0.796      0.862      0.694

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     65/400        16G     0.6744     0.5625      1.005         25       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        560      0.913      0.785      0.856      0.668

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     66/400        16G     0.6719     0.5342     0.9925         25       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        560      0.821      0.732      0.794      0.646

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     67/400        16G     0.7307     0.5646      1.022         25       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        560      0.876      0.776      0.847      0.612

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     68/400        16G     0.7616     0.5975      1.027         36       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        560      0.904      0.776      0.851      0.518

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     69/400        16G     0.6939     0.5659      1.013         26       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        560      0.887      0.716      0.843      0.669

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     70/400      15.9G        0.7     0.5623      1.004         63       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        560      0.887      0.757      0.855      0.686


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    101/400        16G     0.6495     0.5045     0.9885         39       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        560      0.866      0.804      0.833      0.608

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    102/400        16G     0.6018     0.4729     0.9551         60       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        560      0.898      0.777       0.86      0.711

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    103/400        16G     0.6129     0.4951     0.9717         31       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        560      0.851      0.836      0.883      0.721

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    104/400        16G     0.6101     0.4926     0.9734         32       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        560      0.925      0.761      0.886      0.741

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    105/400        16G     0.6095     0.5011     0.9683         40       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        560      0.917       0.79      0.896      0.737

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    106/400      15.8G     0.6087     0.4814     0.9732         59       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.3s
                   all         44        560      0.796      0.873      0.873      0.695

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    107/400        16G     0.6376     0.4995     0.9788         32       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        560       0.81      0.877      0.871      0.686

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    108/400      15.9G     0.6533     0.5228     0.9865         47       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        560      0.887      0.771      0.867      0.651

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    109/400        16G     0.6385     0.4943     0.9811         34       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        560      0.855      0.787      0.876      0.689

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    110/400        16G      0.618     0.4733     0.9621         71       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        560      0.887      0.793       0.89      0.697

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    111/400        16G     0.6015     0.4607     0.9581         51       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        560      0.896      0.834       0.89       0.72

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    112/400        16G      0.617     0.4776     0.9636         37       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        560       0.88      0.802      0.859      0.683

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    113/400        16G     0.5922     0.4537     0.9604         38       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        560      0.907      0.787      0.866      0.702

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    114/400        16G     0.5889     0.4592     0.9677         45       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        560      0.814      0.839      0.844      0.681

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    115/400        16G     0.5924     0.4607     0.9541         31       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        560      0.886      0.747      0.844      0.676

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    116/400        16G       0.58     0.4439     0.9517         40       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.3s
                   all         44        560      0.901      0.758      0.856      0.656

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    117/400        16G     0.5774     0.4503     0.9647         37       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        560       0.91      0.833      0.903       0.71

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    118/400        16G     0.5763     0.4353     0.9493         28       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        560      0.894      0.817      0.903      0.691

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    119/400        16G     0.5938     0.4771     0.9557         16       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        560      0.834      0.869      0.909      0.697
EarlyStopping: Training stopped early as no improvement observed in last 15 epochs. Best results observed at epoch 104, best model saved as best.pt.
To update EarlyStopping(patience=15) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

119 epochs completed in 1.315 hours.
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_x_d/weights/weights/last.pt, 114.6MB
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_x_d/weights/weights/best.pt, 114.6MB

Validating /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_x_d/weights/weights/best.pt...
Ultralytics 8.4.43 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
YOLO11x summary (fused): 191 layers, 56,831,644 parameters, 0 gradients, 194.4 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.1it/s 2.9s
                   all         44        560      0.925      0.761      0.886      0.741
             pv_string         35        292      0.979      0.956      0.992      0.908
             pv_module         36        215      0.959      0.898      0.983      0.794
                 other          6          7      0.785      0.714      0.782       0.68
               anomaly         24         46      0.977      0.478      0.789      0.582
Speed: 0.5ms preprocess, 57.0ms inference, 0.1ms loss, 3.7ms postprocess per image
Results saved to /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_x_d/weights
Elapsed: 1:19:27
```

샘플 200개, pv string 및 pv panel, other, anomaly 지정, nagative 추가, x 모델 사용 (스트링 조건을 명확히, 배치 4)<br>
labels_s200_l_2_d = 완전한 pv panel만 pv string으로 저정<br>
GCE g2-standard-8 (vCPU 8개, 메모리 32GB), NVIDIA L4 1개

```bash
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s200_x_2_d \
    --model models/yolo11x.pt \
    --epochs 400 \
    --batch 4 \
    --device cuda \
    --amp True \
    --output ./workspace/train_s200_x_2_d
Seed 이미지 222장 (수동 라벨 완료)
  train: 178장
  val: 44장
New https://pypi.org/project/ultralytics/8.4.48 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.4.43 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=4, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=workspace/train_s200_x_2_d/dataset/data.yaml, degrees=5.0, deterministic=True, device=0, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=400, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.5, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=1280, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=models/yolo11x.pt, momentum=0.937, mosaic=0.5, multi_scale=0.0, name=weights, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=15, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=workspace/train_s200_x_2_d, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/home/sjkim/solar-thermal/runs/detect/workspace/train_s200_x_2_d/weights, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.3, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.05, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=4

                   from  n    params  module                                       arguments                     
  0                  -1  1      2784  ultralytics.nn.modules.conv.Conv             [3, 96, 3, 2]                 
  1                  -1  1    166272  ultralytics.nn.modules.conv.Conv             [96, 192, 3, 2]               
  2                  -1  2    389760  ultralytics.nn.modules.block.C3k2            [192, 384, 2, True, 0.25]     
  3                  -1  1   1327872  ultralytics.nn.modules.conv.Conv             [384, 384, 3, 2]              
  4                  -1  2   1553664  ultralytics.nn.modules.block.C3k2            [384, 768, 2, True, 0.25]     
  5                  -1  1   5309952  ultralytics.nn.modules.conv.Conv             [768, 768, 3, 2]              
  6                  -1  2   5022720  ultralytics.nn.modules.block.C3k2            [768, 768, 2, True]           
  7                  -1  1   5309952  ultralytics.nn.modules.conv.Conv             [768, 768, 3, 2]              
  8                  -1  2   5022720  ultralytics.nn.modules.block.C3k2            [768, 768, 2, True]           
  9                  -1  1   1476864  ultralytics.nn.modules.block.SPPF            [768, 768, 5]                 
 10                  -1  2   3264768  ultralytics.nn.modules.block.C2PSA           [768, 768, 2]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  2   5612544  ultralytics.nn.modules.block.C3k2            [1536, 768, 2, True]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  2   1700352  ultralytics.nn.modules.block.C3k2            [1536, 384, 2, True]          
 17                  -1  1   1327872  ultralytics.nn.modules.conv.Conv             [384, 384, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  2   5317632  ultralytics.nn.modules.block.C3k2            [1152, 768, 2, True]          
 20                  -1  1   5309952  ultralytics.nn.modules.conv.Conv             [768, 768, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  2   5612544  ultralytics.nn.modules.block.C3k2            [1536, 768, 2, True]          
 23        [16, 19, 22]  1   3150172  ultralytics.nn.modules.head.Detect           [4, 16, None, [384, 768, 768]]
YOLO11x summary: 358 layers, 56,878,396 parameters, 56,878,380 gradients, 195.5 GFLOPs

Transferred 1009/1015 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2811.9±1611.0 MB/s, size: 11641.4 KB)
train: Scanning /home/sjkim/solar-thermal/workspace/train_s200_x_2_d/dataset/labels/train... 178 images, 19 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 178/178 1.2Kit/s 0.1s
train: New cache created: /home/sjkim/solar-thermal/workspace/train_s200_x_2_d/dataset/labels/train.cache
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 3172.8±615.0 MB/s, size: 11244.6 KB)
val: Scanning /home/sjkim/solar-thermal/workspace/train_s200_x_2_d/dataset/labels/val... 44 images, 3 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 44/44 981.1it/s 0.0s
val: New cache created: /home/sjkim/solar-thermal/workspace/train_s200_x_2_d/dataset/labels/val.cache
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.00125, momentum=0.9) with parameter groups 167 weight(decay=0.0), 174 weight(decay=0.0005), 173 bias(decay=0.0)
Plotting labels to /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_x_2_d/weights/labels.jpg... 
^[[A^[[A^[[A^[[AImage sizes 1280 train, 1280 val
Using 8 dataloader workers
Logging results to /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_x_2_d/weights
Starting training for 400 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/400      15.3G      1.444      2.145       1.56         44       1280: 100% ━━━━━━━━━━━━ 45/45 1.1s/it 51.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 1.3it/s 4.8s
                   all         44        559      0.269      0.302     0.0131    0.00752

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/400        16G      1.493      2.186      1.554         31       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 34.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.6it/s 2.4s
                   all         44        559    0.00422      0.197     0.0023   0.000719

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      3/400        16G      1.465      2.709      1.577         68       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.5it/s 2.4s
                   all         44        559      0.348      0.537      0.138     0.0745

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      4/400        16G      1.478      3.152      1.546         42       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.6it/s 2.3s
                   all         44        559    0.00186      0.025   0.000206   4.46e-05

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      5/400        16G      1.878      2.317        1.9         62       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.8it/s 2.2s
                   all         44        559          0          0          0          0

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      6/400        16G      1.521      2.025      1.702         48       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.8it/s 2.1s
                   all         44        559          0          0          0          0

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      7/400        16G      1.461      1.644      1.566         40       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        559     0.0759     0.0513      0.017     0.0079

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      8/400        16G      1.263      1.368      1.379         35       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.5it/s 2.4s
                   all         44        559      0.196      0.485      0.213      0.148

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      9/400        16G      1.267      1.283      1.361         32       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.6it/s 2.3s
                   all         44        559      0.422      0.481      0.492       0.35

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     10/400      15.8G      1.149      1.196      1.276         44       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        559      0.173      0.245      0.141     0.0717



      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     31/400        16G     0.8853     0.7419        1.1         45       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        559      0.935      0.629       0.68      0.452

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     32/400        16G     0.8205     0.7359      1.067         55       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        559      0.802      0.429       0.46      0.331

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     33/400        16G     0.8375     0.7163      1.075         46       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        559      0.815      0.671      0.638      0.454

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     34/400        16G     0.8586     0.7468      1.072         19       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        559       0.71      0.335      0.294      0.215

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     35/400        16G     0.8418      0.736      1.067         45       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.6it/s 2.3s
                   all         44        559      0.901      0.547      0.651      0.502

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     36/400        16G     0.8263     0.7204      1.091         42       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        559      0.929      0.436      0.505      0.344

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     37/400        16G     0.7855     0.6848       1.05         26       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.3s
                   all         44        559      0.739      0.362      0.304      0.227

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     38/400        16G     0.8247     0.7287      1.077         17       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        559      0.929       0.62      0.674      0.434

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     39/400        16G      0.817     0.7088      1.062         43       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        559      0.537      0.436       0.48      0.329

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     40/400        16G     0.8273     0.7038      1.068         37       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        559      0.927      0.641      0.704      0.518


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     61/400        16G     0.7449     0.5694      1.026         32       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        559      0.866      0.499      0.546      0.397

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     62/400        16G     0.7453     0.5863      1.024         35       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        559      0.874      0.444      0.538      0.401

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     63/400        16G     0.7357     0.5699      1.025          9       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        559      0.817      0.702      0.737      0.548

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     64/400        16G     0.7188     0.5419      1.013         28       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        559      0.868       0.74      0.791      0.604

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     65/400        16G      0.743     0.5876      1.028         25       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        559       0.86      0.792      0.798        0.6

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     66/400        16G     0.7345      0.565      1.018         25       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.3s
                   all         44        559      0.858      0.719      0.754      0.529

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     67/400        16G     0.7431     0.5616       1.03         25       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.3s
                   all         44        559      0.752      0.514      0.527      0.412

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     68/400        16G     0.7751     0.5793      1.027         36       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        559      0.722      0.524      0.478       0.36

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     69/400        16G     0.7273     0.5681      1.019         26       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        559      0.908      0.571      0.663      0.483

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     70/400      15.9G     0.7264     0.5616      1.013         63       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        559      0.918      0.714      0.799      0.635


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     91/400        16G     0.7463     0.5284      1.008         37       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        559      0.954      0.811      0.896      0.597

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     92/400        16G     0.7084      0.512          1         27       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        559      0.923      0.709      0.807      0.656

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     93/400        16G     0.6797     0.4969     0.9915         19       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        559      0.854      0.745      0.824       0.65

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     94/400        16G     0.6477     0.4799     0.9795         26       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        559      0.903      0.769      0.857       0.68

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     95/400        16G     0.7117      0.498     0.9983         46       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        559      0.901      0.753      0.832      0.646

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     96/400        16G     0.6707     0.4962     0.9835         34       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        559      0.924      0.799      0.879      0.692

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     97/400        16G     0.6595     0.4872     0.9882         39       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        559      0.897      0.798      0.861      0.681

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     98/400        16G     0.6714      0.511     0.9972         57       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        559      0.903      0.798      0.866      0.698

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     99/400        16G     0.6727      0.505     0.9884         11       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        559      0.909      0.804      0.877      0.703

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    100/400        16G     0.7109     0.5225     0.9994         45       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        559      0.873      0.775       0.83      0.628

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    101/400        16G     0.7075     0.5239     0.9998         39       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.3s
                   all         44        559      0.898      0.681       0.78       0.61

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    102/400        16G     0.6544     0.4866     0.9713         60       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        559      0.919        0.8      0.839      0.669

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    103/400        16G     0.6771     0.5005     0.9857         31       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        559      0.864      0.789      0.842      0.695

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    104/400        16G     0.6742     0.4747     0.9991         32       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.6it/s 2.3s
                   all         44        559      0.921      0.754      0.846      0.686

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    105/400        16G     0.6671     0.4812     0.9845         40       1280: 100% ━━━━━━━━━━━━ 45/45 1.3it/s 33.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.7it/s 2.2s
                   all         44        559      0.903       0.81      0.864      0.711
EarlyStopping: Training stopped early as no improvement observed in last 15 epochs. Best results observed at epoch 90, best model saved as best.pt.
To update EarlyStopping(patience=15) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

105 epochs completed in 1.158 hours.
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_x_2_d/weights/weights/last.pt, 114.6MB
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_x_2_d/weights/weights/best.pt, 114.6MB

Validating /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_x_2_d/weights/weights/best.pt...
Ultralytics 8.4.43 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
YOLO11x summary (fused): 191 layers, 56,831,644 parameters, 0 gradients, 194.4 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.1it/s 2.9s
                   all         44        559       0.88      0.789      0.868      0.714
             pv_string         31        167      0.962      0.994      0.989      0.882
             pv_module         41        340      0.991      0.938       0.99      0.827
                 other          6          7      0.587      0.714      0.771      0.609
               anomaly         23         45      0.981      0.511      0.722       0.54
Speed: 0.5ms preprocess, 46.5ms inference, 0.0ms loss, 12.5ms postprocess per image
Results saved to /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_x_2_d/weights
Elapsed: 1:09:52
```

샘플 200개, pv string 및 pv panel, other, anomaly 지정, nagative 추가, l 모델 사용 (스트링 조건을 명확히, 배치 8)<br>
labels_s200_l_2_d = 완전한 pv panel만 pv string으로 저정<br>
GCE g2-standard-8 (vCPU 8개, 메모리 32GB), NVIDIA L4 1개

```bash
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s200_l_2_d \
    --model models/yolo11l.pt \
    --epochs 400 \
    --batch 8 \
    --device cuda \
    --amp True \
    --output ./workspace/train_s200_l_2_d
Seed 이미지 222장 (수동 라벨 완료)
  train: 178장
  val: 44장
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11l.pt to 'models/yolo11l.pt': 100% ━━━━━━━━━━━━ 49.0MB 280.8MB/s 0.2s
New https://pypi.org/project/ultralytics/8.4.48 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.4.43 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=8, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=workspace/train_s200_l_2_d/dataset/data.yaml, degrees=5.0, deterministic=True, device=0, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=400, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.5, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=1280, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=models/yolo11l.pt, momentum=0.937, mosaic=0.5, multi_scale=0.0, name=weights, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=15, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=workspace/train_s200_l_2_d, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/home/sjkim/solar-thermal/runs/detect/workspace/train_s200_l_2_d/weights, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.3, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.05, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=4

                   from  n    params  module                                       arguments
  0                  -1  1      1856  ultralytics.nn.modules.conv.Conv             [3, 64, 3, 2]
  1                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]
  2                  -1  2    173824  ultralytics.nn.modules.block.C3k2            [128, 256, 2, True, 0.25]
  3                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]
  4                  -1  2    691712  ultralytics.nn.modules.block.C3k2            [256, 512, 2, True, 0.25]
  5                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]
  6                  -1  2   2234368  ultralytics.nn.modules.block.C3k2            [512, 512, 2, True]
  7                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]
  8                  -1  2   2234368  ultralytics.nn.modules.block.C3k2            [512, 512, 2, True]
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]
 10                  -1  2   1455616  ultralytics.nn.modules.block.C2PSA           [512, 512, 2]
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 13                  -1  2   2496512  ultralytics.nn.modules.block.C3k2            [1024, 512, 2, True]
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 16                  -1  2    756736  ultralytics.nn.modules.block.C3k2            [1024, 256, 2, True]
 17                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 19                  -1  2   2365440  ultralytics.nn.modules.block.C3k2            [768, 512, 2, True]
 20                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 22                  -1  2   2496512  ultralytics.nn.modules.block.C3k2            [1024, 512, 2, True]
 23        [16, 19, 22]  1   1414108  ultralytics.nn.modules.head.Detect           [4, 16, None, [256, 512, 512]]
YOLO11l summary: 358 layers, 25,313,564 parameters, 25,313,548 gradients, 87.3 GFLOPs

Transferred 1009/1015 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1551.0±38.5 MB/s, size: 11641.4 KB)
train: Scanning /home/sjkim/solar-thermal/workspace/train_s200_l_2_d/dataset/labels/train... 178 images, 19 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 178/178 1.2Kit/s 0.2s
train: New cache created: /home/sjkim/solar-thermal/workspace/train_s200_l_2_d/dataset/labels/train.cache
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2867.2±1508.0 MB/s, size: 11244.6 KB)
val: Scanning /home/sjkim/solar-thermal/workspace/train_s200_l_2_d/dataset/labels/val... 44 images, 3 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 44/44 971.3it/s 0.0s
val: New cache created: /home/sjkim/solar-thermal/workspace/train_s200_l_2_d/dataset/labels/val.cache
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically...
optimizer: AdamW(lr=0.00125, momentum=0.9) with parameter groups 167 weight(decay=0.0), 174 weight(decay=0.0005), 173 bias(decay=0.0)
Plotting labels to /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_l_2_d/weights/labels.jpg...
Image sizes 1280 train, 1280 val
Using 8 dataloader workers
Logging results to /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_l_2_d/weights
Starting training for 400 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/400      20.9G      1.542      2.385      1.624         41       1280: 100% ━━━━━━━━━━━━ 23/23 2.2s/it 51.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.1s/it 3.3s
                   all         44        559      0.691      0.353      0.331       0.23

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/400      21.2G      1.007       1.26      1.136         32       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.9it/s 1.5s
                   all         44        559       0.78      0.574      0.528      0.336

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      3/400      21.2G      1.136       1.14      1.222         39       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.8it/s 1.7s
                   all         44        559     0.0807      0.184     0.0264     0.0142

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      4/400      21.2G      1.109      1.129      1.192         42       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.9it/s 1.6s
                   all         44        559      0.462      0.132      0.108     0.0732

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      5/400      21.2G      1.103       1.05      1.177         49       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.9it/s 1.6s
                   all         44        559       0.53      0.397      0.217      0.136

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      6/400      21.1G      1.063      1.013      1.195         40       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.765      0.316      0.273      0.179

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      7/400      21.1G      1.048      1.037       1.18         19       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.536      0.313      0.281      0.168

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      8/400      21.3G      1.006     0.8912      1.136         35       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.717        0.2      0.136     0.0704

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      9/400      21.3G     0.9864     0.8209      1.119         26       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.9it/s 1.6s
                   all         44        559      0.701      0.333       0.36      0.252

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     10/400      21.2G     0.9183     0.7932      1.087         47       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.809      0.492      0.578        0.4

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     11/400      21.2G     0.9293     0.7669      1.092         24       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.742      0.635      0.587      0.442

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     12/400      21.2G     0.9315     0.7879      1.077          7       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.712      0.614      0.672      0.497

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     13/400      21.1G     0.9215     0.7058      1.081         29       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.9it/s 1.5s
                   all         44        559      0.724      0.522      0.491      0.397

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     14/400      21.2G      0.913     0.7631      1.091         22       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.798      0.449      0.458      0.336

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     15/400      21.2G     0.9125     0.7651      1.088         43       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.769      0.527      0.623      0.504

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     16/400      21.2G     0.8679     0.7133      1.065         30       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.963       0.32       0.39      0.247

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     17/400      21.2G     0.8099     0.6566      1.031         51       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.849      0.644      0.692      0.456

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     18/400      21.1G     0.8898     0.6708      1.071         20       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.712      0.637       0.65      0.478

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     19/400      21.2G     0.8868     0.6615      1.072         20       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.872      0.748      0.762      0.514

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     20/400      21.2G     0.8428       0.62      1.047          9       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.855      0.665      0.716      0.541

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     21/400      21.2G     0.8581     0.6269      1.056         81       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.709      0.656      0.672      0.537

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     22/400      21.1G     0.8448     0.6271      1.054         53       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.784      0.697      0.753      0.516

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     23/400      21.2G     0.8247        0.6      1.041         46       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.914      0.682      0.747      0.504

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     24/400      21.2G     0.7997     0.5717      1.024         30       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.788      0.683      0.797      0.615

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     25/400      21.3G     0.8111     0.5901      1.024         33       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.786      0.805       0.78       0.63

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     26/400      21.2G     0.7826     0.5981      1.015         36       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.906      0.678      0.749      0.591

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     27/400      21.2G     0.7793     0.5819      1.028         25       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.857      0.746      0.774      0.548

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     28/400      21.2G     0.7696     0.5717      1.018         30       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559       0.84      0.721      0.823      0.613

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     29/400      21.2G      0.818     0.5529      1.033         32       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.834       0.61      0.741      0.559

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     30/400      21.1G     0.8068     0.5658      1.022         18       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.798      0.651      0.728      0.549

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     31/400      21.2G     0.7821     0.5657      1.025         81       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.916      0.789      0.796      0.631

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     32/400      21.2G     0.7822     0.5471      1.024         42       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.867      0.698       0.77      0.581

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     33/400      21.2G     0.7515     0.5497      1.007         42       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.857      0.573      0.717      0.581

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     34/400      21.2G     0.7198      0.529      1.002         19       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.755      0.573      0.608      0.466

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     35/400      21.2G      0.737     0.5318     0.9875         45       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.803      0.671      0.728      0.531

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     36/400      21.2G     0.7288      0.509      1.007         48       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.794      0.716      0.739      0.565

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     37/400      21.2G     0.7234     0.5228     0.9931         27       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.717      0.716      0.674      0.534

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     38/400      21.1G     0.7621     0.5159      1.019         47       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.791      0.719      0.735      0.519

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     39/400      21.2G     0.7195     0.4783      1.005         40       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.853       0.67       0.75      0.575

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     40/400      21.2G     0.6994     0.4784      0.994         28       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.936       0.77      0.834      0.622

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     41/400      21.3G     0.7192     0.4655     0.9976         20       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.925      0.692      0.802      0.561

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     42/400      21.2G     0.7021     0.8698     0.9855          2       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.895      0.743      0.813      0.639

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     43/400      21.1G     0.7149      0.499     0.9876         47       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.846       0.77       0.82      0.557

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     44/400      21.3G     0.7286     0.5096      1.002         21       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.801      0.781      0.815      0.546

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     45/400      21.2G     0.7059     0.5053     0.9952         41       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.898      0.704      0.789      0.634

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     46/400      21.2G     0.6993      0.466     0.9864         36       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.917      0.753      0.831      0.605

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     47/400      21.1G     0.7142     0.4744     0.9914         30       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.864      0.748      0.816      0.669

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     48/400      21.2G     0.7151     0.4831      0.994         39       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.913      0.728      0.869      0.692

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     49/400      21.2G      0.711     0.4827     0.9873         51       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.922      0.732      0.828      0.676

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     50/400      21.2G     0.7061     0.4803     0.9807         44       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.877      0.767      0.846      0.705

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     51/400      21.2G     0.7022     0.4862      1.001         29       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.882      0.788      0.844      0.668

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     52/400      21.2G     0.6935      0.472     0.9866         29       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.867      0.793      0.817      0.676

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     53/400      21.3G     0.7202     0.4708      1.005         33       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.903      0.729      0.833      0.606

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     54/400      21.1G     0.6705     0.4445     0.9816         48       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.882        0.8      0.889      0.718

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     55/400      21.2G     0.6598      0.426     0.9767         33       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559       0.89      0.793      0.873      0.672

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     56/400      21.2G     0.6894     0.4638     0.9835         34       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.823      0.838      0.867      0.691

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     57/400      21.2G     0.7007     0.4646     0.9928         30       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.813      0.764      0.843      0.663

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     58/400      21.3G     0.6603     0.4406     0.9803         54       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.843      0.775      0.869      0.632

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     59/400      21.1G     0.6821     0.4523     0.9852         41       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.865       0.79       0.82      0.672

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     60/400      21.2G      0.662     0.4618      0.978         46       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.861      0.789      0.867      0.667

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     61/400      21.2G     0.6596     0.4335     0.9797         32       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.842      0.854      0.888      0.705

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     62/400      21.1G     0.6738     0.4445     0.9785         44       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.843      0.775      0.862      0.666

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     63/400      21.2G     0.6196      0.428     0.9629          9       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.867      0.752      0.879      0.688

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     64/400      21.2G      0.629     0.4076     0.9709         28       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.943      0.793      0.879      0.677

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     65/400      21.2G     0.6295     0.4225     0.9646         27       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.836      0.846      0.892      0.683

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     66/400      21.2G     0.6698     0.4921      0.977         12       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.823       0.84      0.878      0.714

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     67/400      21.2G     0.6314     0.4267     0.9521         25       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.837      0.766      0.839      0.681

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     68/400      21.2G     0.6382     0.4446     0.9627         35       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.912      0.768      0.864      0.694

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     69/400      21.2G     0.6561     0.4449     0.9694         57       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559       0.92       0.82      0.889      0.646
EarlyStopping: Training stopped early as no improvement observed in last 15 epochs. Best results observed at epoch 54, best model saved as best.pt.
To update EarlyStopping(patience=15) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

69 epochs completed in 0.502 hours.
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_l_2_d/weights/weights/last.pt, 51.4MB
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_l_2_d/weights/weights/best.pt, 51.4MB

Validating /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_l_2_d/weights/weights/best.pt...
Ultralytics 8.4.43 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
YOLO11l summary (fused): 191 layers, 25,282,396 parameters, 0 gradients, 86.6 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        559      0.883        0.8      0.889      0.718
             pv_string         31        167      0.994      0.966      0.994      0.854
             pv_module         41        340      0.982      0.962      0.993      0.814
                 other          6          7      0.604      0.714      0.808       0.65
               anomaly         23         45      0.952      0.556      0.762      0.556
Speed: 0.5ms preprocess, 27.9ms inference, 0.0ms loss, 2.2ms postprocess per image
Results saved to /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_l_2_d/weights
Elapsed: 0:30:39
```

샘플 200개, pv string 및 pv panel, other, anomaly 지정, nagative 추가, l 모델 사용 (배치 8)<br>
GCE g2-standard-8 (vCPU 8개, 메모리 32GB), NVIDIA L4 1개

```bash
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s200_l_d \
    --model models/yolo11l.pt \
    --epochs 400 \
    --batch 8 \
    --device cuda \
    --amp True \
    --output ./workspace/train_s200_l_d
Seed 이미지 222장 (수동 라벨 완료)
  train: 178장
  val: 44장
New https://pypi.org/project/ultralytics/8.4.48 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.4.43 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=8, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=workspace/train_s200_l_d/dataset/data.yaml, degrees=5.0, deterministic=True, device=0, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=400, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.5, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=1280, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=models/yolo11l.pt, momentum=0.937, mosaic=0.5, multi_scale=0.0, name=weights, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=15, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=workspace/train_s200_l_d, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/home/sjkim/solar-thermal/runs/detect/workspace/train_s200_l_d/weights, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.3, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.05, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=4

                   from  n    params  module                                       arguments                     
  0                  -1  1      1856  ultralytics.nn.modules.conv.Conv             [3, 64, 3, 2]                 
  1                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  2                  -1  2    173824  ultralytics.nn.modules.block.C3k2            [128, 256, 2, True, 0.25]     
  3                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
  4                  -1  2    691712  ultralytics.nn.modules.block.C3k2            [256, 512, 2, True, 0.25]     
  5                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  6                  -1  2   2234368  ultralytics.nn.modules.block.C3k2            [512, 512, 2, True]           
  7                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  8                  -1  2   2234368  ultralytics.nn.modules.block.C3k2            [512, 512, 2, True]           
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
 10                  -1  2   1455616  ultralytics.nn.modules.block.C2PSA           [512, 512, 2]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  2   2496512  ultralytics.nn.modules.block.C3k2            [1024, 512, 2, True]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  2    756736  ultralytics.nn.modules.block.C3k2            [1024, 256, 2, True]          
 17                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  2   2365440  ultralytics.nn.modules.block.C3k2            [768, 512, 2, True]           
 20                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  2   2496512  ultralytics.nn.modules.block.C3k2            [1024, 512, 2, True]          
 23        [16, 19, 22]  1   1414108  ultralytics.nn.modules.head.Detect           [4, 16, None, [256, 512, 512]]
YOLO11l summary: 358 layers, 25,313,564 parameters, 25,313,548 gradients, 87.3 GFLOPs

Transferred 1009/1015 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1620.1±87.3 MB/s, size: 11641.4 KB)
train: Scanning /home/sjkim/solar-thermal/workspace/train_s200_l_d/dataset/labels/train... 178 images, 19 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 178/178 1.3Kit/s 0.1s
train: New cache created: /home/sjkim/solar-thermal/workspace/train_s200_l_d/dataset/labels/train.cache
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2653.6±1522.6 MB/s, size: 11244.6 KB)
val: Scanning /home/sjkim/solar-thermal/workspace/train_s200_l_d/dataset/labels/val... 44 images, 3 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 44/44 944.2it/s 0.0s
val: /home/sjkim/solar-thermal/workspace/train_s200_l_d/dataset/images/val/DJI_20251217130705_0114_Z.JPG: 1 duplicate labels removed
val: New cache created: /home/sjkim/solar-thermal/workspace/train_s200_l_d/dataset/labels/val.cache
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.00125, momentum=0.9) with parameter groups 167 weight(decay=0.0), 174 weight(decay=0.0005), 173 bias(decay=0.0)
Plotting labels to /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_l_d/weights/labels.jpg... 
Image sizes 1280 train, 1280 val
Using 8 dataloader workers
Logging results to /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_l_d/weights
Starting training for 400 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/400      20.9G      1.572      2.366      1.645         41       1280: 100% ━━━━━━━━━━━━ 23/23 2.2s/it 49.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.1s/it 3.3s
                   all         44        560      0.585      0.399      0.244      0.127

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/400      21.2G      1.148      1.297      1.233         32       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.7it/s 1.7s
                   all         44        560          0          0          0          0

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      3/400      21.2G      1.066      1.131      1.199         39       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.6it/s 1.9s
                   all         44        560    0.00181     0.0107    3.9e-05   3.94e-06

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      4/400      21.2G      1.166       1.06      1.275         42       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.9it/s 1.6s
                   all         44        560     0.0715    0.00985    0.00303   0.000663

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      5/400      21.2G      1.171      1.145       1.25         49       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560    0.00812      0.036    0.00339   0.000726

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      6/400      21.1G      1.103      1.134      1.259         40       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.602       0.13     0.0457     0.0201

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      7/400      21.1G      1.042       1.01      1.161         19       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.832      0.372      0.379      0.204

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      8/400      21.3G      1.018     0.9359      1.142         35       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.364      0.059     0.0561     0.0424

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      9/400      21.3G     0.9829     0.8814      1.122         26       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.671      0.392      0.444      0.323

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     10/400      21.2G     0.9358     0.8305      1.107         47       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.518      0.201      0.192      0.123

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     11/400      21.2G     0.9132     0.8057      1.093         24       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.671       0.46      0.513      0.387

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     12/400      21.2G     0.9103     0.8124      1.082          7       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.675      0.526      0.474      0.333

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     13/400      21.2G     0.8909      0.778      1.072         29       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.752       0.59      0.591       0.44

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     14/400      21.2G     0.8625     0.7705      1.059         22       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.862      0.498       0.59      0.424

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     15/400      21.2G     0.8686     0.7881      1.059         43       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.779      0.438      0.486      0.371

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     16/400      21.2G     0.8734     0.7682      1.065         30       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560       0.88      0.507      0.579      0.383

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     17/400      21.2G     0.8429     0.7402      1.032         51       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560       0.87      0.601      0.675      0.517

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     18/400      21.1G     0.8574     0.7087      1.053         20       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.857       0.49      0.565      0.424

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     19/400      21.2G     0.8262     0.6888      1.048         20       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.455      0.284      0.232      0.167

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     20/400      21.2G     0.7692     0.6607      1.019          9       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560       0.74       0.36       0.41      0.301

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     21/400      21.2G      0.817     0.6975      1.032         81       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.636      0.658      0.682      0.516

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     22/400      21.1G     0.7885     0.6679      1.025         53       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.818      0.712      0.757      0.507

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     23/400      21.2G     0.8016     0.6483      1.029         46       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.853       0.75      0.786      0.538

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     24/400      21.2G      0.736     0.6318      1.001         30       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.889      0.697      0.776      0.589

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     25/400      21.3G     0.7482     0.6193     0.9991         33       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560        0.7      0.474      0.524      0.339

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     26/400      21.2G     0.7595     0.6068      1.007         36       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.793      0.685      0.771      0.569

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     27/400      21.2G     0.7246     0.6052      1.015         25       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.841      0.695      0.768      0.595

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     28/400      21.2G     0.7195     0.5932      1.006         30       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560       0.88      0.585      0.728      0.558

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     29/400      21.2G     0.7499     0.6157      1.015         32       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560       0.87      0.584      0.651      0.508

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     30/400      21.1G     0.7894     0.6098      1.018         18       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.756      0.761      0.732      0.574

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     31/400      21.2G     0.8613     0.6794      1.047         85       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.863      0.616      0.708      0.484

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     32/400      21.2G     0.7889     0.6041      1.019         42       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.795      0.677       0.75      0.578

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     33/400      21.2G      0.715     0.5718     0.9932         42       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.819      0.713       0.79      0.622

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     34/400      21.2G     0.7154      0.571     0.9995         19       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.918      0.698      0.806      0.535

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     35/400      21.2G     0.7264     0.5756       0.98         45       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.898      0.714      0.801       0.62

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     36/400      21.2G     0.7057     0.5789     0.9951         48       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560        0.9      0.689       0.82      0.606

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     37/400      21.2G     0.7166     0.5647     0.9924         27       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.886       0.68      0.758      0.615

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     38/400      21.1G     0.7717     0.5962      1.014         47       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.824       0.71      0.787      0.627

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     39/400      21.2G      0.701     0.5813     0.9938         40       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.861      0.725        0.8      0.643

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     40/400      21.2G     0.6806     0.5453     0.9802         28       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560       0.77      0.755      0.757      0.607

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     41/400      21.3G      0.684     0.5255     0.9786         20       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.839      0.719      0.756       0.59

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     42/400      21.2G     0.7214     0.6494     0.9915          2       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.898      0.665      0.788      0.614

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     43/400      21.1G     0.7002     0.5927       0.98         47       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.688      0.655      0.677      0.532

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     44/400      21.3G     0.6826     0.5505     0.9856         21       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.872       0.76      0.794      0.587

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     45/400      21.2G      0.745     0.5794      1.002         41       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.868      0.758      0.829      0.665

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     46/400      21.2G     0.7116     0.5547     0.9845         36       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560       0.85      0.727      0.805       0.63

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     47/400      21.1G     0.7913     0.5874       1.01         30       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.852      0.639      0.779      0.619

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     48/400      21.2G       0.75     0.5713     0.9941         39       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.872      0.732      0.826      0.672

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     49/400      21.2G     0.6952     0.5428      0.979         51       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.906      0.754      0.867      0.679

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     50/400      21.2G     0.6799      0.518     0.9706         44       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.908      0.769      0.878      0.696

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     51/400      21.2G     0.6815     0.5553      0.989         29       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.906      0.765      0.869      0.616

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     52/400      21.2G     0.6805     0.5469     0.9745         29       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560        0.9      0.804       0.88      0.697

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     53/400      21.3G     0.7208     0.5443     0.9993         33       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.857      0.816      0.866       0.69

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     54/400      21.1G     0.6834     0.5224     0.9773         48       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.808      0.834      0.867      0.696

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     55/400      21.2G     0.6649     0.5261      0.972         33       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.857      0.775      0.847      0.638

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     56/400      21.2G     0.6844     0.5376     0.9776         34       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.828      0.812      0.839      0.675

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     57/400      21.2G     0.7211     0.5171     0.9882         30       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560       0.88      0.753      0.833      0.663

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     58/400      21.3G      0.658     0.5132     0.9723         54       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.799      0.797       0.83      0.624

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     59/400      21.1G     0.6784      0.503     0.9752         41       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560       0.86      0.762      0.836      0.671

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     60/400      21.2G     0.6528     0.5209     0.9663         46       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.895      0.772      0.883       0.71

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     61/400      21.3G     0.6641     0.4942     0.9765         32       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.915      0.786      0.879      0.716

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     62/400      21.1G     0.6648     0.5311     0.9707         44       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.837      0.783      0.877      0.697

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     63/400      21.2G     0.6084     0.4972     0.9508          9       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.798      0.809      0.848      0.678

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     64/400      21.2G     0.6187     0.4699     0.9554         28       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.893      0.779       0.85      0.687

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     65/400      21.2G     0.6133     0.4879     0.9517         27       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.832      0.774      0.813      0.624

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     66/400      21.2G     0.6785     0.5406     0.9759         12       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.888      0.765      0.855      0.647

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     67/400      21.2G     0.6246     0.4799     0.9471         25       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.883      0.793      0.833      0.664

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     68/400      21.2G     0.6522      0.486      0.961         35       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560       0.84      0.806      0.871        0.7

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     69/400      21.2G     0.6327     0.4821     0.9542         57       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.917      0.772      0.861      0.649

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     70/400      21.1G     0.6991     0.4995     0.9861         42       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.823      0.824      0.872      0.675

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     71/400      21.2G     0.6821     0.4774     0.9763         64       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.861      0.796      0.887      0.692

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     72/400      21.2G     0.6435     0.4689     0.9486         28       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.863      0.788      0.871      0.688

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     73/400      21.2G     0.6361     0.4878     0.9509         63       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.929      0.804      0.902      0.723

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     74/400      21.2G     0.6148     0.4754     0.9504         40       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560       0.92      0.787      0.874      0.729

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     75/400      21.2G     0.6148     0.4787     0.9407         43       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560       0.88      0.797      0.867      0.671

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     76/400      21.2G      0.659     0.4951      0.961         30       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.846       0.83      0.864      0.663

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     77/400      21.2G     0.6432     0.4842     0.9606         41       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560       0.89      0.762       0.89      0.707

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     78/400      21.1G     0.6093     0.4883     0.9434         32       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.779      0.817      0.871      0.712

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     79/400      21.1G      0.607     0.4648     0.9462         21       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.827       0.76       0.83      0.668

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     80/400      21.2G     0.6519      0.494     0.9617         43       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.867      0.804      0.889      0.723

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     81/400      21.2G     0.6608     0.4735     0.9666         19       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.908      0.786       0.89      0.709

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     82/400      21.2G     0.6383       0.47     0.9559         35       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.859       0.75      0.867       0.68

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     83/400      21.2G     0.6312      0.479     0.9544         41       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.856       0.83      0.853      0.683

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     84/400      21.2G     0.6718     0.5093     0.9689         35       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560       0.84      0.833      0.865       0.69

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     85/400      21.3G     0.6247     0.4538     0.9512         37       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.884      0.766      0.883      0.706

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     86/400      21.1G     0.6455     0.4638      0.953         29       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.916      0.723      0.868       0.69

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     87/400      21.2G     0.6233     0.4884     0.9494         18       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.882      0.794      0.859      0.686

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     88/400      21.2G     0.6135     0.4732     0.9496         34       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560      0.931      0.798      0.884      0.684

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     89/400      21.3G     0.6058     0.4455     0.9392         39       1280: 100% ━━━━━━━━━━━━ 23/23 1.1it/s 21.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560       0.84      0.824      0.882      0.699
EarlyStopping: Training stopped early as no improvement observed in last 15 epochs. Best results observed at epoch 74, best model saved as best.pt.
To update EarlyStopping(patience=15) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

89 epochs completed in 0.645 hours.
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_l_d/weights/weights/last.pt, 51.4MB
Optimizer stripped from /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_l_d/weights/weights/best.pt, 51.4MB

Validating /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_l_d/weights/weights/best.pt...
Ultralytics 8.4.43 🚀 Python-3.14.4 torch-2.11.0+cu130 CUDA:0 (NVIDIA L4, 22565MiB)
YOLO11l summary (fused): 191 layers, 25,282,396 parameters, 0 gradients, 86.6 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all         44        560       0.92      0.786      0.874      0.728
             pv_string         35        292       0.99      0.935      0.993      0.877
             pv_module         36        215      0.896       0.94      0.971      0.788
                 other          6          7      0.794      0.714      0.773      0.682
               anomaly         24         46          1      0.556       0.76      0.563
Speed: 0.5ms preprocess, 27.8ms inference, 0.0ms loss, 2.1ms postprocess per image
Results saved to /home/sjkim/solar-thermal/runs/detect/workspace/train_s200_l_d/weights
Elapsed: 0:39:03
```