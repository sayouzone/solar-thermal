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