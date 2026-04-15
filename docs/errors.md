#### Yolo training

```bash
Traceback (most recent call last):
  File "/Users/seongjungkim/Development/sayouzone/solar-thermal/scripts/train_yolo.py", line 63, in <module>
    main()
  File "/Users/seongjungkim/Development/sayouzone/solar-thermal/scripts/train_yolo.py", line 41, in main
    model.train(
  File "/Users/seongjungkim/Development/sayouzone/.venv/lib/python3.11/site-packages/ultralytics/engine/model.py", line 781, in train
    self.trainer = (trainer or self._smart_load("trainer"))(overrides=args, _callbacks=self.callbacks)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/seongjungkim/Development/sayouzone/.venv/lib/python3.11/site-packages/ultralytics/models/yolo/detect/train.py", line 63, in __init__
    super().__init__(cfg, overrides, _callbacks)
  File "/Users/seongjungkim/Development/sayouzone/.venv/lib/python3.11/site-packages/ultralytics/engine/trainer.py", line 128, in __init__
    self.device = select_device(self.args.device)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/seongjungkim/Development/sayouzone/.venv/lib/python3.11/site-packages/ultralytics/utils/torch_utils.py", line 230, in select_device
    raise ValueError(
ValueError: Invalid CUDA 'device=0' requested. Use 'device=cpu' or pass valid CUDA device(s) if available, i.e. 'device=0' or 'device=0,1,2,3' for Multi-GPU.

torch.cuda.is_available(): False
torch.cuda.device_count(): 0
os.environ['CUDA_VISIBLE_DEVICES']: None
See https://pytorch.org/get-started/locally/ for up-to-date torch install instructions if no CUDA devices are seen by torch.
```

####

````bash
Traceback (most recent call last):
  File "/Users/seongjungkim/Development/sayouzone/.venv/lib/python3.11/site-packages/ultralytics/engine/trainer.py", line 705, in get_dataset
    data = check_det_dataset(self.args.data)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/seongjungkim/Development/sayouzone/.venv/lib/python3.11/site-packages/ultralytics/data/utils.py", line 415, in check_det_dataset
    file = Path(check_file(dataset))
                ^^^^^^^^^^^^^^^^^^^
  File "/Users/seongjungkim/Development/sayouzone/.venv/lib/python3.11/site-packages/ultralytics/utils/checks.py", line 683, in check_file
    raise FileNotFoundError(f"'{file}' does not exist")
FileNotFoundError: '/data/solar/data.yaml' does not exist

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/seongjungkim/Development/sayouzone/solar-thermal/scripts/train_yolo.py", line 63, in <module>
    main()
  File "/Users/seongjungkim/Development/sayouzone/solar-thermal/scripts/train_yolo.py", line 41, in main
    model.train(
  File "/Users/seongjungkim/Development/sayouzone/.venv/lib/python3.11/site-packages/ultralytics/engine/model.py", line 781, in train
    self.trainer = (trainer or self._smart_load("trainer"))(overrides=args, _callbacks=self.callbacks)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/seongjungkim/Development/sayouzone/.venv/lib/python3.11/site-packages/ultralytics/models/yolo/detect/train.py", line 63, in __init__
    super().__init__(cfg, overrides, _callbacks)
  File "/Users/seongjungkim/Development/sayouzone/.venv/lib/python3.11/site-packages/ultralytics/engine/trainer.py", line 186, in __init__
    self.data = self.get_dataset()
                ^^^^^^^^^^^^^^^^^^
  File "/Users/seongjungkim/Development/sayouzone/.venv/lib/python3.11/site-packages/ultralytics/engine/trainer.py", line 709, in get_dataset
    raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e
RuntimeError: Dataset '/data/solar/data.yaml' error ❌ '/data/solar/data.yaml' does not exist
````

#### 

```bash
python scripts/train_yolo.py --data configs/dataset.yaml --base yolov8m.pt --device cpu
```

dataset.yaml에서 "path: /data/solar" 에 이미지 파일 경로 변경 필요

```bash
[error] dataset root does not exist: /data/solar
        Fix the `path:` value in configs/dataset.yaml or mount/download the dataset.
```