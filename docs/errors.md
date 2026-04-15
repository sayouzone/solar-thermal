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

#### Solar Data

```bash
python scripts/train_yolo.py --data configs/dataset.yaml --base yolov8m.pt --device cpu
```

dataset.yaml에서 "path: /data/solar" 에 이미지 파일 경로 변경 필요

```bash
[error] dataset root does not exist: /data/solar
        Fix the `path:` value in configs/dataset.yaml or mount/download the dataset.
```

->

dataset.yaml에서 "path: ./data/solar"으로 변경

#### Training / Eval Data

```bash
Traceback (most recent call last):
  File "/Users/seongjungkim/Development/sayouzone/solar-thermal/scripts/train_yolo.py", line 117, in <module>
    main()
  File "/Users/seongjungkim/Development/sayouzone/solar-thermal/scripts/train_yolo.py", line 95, in main
    model.train(
  File "/Users/seongjungkim/Development/sayouzone/.venv/lib/python3.11/site-packages/ultralytics/engine/model.py", line 787, in train
    self.trainer.train()
  File "/Users/seongjungkim/Development/sayouzone/.venv/lib/python3.11/site-packages/ultralytics/engine/trainer.py", line 246, in train
    self._do_train()
  File "/Users/seongjungkim/Development/sayouzone/.venv/lib/python3.11/site-packages/ultralytics/engine/trainer.py", line 369, in _do_train
    self._setup_train()
  File "/Users/seongjungkim/Development/sayouzone/.venv/lib/python3.11/site-packages/ultralytics/engine/trainer.py", line 350, in _setup_train
    self._build_train_pipeline()
  File "/Users/seongjungkim/Development/sayouzone/.venv/lib/python3.11/site-packages/ultralytics/engine/trainer.py", line 271, in _build_train_pipeline
    self.train_loader = self.get_dataloader(
                        ^^^^^^^^^^^^^^^^^^^^
  File "/Users/seongjungkim/Development/sayouzone/.venv/lib/python3.11/site-packages/ultralytics/models/yolo/detect/train.py", line 93, in get_dataloader
    dataset = self.build_dataset(dataset_path, mode, batch_size)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/seongjungkim/Development/sayouzone/.venv/lib/python3.11/site-packages/ultralytics/models/yolo/detect/train.py", line 77, in build_dataset
    return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/seongjungkim/Development/sayouzone/.venv/lib/python3.11/site-packages/ultralytics/data/build.py", line 236, in build_yolo_dataset
    return dataset(
           ^^^^^^^^
  File "/Users/seongjungkim/Development/sayouzone/.venv/lib/python3.11/site-packages/ultralytics/data/dataset.py", line 88, in __init__
    super().__init__(*args, channels=self.data.get("channels", 3), **kwargs)
  File "/Users/seongjungkim/Development/sayouzone/.venv/lib/python3.11/site-packages/ultralytics/data/base.py", line 117, in __init__
    self.im_files = self.get_img_files(self.img_path)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/seongjungkim/Development/sayouzone/.venv/lib/python3.11/site-packages/ultralytics/data/base.py", line 181, in get_img_files
    raise FileNotFoundError(f"{self.prefix}Error loading data from {img_path}\n{HELP_URL}") from e
FileNotFoundError: train: Error loading data from /Users/seongjungkim/Development/sayouzone/solar-thermal/data/solar/images/train
See https://docs.ultralytics.com/datasets for dataset formatting guidance.
```