#

Ubuntu 24.04 LTS Minimal
x86/64, amd64 noble minimal image
500GB

##

```bash
sudo apt update
sudo apt -y upgrade
sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev
sudo apt install -y nano screen
```

NVIDIA 드라이버 설치 후 재부팅

```bash
sudo apt update
sudo apt upgrade -y
sudo ubuntu-drivers devices
```

#### 수동 설치

```bash
ERROR:root:aplay command not found
== /sys/devices/pci0000:00/0000:00:03.0 ==
modalias : pci:v000010DEd000027B8sv000010DEsd000016EEbc03sc02i00
vendor   : NVIDIA Corporation
model    : AD104GL [L4]
driver   : nvidia-driver-580-server-open - distro non-free
driver   : nvidia-driver-580-server - distro non-free
driver   : nvidia-driver-580 - distro non-free
driver   : nvidia-driver-595-open - distro non-free recommended
driver   : nvidia-driver-595-server - distro non-free
driver   : nvidia-driver-595-server-open - distro non-free
driver   : nvidia-driver-595 - distro non-free
driver   : nvidia-driver-580-open - distro non-free
driver   : xserver-xorg-video-nouveau - distro free builtin
```

```bash
sudo apt install nvidia-driver-595 -y
```

#### 자동 설치

```bash
sudo apt update
sudo apt upgrade
sudo ubuntu-drivers autoinstall
```

#### 재부팅 및 nvidia-smi 확인

```bash
sudo reboot

nvidia-smi
```

```bash
Sun Aug 17 23:21:13 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 575.64.03              Driver Version: 575.64.03      CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   52C    P8             10W /   70W |       0MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

```bash
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.
```

문제 해결: reboot Compute Engine

```bash
sudo reboot
```

#### Root password

```bash
sudo passwd root
```


```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Clone Source from Github

```bash
git clone https://github.com/sayouzone/solar-thermal.git
cd solar-thermal
```

#### Download images & workspace from Google Cloud Storage

```bash
mkdir -p data/solar/images
gcloud storage cp -r gs://solar-plant/drone1 data/solar/images/
```

```bash
gcloud storage cp -r gs://solar-plant/workspace .
```

#### 로컬 라벨링 데이터를 GCS으로 복사

```bash
gcloud storage cp -r workspace/labels_s50_m_3 gs://solar-plant/workspace/
```

#### GCS 라벨링 데이터를 GCE으로 복사

```bash
gcloud storage cp -r gs://solar-plant/workspace/labels_s50_m_3 workspace/
```

#### 학습을 위해 불필요한 파일들 지우기

```bash
# 확인
ls data/solar/images/RGB | awk '$0 > "DJI_20251217130416_0051_Z.JPG"'

# 삭제
ls data/solar/images/RGB | awk -v dir="data/solar/images/RGB" '$0 > "DJI_20251217130413_0050_Z.JPG" {print dir "/" $0}' | xargs rm -v
```

#### Python 패키지 설치

```bash
pip3 install -r requirements.txt
```

exiftool 설치

```bash
pip3 install piexif pillow --break-system-packages -q && apt-get install -y exiftool 2>/dev/null | tail -1
```

#### Cloud Storage 접근 오류

```bash
gcloud storage cp -r ~/solar-thermal/runs/detect/workspace/train_s50_m gs://solar-plant/workspace/
ERROR: (gcloud.storage.cp) HTTPError 403: 1037372895180-compute@developer.gserviceaccount.com does not have storage.objects.get access to the Google Cloud Storage object. Permission 'storage.objects.get' denied on resource (or it may not exist). This command is authenticated as 1037372895180-compute@developer.gserviceaccount.com which is the active account specified by the [core/account] property.
```

```bash
gcloud config set project sayouzone-ai
gcloud auth login
```

#### GCE 학습된 모델을 GCS으로 복사

```bash
gcloud storage cp -r ~/solar-thermal/runs/detect/workspace/train_s50_m_2 gs://solar-plant/workspace/
```

#### GCS에 저장된 학습된 모델을 로컬로 복사

```bash
gcloud storage cp -r gs://solar-plant/workspace/train_s50_m_2 runs/detect/workspace/
```

##

**Python 3.12 Ubuntu 패키지 설치**

```bash
sudo apt install -y python3.14 python3-pip python3-venv
```

**Python 3.12 수작업 설치**

```bash
wget https://www.python.org/ftp/python/3.12.0/Python-3.12.0.tar.xz  # 또는 최신 버전 다운로드
tar -xf Python-3.12.0.tar.xz

cd Python-3.12.0
./configure --enable-optimizations
make -j 8  # CPU 코어 수에 맞게 조정
sudo make altinstall

python3 --version
python --version
pip --version
```

####

```bash
sudo apt install ffmpeg -y
```

```bash
ffmpeg -i data/solar/images/RGB/DJI_20251217130214_0006_Z.JPG \
    -vf "crop=iw*0.55:ih:0:0" \
    data/solar/images/RGB/DJI_20251217130214_0006_negative.JPG
```

```bash
ffmpeg -i data/solar/images/RGB/DJI_20251217130223_0009_Z.JPG \
    -vf "crop=iw*0.55:ih:0:0" \
    data/solar/images/RGB/DJI_20251217130223_0009_negative.JPG
```

```bash
ffmpeg -i data/solar/images/RGB/DJI_20251217130242_0016_Z.JPG \
    -vf "crop=iw*0.55:ih:0:0" \
    data/solar/images/RGB/DJI_20251217130242_0016_negative.JPG
```

```bash
ffmpeg -i data/solar/images/RGB/DJI_20251217130317_0029_Z.JPG \
    -vf "crop=iw*0.55:ih:0:0" \
    data/solar/images/RGB/DJI_20251217130317_0029_negative.JPG
```

```bash
ffmpeg -i data/solar/images/RGB/DJI_20251217130411_0049_Z.JPG \
    -vf "crop=ih*0.45:ih:iw*0.55:0" \
    data/solar/images/RGB/DJI_20251217130411_0049_Z_negative.JPG
```

```bash
ffmpeg -i data/solar/images/RGB/DJI_20251217130413_0050_Z.JPG \
    -vf "crop=ih*0.45:ih:iw*0.55:0" \
    data/solar/images/RGB/DJI_20251217130413_0050_Z_negative.JPG
```

```bash
touch workspace/labels_negative/DJI_20251217130413_0050_Z_negative.txt
touch workspace/labels_negative/DJI_20251217130411_0049_Z_negative.txt
touch workspace/labels_negative/DJI_20251217130317_0029_Z_negative.txt
touch workspace/labels_negative/DJI_20251217130242_0016_Z_negative.txt
touch workspace/labels_negative/DJI_20251217130223_0009_Z_negative.txt
touch workspace/labels_negative/DJI_20251217130214_0006_Z_negative.txt
```

```bash
cp workspace/labels_negative/*.txt workspace/labels_s50_m_3
```