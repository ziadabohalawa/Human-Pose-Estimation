# Human-Pose Estimation on AWS (CPU-only)

End-to-end demo that runs **Lightweight OpenPose** on images stored in **Amazon S3**, using an **EC2 t-instance**.  
A tiny Flask UI lets you upload an image → returns an annotated image + JSON key-points.

---

##  Architecture

![architecture_diagram](https://github.com/ziadabohalawa/Human-Pose-Estimation/blob/1ccb9b36c684d75e140ef20b44fd1d7dbd5cbfb5/pose_etimation_diagram.png)


---

## Tech Stack

| Layer / Purpose | Technology |
|-----------------|------------|
| Web backend     | Flask 2 |
| ML framework    | PyTorch CPU |
| Pose model      | Lightweight Human Pose Estimation (MobileNet-v1) |
| Storage         | Amazon S3 (`images/`, `results/`) |
| Compute         | EC2 t3.large (Ubuntu 22.04) |
| Image libs      | OpenCV 4, Pillow |
| Other           | boto3, python-magic |

---

## Quick Start (Step-by-Step)


### 0  Prerequisites
| What | Value |
|------|-------|
| **S3 bucket** | Example: `pose-estimation-bucket` |
| **Security Group** | Inbound: TCP 22 (SSH), TCP 5000 (HTTP) |

### 1  Launch EC2

- AMI : Ubuntu 22.04
- Type : t3.large
- EBS size : 20 GB
- Key pair : pose-estimation.pem
- IAM role : AmazonS3FullAccess (or scoped S3 role)


### 2  SSH & **clone the repo**

```
ssh -i pose-estimation.pem ubuntu@<EC2-IP>
git clone https://github.com/ziadabohalawa/human-pose-estimation.git
cd human-pose-estimation
```
### 3 Install system packages
```
sudo apt update
sudo apt install -y python3-pip libgl1 libglib2.0-0 build-essential python3-dev
```
### 4 Install Python dependencies
```
python3 -m pip install -r requirements.txt
# if torch fails, use the CPU wheel:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
### 5 Download model weights
```
mkdir -p models
wget https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth \
     -O models/human-pose.pth
```

### 6 Run the web app
```
python3 app.py          
```
### 7 Open in browser
```
http://<EC2-Public-IP>:5000
```
Upload and Image ➜ get skeleton overlay + keypoints JSON.

## Acknowledgements
Lightweight Human Pose Estimation
https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch
(Apache 2.0 License)
