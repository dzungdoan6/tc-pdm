FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
RUN apt-get update && apt-get -y install git ca-certificates python3-pip python3-dev ffmpeg libsm6 libxext6 && apt-get clean


# Install PyTorch and torchvision
RUN pip3 install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 -f https://download.pytorch.org/whl/cu118/torch_stable.html


COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt && pip install --upgrade google-cloud-storage 
