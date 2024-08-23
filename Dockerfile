FROM nvidia/cuda:11.4.1-cudnn8-devel-ubuntu20.04

SHELL ["/bin/bash", "-c"]

#ARG USER=user
#RUN useradd -ms /bin/bash ${USER}

RUN apt update && apt -y upgrade
RUN apt install -y --no-install-recommends \
                   python3 \
                   python3-pip \
                   vim \
                   tmux \
                   wget \
                   git \
                   curl \
                   unzip \
                   libgl1-mesa-dev

ENV PATH=/Conda/bin:${PATH}

RUN curl --silent -O https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh && \
    bash Anaconda3-2021.11-Linux-x86_64.sh -b -p /Conda && \
    rm -f Anaconda3-2021.11-Linux-x86_64.sh
RUN echo "source /Conda/etc/profile.d/conda.sh" >> ~/.bashrc

RUN pip3 install --upgrade pip
RUN pip3 install timm==0.3.2 \
                 jupyterlab \
                 matplotlib \
                 opencv-python-headless \
                 tensorboard \
                 umap-learn \
                 pytorch-gradcam \
                 einops

RUN conda install -y conda-build
RUN conda update --all
RUN conda install -c pytorch torchvision 

#USER ${USER}
#WORKDIR /home/${USER}

EXPOSE 8889
