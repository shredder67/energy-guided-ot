FROM nvidia/cuda:11.6.2-base-ubuntu20.04

WORKDIR /tmp

RUN apt-get update -qq && \
	apt-get install --yes --no-install-recommends \
	zip unzip pigz jq moreutils time wget inotify-tools vim parallel curl libxmu6 git

# Install environment
RUN wget --progress=bar:force:noscroll https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh && \
    mkdir /root/.conda && \
    bash Miniconda3-py38_4.11.0-Linux-x86_64.sh -b && \
    rm -f Miniconda3-py38_4.11.0-Linux-x86_64.sh
ENV PATH=/root/miniconda3/bin:${PATH}
RUN conda --version

RUN conda install -y -q -f -c conda-forge dcm2niix=1.0.20190902 && \
	sync

# TODO: test installation on remote
RUN pip install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia

COPY ./requirements.txt /code/requirements.txt
#RUN conda install -y --file /code/requirements.txt
RUN pip install --upgrade pip && \
	pip install -r /code/requirements.txt && \
	sync
