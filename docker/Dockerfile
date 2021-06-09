FROM nvcr.io/nvidia/pytorch:20.10-py3

RUN pip install jsonlines

RUN apt purge --auto-remove cmake

RUN apt-get update && apt-get install -y \
    bash-completion \
    emacs \
    git \
    graphviz \
    htop \
    libopenexr-dev \
    rsync \
    wget \
&& rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get -y install sudo

RUN apt remove --purge --auto-remove cmake

RUN pip install --upgrade cmake

RUN apt-get install -y swig

RUN git clone https://github.com/facebookresearch/faiss.git ~/faiss && cd ~/faiss && \
cmake -DCMAKE_CUDA_ARCHITECTURES="70;80" -B build . && make -C build && \
cd build/faiss/python && python setup.py install
