CC = gcc
UCX_PATH  ?= "$(abspath $(shell pwd))/../ucx/install"
CUDA_PATH ?= "/usr/local/cuda"
PYTHON    ?= python3

CFLAGS  = "-I$(UCX_PATH)/include -I$(CUDA_PATH)/include"
LDFLAGS = "-L$(UCX_PATH)/lib -L$(CUDA_PATH)/lib64"

install:
	LDFLAGS=$(LDFLAGS) CFLAGS=$(CFLAGS) $(PYTHON) setup.py build_ext -i --with-cuda
	$(PYTHON) -m pip install -e .

install-cpu:
	LDFLAGS=$(LDFLAGS) CFLAGS=$(CFLAGS) $(PYTHON) setup.py build_ext -i
	$(PYTHON) -m pip install -e .

conda-install:
	LDFLAGS=$(LDFLAGS) CFLAGS=$(CFLAGS) $(PYTHON) setup.py build_ext -i --with-cuda install

conda-install-cpu:
	LDFLAGS=$(LDFLAGS) CFLAGS=$(CFLAGS) $(PYTHON) setup.py build_ext -i install


clean:
	rm -rf build
	rm -rf ucp/_libs/*.c
	rm -rf ucp/_libs/*.so
	rm -rf *.egg-info

test:
	$(PYTHON) -m pytest tests

conda-packages:
	conda build --numpy=1.14 --python=3.7 ucx
	conda build --numpy=1.14 --python=3.7 ucx-gpu
	conda build --numpy=1.14 --python=3.7 ucx-py     -c defaults -c conda-forge
	conda build --numpy=1.14 --python=3.7 ucx-py-gpu -c defaults -c conda-forge
