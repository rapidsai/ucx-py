CC = gcc
UCX_PY_UCX_PATH ?= /usr/local
UCX_PY_CUDA_PATH ?= /usr/local/cuda

CPPFLAGS := -I$(UCX_PY_CUDA_PATH)/include -I$(UCX_PY_UCX_PATH)/include
LDFLAGS := -L$(UCX_PY_CUDA_PATH)/lib64 -lcuda -L$(UCX_PY_UCX_PATH)/lib -lucp -luct -lucm -lucs

ucp_py/_libs/libucp_py_ucp_fxns.a: ucp_py/_libs/ucp_py_ucp_fxns.o ucp_py/_libs/buffer_ops.o
	ar rcs $@ $^

ucp_py/_libs/%.o: ucp_py/_libs/%.c
	$(CC) -shared -fPIC $(CPPFLAGS) -c $^ -o $@ $(LDFLAGS)

install: ucp_py/_libs/libucp_py_ucp_fxns.a
	python3 setup.py build_ext && \
	python3 -m pip install -e .

clean:
	rm ucp_py/_libs/*.o ucp_py/_libs/*.a
	rm -rf ucp_py/_libs/build ucp_py/_libs/ucx_py.egg-info
	rm ucp_py/_libs/ucp_py.c
