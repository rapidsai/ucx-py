CC = gcc

pybind/libucp_py_ucp_fxns.a: pybind/ucp_py_ucp_fxns.o pybind/buffer_ops.o
	ar rcs $@ $^

pybind/%.o: pybind/%.c
	$(CC) -shared -fPIC -c $^ -o $@

install: pybind/libucp_py_ucp_fxns.a
	cd pybind && \
	python3 setup.py build_ext && \
	python3 -m pip install -e .

clean:
	rm pybind/*.o pybind/*.a
	rm -rf pybind/build pybind/ucx_py.egg-info
	rm pybind/ucp_py.c
