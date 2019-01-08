CC = gcc

pybind/libucp_py_ucp_fxns.a: pybind/ucp_py_ucp_fxns.o pybind/buffer_ops.o
	ar rcs $@ $^

pybind/%.o: pybind/%.c
	$(CC) -shared -fPIC -c $^ -o $@

install: pybind/libucp_py_ucp_fxns.a
	cd pybind && \
	python setup.py build_ext && \
	python -m pip install -e .

clean:
	rm pybind/*.o pybind/*.a
