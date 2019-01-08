CC = gcc

pybind/libmyucp.a: pybind/myucp.o
	ar rcs $@ $^

pybind/myucp.o: pybind/myucp.c
	$(CC) -shared -fPIC -c pybind/myucp.c -o pybind/myucp.o

install: pybind/libmyucp.a
	cd pybind && \
	python setup.py build_ext && \
	python -m pip install -e .
