# Conda recipes for UCX and ucx-py.


*note: these are still quite experimental*

Building
--------

debug


```
$ conda debug conda-recipes/<dir>/
$ # <paste command printed by conda-build>
$ ./conda_build.sh
```


TODO
----

- [  ] proper handling of ibverbs, libnl (package them?)
- [  ] cuda 10 packages
- [  ] write Dockerfile for building

References
----------

- https://github.com/AnacondaRecipes/tensorflow_recipes/
