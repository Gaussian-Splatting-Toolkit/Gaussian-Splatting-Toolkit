import sys


class _LazyError:
    def __init__(self, data):
        self.__data = data  # pylint: disable=unused-private-member

    class LazyErrorObj:
        def __init__(self, data):
            self.__data = data  # pylint: disable=unused-private-member

        def __call__(self, *args, **kwds):
            name, exc = object.__getattribute__(self, "__data")
            raise RuntimeError(f"Could not load package {name}.") from exc

        def __getattr__(self, __name: str):
            name, exc = object.__getattribute__(self, "__data")
            raise RuntimeError(f"Could not load package {name}") from exc

    def __getattr__(self, __name: str):
        return _LazyError.LazyErrorObj(object.__getattribute__(self, "__data"))


TCNN_EXISTS = False
tcnn_import_exception = None
tcnn = None
try:
    import tinycudann

    tcnn = tinycudann
    del tinycudann
    TCNN_EXISTS = True
except ModuleNotFoundError as _exp:
    tcnn_import_exception = _exp
except ImportError as _exp:
    tcnn_import_exception = _exp
except EnvironmentError as _exp:
    if "Unknown compute capability" not in _exp.args[0]:
        raise _exp
    print("Could not load tinycudann: " + str(_exp), file=sys.stderr)
    tcnn_import_exception = _exp

if tcnn_import_exception is not None:
    tcnn = _LazyError(tcnn_import_exception)
