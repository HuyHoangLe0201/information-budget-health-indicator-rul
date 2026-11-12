"""Restricted unpickler: only plain containers and NumPy/pandas array reconstruction are allowed, so a downloaded pickle cannot run arbitrary code."""
import pickle, io
ALLOWED = {('numpy.core.multiarray', '_reconstruct'), ('numpy._core.multiarray', '_reconstruct'), ('numpy.core.multiarray', 'scalar'), ('numpy._core.multiarray', 'scalar'),
           ('numpy', 'ndarray'), ('numpy', 'dtype'), ('collections', 'OrderedDict'), ('builtins', 'dict'), ('builtins', 'list'), ('builtins', 'set'), ('builtins', 'tuple'),
           ('numpy.core.numeric', '_frombuffer'), ('numpy._core.numeric', '_frombuffer'), ('_codecs', 'encode')}
class Safe(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError('blocked global %s.%s' % (module, name))
def loads(b): return Safe(io.BytesIO(b)).load()
def load(path):
    with open(path, 'rb') as f: return loads(f.read())
