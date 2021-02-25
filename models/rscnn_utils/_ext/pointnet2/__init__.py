import torch
from functools import wraps
try:
    import cffi
except ImportError:
    raise ImportError("torch.utils.ffi requires the cffi package")


if cffi.__version_info__ < (1, 4, 0):
    raise ImportError("torch.utils.ffi requires cffi version >= 1.4, but "
                      "got " + '.'.join(map(str, cffi.__version_info__)))


#from torch.utils.ffi import _wrap_function
from ._pointnet2 import lib as _lib, ffi as _ffi


def _generate_typedefs():
    typedefs = []
    for t in ['Double', 'Float', 'Long', 'Int', 'Short', 'Char', 'Byte']:
        for lib in ['TH', 'THCuda']:
            for kind in ['Tensor', 'Storage']:
                python_name = t + kind
                if t == 'Float' and lib == 'THCuda':
                    th_name = 'THCuda' + kind
                else:
                    th_name = lib + t + kind
                th_struct = 'struct ' + th_name

                typedefs += ['typedef {} {};'.format(th_struct, th_name)]
                # We have to assemble a string here, because we're going to
                # do this lookup based on tensor.type(), which returns a
                # string (not a type object, as this code was before)
                python_module = 'torch.cuda' if lib == 'THCuda' else 'torch'
                python_class = python_module + '.' + python_name
                _cffi_to_torch[th_struct] = python_class
                _torch_to_cffi[python_class] = th_struct
    return '\n'.join(typedefs) + '\n'
_cffi_to_torch = {}
_torch_to_cffi = {}
_typedefs = _generate_typedefs()



def _wrap_function(function, ffi):
    @wraps(function)
    def safe_call(*args, **kwargs):
        args = tuple(ffi.cast(_torch_to_cffi.get(arg.type(), 'void') + '*', arg._cdata)
                     if isinstance(arg, torch.Tensor) or torch.is_storage(arg)
                     else arg
                     for arg in args)
        args = (function,) + args
        result = torch._C._safe_call(*args, **kwargs)
        if isinstance(result, ffi.CData):
            typeof = ffi.typeof(result)
            if typeof.kind == 'pointer':
                cdata = int(ffi.cast('uintptr_t', result))
                cname = typeof.item.cname
                if cname in _cffi_to_torch:
                    # TODO: Maybe there is a less janky way to eval
                    # off of this
                    return eval(_cffi_to_torch[cname])(cdata=cdata)
        return result
    return safe_call


__all__ = []
def _import_symbols(locals):
    for symbol in dir(_lib):
        fn = getattr(_lib, symbol)
        if callable(fn):
            locals[symbol] = _wrap_function(fn, _ffi)
        else:
            locals[symbol] = fn
        __all__.append(symbol)

_import_symbols(locals())


