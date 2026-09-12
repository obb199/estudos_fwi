"""
fwikit.compat
=============
Camada de compatibilidade para rodar o PyFWI 0.1.10 em stacks modernos.

POR QUE ISSO EXISTE
-------------------
O PyFWI 0.1.10 foi escrito na era numpy 1.x / scipy 1.8. Ele importa tres
simbolos que foram REMOVIDOS das bibliotecas depois disso:

  1. `numpy.lib.function_base`  -> removido no numpy 2.0
     (usado em PyFWI/seismic_io.py: `from numpy.lib.function_base import kaiser`)

  2. `scipy.optimize.optimize.MemoizeJac` -> removido no scipy 1.12
     (usado em PyFWI/fwi.py e PyFWI/optimization.py)

  3. `scipy.optimize.lbfgsb.fmin_l_bfgs_b` -> namespace depreciado
     (ainda funciona no scipy 1.17, mas emite DeprecationWarning)

Em vez de rebaixar todo o ambiente (o que travaria voce no numpy 1.x para
sempre), reconstruimos os simbolos ausentes ANTES do PyFWI ser importado.
E um remendo de 20 linhas que economiza horas de dor de cabeca.

USO
---
    import fwikit          # ja aplica o patch automaticamente
    import PyFWI.fwi       # agora importa sem erro

LICAO DE ENGENHARIA
-------------------
Voce vai encontrar isso o tempo todo em geofisica computacional: codigo
academico publicado junto com um artigo, congelado no tempo, que precisa
rodar em 2026. Saber diagnosticar (`ImportError: cannot import name X`) e
remendar sem quebrar o resto e parte do trabalho.
"""
from __future__ import annotations

import sys
import types
import warnings

_APLICADO = False
_RELATORIO: list[str] = []


def aplicar(verboso: bool = False) -> list[str]:
    """Aplica os remendos de compatibilidade. Idempotente."""
    global _APLICADO
    if _APLICADO:
        return _RELATORIO

    import numpy as np

    # ---- Remendo 1: numpy.lib.function_base (removido no numpy 2.0) --------
    if "numpy.lib.function_base" not in sys.modules:
        try:
            import numpy.lib.function_base  # noqa: F401
        except ModuleNotFoundError:
            mod = types.ModuleType("numpy.lib.function_base")
            for nome in ("kaiser", "hamming", "hanning", "blackman", "bartlett",
                         "interp", "copy", "average", "median", "percentile",
                         "gradient", "diff", "trapz"):
                if hasattr(np, nome):
                    setattr(mod, nome, getattr(np, nome))
            sys.modules["numpy.lib.function_base"] = mod
            _RELATORIO.append("numpy.lib.function_base reconstruido (numpy>=2.0)")

    # ---- Remendo 2: scipy.optimize.optimize.MemoizeJac -------------------
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import scipy.optimize as _so
            import scipy.optimize.optimize as _soo
        if not hasattr(_soo, "MemoizeJac"):
            _soo.MemoizeJac = _so._optimize.MemoizeJac
            _RELATORIO.append("scipy.optimize.optimize.MemoizeJac restaurado (scipy>=1.12)")
    except Exception as exc:                                    # pragma: no cover
        _RELATORIO.append(f"AVISO: nao consegui remendar MemoizeJac ({exc})")

    # ---- Remendo 3: scipy.optimize.lbfgsb ---------------------------------
    if "scipy.optimize.lbfgsb" not in sys.modules:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import scipy.optimize.lbfgsb  # noqa: F401
        except ModuleNotFoundError:                             # pragma: no cover
            import scipy.optimize as _so2
            mod = types.ModuleType("scipy.optimize.lbfgsb")
            mod.fmin_l_bfgs_b = _so2.fmin_l_bfgs_b
            sys.modules["scipy.optimize.lbfgsb"] = mod
            _RELATORIO.append("scipy.optimize.lbfgsb reconstruido")

    _APLICADO = True
    if verboso:
        for linha in _RELATORIO:
            print("  [compat]", linha)
    return _RELATORIO


def silenciar_ruido() -> None:
    """
    O PyFWI + pyopencl despejam centenas de warnings por execucao
    (recompilacao de kernel, `RepeatedKernelRetrieval`, depreciacoes do numpy).
    Nada disso e erro. Silenciamos para o terminal da aula ficar legivel.

    Se voce estiver DEPURANDO, chame `fwikit.compat.reativar_avisos()`.
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    try:
        import pyopencl as cl
        warnings.filterwarnings("ignore", category=cl.CompilerWarning)
        from pyopencl import RepeatedKernelRetrieval
        warnings.filterwarnings("ignore", category=RepeatedKernelRetrieval)
    except Exception:
        pass
    import os
    os.environ.setdefault("PYOPENCL_COMPILER_OUTPUT", "0")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")


def reativar_avisos() -> None:
    """Desfaz `silenciar_ruido()` -- util quando algo da errado de verdade."""
    warnings.resetwarnings()
