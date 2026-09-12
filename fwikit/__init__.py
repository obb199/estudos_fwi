"""
fwikit -- caixa de ferramentas do Curso de FWI com PyFWI
========================================================

Importar este pacote ja:
  1. aplica os remendos de compatibilidade do PyFWI (numpy 2 / scipy 1.17);
  2. silencia o ruido de warnings do pyopencl.

Modulos
-------
  fwikit.aula            framework didatico de terminal (secoes, quiz, pausas)
  fwikit.plot            visualizacao padronizada (modelos, sismogramas, espectros)
  fwikit.acustico        propagador acustico 2D + estado adjunto (do zero)
  fwikit.inversao        otimizadores (descida maxima, CG, l-BFGS) e busca linear
  fwikit.verificacao     testes de gradiente (dif. finitas, direcional, Taylor)
  fwikit.metricas        metricas de avaliacao de modelos e dados
  fwikit.compat          camada de compatibilidade do PyFWI
"""
from fwikit import compat as _compat

_compat.aplicar()
_compat.silenciar_ruido()

__version__ = "1.0"
__all__ = ["aula", "plot", "acustico", "inversao",
           "metricas", "verificacao", "compat"]
