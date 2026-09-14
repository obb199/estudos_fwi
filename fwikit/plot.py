"""
fwikit.plot
===========
Utilitarios de visualizacao com uma convencao unica para todo o curso:

  * toda figura e SEMPRE salva em PNG dentro de `saidas/aulaXX/`;
  * se houver display disponivel e a aula nao estiver em `--rapido`,
    a janela tambem e aberta (inclusive as interativas com sliders).

Convencoes de exibicao usadas no curso inteiro:
  * modelos de velocidade: eixo z para baixo, colormap perceptual
  * sismogramas: tempo para baixo, cinza simetrico com clip de percentil
  * gradientes / kernels: colormap divergente centrado em zero
"""
from __future__ import annotations

import os

import matplotlib
import numpy as np

from fwikit.aula import mostrar_figuras

if not mostrar_figuras():
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
})


# --------------------------------------------------------------------------
def salvar(fig, aula, nome: str, mostrar: bool | None = None) -> str:
    """Salva a figura em `saidas/aulaXX/nome.png` e opcionalmente exibe."""
    caminho = os.path.join(aula.dir_saida, f"{nome}.png")
    fig.savefig(caminho, bbox_inches="tight")
    print(f"    [figura] {caminho}")
    if mostrar is None:
        mostrar = mostrar_figuras()
    if mostrar:
        plt.show()
    else:
        plt.close(fig)
    return caminho


def _clip(d: np.ndarray, p: float = 99.0) -> float:
    v = np.percentile(np.abs(d[np.isfinite(d)]), p)
    return float(v) if v > 0 else 1.0


# --------------------------------------------------------------------------
def modelo(ax, m: np.ndarray, dh: float, titulo: str = "",
           unidade: str = "m/s", cmap: str = "viridis",
           vmin=None, vmax=None, colorbar: bool = True):
    """Plota um modelo 2D (nz, nx) com eixos em metros e z para baixo."""
    nz, nx = m.shape
    im = ax.imshow(m, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax,
                   extent=[0, nx * dh, nz * dh, 0], interpolation="bilinear")
    ax.set_xlabel("distancia x (m)")
    ax.set_ylabel("profundidade z (m)")
    if titulo:
        ax.set_title(titulo)
    ax.grid(False)
    if colorbar:
        cb = ax.figure.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
        cb.set_label(unidade)
    return im


def perturbacao(ax, dm: np.ndarray, dh: float, titulo: str = "",
                unidade: str = "", simetrico: bool = True, colorbar: bool = True):
    """Plota um campo assinalado (gradiente, diferenca) com cmap divergente."""
    lim = _clip(dm, 99.5) if simetrico else None
    im = ax.imshow(dm, cmap="RdBu_r", aspect="auto",
                   vmin=-lim if lim else None, vmax=lim,
                   extent=[0, dm.shape[1] * dh, dm.shape[0] * dh, 0],
                   interpolation="bilinear")
    ax.set_xlabel("distancia x (m)")
    ax.set_ylabel("profundidade z (m)")
    if titulo:
        ax.set_title(titulo)
    ax.grid(False)
    if colorbar:
        cb = ax.figure.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
        if unidade:
            cb.set_label(unidade)
    return im


def sismograma(ax, d: np.ndarray, dt: float, titulo: str = "",
               dx_rec: float | None = None, perc: float = 98.0,
               cmap: str = "gray"):
    """Plota um sismograma (nt, nr): tempo para baixo."""
    nt, nr = d.shape
    lim = _clip(d, perc)
    ext = [0, nr * (dx_rec or 1.0), nt * dt, 0]
    im = ax.imshow(d, cmap=cmap, aspect="auto", vmin=-lim, vmax=lim,
                   extent=ext, interpolation="bilinear")
    ax.set_xlabel("posicao ao longo do arranjo (m)" if dx_rec else "receptor")
    ax.set_ylabel("tempo (s)")
    if titulo:
        ax.set_title(titulo)
    ax.grid(False)
    return im


def tracos(ax, d: np.ndarray, dt: float, indices=None, escala: float = 1.0,
           titulo: str = "", cor="k", rotulo=None):
    """Wiggle plot simples de alguns tracos."""
    nt, nr = d.shape
    t = np.arange(nt) * dt
    indices = indices if indices is not None else range(0, nr, max(1, nr // 12))
    amp = escala * _clip(d, 99.0)
    for k, i in enumerate(indices):
        x = i + d[:, i] / (amp + 1e-30)
        ax.plot(x, t, lw=0.7, color=cor,
                label=rotulo if (rotulo and k == 0) else None)
        ax.fill_betweenx(t, i, x, where=(x > i), color=cor, alpha=0.35, lw=0)
    ax.set_ylim(t[-1], 0)
    ax.set_xlabel("receptor")
    ax.set_ylabel("tempo (s)")
    if titulo:
        ax.set_title(titulo)
    return ax


def espectro(ax, sinal: np.ndarray, dt: float, rotulo: str = "",
             fmax: float | None = None, db: bool = False, **kw):
    """Espectro de amplitude de um traco 1D."""
    n = len(sinal)
    nfft = int(2 ** np.ceil(np.log2(max(n, 2) * 4)))
    S = np.abs(np.fft.rfft(sinal, nfft))
    f = np.fft.rfftfreq(nfft, dt)
    if db:
        S = 20 * np.log10(S / (S.max() + 1e-30) + 1e-12)
    else:
        S = S / (S.max() + 1e-30)
    ax.plot(f, S, label=rotulo, **kw)
    ax.set_xlim(0, fmax or min(f[-1], 120))
    ax.set_xlabel("frequencia (Hz)")
    ax.set_ylabel("amplitude (dB)" if db else "amplitude normalizada")
    return f, S
