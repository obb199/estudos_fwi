"""
fwikit.acustico
===============
Propagador acustico 2D em diferencas finitas, escrito do zero em NumPy,
com o operador adjunto e o gradiente pelo metodo do estado adjunto.

POR QUE ESCREVER DO ZERO SE EXISTE PyFWI/Devito?
------------------------------------------------
Porque na sua dissertacao voce vai SUBSTITUIR o operador de propagacao
(acustico -> viscoacustico de Kjartansson) e REESCREVER o adjunto
correspondente. Isso so e possivel se voce souber exatamente onde cada
termo entra. Este modulo e deliberadamente transparente: sem GPU, sem
abstracao, e com mais comentario que codigo. O PyFWI entra depois como implementacao de
producao e benchmark.

FORMULACAO
----------
Equacao da onda acustica, densidade constante, em termos da vagarosidade
ao quadrado m(x) = 1/c(x)^2:

        m(x) d2p/dt2 - laplaciano(p) = s(t) delta(x - xs)          (direto)

Discretizacao:
  * tempo   : diferenca centrada de 2a ordem (leap-frog)
  * espaco  : diferencas centradas de ordem 4 ou 8
  * bordas  : camada absorvente de Cerjan (sponge) -- operador DIAGONAL,
              portanto auto-adjunto, o que preserva a validade do teste
              de gradiente. (O PyFWI usa CPML, mais eficiente e mais
              complexo de adjuntar.)

O PARAMETRO DE INVERSAO
-----------------------
Invertemos em m = 1/c^2 (vagarosidade ao quadrado) porque a equacao e
LINEAR em m. Isso torna o gradiente limpo:

        dJ/dm(x) = soma_tiros integral_t  lambda(x,t) d2p/dt2(x,t) dt

A conversao para velocidade e feita pela regra da cadeia:

        dJ/dc = dJ/dm * dm/dc = dJ/dm * (-2/c^3)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------
# Coeficientes de diferencas finitas centradas para a segunda derivada
#   d2f/dx2 ~ (1/h^2) [ c0 f_i + soma_k ck (f_{i+k} + f_{i-k}) ]
# --------------------------------------------------------------------------
COEF_D2 = {
    2: np.array([-2.0, 1.0]),
    4: np.array([-5.0 / 2.0, 4.0 / 3.0, -1.0 / 12.0]),
    8: np.array([-205.0 / 72.0, 8.0 / 5.0, -1.0 / 5.0, 8.0 / 315.0, -1.0 / 560.0]),
}


# --------------------------------------------------------------------------
# Fonte
# --------------------------------------------------------------------------
def ricker(f0: float, dt: float, nt: int, atraso: float | None = None) -> np.ndarray:
    """
    Wavelet de Ricker (segunda derivada da gaussiana), amostrada em `nt`
    pontos com passo `dt`.

        w(t) = (1 - 2 (pi f0 (t-t0))^2) exp(-(pi f0 (t-t0))^2)

    O atraso padrao t0 = 1/f0 garante que a wavelet comece praticamente
    em zero (causalidade numerica). Frequencia de pico = f0;
    frequencia maxima util ~ 2.5 f0 (ver aula 01).
    """
    t0 = atraso if atraso is not None else 1.0 / f0
    t = np.arange(nt) * dt - t0
    a = (np.pi * f0 * t) ** 2
    return ((1.0 - 2.0 * a) * np.exp(-a)).astype(np.float32)


def espectro_amplitude(w: np.ndarray, dt: float):
    """Retorna (frequencias, amplitude normalizada) de um sinal 1D."""
    nfft = int(2 ** np.ceil(np.log2(len(w) * 4)))
    S = np.abs(np.fft.rfft(w, nfft))
    f = np.fft.rfftfreq(nfft, dt)
    return f, S / (S.max() + 1e-30)


def largura_banda(w: np.ndarray, dt: float, queda_db: float = -6.0):
    """Frequencias inferior/superior na queda de `queda_db` do pico."""
    f, S = espectro_amplitude(w, dt)
    S_db = 20 * np.log10(S + 1e-12)
    acima = np.where(S_db >= queda_db)[0]
    if acima.size == 0:
        return 0.0, 0.0
    return float(f[acima[0]]), float(f[acima[-1]])


# --------------------------------------------------------------------------
# Configuracao do modelo/propagacao
# --------------------------------------------------------------------------
@dataclass
class Config:
    """Todos os parametros numericos de uma simulacao, em um lugar so."""
    dh: float = 10.0            # espacamento da malha (m), isotropico
    dt: float = 1.0e-3          # passo de tempo (s)
    nt: int = 1000              # numero de amostras no tempo
    ordem: int = 4              # ordem espacial (2, 4 ou 8)
    n_abs: int = 30             # espessura da camada absorvente (pontos)
    fator_abs: float | None = None   # intensidade do sponge; None = automatico
    superficie_livre: bool = False   # p = 0 no topo (mar) ou absorvente
    f0: float = 10.0            # frequencia de pico da fonte (Hz)

    @property
    def fator_absorcao(self) -> float:
        """
        Intensidade do taper de Cerjan.

        Existe um OTIMO: taper fraco demais nao absorve; taper forte demais
        cria um degrau de impedancia que ele mesmo reflete. Calibrado
        numericamente neste curso (ver aula 03), o otimo fica em torno de

            fator ~ 0.25 / n_abs

        que entrega -16.5 dB de reflexao com n_abs = 30 e -18.4 dB com
        n_abs = 45 (medido na aula 03). Para comparacao, a CPML do PyFWI
        chega a -51 dB com 10 pontos e -70 dB com 20 (medido na aula 06).
        E por isso que codigos de producao usam CPML.
        """
        if self.fator_abs is not None:
            return self.fator_abs
        return 0.25 / max(self.n_abs, 1)

    @property
    def t(self) -> np.ndarray:
        return np.arange(self.nt) * self.dt

    @property
    def duracao(self) -> float:
        return (self.nt - 1) * self.dt


@dataclass
class Geometria:
    """Posicoes de fontes e receptores em INDICES da malha fisica (iz, ix)."""
    fontes: np.ndarray            # (ns, 2) -> [iz, ix]
    receptores: np.ndarray        # (nr, 2) -> [iz, ix]

    @property
    def ns(self) -> int:
        return len(self.fontes)

    @property
    def nr(self) -> int:
        return len(self.receptores)


# --------------------------------------------------------------------------
# Estabilidade e dispersao
# --------------------------------------------------------------------------
def cfl(c_max: float, dt: float, dh: float, ordem: int = 4, ndim: int = 2) -> float:
    """
    Numero de Courant  C = c_max * dt / dh.

    A simulacao e estavel enquanto C <= `cfl_limite(ordem, ndim)`.
    """
    return c_max * dt / dh


def cfl_limite(ordem: int = 4, ndim: int = 2) -> float:
    """
    Limite de estabilidade de von Neumann para o esquema leap-frog.

    Substituindo uma onda plana no esquema discreto, o fator de amplificacao
    so tem modulo 1 enquanto  c^2 dt^2 Lambda_max <= 4, onde Lambda_max e o
    maior autovalor do laplaciano discreto. Esse maximo ocorre em k*dh = pi e
    vale  ndim * S / dh^2  com  S = |c0| + 2 soma|ck|.  Logo

        C_limite = 2 / sqrt(ndim * S).

    Da 0.7071 (O2), 0.6124 (O4) e 0.5546 (O8) em 2D -- ordem espacial MAIOR
    aperta o limite, porque o estencil mais largo amplifica mais os numeros
    de onda altos.
    """
    c = COEF_D2[ordem]
    soma = abs(c[0]) + 2 * np.abs(c[1:]).sum()
    return float(2.0 / np.sqrt(ndim * soma))


def pontos_por_comprimento_onda(c_min: float, f_max: float, dh: float) -> float:
    """G = lambda_min / dh. Regra pratica: G >= 5 (O8) ou G >= 8..10 (O4)."""
    return c_min / (f_max * dh)


def dt_maximo(c_max: float, dh: float, ordem: int = 4, seguranca: float = 0.9) -> float:
    """Maior dt estavel, com margem de seguranca."""
    return seguranca * cfl_limite(ordem) * dh / c_max


# --------------------------------------------------------------------------
# Nucleo: laplaciano e passo no tempo
# --------------------------------------------------------------------------
def _laplaciano(p: np.ndarray, dh: float, coef: np.ndarray,
                saida: np.ndarray) -> np.ndarray:
    """Laplaciano 2D por diferencas centradas. Escreve em `saida` (in-place)."""
    N = len(coef) - 1
    nz, nx = p.shape
    saida[:] = 0.0
    acc = 2.0 * coef[0] * p[N:nz - N, N:nx - N]
    for k in range(1, N + 1):
        acc = acc + coef[k] * (p[N + k:nz - N + k, N:nx - N] +
                               p[N - k:nz - N - k, N:nx - N] +
                               p[N:nz - N, N + k:nx - N + k] +
                               p[N:nz - N, N - k:nx - N - k])
    saida[N:nz - N, N:nx - N] = acc / (dh * dh)
    return saida


def _mascara_sponge(nz: int, nx: int, n_abs: int, fator: float,
                    superficie_livre: bool) -> np.ndarray:
    """
    Camada absorvente de Cerjan (1985): multiplicador gaussiano em (0,1]
    aplicado ao campo a cada passo dentro da moldura absorvente.

    E um operador DIAGONAL -> auto-adjunto -> nao quebra o teste de
    gradiente (ao contrario de uma CPML implementada sem cuidado).
    """
    m = np.ones((nz, nx), dtype=np.float32)
    if n_abs <= 0:
        return m
    d = np.arange(n_abs, 0, -1, dtype=np.float32)
    taper = np.exp(-((fator * d) ** 2)).astype(np.float32)
    for i, v in enumerate(taper):
        m[:, i] = np.minimum(m[:, i], v)           # esquerda
        m[:, nx - 1 - i] = np.minimum(m[:, nx - 1 - i], v)   # direita
        m[nz - 1 - i, :] = np.minimum(m[nz - 1 - i, :], v)   # fundo
        if not superficie_livre:
            m[i, :] = np.minimum(m[i, :], v)       # topo
    return m


class Acustico2D:
    """
    Propagador acustico 2D com camada absorvente.

    O modelo fisico tem forma (nz, nx). Internamente o solver trabalha num
    dominio ESTENDIDO com `n_abs` pontos de moldura absorvente -- exatamente
    o papel do `npml` no PyFWI.
    """

    def __init__(self, cfg: Config, forma_fisica: tuple[int, int]):
        self.cfg = cfg
        self.nz, self.nx = forma_fisica
        n = cfg.n_abs
        self.topo = 0 if cfg.superficie_livre else n
        self.nze = self.nz + self.topo + n
        self.nxe = self.nx + 2 * n
        self.coef = COEF_D2[cfg.ordem]
        self.sponge = _mascara_sponge(self.nze, self.nxe, n, cfg.fator_absorcao,
                                      cfg.superficie_livre)

    # ---------------------------------------------------------------- malha
    def expandir(self, campo: np.ndarray) -> np.ndarray:
        """Estende o modelo fisico para o dominio com moldura absorvente."""
        n = self.cfg.n_abs
        return np.pad(campo, ((self.topo, n), (n, n)), mode="edge").astype(np.float32)

    def contrair(self, campo: np.ndarray) -> np.ndarray:
        """Recorta o dominio estendido de volta ao modelo fisico."""
        n = self.cfg.n_abs
        return campo[self.topo:self.topo + self.nz, n:n + self.nx]

    def contrair_adjunto(self, campo: np.ndarray) -> np.ndarray:
        """
        ADJUNTO de `expandir`.

        `expandir` usa mode="edge": cada pixel da borda do modelo fisico e
        REPLICADO por toda a moldura absorvente. Isso e um operador linear E.
        O gradiente precisa de E^T, nao de um simples recorte: a sensibilidade
        acumulada dentro da moldura tem de ser DOBRADA de volta sobre o pixel
        de borda que a gerou.

        Esquecer isso e um dos erros mais comuns -- e mais dificeis de achar --
        em implementacoes de FWI. O sintoma e um gradiente que passa no teste
        de diferencas finitas no MIOLO do modelo mas erra sistematicamente
        perto das bordas e da superficie (justamente onde ficam fontes e
        receptores). No nosso caso isso valia ~10% de erro na derivada
        direcional. Ver aula 09.
        """
        n = self.cfg.n_abs
        g = np.array(campo, dtype=np.float64, copy=True)
        # dobra topo e fundo
        if self.topo > 0:
            g[self.topo] += g[:self.topo].sum(axis=0)
        g[self.topo + self.nz - 1] += g[self.topo + self.nz:].sum(axis=0)
        g = g[self.topo:self.topo + self.nz]
        # dobra esquerda e direita
        g[:, n] += g[:, :n].sum(axis=1)
        g[:, n + self.nx - 1] += g[:, n + self.nx:].sum(axis=1)
        return g[:, n:n + self.nx]

    def indice(self, iz: int, ix: int) -> tuple[int, int]:
        """Converte indice fisico -> indice no dominio estendido."""
        return iz + self.topo, ix + self.cfg.n_abs

    # ------------------------------------------------------------- direto
    def _propagar(self, c: np.ndarray, pos_fonte, sinais: np.ndarray,
                  rec_iz=None, rec_ix=None, guardar: str | None = None,
                  snapshots: list[int] | None = None):
        """
        NUCLEO da propagacao -- usado tanto pelo problema DIRETO quanto pelo
        ADJUNTO. Um unico laco no tempo, escrito uma vez so.

        `pos_fonte` : lista de (iz, ix) JA no dominio estendido
        `sinais`    : (nt, n_fontes) series temporais injetadas
        `guardar`   : None | 'p' (campo de pressao) | 'dtt' (d2p/dt2)

        Convencao de indices (importante para o adjunto ser exato):
          no passo `it` o laco le p^it, grava o dado p^it, calcula
          d2p/dt2 em `it` e produz p^(it+1).
        """
        cfg = self.cfg
        ce = self.expandir(c)
        c2 = (ce ** 2).astype(np.float32)
        c2dt2 = (c2 * cfg.dt ** 2).astype(np.float32)

        p_ant = np.zeros((self.nze, self.nxe), dtype=np.float32)
        p_atu = np.zeros_like(p_ant)
        p_pro = np.zeros_like(p_ant)
        lap = np.zeros_like(p_ant)

        fz = np.asarray([q[0] for q in pos_fonte], dtype=np.intp)
        fx = np.asarray([q[1] for q in pos_fonte], dtype=np.intp)
        grava_dado = rec_iz is not None
        dados = np.zeros((cfg.nt, len(rec_iz)), dtype=np.float32) \
            if grava_dado else None
        campo = np.zeros((cfg.nt, self.nze, self.nxe), dtype=np.float32) \
            if guardar else None
        snaps = {}
        alvo = set(snapshots or [])
        esc = np.float32(1.0 / (cfg.dh ** 2))

        for it in range(cfg.nt):
            if grava_dado:
                dados[it] = p_atu[rec_iz, rec_ix]
            _laplaciano(p_atu, cfg.dh, self.coef, lap)
            if it < len(sinais):
                np.add.at(lap, (fz, fx), sinais[it] * esc)
            if guardar == "dtt":
                campo[it] = c2 * lap          # d2p/dt2 no instante `it`
            elif guardar == "p":
                campo[it] = p_atu
            if it in alvo:
                snaps[it] = p_atu.copy()
            p_pro[:] = 2.0 * p_atu - p_ant + c2dt2 * lap
            if cfg.superficie_livre:
                p_pro[0, :] = 0.0
            p_pro *= self.sponge
            p_ant, p_atu, p_pro = p_atu, p_pro, p_ant

        return dados, campo, snaps

    def modelar(self, c: np.ndarray, geom: Geometria, wavelet: np.ndarray,
                i_fonte: int = 0, guardar_campo: bool = False,
                snapshots: list[int] | None = None):
        """
        Modelagem direta para UM tiro.

        Parametros
        ----------
        c        : (nz, nx) velocidade em m/s
        geom     : geometria de aquisicao
        wavelet  : (nt,) assinatura da fonte
        i_fonte  : indice do tiro em `geom.fontes`
        guardar_campo : se True, guarda d2p/dt2 (necessario para o gradiente)
        snapshots: indices de tempo para guardar o campo de pressao completo

        Retorna
        -------
        dados (nt, nr), extras (dict com 'dtt' e/ou 'snapshots')
        """
        sz, sx = self.indice(*geom.fontes[i_fonte])
        rz = np.array([self.indice(*r)[0] for r in geom.receptores], dtype=np.intp)
        rx = np.array([self.indice(*r)[1] for r in geom.receptores], dtype=np.intp)
        sinais = np.asarray(wavelet, dtype=np.float32).reshape(-1, 1)
        dados, campo, snaps = self._propagar(
            c, [(sz, sx)], sinais, rz, rx,
            guardar="dtt" if guardar_campo else None, snapshots=snapshots)
        extras = {}
        if campo is not None:
            extras["dtt"] = campo
        if snaps:
            extras["snapshots"] = snaps
        return dados, extras

    def campo_de_onda(self, c: np.ndarray, geom: Geometria, wavelet: np.ndarray,
                      i_fonte: int = 0, snapshots: list[int] | None = None):
        """Igual a `modelar`, mas devolve tambem instantaneos do campo p."""
        return self.modelar(c, geom, wavelet, i_fonte, False, snapshots)

    # ------------------------------------------------------------- adjunto
    def retropropagar(self, c: np.ndarray, geom: Geometria,
                      residuo: np.ndarray) -> np.ndarray:
        """
        Propagacao ADJUNTA.

        A equacao adjunta da onda acustica e:

            m d2(lambda)/dt2 - laplaciano(lambda) = - soma_r r(t) delta(x-x_r)

        com CONDICOES FINAIS nulas (lambda = 0 em t = T). Ou seja: a mesma
        equacao de onda, com o RESIDUO COM SINAL TROCADO injetado em todos os
        receptores ao mesmo tempo, resolvida de tras para frente.

        Implementacao: como o operador e auto-adjunto (invariante por reversao
        temporal), reutilizamos EXATAMENTE o mesmo laco `_propagar`, injetando
        o residuo invertido no tempo, e depois desinvertemos o campo.

        GUARDE ESTE PONTO PARA A DISSERTACAO: essa simetria vale porque nao ha
        dissipacao. No operador viscoacustico de Kjartansson o adjunto do termo
        atenuante e um termo que AMPLIFICA, e o campo adjunto deixa de ser uma
        simples reversao temporal. E por isso que a Fase IV do seu projeto e
        uma etapa propria.

        Retorna lambda(t, z, x) no dominio estendido, indexado no tempo direto.
        """
        rz = [self.indice(*r)[0] for r in geom.receptores]
        rx = [self.indice(*r)[1] for r in geom.receptores]
        fontes_adj = list(zip(rz, rx))
        sinais = (-np.asarray(residuo, dtype=np.float32))[::-1].copy()
        _, campo, _ = self._propagar(c, fontes_adj, sinais, guardar="p")
        return campo[::-1]        # volta ao tempo direto


# --------------------------------------------------------------------------
# Funcao objetivo e gradiente
# --------------------------------------------------------------------------
def misfit_l2(d_calc: np.ndarray, d_obs: np.ndarray, dt: float = 1.0):
    """
    Funcional de minimos quadrados e residuo.

        J(m) = 1/2 * soma_r integral_t (d_calc - d_obs)^2 dt

    Retorna (J, residuo).
    """
    r = (d_calc - d_obs).astype(np.float32)
    return 0.5 * float(np.sum(r ** 2)) * dt, r


def gradiente_adjunto(solver: Acustico2D, c: np.ndarray, geom: Geometria,
                      wavelet: np.ndarray, d_obs: np.ndarray,
                      parametro: str = "c", tiros=None):
    """
    Gradiente de J pelo metodo do estado adjunto, somado sobre os tiros.

        dJ/dm(x) = soma_tiros integral_t lambda(x,t) d2p/dt2(x,t) dt

    `d_obs` tem forma (nt, nr, ns) ou (nt, nr) se houver um tiro so.

    Retorna (J, gradiente (nz,nx), residuos)
    """
    if d_obs.ndim == 2:
        d_obs = d_obs[:, :, None]
    tiros = range(geom.ns) if tiros is None else tiros
    dt = solver.cfg.dt

    J = 0.0
    g_ext = np.zeros((solver.nze, solver.nxe), dtype=np.float64)
    residuos = np.zeros_like(d_obs)

    for s in tiros:
        d_calc, extras = solver.modelar(c, geom, wavelet, i_fonte=s,
                                        guardar_campo=True)
        Js, r = misfit_l2(d_calc, d_obs[:, :, s], dt)
        J += Js
        residuos[:, :, s] = r
        lam = solver.retropropagar(c, geom, r)
        # correlacao cruzada no tempo entre campo adjunto e d2p/dt2
        g_ext += np.einsum("tzx,tzx->zx", lam, extras["dtt"],
                           optimize=True) * (dt * solver.cfg.dh ** 2)

    # E^T: dobra a moldura absorvente de volta sobre as bordas do modelo
    g_m = solver.contrair_adjunto(g_ext)   # gradiente em m = 1/c^2
    if parametro == "m":
        return J, g_m, residuos
    # regra da cadeia para velocidade: dm/dc = -2/c^3
    g_c = g_m * (-2.0 / c ** 3)
    return J, g_c, residuos
