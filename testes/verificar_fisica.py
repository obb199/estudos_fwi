#!/usr/bin/env python3
"""
Verificacao independente da fisica e da matematica do curso
===========================================================
    python testes/verificar_fisica.py

Este script NAO testa se o codigo roda -- testa se ele esta CERTO. Cada
bloco confronta uma formula usada no curso com sua derivacao analitica ou
com uma referencia independente, e imprime OK ou FALHA.

Rode depois de qualquer alteracao em `fwikit/`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                          # noqa: E402
import fwikit                                               # noqa: E402
from fwikit.acustico import (COEF_D2, Acustico2D, Config,   # noqa: E402
                             Geometria, cfl_limite, ricker,
                             espectro_amplitude, misfit_l2,
                             gradiente_adjunto, dt_maximo)
from fwikit.inversao import tikhonov, variacao_total, modelo_a_priori  # noqa: E402
from fwikit.verificacao import (teste_derivada_direcional,  # noqa: E402
                                teste_taylor, direcao_suave, veredito)

falhas = []


def checa(nome, condicao, detalhe=""):
    marca = "OK   " if condicao else "FALHA"
    if not condicao:
        falhas.append(nome)
    print(f"  [{marca}] {nome}" + (f"   {detalhe}" if detalhe else ""))


def secao(t):
    print(f"\n{'-'*72}\n{t}\n{'-'*72}")


# ==========================================================================
secao("1. Coeficientes de diferencas finitas (contra Taylor exato)")
# o estencil de ordem 2N deve dar  soma_j cj j^p = 2 se p=2, e 0 caso contrario
for ordem in (2, 4, 8):
    c = COEF_D2[ordem]
    N = len(c) - 1
    j = np.arange(-N, N + 1, dtype=float)
    cj = np.array([c[abs(int(k))] for k in j])
    erro = max(abs(np.sum(cj * j ** p) - (2.0 if p == 2 else 0.0))
               for p in range(0, 2 * N + 2))
    checa(f"O({ordem}) anula os momentos de Taylor ate ordem {2*N+1}",
          erro < 1e-10, f"erro max = {erro:.2e}")

# ==========================================================================
secao("2. Limite CFL (von Neumann analitico x implementado)")
_theta = np.linspace(0.0, np.pi, 20001)
for ordem in (2, 4, 8):
    c = COEF_D2[ordem]
    # maximo do simbolo do laplaciano discreto, procurado NUMERICAMENTE em todo
    # k*dh -- e nao so em k*dh = pi, que e o que a formula implementada assume
    simb = -(c[0] + 2 * sum(c[k] * np.cos(k * _theta) for k in range(1, len(c))))
    S = float(simb.max())
    lim = 2.0 / np.sqrt(2 * S)
    checa(f"O({ordem}) C_limite = 2/sqrt(ndim*S)",
          abs(lim - cfl_limite(ordem)) < 1e-12,
          f"{lim:.6f} vs {cfl_limite(ordem):.6f}")

secao("3. CFL: estavel abaixo do limite, instavel acima")
c = np.full((50, 60), 2500.0, dtype=np.float32)
dh = 10.0
dt_lim = cfl_limite(4) * dh / c.max()
for fator, esperado in [(0.5, True), (0.9, True), (1.05, False), (1.2, False)]:
    cfg = Config(dh=dh, dt=fator * dt_lim, nt=350, ordem=4, n_abs=15, f0=15.0)
    sv = Acustico2D(cfg, c.shape)
    g = Geometria(fontes=np.array([[25, 30]]), receptores=np.array([[5, 30]]))
    # o overflow nos casos instaveis E o resultado esperado deste teste
    with np.errstate(over="ignore", invalid="ignore"):
        d, _ = sv.modelar(c, g, ricker(15.0, cfg.dt, cfg.nt))
        estavel = bool(np.isfinite(d).all() and np.abs(d).max() < 1e3)
    checa(f"C = {fator*cfl_limite(4):.3f} -> {'estavel' if esperado else 'instavel'}",
          estavel == esperado)

# ==========================================================================
secao("4. Wavelet de Ricker")
dt = 2e-4
nt = 6000
for f0 in (5.0, 12.0, 25.0, 40.0):
    w = ricker(f0, dt, nt)
    f, S = espectro_amplitude(w, dt)
    fp = f[np.argmax(S)]
    largura_bin = f[1] - f[0]
    checa(f"pico do espectro em f0 = {f0:.0f} Hz",
          abs(fp - f0) <= max(largura_bin, 0.02 * f0), f"medido {fp:.2f} Hz")
w = ricker(20.0, 1e-4, 8000)
checa("integral da wavelet e nula (media zero)",
      abs(np.sum(w) * 1e-4) < 1e-4, f"{np.sum(w)*1e-4:.2e}")
# causalidade: com t0 = 1/f0 a amplitude truncada fica abaixo de 0.1% do pico
f0 = 10.0
t = np.arange(-6 / f0, 6 / f0, 1e-5)
a = (np.pi * f0 * t) ** 2
wv = (1 - 2 * a) * np.exp(-a)
trunc = np.abs(wv[t + 1 / f0 < 0]).max() / np.abs(wv).max()
checa("t0 = 1/f0 trunca menos de 0.1% da amplitude de pico",
      trunc < 1.0e-3, f"{100*trunc:.4f}%")

# ==========================================================================
secao("5. Reducao acustica e relacoes elasticas")
rho, lam = 2200.0, 1.1e10
# K = lam + 2mu/3. A igualdade vp = sqrt(K/rho) vale SO no limite mu -> 0:
# o teste confere os dois lados -- que com mu != 0 elas DIFEREM (senao o teste
# nao testaria nada) e que com mu = 0 elas coincidem e lam passa a SER K.
def _vp(mu):
    return np.sqrt((lam + 2 * mu) / rho)
def _vp_de_K(mu):
    return np.sqrt((lam + 2 * mu / 3) / rho)
_mu = 0.9e10
_difere = abs(_vp(_mu) - _vp_de_K(_mu)) / _vp(_mu) > 0.1
_coincide = (abs(_vp(0.0) - _vp_de_K(0.0)) < 1e-9
             and abs((lam + 2 * 0.0 / 3) - lam) < 1e-9)
checa("vp = sqrt(K/rho) so no limite mu -> 0 (e K -> lam)",
      _difere and _coincide,
      f"mu=9e9: {_vp(_mu):.0f} vs {_vp_de_K(_mu):.0f} m/s; mu=0: iguais")
nu = 0.25
checa("Poisson 0.25 -> vp/vs = sqrt(3)",
      abs(np.sqrt((2 - 2 * nu) / (1 - 2 * nu)) - np.sqrt(3)) < 1e-9)

# ==========================================================================
secao("6. Gradiente adjunto (o teste que mais importa)")
nz, nx, dh = 50, 80, 10.0
c_verd = np.full((nz, nx), 2000.0, dtype=np.float32)
zz, xx = np.mgrid[0:nz, 0:nx]
c_verd[((zz - 26) ** 2 + (xx - 40) ** 2) < 7 ** 2] = 2300.0
c_ini = np.full((nz, nx), 2000.0, dtype=np.float32)
dt = dt_maximo(2400.0, dh, 4)
cfg = Config(dh=dh, dt=dt, nt=int(0.8 / dt), ordem=4, n_abs=25, f0=12.0)
solver = Acustico2D(cfg, (nz, nx))
w = ricker(12.0, dt, cfg.nt)
geom = Geometria(fontes=np.array([[3, 18], [3, 62]]),
                 receptores=np.array([[3, i] for i in range(4, 78, 2)]))
d_obs = np.stack([solver.modelar(c_verd, geom, w, i)[0]
                  for i in range(geom.ns)], axis=-1)


def J_de_m(m):
    cc = (1.0 / np.sqrt(m)).astype(np.float32)
    return sum(misfit_l2(solver.modelar(cc, geom, w, s)[0], d_obs[:, :, s], dt)[0]
               for s in range(geom.ns))


m0 = (1.0 / c_ini ** 2).astype(np.float64)
J0, g_m, _ = gradiente_adjunto(solver, c_ini, geom, w, d_obs, parametro="m")
h = direcao_suave(g_m, m0, 1e-2)
gh, lin = teste_derivada_direcional(J_de_m, m0, g_m, h,
                                    alphas=(0.4, 0.2, 0.1, 0.05, 0.02))
ok, msg = veredito(lin, tol=0.02)
checa("derivada direcional bate com diferencas centradas", ok, msg)

_, _, lt = teste_taylor(J_de_m, m0, g_m, h, alphas=(0.4, 0.2, 0.1, 0.05), J0=J0)
ordens = [L["ordem_E1"] for L in lt if np.isfinite(L["ordem_E1"])]
media = float(np.mean(ordens)) if ordens else 0.0
checa("teste de Taylor: E1 converge em segunda ordem",
      1.85 < media < 2.15, f"ordem media de E1 = {media:.2f}")

# ==========================================================================
secao("7. Gradientes dos termos de regularizacao")
rng = np.random.default_rng(0)
mreg = rng.standard_normal((12, 15)) * 10 + 2000
for nome, fun in [("Tikhonov", lambda x: tikhonov(x, dh=10.0)),
                  ("Variacao Total", lambda x: variacao_total(x, dh=10.0)),
                  ("modelo a priori", lambda x: modelo_a_priori(x, mreg * 0 + 2000.0))]:
    _, g = fun(mreg)
    eps = 1e-3
    pior = 0.0
    for _ in range(8):
        i, j = int(rng.integers(0, 12)), int(rng.integers(0, 15))
        mp = mreg.copy(); mp[i, j] += eps
        mm = mreg.copy(); mm[i, j] -= eps
        gfd = (fun(mp)[0] - fun(mm)[0]) / (2 * eps)
        pior = max(pior, abs(g[i, j] - gfd) / (abs(gfd) + 1e-30))
    checa(f"gradiente de {nome}", pior < 1e-5, f"erro rel. max = {pior:.2e}")

# ==========================================================================
secao("8. Modelo constant-Q de Kjartansson")
for Q in (10.0, 20.0, 50.0, 100.0):
    gamma = np.arctan(1 / Q) / np.pi
    exato = np.tan(np.pi * gamma / 2)
    checa(f"Q={Q:>5.0f}: forma de alto Q 1/(2Q) aproxima tan(pi.gamma/2)",
          abs(exato - 1 / (2 * Q)) / exato < 3e-3,
          f"erro {100*abs(exato-1/(2*Q))/exato:.3f}%")
# "Q ciclos levam a amplitude a e^-pi" NAO e uma identidade trivial: sai da
# formula de atenuacao A = exp(-w L / (2 Q c)) usada na aula 14. Percorrendo
# L = Q comprimentos de onda (L = Q c / f), o expoente vira -pi Q / Q = -pi.
_f, _c, _Q = 20.0, 2000.0, 37.0        # Q nao-redondo, de proposito
_L = _Q * _c / _f                       # Q comprimentos de onda
_A = np.exp(-2 * np.pi * _f * _L / (2 * _Q * _c))
checa("Q ciclos levam a amplitude a e^-pi (via A = exp(-wL/2Qc))",
      abs(_A - np.exp(-np.pi)) < 1e-12,
      f"A = {_A:.6f} vs e^-pi = {np.exp(-np.pi):.6f}")

# ==========================================================================
secao("9. Refracao: distancia critica != distancia de cruzamento")
z1, c1, c2 = 400.0, 1800.0, 2300.0
x_cross = 2 * z1 * np.sqrt((c2 + c1) / (c2 - c1))
x_crit = 2 * z1 * np.tan(np.arcsin(c1 / c2))
checa("as duas grandezas sao mesmo diferentes",
      abs(x_cross - x_crit) > 1000.0,
      f"cruzamento {x_cross:.0f} m, critica {x_crit:.0f} m")

# ==========================================================================
secao("10. Dispersao numerica: medida no propagador x von Neumann")
# Relacao de dispersao do esquema COMPLETO (leap-frog + estencil espacial),
# para propagacao ao longo de um eixo da malha:
#     (2/dt)^2 sin^2(w dt/2) = c^2 S(k dh) / dh^2
# e do esquema SEMI-DISCRETO (tempo continuo):  w^2 = c^2 S(k dh) / dh^2.
# A velocidade de fase do propagador e medida pela diferenca de fase entre
# dois receptores alinhados com a fonte (campo distante: a fase 2D e kr - pi/4
# nos dois, e o -pi/4 se cancela) e confrontada com as duas previsoes.
from scipy.optimize import brentq                           # noqa: E402
from scipy.signal import correlate                          # noqa: E402


def _simbolo(ordem, th):
    cc = COEF_D2[ordem]
    return -(cc[0] + 2 * sum(cc[j] * np.cos(j * th) for j in range(1, len(cc))))


def _c_rel(ordem, f, dh, c0, C=None):
    """c_numerico / c pela relacao de dispersao; C=None -> semi-discreto."""
    w_ = 2 * np.pi * f
    if C is None:
        alvo = (w_ * dh / c0) ** 2
    else:
        alvo = (4.0 / C ** 2) * np.sin(w_ * (C * dh / c0) / 2) ** 2
    th = brentq(lambda t_: _simbolo(ordem, t_) - alvo, 1e-9, np.pi)
    return w_ * dh / (th * c0)


c0, dh = 2000.0, 10.0
for ordem, G in [(4, 6), (8, 5)]:
    C = 0.9 * cfl_limite(ordem)             # o dt que dt_maximo() escolhe
    dt = C * dh / c0
    cfg = Config(dh=dh, dt=dt, nt=int(0.95 / dt), ordem=ordem, n_abs=40, f0=25.0)
    nzd, nxd, r1, r2 = 260, 300, 50, 100
    sv = Acustico2D(cfg, (nzd, nxd))
    gd = Geometria(fontes=np.array([[nzd // 2, 20]]),
                   receptores=np.array([[nzd // 2, 20 + r1], [nzd // 2, 20 + r2]]))
    d = sv.modelar(np.full((nzd, nxd), c0, np.float32), gd,
                   ricker(25.0, dt, cfg.nt))[0].astype(np.float64)
    nfft = 2 ** 16
    freq = np.fft.rfftfreq(nfft, dt)

    def _espectro(tr, t_chegada):
        i0, n = int((t_chegada - 0.08) / dt), int(0.45 / dt)
        X = np.fft.rfft(tr[i0:i0 + n] * np.hanning(n), nfft)
        return X * np.exp(-2j * np.pi * freq * i0 * dt)   # fase relativa a t = 0

    fase = np.unwrap(np.angle(_espectro(d[:, 0], 0.04 + r1 * dh / c0) *
                              np.conj(_espectro(d[:, 1], 0.04 + r2 * dh / c0))))
    i = int(np.argmin(np.abs(freq - c0 / (G * dh))))
    medido = 2 * np.pi * freq[i] * (r2 - r1) * dh / fase[i] / c0
    completo = _c_rel(ordem, freq[i], dh, c0, C)
    semi = _c_rel(ordem, freq[i], dh, c0)
    checa(f"O({ordem}), G={G}, C={C:.3f}: medido bate com o esquema completo",
          abs(medido - completo) < 3e-3,
          f"c/c0 medido {medido:.4f}, completo {completo:.4f}, "
          f"semi-discreto {semi:.4f}")
    if ordem == 8:
        checa("O(8) perto do CFL: a curva so espacial NAO e conservadora",
              abs(completo - 1) > 5 * abs(semi - 1),
              f"erro real {100*abs(completo-1):.2f}% x previsto so pelo "
              f"espaco {100*abs(semi-1):.2f}%")

# ==========================================================================
secao("11. Superficie livre: p = 0 exatamente em z = 0")
# A superficie livre gera um GHOST: reflexao com polaridade invertida, que
# equivale a uma fonte-imagem espelhada acima de z = 0. Com a fonte a 30
# celulas de profundidade e o receptor a 70, a imagem fica a 100 celulas do
# receptor -- SE a superficie estiver mesmo em z = 0. O atraso do ghost contra
# a onda direta de um meio sem superficie a 100 celulas mede onde ela esta.
for ordem in (4, 8):
    dt = 0.3 * cfl_limite(ordem) * dh / c0
    nt = int(0.9 / dt)
    w = ricker(15.0, dt, nt)
    cfg_l = Config(dh=dh, dt=dt, nt=nt, ordem=ordem, n_abs=40, f0=15.0,
                   superficie_livre=True)
    d_livre = Acustico2D(cfg_l, (260, 240)).modelar(
        np.full((260, 240), c0, np.float32),
        Geometria(fontes=np.array([[30, 120]]), receptores=np.array([[70, 120]])),
        w)[0][:, 0].astype(np.float64)
    cfg_i = Config(dh=dh, dt=dt, nt=nt, ordem=ordem, n_abs=40, f0=15.0)
    d_inf = Acustico2D(cfg_i, (500, 240)).modelar(
        np.full((500, 240), c0, np.float32),
        Geometria(fontes=np.array([[230, 120]]),
                  receptores=np.array([[270, 120], [330, 120]])),
        w)[0].astype(np.float64)
    ghost = -(d_livre - d_inf[:, 0])        # isola o ghost e desfaz a inversao
    up = 20
    t_ = np.arange(nt)
    t_fino = np.arange(0, nt - 1, 1 / up)
    xc = correlate(np.interp(t_fino, t_, ghost),
                   np.interp(t_fino, t_, d_inf[:, 1]), mode="full", method="fft")
    atraso = (np.argmax(xc) - (len(t_fino) - 1)) / up * dt
    z_ef = -atraso * c0 / 2                  # posicao efetiva da superficie
    checa(f"O({ordem}): superficie efetiva em z = 0 (tolerancia 0.1 dh)",
          abs(z_ef) < 0.1 * dh, f"z efetivo = {z_ef:+.2f} m ({z_ef/dh:+.2f} dh)")

# ==========================================================================
print(f"\n{'='*72}")
if falhas:
    print(f"  {len(falhas)} VERIFICACAO(OES) FALHOU(RAM):")
    for f_ in falhas:
        print(f"    - {f_}")
    sys.exit(1)
print("  TODAS AS VERIFICACOES PASSARAM")
print(f"{'='*72}\n")
