"""
fwikit.inversao
===============
Algoritmos de otimizacao para FWI, escritos do zero e comentados.

A FWI e um problema de otimizacao nao linear de grande porte:

        m* = argmin_m  J(m) = 1/2 || F(m) - d_obs ||^2

com m tendo 10^4-10^8 incognitas. Isso exclui qualquer metodo que precise
montar a Hessiana. Sobram os metodos de primeira ordem e quase-Newton, que
usam SO o gradiente:

    descida maxima   ->  d = -g                      (simples, lento)
    grad. conjugado  ->  d = -g + beta d_ant         (bom custo/beneficio)
    l-BFGS           ->  d = -H_aprox g              (padrao de mercado)

Todos precisam de uma BUSCA LINEAR: dado d, achar o passo alpha que
reduz J o suficiente. Em FWI a busca linear e cara (cada tentativa = uma
modelagem direta por tiro), entao a qualidade do criterio importa muito.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Historico:
    """Registro da convergencia, para as curvas que vao na dissertacao."""
    J: list = field(default_factory=list)
    passo: list = field(default_factory=list)
    norma_grad: list = field(default_factory=list)
    avaliacoes: int = 0
    modelos: list = field(default_factory=list)

    def registrar(self, J, alpha, g, m=None, guardar_modelo=False):
        self.J.append(float(J))
        self.passo.append(float(alpha))
        self.norma_grad.append(float(np.linalg.norm(g)))
        if guardar_modelo and m is not None:
            self.modelos.append(np.array(m, copy=True))

    @property
    def J_normalizado(self):
        return np.array(self.J) / (self.J[0] + 1e-30)


# --------------------------------------------------------------------------
# Busca linear
# --------------------------------------------------------------------------
def busca_linear_armijo(f, m, J0, g, d, alpha0: float,
                        c1: float = 1e-4, reducao: float = 0.5,
                        max_tent: int = 12):
    """
    Busca linear por retrocesso (backtracking) com a condicao de Armijo:

        J(m + alpha d) <= J(m) + c1 * alpha * <g, d>

    Ou seja: aceite o passo se ele reduziu J pelo menos uma fracao c1 do
    que a aproximacao linear prometia. E o criterio mais barato que ainda
    garante convergencia. Cada tentativa custa UMA modelagem direta
    completa (todos os tiros) -- por isso `max_tent` e pequeno.

    Retorna (alpha, J_novo, n_avaliacoes). alpha=0 significa falha.
    """
    inclinacao = float(np.dot(g.ravel(), d.ravel()))
    if inclinacao >= 0:
        # d nao e direcao de descida (pode acontecer no CG/l-BFGS)
        return 0.0, J0, 0
    alpha = alpha0
    for k in range(max_tent):
        J_novo = f(m + alpha * d)
        if J_novo <= J0 + c1 * alpha * inclinacao:
            return alpha, J_novo, k + 1
        alpha *= reducao
    return 0.0, J0, max_tent


def busca_linear_parabolica(f, m, J0, g, d, alpha0: float,
                            c1: float = 1e-4, max_tent: int = 8):
    """
    Busca linear por AJUSTE PARABOLICO -- o padrao pratico em FWI.

    Ja conhecemos J(0) = J0 e a inclinacao J'(0) = <g,d>. Com UMA avaliacao
    extra, J(alpha0), fica determinada a parabola

        q(a) = J0 + <g,d> a + c a^2,    c = (J(a0) - J0 - <g,d> a0) / a0^2

    cujo minimo esta em  a* = -<g,d> / (2c)  (se c > 0, isto e, se a
    parabola abre para cima). Isso costuma acertar o passo em 2 avaliacoes,
    contra 5-10 do retrocesso puro. Em FWI cada avaliacao e uma modelagem
    direta de TODOS os tiros, entao a economia e enorme.

    Se a parabola nao ajudar, caimos no retrocesso de Armijo.
    """
    incl = float(np.dot(g.ravel(), d.ravel()))
    if incl >= 0:
        return 0.0, J0, 0
    n_aval = 0
    J1 = f(m + alpha0 * d)
    n_aval += 1
    candidatos = [(alpha0, J1)]

    c = (J1 - J0 - incl * alpha0) / (alpha0 ** 2)
    if c > 0:
        a_est = -incl / (2.0 * c)
        a_est = float(np.clip(a_est, 0.05 * alpha0, 20.0 * alpha0))
        if abs(a_est - alpha0) > 1e-3 * alpha0:
            J2 = f(m + a_est * d)
            n_aval += 1
            candidatos.append((a_est, J2))

    validos = [(a, J) for a, J in candidatos
               if np.isfinite(J) and J <= J0 + c1 * a * incl]
    if validos:
        a, J = min(validos, key=lambda p: p[1])
        return a, J, n_aval

    # plano B: retrocesso
    alpha = alpha0 * 0.5
    for k in range(max_tent):
        J_novo = f(m + alpha * d)
        n_aval += 1
        if np.isfinite(J_novo) and J_novo <= J0 + c1 * alpha * incl:
            return alpha, J_novo, n_aval
        alpha *= 0.5
    return 0.0, J0, n_aval


def passo_inicial_por_escala(m, d, J0=None, g=None,
                             fracao: float = 0.02) -> float:
    """
    Chute inicial do passo. Em FWI isso importa MUITO: um alpha0 ruim gasta
    toda a busca linear em tentativas inuteis.

    Combinamos dois criterios e ficamos com o mais conservador:

    1. ESCALA DO MODELO -- o passo que altera o modelo em `fracao` do seu
       valor tipico (regra pratica: 1-3% por iteracao). E o criterio
       preferido em FWI, porque tem significado fisico: "nao mexa mais do
       que 2% na velocidade de uma vez".

    2. MODELO QUADRATICO -- alpha = 2 J0 / |<g,d>|, o passo que zeraria J
       se ele fosse linear. Serve de teto e de plano B quando o modelo
       ainda e nulo (caso de testes analiticos).
    """
    esc_d = float(np.max(np.abs(d)))
    if esc_d <= 0 or not np.isfinite(esc_d):
        return 1.0
    a_modelo = fracao * float(np.max(np.abs(m))) / esc_d

    a_quad = None
    if J0 is not None and g is not None:
        incl = abs(float(np.dot(g.ravel(), d.ravel())))
        if incl > 0 and np.isfinite(incl):
            a_quad = 2.0 * abs(J0) / incl

    if a_modelo <= 0 or not np.isfinite(a_modelo):
        return a_quad if a_quad else 1.0
    return min(a_modelo, a_quad) if a_quad else a_modelo


# --------------------------------------------------------------------------
# Otimizadores
# --------------------------------------------------------------------------
def descida_maxima(f_e_g, m0, n_iter: int = 20, fracao_passo: float = 0.02,
                   verboso: bool = True, guardar_modelos: bool = False,
                   projecao=None, busca_linha=busca_linear_parabolica):
    """
    Descida maxima (steepest descent): d = -g.

    Simples e robusto, mas converge devagar quando a Hessiana e mal
    condicionada -- que e SEMPRE o caso em FWI (a iluminacao decai com a
    profundidade, entao os autovalores variam por ordens de grandeza).
    Serve de linha de base para mostrar o ganho do CG e do l-BFGS.
    """
    m = np.array(m0, dtype=np.float64, copy=True)
    hist = Historico()
    for it in range(n_iter):
        J, g = f_e_g(m)
        hist.avaliacoes += 1
        d = -g
        a0 = passo_inicial_por_escala(m, d, J, g, fracao_passo)
        alpha, J_novo, nav = busca_linha(
            lambda x: f_e_g(x, so_J=True), m, J, g, d, a0)
        hist.avaliacoes += nav
        hist.registrar(J, alpha, g, m, guardar_modelos)
        if verboso:
            print(f"    it {it:3d}  J = {J:.6e}   |g| = {np.linalg.norm(g):.3e}"
                  f"   alpha = {alpha:.3e}")
        if alpha == 0.0:
            if verboso:
                print("    busca linear falhou -- parando")
            break
        m = m + alpha * d
        if projecao is not None:
            m = projecao(m)
    J, g = f_e_g(m)
    hist.registrar(J, 0.0, g, m, guardar_modelos)
    return m, hist


def gradiente_conjugado(f_e_g, m0, n_iter: int = 20, fracao_passo: float = 0.02,
                        verboso: bool = True, guardar_modelos: bool = False,
                        projecao=None, busca_linha=busca_linear_parabolica):
    """
    Gradiente conjugado nao linear (Polak-Ribiere com reinicio).

        beta = max(0, <g_k, g_k - g_{k-1}> / <g_{k-1}, g_{k-1}>)
        d_k  = -g_k + beta * d_{k-1}

    O `max(0, .)` faz o reinicio automatico: quando beta ficaria negativo,
    o metodo volta a ser descida maxima. Sem isso o CG nao linear pode
    gerar direcoes de subida.
    """
    m = np.array(m0, dtype=np.float64, copy=True)
    hist = Historico()
    g_ant = d_ant = None
    for it in range(n_iter):
        J, g = f_e_g(m)
        hist.avaliacoes += 1
        if g_ant is None:
            d = -g
        else:
            num = float(np.dot(g.ravel(), (g - g_ant).ravel()))
            den = float(np.dot(g_ant.ravel(), g_ant.ravel())) + 1e-30
            beta = max(0.0, num / den)
            d = -g + beta * d_ant
            if float(np.dot(g.ravel(), d.ravel())) >= 0:
                d = -g                       # seguranca: reinicia
        a0 = passo_inicial_por_escala(m, d, J, g, fracao_passo)
        alpha, J_novo, nav = busca_linha(
            lambda x: f_e_g(x, so_J=True), m, J, g, d, a0)
        hist.avaliacoes += nav
        hist.registrar(J, alpha, g, m, guardar_modelos)
        if verboso:
            print(f"    it {it:3d}  J = {J:.6e}   |g| = {np.linalg.norm(g):.3e}"
                  f"   alpha = {alpha:.3e}")
        if alpha == 0.0:
            if verboso:
                print("    busca linear falhou -- parando")
            break
        m = m + alpha * d
        if projecao is not None:
            m = projecao(m)
        g_ant, d_ant = g, d
    J, g = f_e_g(m)
    hist.registrar(J, 0.0, g, m, guardar_modelos)
    return m, hist


def lbfgs(f_e_g, m0, n_iter: int = 20, memoria: int = 8,
          fracao_passo: float = 0.02, verboso: bool = True,
          guardar_modelos: bool = False, projecao=None,
          busca_linha=busca_linear_parabolica):
    """
    l-BFGS com recursao de dois lacos (Nocedal).

    Constroi uma aproximacao da INVERSA da Hessiana usando apenas os
    ultimos `memoria` pares (s_k, y_k):

        s_k = m_{k+1} - m_k        y_k = g_{k+1} - g_k

    Custo de memoria: 2 * memoria * N (contra N^2 da Hessiana cheia).
    E o otimizador padrao em FWI de producao -- inclusive o que o PyFWI
    chama via `fmin_l_bfgs_b`.
    """
    m = np.array(m0, dtype=np.float64, copy=True)
    hist = Historico()
    S, Y = [], []
    g_ant = m_ant = None
    for it in range(n_iter):
        J, g = f_e_g(m)
        hist.avaliacoes += 1
        if g_ant is not None:
            s = (m - m_ant).ravel()
            y = (g - g_ant).ravel()
            if float(np.dot(s, y)) > 1e-20:      # condicao de curvatura
                S.append(s)
                Y.append(y)
                if len(S) > memoria:
                    S.pop(0)
                    Y.pop(0)
        # ---- recursao de dois lacos ----
        q = g.ravel().copy()
        alphas = []
        for s, y in zip(reversed(S), reversed(Y)):
            rho = 1.0 / np.dot(y, s)
            a = rho * np.dot(s, q)
            q -= a * y
            alphas.append((rho, a, s, y))
        if S:
            gamma = np.dot(S[-1], Y[-1]) / np.dot(Y[-1], Y[-1])
            q *= gamma
        for rho, a, s, y in reversed(alphas):
            b = rho * np.dot(y, q)
            q += (a - b) * s
        d = -q.reshape(g.shape)
        if float(np.dot(g.ravel(), d.ravel())) >= 0:
            d = -g
            S.clear(); Y.clear()
        a0 = 1.0 if S else passo_inicial_por_escala(m, d, J, g, fracao_passo)
        alpha, J_novo, nav = busca_linha(
            lambda x: f_e_g(x, so_J=True), m, J, g, d, a0)
        hist.avaliacoes += nav
        hist.registrar(J, alpha, g, m, guardar_modelos)
        if verboso:
            print(f"    it {it:3d}  J = {J:.6e}   |g| = {np.linalg.norm(g):.3e}"
                  f"   alpha = {alpha:.3e}")
        if alpha == 0.0:
            if verboso:
                print("    busca linear falhou -- parando")
            break
        g_ant, m_ant = g, m.copy()
        m = m + alpha * d
        if projecao is not None:
            m = projecao(m)
    J, g = f_e_g(m)
    hist.registrar(J, 0.0, g, m, guardar_modelos)
    return m, hist


OTIMIZADORES = {
    "descida_maxima": descida_maxima,
    "cg": gradiente_conjugado,
    "lbfgs": lbfgs,
}


# --------------------------------------------------------------------------
# Vinculos e pre-condicionamento
# --------------------------------------------------------------------------
def limitar(vmin: float, vmax: float):
    """Projecao em caixa: mantem o modelo dentro de limites fisicos."""
    return lambda m: np.clip(m, vmin, vmax)


def precondicionar_profundidade(g: np.ndarray, potencia: float = 1.0,
                                z0: int = 0) -> np.ndarray:
    """
    Pre-condicionador de iluminacao mais simples que existe: compensa o
    decaimento geometrico da amplitude com a profundidade multiplicando o
    gradiente por z^potencia.

    Nao e a inversa da Hessiana, mas resolve o sintoma mais visivel do
    gradiente bruto: tudo concentrado perto das fontes e receptores.
    """
    nz = g.shape[0]
    peso = (np.arange(nz, dtype=np.float64) + 1.0 - z0).clip(min=1.0) ** potencia
    return g * (peso / peso.max())[:, None]


def suavizar(g: np.ndarray, sigma: float) -> np.ndarray:
    """Suavizacao gaussiana do gradiente (regularizacao implicita)."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(g, sigma) if sigma > 0 else g


# --------------------------------------------------------------------------
# Regularizacao
# --------------------------------------------------------------------------
def tikhonov(m: np.ndarray, dh: float = 1.0, az: float = 1.0, ax: float = 1.0):
    """
    Regularizacao de Tikhonov (suavidade / gradiente L2):

        R(m) = (1/2) integral [ az (dm/dz)^2 + ax (dm/dx)^2 ]

    Penaliza variacao BRUSCA, favorecendo modelos suaves. Barata, estavel,
    e a escolha padrao quando o alvo nao tem contrastes nitidos.

    O preco: ela suaviza TUDO, inclusive as interfaces que voce queria ver.
    Retorna (valor, gradiente).
    """
    gz = np.zeros_like(m)
    gx = np.zeros_like(m)
    gz[:-1, :] = (m[1:, :] - m[:-1, :]) / dh
    gx[:, :-1] = (m[:, 1:] - m[:, :-1]) / dh
    f = 0.5 * float(np.sum(az * gz ** 2 + ax * gx ** 2)) * dh * dh
    # gradiente: adjunto do operador de diferenca (o mesmo padrao usado na TV)
    g = np.zeros_like(m)
    g[1:, :] += az * gz[:-1, :]
    g[:-1, :] -= az * gz[:-1, :]
    g[:, 1:] += ax * gx[:, :-1]
    g[:, :-1] -= ax * gx[:, :-1]
    return f, g * dh


def variacao_total(m: np.ndarray, dh: float = 1.0, eps: float = 1e-7,
                   az: float = 1.0, ax: float = 1.0):
    """
    Variacao Total (TV):

        R(m) = integral sqrt( az (dm/dz)^2 + ax (dm/dx)^2 + eps )

    Penaliza a norma L1 do gradiente, nao a L2. A diferenca e decisiva:
    a L1 e "barata" para poucos saltos grandes e "cara" para muitas
    oscilacoes pequenas. Resultado: TV PRESERVA INTERFACES nitidas enquanto
    remove ruido -- exatamente o que se quer em geologia estratificada.

    `eps` evita divisao por zero onde o gradiente e nulo.
    Retorna (valor, gradiente).
    """
    gz = np.zeros_like(m)
    gx = np.zeros_like(m)
    gz[:-1, :] = (m[1:, :] - m[:-1, :]) / dh
    gx[:, :-1] = (m[:, 1:] - m[:, :-1]) / dh
    raiz = np.sqrt(az * gz ** 2 + ax * gx ** 2 + eps)
    f = float(np.sum(raiz)) * dh * dh
    pz = az * gz / raiz
    px = ax * gx / raiz
    g = np.zeros_like(m)
    g[:-1, :] -= pz[:-1, :]
    g[1:, :] += pz[:-1, :]
    g[:, :-1] -= px[:, :-1]
    g[:, 1:] += px[:, :-1]
    return f, g * dh


def modelo_a_priori(m: np.ndarray, m_prior: np.ndarray, peso=None):
    """
    Regularizacao por MODELO A PRIORI (Asnaashari et al., 2013):

        R(m) = (1/2) || W (m - m_prior) ||^2

    Puxa a solucao na direcao de um modelo conhecido -- tipicamente um perfil
    de poco, um modelo de tomografia ou o proprio modelo inicial. `peso`
    permite confiar mais em algumas regioes (por exemplo, alto peso perto do
    poco e zero longe dele).

    E a forma mais direta de injetar conhecimento geologico num problema mal
    posto, e a que menos distorce o que os dados dizem, porque atua so onde o
    peso e nao nulo.
    """
    d = m - m_prior
    if peso is None:
        return 0.5 * float(np.sum(d ** 2)), d
    return 0.5 * float(np.sum((peso * d) ** 2)), peso ** 2 * d
