"""
fwikit.verificacao
==================
Testes para PROVAR que um gradiente esta correto.

Por que isso merece um modulo proprio: um gradiente errado nao causa erro
nem crash. Ele causa uma inversao que roda, converge, produz uma figura
plausivel -- e esta errada. E o tipo de defeito que sobrevive a revisao e
so aparece quando alguem tenta reproduzir o resultado.

Os tres testes, do mais simples ao mais rigoroso:

  1. DIFERENCAS FINITAS PONTUAIS -- compara dJ/dm_i em alguns pixels.
     Bom para localizar ONDE o gradiente erra.

  2. DERIVADA DIRECIONAL -- compara <g, h> com (J(m+ah)-J(m-ah))/2a.
     Testa o gradiente inteiro de uma vez, com um unico numero.
     Custa 2 modelagens por alpha. E o teste do dia a dia.

  3. TESTE DE TAYLOR -- verifica as ORDENS de convergencia:
         E0(a) = |J(m+ah) - J(m)|            deve cair como O(a)
         E1(a) = |J(m+ah) - J(m) - a<g,h>|   deve cair como O(a^2)
     E o mais rigoroso: E1 so e de segunda ordem se o gradiente estiver
     exato. E o teste padrao exigido em trabalho academico.
"""
from __future__ import annotations

import numpy as np


def teste_diferencas_finitas(J_de_m, m, g, pontos, eps: float = 1e-10,
                             central: bool = True):
    """
    Compara g[i] com a derivada numerica em cada ponto de `pontos`.

    Retorna lista de dicionarios com adjunto, diferencas finitas e razao.
    Razao proxima de 1 = correto. Espere degradacao em pontos de amplitude
    baixa (o ruido numerico domina) e perto de fontes/receptores.
    """
    J0 = None if central else J_de_m(m)
    linhas = []
    for (iz, ix) in pontos:
        mp = np.array(m, copy=True)
        mp[iz, ix] += eps
        if central:
            mm = np.array(m, copy=True)
            mm[iz, ix] -= eps
            gfd = (J_de_m(mp) - J_de_m(mm)) / (2 * eps)
        else:
            gfd = (J_de_m(mp) - J0) / eps
        ga = float(g[iz, ix])
        linhas.append({"ponto": (iz, ix), "adjunto": ga, "dif_finitas": gfd,
                       "razao": ga / gfd if gfd != 0 else float("nan")})
    return linhas


def teste_derivada_direcional(J_de_m, m, g, h, alphas=(0.4, 0.2, 0.1, 0.05)):
    """
    Compara a derivada direcional analitica <g, h> com a numerica por
    diferencas CENTRADAS.

    A diferenca centrada tem erro O(a^2), entao a razao deve ficar proxima
    de 1 e ESTAVEL ao variar alpha. Razao estavel mas diferente de 1
    denuncia erro sistematico (fator de escala, sinal, termo faltando).
    Razao instavel denuncia ruido numerico -- reduza o alcance de alpha.
    """
    gh = float(np.sum(g * h))
    linhas = []
    for a in alphas:
        gfd = (J_de_m(m + a * h) - J_de_m(m - a * h)) / (2 * a)
        linhas.append({"alpha": a, "analitico": gh, "numerico": gfd,
                       "razao": gh / gfd if gfd != 0 else float("nan")})
    return gh, linhas


def teste_taylor(J_de_m, m, g, h, alphas=(0.4, 0.2, 0.1, 0.05, 0.025),
                 J0=None):
    """
    Teste de Taylor. Retorna (J0, gh, linhas) com E0, E1 e as ordens medidas.

        E0(a) = |J(m+ah) - J(m)|              -> ordem ~ 1
        E1(a) = |J(m+ah) - J(m) - a <g,h>|    -> ordem ~ 2

    A ordem e estimada como log2(E(a) / E(a/2)).

    ATENCAO ao escolher `h`: uma direcao aleatoria costuma ser quase
    ortogonal ao gradiente (que e suave), deixando <g,h> pequeno e o teste
    dominado pela curvatura. Use uma direcao SUAVE -- o proprio gradiente
    normalizado e a escolha mais confiavel.
    """
    if J0 is None:
        J0 = J_de_m(m)
    gh = float(np.sum(g * h))
    linhas = []
    E0_ant = E1_ant = None
    for a in alphas:
        Ja = J_de_m(m + a * h)
        E0 = abs(Ja - J0)
        E1 = abs(Ja - J0 - a * gh)
        o0 = np.log2(E0_ant / E0) if E0_ant and E0 > 0 else float("nan")
        o1 = np.log2(E1_ant / E1) if E1_ant and E1 > 0 else float("nan")
        linhas.append({"alpha": a, "E0": E0, "E1": E1,
                       "ordem_E0": o0, "ordem_E1": o1})
        E0_ant, E1_ant = E0, E1
    return J0, gh, linhas


def direcao_suave(g, m_ref, amplitude: float = 1e-2):
    """
    Direcao de perturbacao recomendada para os testes: o proprio gradiente
    normalizado, escalado para `amplitude` do valor tipico do modelo.
    """
    pico = float(np.abs(g).max())
    if pico == 0:
        raise ValueError("gradiente nulo")
    return g / pico * float(np.mean(np.abs(m_ref))) * amplitude


def veredito(linhas_direcional, tol: float = 0.02) -> tuple[bool, str]:
    """Avalia o teste de derivada direcional e devolve (ok, mensagem)."""
    razoes = np.array([l["razao"] for l in linhas_direcional])
    finitas = razoes[np.isfinite(razoes)]
    if finitas.size == 0:
        return False, "sem razoes finitas -- o teste nao pode ser avaliado"
    erro = float(np.max(np.abs(finitas - 1.0)))
    espalhamento = float(np.std(finitas))
    if erro < tol:
        return True, (f"APROVADO: erro maximo de {100*erro:.2f}% "
                      f"(tolerancia {100*tol:.0f}%)")
    if espalhamento < 0.1 * erro:
        return False, (f"REPROVADO: razao ESTAVEL em ~{finitas.mean():.4f}. "
                       "Erro sistematico -- procure fator de escala, sinal "
                       "trocado ou um termo faltando no adjunto.")
    return False, (f"REPROVADO: razao INSTAVEL ({finitas.min():.3f} a "
                   f"{finitas.max():.3f}). Provavel ruido numerico -- "
                   "ajuste a faixa de alpha ou use precisao dupla.")
