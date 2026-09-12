"""
fwikit.metricas
===============
Metricas quantitativas para avaliar uma FWI. Sem elas voce so tem
"ficou parecido" -- que nao sustenta uma dissertacao.

As definicoes seguem o que se usa na literatura de FWI (Virieux et al., 2017)
e sao as mesmas que voce vai reportar em tabelas e curvas.
"""
from __future__ import annotations

import numpy as np


def erro_relativo_modelo(m_est: np.ndarray, m_true: np.ndarray) -> float:
    """
    Erro relativo em norma L2 -- a metrica-sintese mais usada:

        E = ||m_est - m_true||_2 / ||m_true||_2

    Interpretacao: 0 = perfeito. Tipicamente 0.01-0.05 e uma FWI boa em
    dado sintetico; acima de 0.10 algo esta errado (modelo inicial ruim,
    cycle skipping, ou fisica incompativel entre dado e operador).
    """
    return float(np.linalg.norm(m_est - m_true) / np.linalg.norm(m_true))


def erro_relativo_percentual(m_est, m_true) -> float:
    return 100.0 * erro_relativo_modelo(m_est, m_true)


def correlacao(m_est: np.ndarray, m_true: np.ndarray) -> float:
    """Coeficiente de correlacao de Pearson entre os dois modelos."""
    a = np.asarray(m_est).ravel() - np.mean(m_est)
    b = np.asarray(m_true).ravel() - np.mean(m_true)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else 0.0


def r2(m_est: np.ndarray, m_true: np.ndarray) -> float:
    """Coeficiente de determinacao R^2 (fracao da variancia explicada)."""
    ss_res = np.sum((m_true - m_est) ** 2)
    ss_tot = np.sum((m_true - np.mean(m_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def erro_dados(d_est: np.ndarray, d_obs: np.ndarray) -> float:
    """Erro relativo dos DADOS (nao do modelo) -- o que a FWI minimiza."""
    return float(np.linalg.norm(d_est - d_obs) / np.linalg.norm(d_obs))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def ganho(m_est, m_ini, m_true) -> float:
    """
    Quanto a inversao melhorou em relacao ao modelo inicial, em %.
    Positivo = melhorou. Negativo = a inversao PIOROU o modelo (acontece!).
    """
    e0 = erro_relativo_modelo(m_ini, m_true)
    e1 = erro_relativo_modelo(m_est, m_true)
    return 100.0 * (e0 - e1) / e0 if e0 > 0 else 0.0


def resumo(m_est, m_ini, m_true, titulo: str = "") -> dict:
    """Dicionario com todas as metricas de uma vez."""
    return {
        "titulo": titulo,
        "erro_inicial_%": erro_relativo_percentual(m_ini, m_true),
        "erro_final_%": erro_relativo_percentual(m_est, m_true),
        "ganho_%": ganho(m_est, m_ini, m_true),
        "correlacao": correlacao(m_est, m_true),
        "r2": r2(m_est, m_true),
    }


def imprimir_resumo(d: dict) -> None:
    print(f"    {'metrica':<28} {'valor':>10}")
    print(f"    {'-'*28} {'-'*10}")
    for k, v in d.items():
        if k == "titulo":
            continue
        print(f"    {k:<28} {v:>10.4f}")
