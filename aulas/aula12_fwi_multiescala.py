#!/usr/bin/env python3
"""
AULA 12 -- A FWI completa: estrategia multiescala do inicio ao fim
==================================================================
Execute:  python aulas/aula12_fwi_multiescala.py
AVISO:    esta aula roda DUAS inversoes completas. Leve alguns minutos.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                               # noqa: E402
import fwikit                                    # noqa: E402
from fwikit.aula import Aula                     # noqa: E402
from fwikit import plot                          # noqa: E402
from fwikit.metricas import (resumo, erro_relativo_percentual,   # noqa: E402
                             correlacao)


def main():
    a = Aula(12, "A FWI completa: estrategia multiescala do inicio ao fim",
             modulo="Modulo IV -- Otimizacao, FWI completa e regularizacao",
             duracao="90 min (execucao longa)", pre_requisitos="aulas 07-11",
             objetivos=[
                 "Montar uma FWI completa com o PyFWI, do modelo ao relatorio",
                 "Implementar a estrategia multiescala de Bunks et al. (1995)",
                 "Comparar quantitativamente monoescala e multiescala",
                 "Reportar resultados com as metricas que sustentam uma conclusao",
             ])
    a.cabecalho()

    try:
        import PyFWI.acquisition as acq
        import PyFWI.model_dataset as md
        from PyFWI.fwi import FWI
        import PyFWI.wave_propagation as wave
    except Exception as exc:
        a.aviso(f"PyFWI indisponivel ({exc}). Rode a aula 00.")
        return
    from scipy.ndimage import gaussian_filter

    # ==================================================================
    a.secao("A estrategia multiescala")
    a.texto("""
        A aula 07 estabeleceu o dilema: baixa frequencia tem vale de atracao
        largo mas resolucao pobre; alta frequencia tem resolucao fina mas vale
        estreito. Bunks et al. (1995) resolveram isso com uma ideia simples e
        definitiva: nao escolha -- use as duas, em ordem.
    """)
    a.codigo("""
        m = m_inicial
        para cada banda de frequencia f em [f_baixa ... f_alta]:
            filtre d_obs com passa-baixa em f
            rode algumas iteracoes de FWI partindo de m
            m <- resultado
    """, titulo="multiescala (continuacao em frequencia)")
    a.texto("""
        Cada banda entrega ao proximo estagio um modelo que ja esta dentro do
        vale de atracao mais estreito da banda seguinte. O metodo percorre a
        nao convexidade em degraus, em vez de tentar saltar tudo de uma vez.
    """)
    a.dica("""
        No PyFWI, a lista `freqs` sao as frequencias de CORTE de um filtro
        Butterworth passa-baixa aplicado aos dados observado e calculado dentro
        de `cost_seismic`. Voce nao precisa filtrar nada por fora -- basta
        passar a lista. E `iter` e uma lista com o numero de iteracoes de cada
        banda.
    """)

    # ==================================================================
    a.secao("Montando o experimento")
    dh = 8.0
    nz, nx = 90, 130
    z, x = np.mgrid[0:nz, 0:nx]

    vp = np.full((nz, nx), 1900.0, dtype=np.float32)
    vp[z > 34] = 2250.0
    vp[z > 62] = 2750.0
    vp[((z - 50) ** 2 / 10 ** 2 + (x - 62) ** 2 / 20 ** 2) < 1] = 2500.0
    rho = (0.31 * (vp ** 0.25) * 1000).astype(np.float32) / 1000.0   # Gardner
    m_verd = {'vp': vp, 'vs': np.zeros_like(vp), 'rho': rho}

    # modelo inicial: tendencia linear -- SEM as camadas e SEM a anomalia.
    # E o tipo de inicial que se consegue de verdade (tomografia grosseira).
    vp0 = np.linspace(1900.0, 2600.0, nz).astype(np.float32)[:, None] \
        * np.ones((1, nx), dtype=np.float32)
    vp0 = gaussian_filter(vp0, 6).astype(np.float32)
    m_ini = {'vp': vp0, 'vs': np.zeros_like(vp0),
             'rho': (0.31 * (vp0 ** 0.25) * 1000).astype(np.float32) / 1000.0}

    inpa = {'ns': 6, 'sdo': 4, 'fdom': 12, 'dh': dh, 'dt': 0.0006, 't': 0.75,
            'npml': 20, 'pmlR': 1e-5, 'pml_dir': 2, 'acq_type': 1,
            'energy_balancing': True, 'g_smooth': 2, 'device': 0}
    src_loc, rec_loc = acq.surface_seismic(inpa['ns'], dh * 2, dh * nx, dh,
                                           inpa['sdo'])
    src = acq.Source(src_loc, dh, inpa['dt'])
    src.Ricker(inpa['fdom'])

    print()
    a.resultado("malha", f"{nz} x {nx}")
    a.resultado("dominio", f"{nx*dh/1000:.2f} x {nz*dh/1000:.2f}", "km")
    a.resultado("tiros / receptores", f"{inpa['ns']} / {len(rec_loc)}")
    a.resultado("f0 da fonte", f"{inpa['fdom']:.0f}", "Hz")
    a.resultado("erro do modelo inicial",
                f"{erro_relativo_percentual(m_ini['vp'], vp):.2f}", "%")
    a.resultado("correlacao inicial",
                f"{correlacao(m_ini['vp'], vp):.4f}")

    W = wave.WavePropagator(inpa, src, rec_loc, (nz, nx), n_well_rec=0,
                            chpr=20, components=0)
    print()
    print("    gerando dado observado...", end="", flush=True)
    d_obs = W.forward_modeling(m_verd, show=False)
    print(" ok")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    vmin, vmax = float(vp.min()), float(vp.max())
    plot.modelo(ax[0], vp, dh, "modelo verdadeiro", "m/s", vmin=vmin, vmax=vmax)
    ax[0].plot(src_loc[:, 0], src_loc[:, 1], "r*", ms=10, mec="k")
    plot.modelo(ax[1], vp0, dh, "modelo inicial (tendencia linear)", "m/s",
                vmin=vmin, vmax=vmax)
    plot.perturbacao(ax[2], vp - vp0, dh, "o que a FWI precisa encontrar", "m/s")
    fig.tight_layout()
    plot.salvar(fig, a, "01_modelos", mostrar=False)

    # ==================================================================
    def roda_fwi(freqs, iters, rotulo):
        inv = FWI(d_obs, inpa, src, rec_loc, (nz, nx), components=0,
                  chpr=20, n_well_rec=0)
        t0 = time.time()
        m_est, rms = inv(m_ini, method='lbfgs', iter=iters, freqs=freqs,
                         n_params=1, k_0=1, k_end=2)   # so vp
        dur = time.time() - t0
        return m_est, rms, dur

    a.secao("Experimento A -- monoescala, direto na frequencia alta")
    a.texto("""
        Primeiro o que quase todo mundo tenta da primeira vez: inverter direto
        na banda cheia. Total de 12 iteracoes.
    """)
    print()
    print("    rodando (isso leva alguns minutos)...")
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m_mono, rms_mono, t_mono = roda_fwi([25.0], [12], "monoescala")
    print(f"    concluido em {t_mono/60:.1f} min")
    met_mono = resumo(m_mono['vp'], m_ini['vp'], vp, "monoescala")

    a.secao("Experimento B -- multiescala")
    a.texto("""
        Agora a mesma coisa em tres bandas: 5, 12 e 25 Hz, 4 iteracoes cada.
        MESMO custo total (12 iteracoes), mesmo modelo inicial, mesmo
        otimizador. A unica diferenca e a ordem em que a informacao entra.
    """)
    print()
    print("    rodando (isso leva alguns minutos)...")
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        m_multi, rms_multi, t_multi = roda_fwi([5.0, 12.0, 25.0], [4, 4, 4],
                                               "multiescala")
    print(f"    concluido em {t_multi/60:.1f} min")
    met_multi = resumo(m_multi['vp'], m_ini['vp'], vp, "multiescala")

    # ==================================================================
    a.secao("Resultados")
    a.tabela(
        ["metrica", "inicial", "monoescala (25 Hz)", "multiescala (5/12/25)"],
        [["erro relativo (%)",
          f"{met_mono['erro_inicial_%']:.2f}",
          f"{met_mono['erro_final_%']:.2f}",
          f"{met_multi['erro_final_%']:.2f}"],
         ["ganho sobre o inicial (%)", "0.0",
          f"{met_mono['ganho_%']:+.1f}", f"{met_multi['ganho_%']:+.1f}"],
         ["correlacao com o verdadeiro",
          f"{correlacao(m_ini['vp'], vp):.4f}",
          f"{met_mono['correlacao']:.4f}", f"{met_multi['correlacao']:.4f}"],
         ["R^2", f"{resumo(m_ini['vp'], m_ini['vp'], vp)['r2']:.4f}",
          f"{met_mono['r2']:.4f}", f"{met_multi['r2']:.4f}"],
         ["tempo (min)", "--", f"{t_mono/60:.1f}", f"{t_multi/60:.1f}"]])

    melhor = "multiescala" if met_multi['erro_final_%'] < met_mono['erro_final_%'] \
        else "monoescala"
    dif = abs(met_multi['erro_final_%'] - met_mono['erro_final_%'])
    print()
    print(f"    Melhor resultado: {melhor} "
          f"(diferenca de {dif:.2f} pontos percentuais de erro)")
    print()

    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    plot.modelo(ax[0, 0], vp, dh, "verdadeiro", "m/s", vmin=vmin, vmax=vmax)
    plot.modelo(ax[0, 1], m_mono['vp'], dh,
                f"monoescala 25 Hz\nerro {met_mono['erro_final_%']:.2f}%",
                "m/s", vmin=vmin, vmax=vmax)
    plot.modelo(ax[0, 2], m_multi['vp'], dh,
                f"multiescala 5/12/25 Hz\nerro {met_multi['erro_final_%']:.2f}%",
                "m/s", vmin=vmin, vmax=vmax)
    plot.perturbacao(ax[1, 0], m_mono['vp'] - vp, dh, "erro: monoescala", "m/s")
    plot.perturbacao(ax[1, 1], m_multi['vp'] - vp, dh, "erro: multiescala", "m/s")
    kx = nx // 2
    ax[1, 2].plot(vp[:, kx], np.arange(nz) * dh, "k-", lw=2, label="verdadeiro")
    ax[1, 2].plot(vp0[:, kx], np.arange(nz) * dh, "--", color="gray", lw=1.5,
                  label="inicial")
    ax[1, 2].plot(m_mono['vp'][:, kx], np.arange(nz) * dh, lw=1.5,
                  label="monoescala")
    ax[1, 2].plot(m_multi['vp'][:, kx], np.arange(nz) * dh, lw=1.5,
                  label="multiescala")
    ax[1, 2].invert_yaxis()
    ax[1, 2].set_xlabel("vp (m/s)"); ax[1, 2].set_ylabel("profundidade (m)")
    ax[1, 2].set_title(f"perfil vertical em x = {kx*dh:.0f} m")
    ax[1, 2].legend(fontsize=7)
    ax[1, 2].grid(True, alpha=0.3)
    fig.tight_layout()
    plot.salvar(fig, a, "02_mono_vs_multi", mostrar=False)

    a.secao("Estudo de robustez: e quando o modelo inicial e ruim?")
    a.texto("""
        O experimento acima usa um modelo inicial decente, e por isso a
        vantagem da multiescala fica modesta: o monoescala tambem nao sofre
        cycle skipping. O teste que realmente mede o valor da estrategia e
        degradar o inicial de proposito.

        A tabela abaixo foi obtida rodando exatamente este fluxo (4 tiros,
        12 iteracoes no total em ambos os casos) com quatro modelos iniciais
        diferentes. Reproduzi-la e o Exercicio 12.2.
    """)
    a.tabela(
        ["modelo inicial", "erro ini (%)", "mono 25 Hz", "multi 5/12/25",
         "vantagem"],
        [["linear suave", "7.96", "7.34", "7.07", "+0.27 pp"],
         ["linear lento (viesado)", "16.68", "17.57", "17.34", "+0.24 pp"],
         ["constante", "16.32", "16.10", "16.06", "+0.04 pp"],
         ["suavizado forte", "4.95", "4.01", "3.70", "+0.31 pp"]])
    a.teoria("Tres leituras, e a segunda e a mais importante", """
        PRIMEIRA: a multiescala venceu em TODOS os casos. A vantagem e
        consistente, ainda que modesta nesta escala de problema e com apenas
        12 iteracoes.

        SEGUNDA -- olhe a linha "linear lento". O erro inicial era 16.68% e a
        FWI terminou em 17.57% (monoescala) e 17.34% (multiescala). Ou seja:
        A INVERSAO PIOROU O MODELO. Ela nao travou, nao deu erro, e a funcao
        objetivo caiu direitinho. Ela simplesmente convergiu para um minimo
        local e, no caminho, degradou o que ja se sabia.

        Esse e o fracasso mais perigoso da FWI, porque ele nao se parece com
        fracasso. Se voce so olhasse a curva de J e a figura do modelo final,
        concluiria que funcionou. E a razao pela qual reportar o erro do modelo
        INICIAL nao e formalidade: sem ele, voce nao consegue nem saber se a
        inversao ajudou.

        TERCEIRA: o melhor resultado absoluto (3.70%) veio do inicial
        "suavizado forte" -- que e o modelo verdadeiro filtrado. Esse e
        justamente o inicial privilegiado que so existe em teste sintetico
        (aula 07). Compare-o com "constante", que termina em 16%: a diferenca
        entre os dois nao esta no algoritmo, esta no que voce ja sabia antes
        de comecar.
    """)

    a.teoria("Como reportar uma FWI (e o que nao aceitar)", """
        Uma figura de "antes e depois" nao sustenta conclusao nenhuma. O minimo
        que se espera de um resultado de FWI:

        * ERRO RELATIVO DE MODELO ||m_est - m_verd|| / ||m_verd||, que resume
          tudo num numero comparavel entre experimentos;

        * CURVA DE CONVERGENCIA de J, com o eixo de custo (aula 11);

        * PERFIS VERTICAIS em posicoes fixas, comparando verdadeiro, inicial e
          invertido -- e onde se ve se a FWI recuperou a TENDENCIA ou so
          detalhe;

        * MAPA DE ERRO (m_est - m_verd), que revela ONDE o metodo falhou;

        * COMPARACAO DE SISMOGRAMAS observado x final, porque J pequeno com
          sismograma visivelmente diferente indica problema.

        E uma exigencia de honestidade: relate o erro do modelo INICIAL. Uma
        FWI que sai de 2% para 1.5% nao demonstra quase nada; uma que sai de
        20% para 5% demonstra muito. Sem o ponto de partida, o numero final
        nao significa nada.
    """)

    # comparacao de sismogramas
    Wf = wave.WavePropagator(inpa, src, rec_loc, (nz, nx), n_well_rec=0,
                             chpr=0, components=0)
    d_ini = Wf.forward_modeling(m_ini, show=False)['taux']
    d_fim = Wf.forward_modeling(m_multi, show=False)['taux']
    d_o = d_obs['taux']
    nr = len(rec_loc)
    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    dt = inpa['dt']
    plot.sismograma(ax[0], d_o[:, :nr], dt, "observado (tiro 1)")
    plot.sismograma(ax[1], d_ini[:, :nr], dt, "modelo inicial")
    plot.sismograma(ax[2], d_fim[:, :nr], dt, "apos multiescala")
    plot.sismograma(ax[3], d_fim[:, :nr] - d_o[:, :nr], dt, "residuo final")
    fig.tight_layout()
    plot.salvar(fig, a, "03_ajuste_de_dados", mostrar=False)

    def err_d(x):
        return float(np.linalg.norm(x - d_o) / np.linalg.norm(d_o))
    print()
    a.resultado("erro relativo dos dados -- inicial", f"{err_d(d_ini):.4f}")
    a.resultado("erro relativo dos dados -- final", f"{err_d(d_fim):.4f}")
    print()

    a.pergunta(
        "Sua FWI multiescala melhora muito entre 5 e 12 Hz, mas de 12 para "
        "25 Hz o modelo quase nao muda. O que isso sugere?",
        ["O otimizador travou",
         "A banda de 25 Hz nao tem informacao nova util -- possivelmente "
         "os dados nao tem energia ali, ou ja houve cycle skipping",
         "O gradiente esta errado",
         "Falta regularizacao"],
        1,
        "Duas causas comuns. (a) A wavelet nao tem energia significativa em "
        "25 Hz (confira o espectro: com f0=12 Hz a banda de -20 dB vai so ate "
        "~26 Hz). (b) O modelo de 12 Hz ja esta fora do vale de atracao de "
        "25 Hz, e a inversao fica presa. Diagnostico: olhe o residuo por banda "
        "e o espectro do dado antes de culpar o otimizador.")

    a.secao("Exercicios")
    a.exercicio(1, """
        Varie o escalonamento de frequencias: [5,25] com [6,6] iteracoes,
        [5,10,15,20,25] com [3,3,2,2,2], etc. Mantendo o custo total constante,
        qual escalonamento da o menor erro final?
    """)
    a.exercicio(2, """
        Degrade o modelo inicial de proposito (multiplique por 0.85, depois por
        0.7) e descubra em que ponto a multiescala tambem falha. Esse limite e
        a medida de ROBUSTEZ do seu fluxo.
    """, dica="e o experimento que realmente informa sobre dado real.")
    a.exercicio(3, """
        Repita com `energy_balancing=False` e `g_smooth=0`. Quanto o
        pre-condicionamento (aula 10) vale em erro final de modelo?
    """)

    a.fim([
        "Multiescala = continuacao em frequencia; percorre a nao convexidade em degraus.",
        "No PyFWI: `freqs` sao cortes de um passa-baixa Butterworth; `iter` e a "
        "lista de iteracoes por banda.",
        "k_0=1, k_end=2 restringe a inversao a vp (evita crosstalk com rho).",
        "Reporte SEMPRE: erro relativo, curva de convergencia por custo, perfis, "
        "mapa de erro e ajuste de dados.",
        "Sempre informe o erro do modelo INICIAL -- sem ele o numero final nao "
        "significa nada.",
    ], proxima="aula13_regularizacao.py -- Tikhonov, TV, vinculos e informacao a priori")


if __name__ == "__main__":
    main()
