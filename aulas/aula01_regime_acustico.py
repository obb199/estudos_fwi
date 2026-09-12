#!/usr/bin/env python3
"""
AULA 01 -- O regime acustico: de onde vem a equacao que a FWI resolve
=====================================================================
Execute:  python aulas/aula01_regime_acustico.py
          python aulas/aula01_regime_acustico.py --rapido
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                               # noqa: E402
import fwikit                                    # noqa: E402
from fwikit.aula import Aula, mostrar_figuras    # noqa: E402
from fwikit import plot                          # noqa: E402
from fwikit.acustico import (ricker, espectro_amplitude,   # noqa: E402
                             largura_banda)


def main():
    a = Aula(1, "O regime acustico: de onde vem a equacao que a FWI resolve",
             modulo="Modulo I -- Fisica e numerica da propagacao",
             duracao="60 min",
             pre_requisitos="aula 00",
             objetivos=[
                 "Derivar a equacao acustica a partir da elastodinamica e saber "
                 "exatamente qual hipotese foi feita em cada passo",
                 "Explicar o que a aproximacao acustica preserva e o que ela joga fora",
                 "Reconhecer quando o regime acustico e adequado e quando ele mente",
                 "Dominar a wavelet de Ricker: pico, banda, atraso e por que isso "
                 "controla a resolucao da FWI",
             ])
    a.cabecalho()

    # ==================================================================
    a.secao("Ponto de partida: elastodinamica")
    a.texto("""
        Toda a sismologia de exploracao comeca em dois ingredientes: a segunda
        lei de Newton escrita para um meio continuo, e uma lei constitutiva que
        diz como o material reage a deformacao.
    """)
    a.eq("rho dv_i/dt = d(sigma_ij)/dx_j + f_i", rotulo="conservacao do momento")
    a.eq("d(sigma_ij)/dt = lambda delta_ij dv_k/dx_k"
         "  +  mu ( dv_i/dx_j + dv_j/dx_i )", rotulo="Hooke, forma em taxa")
    a.texto("""
        Aqui v e a velocidade de PARTICULA (nao a velocidade de onda), sigma e o
        tensor de tensoes, rho a densidade, e lambda e mu sao os parametros de
        Lame. mu e o modulo de cisalhamento: e ele que resiste a deformacao sem
        mudanca de volume.

        Escrito assim, em 2D, isso vira um sistema de 5 equacoes acopladas nas
        incognitas (vx, vz, sigma_xx, sigma_zz, sigma_xz). E EXATAMENTE esse
        sistema que o kernel OpenCL do PyFWI resolve -- volte nesse ponto na
        aula 05.
    """)
    a.teoria("As duas velocidades de onda", """
        Do sistema elastico saem duas velocidades:

            vp = sqrt( (lambda + 2 mu) / rho )      onda P (compressional)
            vs = sqrt( mu / rho )                   onda S (cisalhante)

        Elas se propagam com velocidades diferentes, se convertem uma na outra
        em interfaces, e geram ondas de superficie (Rayleigh, Scholte). Um dado
        sismico real contem tudo isso.
    """)

    # ==================================================================
    a.secao("A aproximacao acustica: uma unica hipotese")
    a.texto("""
        A aproximacao acustica consiste em UMA hipotese fisica:
        o meio nao resiste ao cisalhamento.
    """)
    a.eq("mu = 0      (equivalentemente  vs = 0)", rotulo="hipotese acustica")
    a.texto("""
        Esse e literalmente o comportamento de um fluido. Veja o que acontece
        no sistema elastico quando mu = 0:
    """)
    a.lista([
        "sigma_xz = 0 -- nao existe tensao cisalhante.",
        "d(sigma_xx)/dt = d(sigma_zz)/dt = lambda (dvx/dx + dvz/dz) -- as duas "
        "tensoes normais passam a obedecer a MESMA equacao. Como partem de "
        "zero, permanecem iguais em todo instante: sigma_xx = sigma_zz. "
        "O tensor de tensoes vira isotropico.",
        "Um tensor isotropico e descrito por um unico escalar. Definimos a "
        "PRESSAO como esse escalar, com o sinal trocado (compressao positiva).",
    ])
    a.eq("p = - (sigma_xx + sigma_zz) / 2  =  - sigma_xx  =  - sigma_zz",
         rotulo="definicao de pressao")
    a.aviso("""
        Guarde esta linha. E exatamente o que o PyFWI faz quando voce pede
        `components=0`: ele devolve (taux + tauz)/2, ou seja, -p. O PyFWI resolve
        o sistema ELASTICO e obtem o caso acustico como o limite vs -> 0.
        Vamos comprovar isso numericamente na aula 05.
    """)
    a.texto("""
        Com mu = 0, lambda passa a ser o modulo de compressibilidade (bulk
        modulus) K, e vp = sqrt(K/rho). O sistema de 5 equacoes colapsa em 3:
    """)
    a.eq("rho dv/dt = - grad p",
         "dp/dt = - K div v  +  s(t) delta(x - xs)",
         rotulo="sistema acustico de 1a ordem (velocidade-pressao)")
    a.texto("""
        Eliminando v (derive a segunda no tempo e substitua a primeira) chega-se
        a forma de segunda ordem, que e a que vamos discretizar no curso:
    """)
    a.eq("(1/K) d2p/dt2  =  div( (1/rho) grad p )  +  s(t) delta(x - xs)",
         rotulo="acustica, densidade variavel")
    a.texto("""
        E, se a densidade for considerada constante (ou lentamente variavel),
        1/rho sai do divergente e sobra a forma classica:
    """)
    a.eq("(1 / c(x)^2) d2p/dt2  -  laplaciano(p)  =  s(t) delta(x - xs)",
         rotulo="EQUACAO ACUSTICA -- densidade constante")
    a.aviso("""
        Um detalhe de notacao que quase todo texto omite e que custa horas de
        depuracao: o `s` da equacao de SEGUNDA ordem NAO e o mesmo `s` da
        equacao de primeira ordem.

        Ao derivar dp/dt no tempo para eliminar v, o termo-fonte vira
        (1/K) ds/dt. Escrever "+ s" na equacao de segunda ordem e redefinir a
        fonte -- pratica padrao na literatura, mas com uma consequencia
        concreta: a wavelet EFETIVA das duas formulacoes difere por uma
        DERIVADA TEMPORAL.

        E por isso que dois codigos corretos, um em velocidade-tensao (1a
        ordem, como o PyFWI) e outro em pressao (2a ordem, como o fwikit),
        produzem tracos com formas diferentes alimentados pela mesma Ricker.
        Na aula 05 medimos isso: a correlacao bruta entre os dois e de apenas
        +0.12, e sobe para +0.85 depois de derivar um deles no tempo.
    """)
    a.dica("""
        Esta ultima e a equacao que `fwikit.acustico` resolve, e e a mesma que
        aparece na formulacao classica de FWI acustica (Tarantola, 1984).

        Repare na forma com que escrevemos o parametro: usamos m(x) = 1/c(x)^2,
        a VAGAROSIDADE AO QUADRADO. A razao e algebrica e vale ouro no Modulo
        III: a equacao e LINEAR em m, mas nao em c. Isso deixa o gradiente
        limpo e a derivada em relacao ao parametro trivial.
    """)

    a.pergunta(
        "O que exatamente se perde ao adotar mu = 0?",
        ["Apenas as ondas S; o resto do dado fica intacto",
         "Ondas S, conversoes P-S, ondas de superficie e o comportamento "
         "correto de amplitude com o angulo",
         "Nada relevante para dado marinho",
         "A cinematica das ondas P (tempos de transito)"],
        1,
        "Some tudo que depende do cisalhamento: ondas S, conversoes em "
        "interfaces, Rayleigh/Scholte, e a dependencia angular correta da "
        "reflexao (AVO). O que SOBREVIVE bem e a cinematica das ondas P -- "
        "e por isso a FWI acustica ainda constroi bons modelos de velocidade.")

    a.pausa()

    # ==================================================================
    a.secao("O que o regime acustico preserva, e o que ele mente")
    a.tabela(
        ["Fenomeno", "Acustico", "Comentario"],
        [["Tempo de transito P", "OK", "preservado; e o que a FWI mais usa"],
         ["Difracao / multipath", "OK", "e a vantagem sobre tomografia de raio"],
         ["Reflexao (baixo angulo)", "OK", "coeficiente aproximadamente correto"],
         ["Reflexao (alto angulo)", "ERRA", "AVO precisa de vs"],
         ["Onda S", "AUSENTE", "nao existe no modelo"],
         ["Conversao P-S", "AUSENTE", "principal contaminante em dado terrestre"],
         ["Onda de superficie", "AUSENTE", "domina o dado terrestre raso"],
         ["Atenuacao (Q)", "AUSENTE", "amplitude e fase ficam erradas"],
         ["Anisotropia", "AUSENTE", "exige VTI/TTI"]])

    a.teoria("Quando o regime acustico e legitimo", """
        Ele funciona bem quando:
          * o dado e MARINHO com hidrofones (a coluna d'agua e literalmente
            acustica, e a fonte/receptor estao no fluido);
          * os angulos de abertura sao moderados;
          * o alvo e o modelo de VELOCIDADE P de grande escala, nao amplitude
            absoluta;
          * as ondas de superficie foram removidas no processamento.

        Ele falha quando:
          * dado terrestre com ground roll forte;
          * contrastes de vs relevantes (sal, carbonatos, fundo marinho duro);
          * o objetivo e amplitude/AVO, e nao cinematica;
          * o meio e fortemente atenuante e a banda e larga.
    """)
    a.texto("""
        Um ponto honesto e pouco dito: mesmo quando a fisica acustica esta
        "errada", a FWI acustica continua util, porque o que ela extrai melhor
        e a informacao CINEMATICA (tempos), e essa sobrevive. O preco e que o
        erro de modelagem que sobra -- a parte do residuo que a fisica acustica
        nao consegue explicar -- e absorvido pelo modelo de velocidade. E esse o
        motivo pelo qual se fala em FWI elastica e viscoacustica. Voltamos a isso
        na aula 14.
    """)

    a.pausa()

    # ==================================================================
    a.secao("A fonte: wavelet de Ricker")
    a.texto("""
        O termo s(t) da equacao e a assinatura da fonte. Em trabalho sintetico
        usa-se quase sempre a wavelet de Ricker: a segunda derivada de uma
        gaussiana, com media zero e forma analitica simples.
    """)
    a.eq("w(t) = [1 - 2 (pi f0 (t-t0))^2] exp[ - (pi f0 (t-t0))^2 ]",
         rotulo="Ricker")
    a.texto("""
        f0 e a frequencia de PICO do espectro (nao a maxima, e nao a media --
        confundir isso e um erro classico).

        O deslocamento t0 existe porque a Ricker e simetrica em torno do pico e
        nao se anula para t < 0: sem atraso, ela seria cortada na origem, e o
        degrau resultante sujaria o espectro. A convencao t0 = 1/f0 nao e
        arbitraria -- e justamente onde a amplitude truncada cai abaixo de 0.1%
        do pico:

            t0 = 0.6/f0  ->  17.5%  da amplitude de pico truncada
            t0 = 0.8/f0  ->   2.1%
            t0 = 1.0/f0  ->   0.10%   <- a convencao
            t0 = 1.2/f0  ->   0.002%
    """)

    dt = 1.0e-3
    nt = 700
    print()
    print("    Banda da Ricker medida numericamente, em duas quedas:")
    print()
    linhas = []
    for f0 in [5, 10, 15, 25, 40]:
        w = ricker(f0, dt, nt)
        _, fmax6 = largura_banda(w, dt, -6.0)
        _, fmax20 = largura_banda(w, dt, -20.0)
        f, S = espectro_amplitude(w, dt)
        f_pico = f[np.argmax(S)]
        linhas.append([f0, f"{f_pico:.1f}", f"{fmax6:.1f}", f"{fmax6/f0:.2f}",
                       f"{fmax20:.1f}", f"{fmax20/f0:.2f}"])
    a.tabela(["f0 (Hz)", "pico", "fmax -6dB", "/f0", "fmax -20dB", "/f0"],
             linhas)
    a.texto("""
        Leia a tabela com atencao, porque as duas colunas servem a proposito
        diferente. A -6 dB (metade da amplitude) a banda vai ate ~1.6 f0: e a
        faixa onde esta a energia que de fato move a inversao. A -20 dB a banda
        vai ate ~2.2 f0: e a faixa que ainda tem energia suficiente para
        DISPERSAR numericamente e estragar a simulacao.

        Por isso as duas regras convivem: use ~1.6 f0 para raciocinar sobre
        resolucao, e arredonde para cima -- 2.5 f0 -- ao dimensionar a malha,
        deixando uma margem sobre os 2.2 f0 medidos.
    """)
    a.dica("""
        Regras de bolso que voce vai usar o curso inteiro:

            frequencia de pico          = f0
            banda energetica (-6 dB)   ~ 1.6 f0    -> pense em RESOLUCAO
            banda conservadora (-20 dB) ~ 2.2 f0   -> dimensione a MALHA com 2.5 f0
            comprimento de onda minimo  lambda_min = c_min / f_max

        lambda_min e o que dita o espacamento da malha (aula 02) E o limite de
        resolucao da FWI: nao se recupera estrutura menor que ~lambda/2.
    """)

    # ---- figura 1: Ricker no tempo e na frequencia --------------------
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    t = np.arange(nt) * dt
    for f0, cor in zip([5, 10, 25], ["#1b6ca8", "#d1495b", "#2a9d8f"]):
        w = ricker(f0, dt, nt)
        ax[0].plot(t, w, color=cor, lw=1.4, label=f"f0 = {f0} Hz")
        plot.espectro(ax[1], w, dt, rotulo=f"f0 = {f0} Hz", fmax=90,
                      color=cor, lw=1.4)
    ax[0].set_xlim(0, 0.45)
    ax[0].set_xlabel("tempo (s)")
    ax[0].set_ylabel("amplitude")
    ax[0].set_title("Ricker no tempo")
    ax[0].legend(fontsize=8)
    ax[1].set_title("espectro de amplitude")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    plot.salvar(fig, a, "01_ricker_tempo_frequencia", mostrar=False)

    # ---- figura 2: resolucao vs frequencia ---------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    c_vals = [1500, 2500, 4000]
    f0s = np.linspace(3, 45, 100)
    for c, cor in zip(c_vals, ["#1b6ca8", "#d1495b", "#2a9d8f"]):
        lam = c / (2.5 * f0s)
        ax[0].plot(f0s, lam, color=cor, lw=1.5, label=f"c = {c} m/s")
        ax[1].plot(f0s, lam / 2, color=cor, lw=1.5, label=f"c = {c} m/s")
    ax[0].set_xlabel("f0 (Hz)"); ax[0].set_ylabel("lambda_min (m)")
    ax[0].set_title("menor comprimento de onda (f_max = 2.5 f0)")
    ax[1].set_xlabel("f0 (Hz)"); ax[1].set_ylabel("lambda_min / 2 (m)")
    ax[1].set_title("limite de resolucao aproximado da FWI")
    for e in ax:
        e.legend(fontsize=8); e.set_yscale("log")
    fig.tight_layout()
    plot.salvar(fig, a, "02_resolucao_vs_frequencia", mostrar=False)

    a.pergunta(
        "Voce inverte com f0 = 8 Hz num meio de 2000 m/s. Qual a menor "
        "estrutura que pode esperar recuperar?",
        ["cerca de 10 m", "cerca de 50 m", "cerca de 250 m", "cerca de 1000 m"],
        1,
        "f_max ~ 2.5 * 8 = 20 Hz. lambda_min = 2000/20 = 100 m. O limite de "
        "resolucao e ~lambda/2 = 50 m. Aumentar a frequencia melhora a "
        "resolucao -- mas piora o cycle skipping (aula 07). Esse e o dilema "
        "central da FWI e a razao da estrategia multiescala (aula 12).")

    # ---- figura interativa -------------------------------------------
    if mostrar_figuras():
        a.secao("Explore: wavelet interativa")
        a.texto("""
            Uma janela vai abrir com dois controles deslizantes: f0 e o atraso
            t0. Observe tres coisas: (1) aumentar f0 estreita a wavelet no tempo
            e alarga o espectro; (2) t0 pequeno demais faz a wavelet ser cortada
            em t=0, o que introduz uma descontinuidade e suja o espectro;
            (3) a banda util e sempre ~2.5 f0, independente de f0. Feche a janela
            para continuar.
        """)
        from matplotlib.widgets import Slider
        figi, (axt, axf) = plt.subplots(1, 2, figsize=(11, 4))
        plt.subplots_adjust(bottom=0.28)
        w0 = ricker(10, dt, nt)
        (lt,) = axt.plot(t, w0, lw=1.5, color="#1b6ca8")
        axt.set_xlim(0, 0.5); axt.set_ylim(-0.8, 1.1)
        axt.set_xlabel("tempo (s)"); axt.set_title("Ricker no tempo")
        f, S = espectro_amplitude(w0, dt)
        (lf,) = axf.plot(f, S, lw=1.5, color="#d1495b")
        axf.set_xlim(0, 100); axf.set_ylim(0, 1.05)
        axf.set_xlabel("frequencia (Hz)"); axf.set_title("espectro")
        s_f0 = Slider(plt.axes([0.12, 0.14, 0.76, 0.03]), "f0 (Hz)", 2.0, 45.0,
                      valinit=10.0)
        s_t0 = Slider(plt.axes([0.12, 0.07, 0.76, 0.03]), "t0 (s)", 0.0, 0.35,
                      valinit=0.10)

        def atualiza(_):
            w = ricker(s_f0.val, dt, nt, atraso=s_t0.val)
            lt.set_ydata(w)
            ff, SS = espectro_amplitude(w, dt)
            lf.set_data(ff, SS)
            figi.canvas.draw_idle()

        s_f0.on_changed(atualiza)
        s_t0.on_changed(atualiza)
        plt.show()

    # ==================================================================
    a.secao("Exercicios")
    a.exercicio(1, """
        Mostre analiticamente que o espectro da Ricker tem maximo em f0.
        Parta da transformada W(f) proporcional a f^2 exp(-f^2/f0^2) e derive.
    """, dica="dW/df = 0 leva a 2f - 2f^3/f0^2 = 0.")
    a.exercicio(2, """
        Meca numericamente a razao f_max/f0 usando quedas de -3, -6, -12 e -20
        dB. Monte uma tabela. Qual criterio voce usaria para dimensionar a malha
        e por que?
    """, dica="use fwikit.acustico.largura_banda(w, dt, queda_db).")
    a.exercicio(3, """
        Escreva, para o caso de densidade VARIAVEL, a diferenca entre
        div((1/rho) grad p) e (1/rho) laplaciano(p). Em que situacao geologica
        essa diferenca deixa de ser desprezivel?
    """, dica="expanda o divergente pela regra do produto; o termo extra "
              "envolve grad(1/rho), que so importa em contrastes bruscos.")

    a.fim([
        "A aproximacao acustica e UMA hipotese: mu = 0 (o meio nao cisalha).",
        "Com mu = 0 o tensor de tensoes fica isotropico e p = -(sigma_xx+sigma_zz)/2.",
        "A equacao resultante: (1/c^2) p_tt - laplaciano(p) = s. Invertemos em m = 1/c^2 "
        "porque a equacao e linear nesse parametro.",
        "O regime acustico preserva a CINEMATICA das ondas P e descarta cisalhamento, "
        "conversoes, ondas de superficie, AVO de alto angulo e atenuacao.",
        "Ricker: pico em f0, banda util ate ~2.5 f0, resolucao ~lambda_min/2.",
    ], proxima="aula02_diferencas_finitas.py -- estabilidade e dispersao numerica")


if __name__ == "__main__":
    main()
