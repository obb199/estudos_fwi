#!/usr/bin/env python3
"""
AULA 04 -- Aquisicao: geometria, sismogramas e o que o dado realmente ve
========================================================================
Execute:  python aulas/aula04_aquisicao.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                               # noqa: E402
import fwikit                                    # noqa: E402
from fwikit.aula import Aula, mostrar_figuras    # noqa: E402
from fwikit import plot                          # noqa: E402
from fwikit.acustico import (Acustico2D, Config, Geometria,   # noqa: E402
                             ricker, dt_maximo)


def main():
    a = Aula(4, "Aquisicao: geometria, sismogramas e o que o dado realmente ve",
             modulo="Modulo I -- Fisica e numerica da propagacao",
             duracao="60 min", pre_requisitos="aulas 01-03",
             objetivos=[
                 "Montar geometrias de aquisicao de superficie e de poco",
                 "Ler um shot gather e identificar direta, refracao e reflexoes",
                 "Relacionar abertura (offset maximo) com profundidade de investigacao",
                 "Entender por que a FWI usa shot gathers e nao dado empilhado",
                 "Avaliar o efeito de ruido e de amostragem espacial insuficiente",
             ])
    a.cabecalho()

    # ==================================================================
    a.secao("Vocabulario: o dado sismico e uma matriz de 4 indices")
    a.texto("""
        Um levantamento sismico produz tracos indexados por (fonte, receptor,
        tempo). Reorganizar esses indices da nomes diferentes ao mesmo dado:
    """)
    a.tabela(
        ["Organizacao", "O que fica fixo", "Uso tipico"],
        [["Shot gather", "a fonte", "modelagem e FWI"],
         ["Common receiver", "o receptor", "analise de estatica"],
         ["Common offset", "a distancia fonte-receptor", "controle de qualidade"],
         ["CMP gather", "o ponto medio", "analise de velocidade, empilhamento"]])
    a.dica("""
        A FWI trabalha SEMPRE em shot gathers, e a razao e conceitual: a
        modelagem direta simula um tiro por vez. F(m) produz naturalmente um
        shot gather, entao o residuo F(m) - d_obs tem que ser calculado nessa
        mesma organizacao.

        E por isso que a FWI nao usa dado empilhado (stack). O empilhamento joga
        fora a dependencia com o offset -- justamente a informacao que permite
        separar velocidade de profundidade.
    """)

    # ==================================================================
    a.secao("Construindo um levantamento")
    nz, nx, dh = 110, 260, 10.0
    c = np.full((nz, nx), 1800.0, dtype=np.float32)
    c[40:, :] = 2300.0
    c[75:, :] = 3000.0
    zz, xx = np.mgrid[0:nz, 0:nx]
    c[((zz - 58) ** 2 / 8 ** 2 + (xx - 150) ** 2 / 26 ** 2) < 1] = 2650.0

    f0 = 10.0
    dt = dt_maximo(c.max(), dh, 4)
    nt = int(2.0 / dt)
    cfg = Config(dh=dh, dt=dt, nt=nt, ordem=4, n_abs=40, f0=f0)

    n_tiros = 9
    x_tiros = np.linspace(20, nx - 21, n_tiros).astype(int)
    fontes = np.array([[4, ix] for ix in x_tiros])
    receptores = np.array([[4, ix] for ix in range(6, nx - 6, 2)])
    geom = Geometria(fontes=fontes, receptores=receptores)
    w = ricker(f0, dt, nt)
    solver = Acustico2D(cfg, c.shape)

    print()
    a.resultado("dominio", f"{nx*dh/1000:.2f} x {nz*dh/1000:.2f}", "km")
    a.resultado("tiros", geom.ns)
    a.resultado("receptores por tiro", geom.nr)
    a.resultado("espacamento de receptores", f"{2*dh:.0f}", "m")
    extensao = (receptores[:, 1].max() - receptores[:, 1].min()) * dh
    off_max = max(abs(receptores[:, 1] * dh - f[1] * dh).max() for f in fontes)
    a.resultado("extensao do arranjo", f"{extensao/1000:.2f}", "km")
    a.resultado("offset maximo (melhor tiro)", f"{off_max/1000:.2f}", "km")
    a.resultado("tempo de registro", f"{nt*dt:.2f}", "s")
    a.resultado("tracos totais", geom.ns * geom.nr)

    a.teoria("Abertura e profundidade de investigacao", """
        A regra pratica mais usada:

            profundidade maxima confiavel ~ offset_maximo / 3  a  offset_maximo / 2

        A razao e geometrica. Para iluminar um ponto em profundidade z com
        angulos suficientemente variados -- e e a variedade de angulos que
        permite separar velocidade de posicao -- e preciso ter raios chegando
        de longe. Offset curto so gera incidencia quase normal, que e
        ambigua: mais lento e mais raso produz o mesmo tempo que mais rapido
        e mais fundo.

        Confira com os numeros deste levantamento, impressos acima: com o
        offset maximo desta geometria, a faixa confiavel vai ate cerca de um
        terco a metade desse valor. Abaixo disso a FWI passa a extrapolar -- e
        extrapolacao, em FWI, tem outro nome: artefato.
    """)

    # ---- modela todos os tiros ---------------------------------------
    print()
    print("    modelando tiros...", end="", flush=True)
    d_obs = np.stack([solver.modelar(c, geom, w, i)[0] for i in range(geom.ns)],
                     axis=-1)
    print(f" pronto. d_obs com forma {d_obs.shape} (nt, nr, ns)")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(15, 4.2))
    plot.modelo(ax[0], c, dh, "modelo + geometria", "m/s")
    ax[0].plot(fontes[:, 1] * dh, fontes[:, 0] * dh, "r*", ms=11, mec="k")
    ax[0].plot(receptores[:, 1] * dh, receptores[:, 0] * dh, "wv", ms=2)
    for k, itiro in enumerate([0, 4, 8]):
        off = (receptores[:, 1] - fontes[itiro, 1]) * dh
        plot.sismograma(ax[k + 1], d_obs[:, :, itiro], dt,
                        f"tiro {itiro+1} (x = {fontes[itiro,1]*dh/1000:.2f} km)")
        ax[k + 1].set_xlabel("receptor")
    fig.tight_layout()
    plot.salvar(fig, a, "01_geometria_e_shot_gathers", mostrar=False)

    # ==================================================================
    a.secao("Lendo um shot gather")
    a.texto("""
        Um shot gather tem uma anatomia reconhecivel. Do mais rapido ao mais
        lento em offset longo:
    """)
    a.lista([
        "ONDA DIRETA -- reta com inclinacao 1/c da camada mais rasa. Passa pela "
        "origem (t=0 em offset 0). Nao carrega informacao de profundidade, e "
        "frequentemente e removida (mute).",
        "REFRACOES (head waves) -- retas com inclinacao 1/c das camadas mais "
        "profundas, que CRUZAM a direta a partir de um offset critico. "
        "Dominam o offset longo. Sao a fonte mais rica de informacao de "
        "velocidade de grande escala para a FWI.",
        "REFLEXOES -- hiperboles. O apice esta em offset zero, no tempo de "
        "ida-e-volta vertical t0 = 2z/c. A curvatura (moveout) e que revela a "
        "velocidade media ate o refletor.",
        "DIFRACOES -- hiperboles com apice deslocado, geradas por "
        "descontinuidades pontuais (bordas de falha, o topo da nossa anomalia).",
    ])
    a.eq("t(x)^2  =  t0^2  +  x^2 / v_rms^2", rotulo="moveout hiperbolico")

    # tempos teoricos dos eventos principais
    itiro = 4
    x_src = fontes[itiro, 1] * dh
    offsets = (receptores[:, 1] * dh - x_src)
    z1, c1 = 40 * dh, 1800.0
    z2, c2 = 75 * dh, 2300.0
    t_dir = np.abs(offsets) / c1
    t_teo = np.sqrt((2 * z1 / c1) ** 2 + (offsets / c1) ** 2)
    v_rms2 = np.sqrt((c1 ** 2 * (2 * z1 / c1) + c2 ** 2 * (2 * (z2 - z1) / c2))
                     / (2 * z1 / c1 + 2 * (z2 - z1) / c2))
    t0_2 = 2 * z1 / c1 + 2 * (z2 - z1) / c2
    t_teo2 = np.sqrt(t0_2 ** 2 + (offsets / v_rms2) ** 2)
    x_crit = 2 * z1 * np.sqrt((c2 + c1) / (c2 - c1))
    print()
    a.resultado("v_rms ate a base da camada 2", f"{v_rms2:.0f}", "m/s")
    a.resultado("t0 da reflexao da camada 1", f"{2*z1/c1:.3f}", "s")
    a.resultado("t0 da reflexao da camada 2", f"{t0_2:.3f}", "s")
    a.resultado("offset critico da refracao c2", f"{x_crit:.0f}", "m")
    print()
    print(f"    {'offset (m)':<14}{'direta (s)':<14}{'refl. 1 (s)':<14}"
          f"{'refl. 2 (s)':<14}")
    for k in [len(offsets) // 2, len(offsets) // 2 + 20,
              len(offsets) // 2 + 40, len(offsets) - 3]:
        print(f"    {offsets[k]:<14.0f}{t_dir[k]:<14.3f}{t_teo[k]:<14.3f}"
              f"{t_teo2[k]:<14.3f}")
    print()
    a.texto("""
        Confira estes numeros contra a figura seguinte, onde as curvas teoricas
        estao sobrepostas ao sismograma. Note como as duas hiperboles ficam
        quase planas em offset curto -- e por isso que offset curto informa
        pouco sobre velocidade -- e como se separam da direta a medida que o
        offset cresce.
    """)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    plot.sismograma(ax[0], d_obs[:, :, itiro], dt,
                    "shot gather central com eventos marcados",
                    dx_rec=None)
    idx = np.arange(len(offsets))
    ax[0].plot(idx, t_dir, "r-", lw=1.2, label="onda direta (1/c1)")
    ax[0].plot(idx, t_teo, "y--", lw=1.2, label="reflexao base camada 1")
    ax[0].plot(idx, t_teo2, "c--", lw=1.2, label="reflexao base camada 2")
    ax[0].legend(fontsize=7, loc="lower right")
    ax[0].set_ylim(nt * dt, 0)
    plot.tracos(ax[1], d_obs[:, :, itiro], dt,
                indices=range(0, geom.nr, 8), titulo="mesmos dados em wiggle")
    fig.tight_layout()
    plot.salvar(fig, a, "02_anatomia_shot_gather", mostrar=False)

    a.pergunta(
        "Voce quer melhorar a recuperacao da velocidade ABAIXO de 2 km. "
        "Qual mudanca de aquisicao ajuda mais?",
        ["Diminuir o espacamento entre receptores",
         "Aumentar o offset maximo",
         "Aumentar a frequencia da fonte",
         "Aumentar o numero de tiros na mesma extensao"],
        1,
        "Profundidade de investigacao e governada pela ABERTURA. Receptores mais "
        "densos combatem aliasing espacial, mais tiros melhoram a razao "
        "sinal/ruido e a cobertura, e frequencia maior melhora resolucao -- mas "
        "nenhum deles ilumina mais fundo. So offset maior faz isso.")

    a.pausa()

    # ==================================================================
    a.secao("Amostragem espacial: aliasing")
    a.texto("""
        Assim como no tempo, a amostragem espacial tem um limite de Nyquist. Um
        evento com inclinacao (dt/dx) precisa de pelo menos dois receptores por
        periodo aparente:
    """)
    a.eq("dx_receptor  <  c_aparente / (2 f_max)", rotulo="Nyquist espacial")
    print()
    for dx_rec in [2, 4, 8, 16]:
        dx_m = dx_rec * dh
        c_ap = 1800.0
        f_alias = c_ap / (2 * dx_m)
        print(f"    espacamento {dx_m:>5.0f} m  ->  aliasing acima de "
              f"{f_alias:>6.1f} Hz   "
              f"({'ok' if f_alias > 2.5*f0 else 'INSUFICIENTE para f0=%.0f Hz' % f0})")
    print()
    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    for k, passo in enumerate([1, 3, 6, 12]):
        plot.sismograma(ax[k], d_obs[:, ::passo, itiro], dt,
                        f"dx = {passo*2*dh:.0f} m ({d_obs[:,::passo,itiro].shape[1]} rec)")
    fig.tight_layout()
    plot.salvar(fig, a, "03_aliasing_espacial", mostrar=False)
    a.texto("""
        Nos paineis mais esparsos os eventos inclinados deixam de ser continuos e
        passam a parecer uma serie de segmentos desconexos. Esse e o rosto do
        aliasing espacial. Em FWI ele e especialmente perverso porque a inversao
        tenta explicar o padrao serrilhado com estrutura no modelo.
    """)

    # ==================================================================
    a.secao("Ruido e o limite pratico da FWI")
    a.texto("""
        Dado real tem ruido. A FWI de minimos quadrados assume implicitamente
        que o residuo restante e ruido gaussiano nao correlacionado -- se o
        ruido for coerente (ground roll, multiplas, interferencia), a inversao
        tenta modela-lo.
    """)
    rng = np.random.default_rng(42)
    sinal_rms = float(np.sqrt(np.mean(d_obs[:, :, itiro] ** 2)))
    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    print()
    print(f"    {'S/R (dB)':<12}{'rms do ruido':<18}{'correlacao com o dado limpo'}")
    for k, snr_db in enumerate([np.inf, 20, 10, 3]):
        if np.isinf(snr_db):
            dr = d_obs[:, :, itiro]
            rms_r = 0.0
        else:
            rms_r = sinal_rms / (10 ** (snr_db / 20))
            dr = d_obs[:, :, itiro] + rng.normal(0, rms_r,
                                                 d_obs[:, :, itiro].shape)
        corr = np.corrcoef(dr.ravel(), d_obs[:, :, itiro].ravel())[0, 1]
        rot = "sem ruido" if np.isinf(snr_db) else f"S/R = {snr_db:.0f} dB"
        print(f"    {rot:<12}{rms_r:<18.4e}{corr:.4f}")
        plot.sismograma(ax[k], dr, dt, rot)
    print()
    fig.tight_layout()
    plot.salvar(fig, a, "04_ruido", mostrar=False)
    a.dica("""
        Na pratica: FWI tolera bem ruido ALEATORIO, porque ele nao se soma
        coerentemente no gradiente quando voce empilha muitos tiros. O que a
        FWI NAO tolera e ruido COERENTE e erro de modelagem sistematico --
        esses somam em fase e viram estrutura no modelo.

        E por isso que, na hora de comparar experimentos, vale mais controlar a
        fisica do operador do que perseguir o ultimo dB de S/R.
    """)

    # ==================================================================
    a.secao("Outras geometrias")
    a.texto("""
        Superficie nao e a unica opcao, e o PyFWI suporta as principais
        (`acq_type`): aquisicao de superficie e crosswell (entre pocos).
    """)
    a.tabela(
        ["Geometria", "Fontes / receptores", "Vantagem", "Limitacao"],
        [["Superficie", "ambos no topo", "barata, cobertura lateral",
          "iluminacao angular limitada"],
         ["VSP", "fonte no topo, rec. no poco", "alta resolucao no poco",
          "cobertura lateral minima"],
         ["Crosswell", "poco a poco", "transmissao pura, otima para FWI",
          "so entre os pocos"]])
    a.texto("""
        Do ponto de vista da FWI, crosswell e o caso mais favoravel que existe:
        as ondas atravessam o alvo em transmissao, com cobertura angular quase
        completa. Superficie e o mais dificil, porque toda a informacao precisa
        voltar para o topo. Quando um artigo mostra uma FWI espetacular, vale
        sempre conferir qual geometria foi usada.
    """)

    if mostrar_figuras():
        a.secao("Explore: geometria interativa")
        a.texto("""
            Mova a posicao do tiro e veja o shot gather mudar. Repare em como a
            hiperbole se desloca e como a anomalia so aparece em alguns tiros --
            e disso que vem a necessidade de cobertura. Feche para continuar.
        """)
        from matplotlib.widgets import Slider
        figi, (axm, axs) = plt.subplots(1, 2, figsize=(13, 4.6))
        plt.subplots_adjust(bottom=0.2)
        plot.modelo(axm, c, dh, "modelo", "m/s")
        (marcador,) = axm.plot([fontes[0, 1] * dh], [fontes[0, 0] * dh], "r*",
                               ms=16, mec="k")
        imgs = plot.sismograma(axs, d_obs[:, :, 0], dt, "shot gather")
        s_t = Slider(plt.axes([0.15, 0.06, 0.7, 0.03]), "tiro", 1, geom.ns,
                     valinit=1, valstep=1)

        def atualiza(_):
            k = int(s_t.val) - 1
            marcador.set_data([fontes[k, 1] * dh], [fontes[k, 0] * dh])
            imgs.set_data(d_obs[:, :, k])
            axs.set_title(f"shot gather {k+1}")
            figi.canvas.draw_idle()

        s_t.on_changed(atualiza)
        plt.show()

    a.secao("Exercicios")
    a.exercicio(1, """
        Meca a velocidade da primeira camada ajustando uma reta a onda direta no
        shot gather. Compare com o valor verdadeiro (1800 m/s). Qual o erro?
    """, dica="use o primeiro cruzamento por zero ou o pico, e np.polyfit.")
    a.exercicio(2, """
        Encontre o offset critico a partir do qual a refracao da segunda camada
        ultrapassa a onda direta. Compare com a previsao teorica
        x_c = 2 z1 sqrt((c2+c1)/(c2-c1)).
    """)
    a.exercicio(3, """
        Monte uma geometria crosswell (fontes num poco a esquerda, receptores num
        poco a direita) para o mesmo modelo e compare o dado com o de superficie.
        Qual dos dois "ve" melhor a anomalia eliptica, e por que?
    """, dica="em crosswell a anomalia e atravessada em transmissao.")

    a.fim([
        "FWI trabalha em shot gathers porque F(m) simula um tiro por vez.",
        "Anatomia do gather: direta (reta), refracoes (retas que cruzam), "
        "reflexoes (hiperboles), difracoes (hiperboles deslocadas).",
        "Profundidade confiavel ~ offset_maximo / 2 a / 3. Abertura e o que ilumina fundo.",
        "Amostragem espacial insuficiente gera aliasing, que a FWI tenta explicar "
        "com estrutura falsa.",
        "Ruido aleatorio a FWI tolera; ruido coerente e erro de modelagem, nao.",
    ], proxima="aula05_pyfwi_modelagem.py -- PyFWI na pratica e a prova do regime acustico")


if __name__ == "__main__":
    main()
