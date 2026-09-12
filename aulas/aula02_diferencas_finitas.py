#!/usr/bin/env python3
"""
AULA 02 -- Diferencas finitas: estabilidade, dispersao e como dimensionar a malha
=================================================================================
Execute:  python aulas/aula02_diferencas_finitas.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                               # noqa: E402
import fwikit                                    # noqa: E402
from fwikit.aula import Aula, mostrar_figuras    # noqa: E402
from fwikit import plot                          # noqa: E402
from fwikit.acustico import (COEF_D2, Acustico2D, Config, Geometria,  # noqa: E402
                             cfl, cfl_limite, dt_maximo, ricker,
                             pontos_por_comprimento_onda)


def main():
    a = Aula(2, "Diferencas finitas: estabilidade, dispersao e a malha",
             modulo="Modulo I -- Fisica e numerica da propagacao",
             duracao="60 min", pre_requisitos="aula 01",
             objetivos=[
                 "Montar operadores de diferencas finitas de ordem 2, 4 e 8",
                 "Deduzir e aplicar a condicao CFL de estabilidade",
                 "Reconhecer dispersao numerica e saber distingui-la de fisica real",
                 "Dimensionar dh e dt para um problema, com criterio, e nao no chute",
             ])
    a.cabecalho()

    # ==================================================================
    a.secao("Como se aproxima uma derivada segunda")
    a.texto("""
        A equacao acustica tem derivadas segundas no tempo e no espaco. A receita
        e a mesma nos dois casos: expandir em Taylor em torno do ponto e combinar
        os vizinhos de modo a cancelar o maximo possivel de termos.
    """)
    a.eq("d2f/dx2  ~  (1/h^2) [ c0 f_i  +  soma_k ck ( f_{i+k} + f_{i-k} ) ]",
         rotulo="estencil centrado")
    a.texto("""
        Os coeficientes saem de um sistema linear que impoe o cancelamento dos
        termos de Taylor de ordem baixa. Sao estes:
    """)
    linhas = []
    for ordem in (2, 4, 8):
        c = COEF_D2[ordem]
        coefs = ", ".join(f"{v:+.5f}" for v in c)
        linhas.append([f"O({ordem})", len(c) * 2 - 1, coefs])
    a.tabela(["ordem", "pontos", "coeficientes (c0, c1, c2, ...)"], linhas)

    a.texto("""
        Vale conferir que eles realmente aproximam a derivada. Testamos em
        f(x) = sin(kx), cuja derivada segunda exata e -k^2 sin(kx).
    """)
    h = 1.0
    k = 0.3
    x = np.arange(-6, 7) * h
    f = np.sin(k * x)
    exato = -k ** 2 * np.sin(0.0)
    print()
    print(f"    {'ordem':<10}{'numerico':<18}{'exato':<18}{'erro relativo'}")
    for ordem in (2, 4, 8):
        c = COEF_D2[ordem]
        N = len(c) - 1
        i0 = len(x) // 2
        num = c[0] * f[i0] + sum(c[j] * (f[i0 + j] + f[i0 - j])
                                 for j in range(1, N + 1))
        num /= h ** 2
        # em x=0 a derivada segunda e 0; usamos x=h/2 deslocado para nao zerar
        xr = x + 0.5
        fr = np.sin(k * xr)
        numr = c[0] * fr[i0] + sum(c[j] * (fr[i0 + j] + fr[i0 - j])
                                   for j in range(1, N + 1))
        numr /= h ** 2
        exr = -k ** 2 * np.sin(k * xr[i0])
        print(f"    O({ordem}){'':<6}{numr:<18.10f}{exr:<18.10f}"
              f"{abs(numr-exr)/abs(exr):.3e}")
    print()
    a.dica("""
        Ordem maior nao e so "mais preciso": e mais preciso PARA O MESMO dh, o
        que permite usar dh MAIOR e economizar memoria e tempo. O custo e um
        estencil mais largo (mais comunicacao, bordas mais espessas). Em FWI de
        producao, O(8) no espaco e O(2) no tempo e a combinacao mais comum --
        e exatamente o que o PyFWI oferece via `inpa['sdo'] = 8`.
    """)

    # ==================================================================
    a.secao("Estabilidade: a condicao CFL")
    a.texto("""
        O esquema leap-frog no tempo e condicionalmente estavel. Se dt for grande
        demais, a solucao numerica cresce exponencialmente e o campo vira NaN em
        poucas centenas de passos. A analise de von Neumann -- substituir uma
        onda plana exp(i(kx - wt)) no esquema discreto e exigir |fator de
        amplificacao| <= 1 -- leva a condicao:
    """)
    a.eq("C = c_max dt / dh  <=  C_limite(ordem, dimensao)", rotulo="CFL")
    a.eq("C_limite = 2 / sqrt( ndim * ( |c0| + 2 soma_k |ck| ) )")
    linhas = []
    for ordem in (2, 4, 8):
        lim = cfl_limite(ordem, 2)
        linhas.append([f"O({ordem})", f"{lim:.4f}",
                       f"{dt_maximo(3000, 10, ordem)*1e3:.3f} ms",
                       f"{dt_maximo(4500, 25, ordem)*1e3:.3f} ms"])
    a.tabela(["ordem espacial", "C_limite (2D)",
              "dt_max (c=3000, dh=10)", "dt_max (c=4500, dh=25)"], linhas)
    a.aviso("""
        Note o contra-senso aparente: ordem espacial MAIOR reduz o dt maximo.
        O estencil mais largo amplifica mais os numeros de onda altos, entao a
        condicao de estabilidade aperta. Voce ganha em dh e perde em dt.

        E note tambem: o limite depende de c_MAX. Uma unica celula com
        velocidade alta (sal, basalto, um erro de interpolacao) derruba o dt
        do modelo inteiro.
    """)

    a.secao("Demonstracao: o que a instabilidade parece")
    a.texto("""
        Vamos rodar a MESMA simulacao com tres valores de dt: um seguro, um no
        limite e um acima do limite. Observe a amplitude maxima do campo.
    """)
    c = np.full((60, 80), 2500.0, dtype=np.float32)
    lim = cfl_limite(4, 2)
    dh = 10.0
    dt_lim = lim * dh / c.max()
    print()
    print(f"    c_max = {c.max():.0f} m/s   dh = {dh:.0f} m   "
          f"C_limite = {lim:.4f}   dt_limite = {dt_lim*1e3:.4f} ms")
    print()
    print(f"    {'dt (ms)':<12}{'C':<10}{'amp. maxima':<18}{'situacao'}")
    resultados = []
    for fator in [0.5, 0.9, 1.02, 1.15]:
        dt = fator * dt_lim
        cfg = Config(dh=dh, dt=dt, nt=400, ordem=4, n_abs=20, f0=15.0)
        s = Acustico2D(cfg, c.shape)
        geom = Geometria(fontes=np.array([[30, 40]]),
                         receptores=np.array([[5, 40]]))
        w = ricker(cfg.f0, cfg.dt, cfg.nt)
        d, _ = s.modelar(c, geom, w)
        amax = float(np.abs(d).max())
        ok = np.isfinite(amax) and amax < 1e3
        estado = "estavel" if ok else "EXPLODIU"
        print(f"    {dt*1e3:<12.4f}{cfl(c.max(), dt, dh):<10.4f}"
              f"{amax:<18.3e}{estado}")
        resultados.append((fator, cfl(c.max(), dt, dh), amax, ok))
    print()
    a.texto("""
        O comportamento nao e gradual: abaixo do limite funciona, acima estoura.
        Nao existe "um pouquinho instavel". Por isso se usa sempre uma margem de
        seguranca de 10-20% (o `dt_maximo` do fwikit ja aplica 0.9).
    """)

    a.pergunta(
        "Voce refina a malha de dh = 20 m para dh = 10 m. O que acontece com o "
        "custo total da simulacao 2D?",
        ["dobra", "quadruplica", "multiplica por 8", "nao muda"],
        2,
        "Metade de dh dobra nz e dobra nx (4x mais pontos). E o CFL obriga a "
        "reduzir dt pela metade tambem, dobrando o numero de passos. "
        "Total: 4 x 2 = 8x. Em 3D seria 16x. E por isso que dimensionar a "
        "malha com criterio -- e nao por excesso de zelo -- vale tanto tempo.")

    a.pausa()

    # ==================================================================
    a.secao("Dispersao numerica: o erro que parece geologia")
    a.texto("""
        Estabilidade so garante que o campo nao explode. Nao garante que ele
        esteja certo. O erro mais traicoeiro das diferencas finitas e a
        DISPERSAO NUMERICA: no esquema discreto, a velocidade de fase passa a
        depender do numero de onda.
    """)
    a.eq("c_num(k)  !=  c   quando  k dh  nao e pequeno", rotulo="dispersao")
    a.texto("""
        Componentes de alta frequencia viajam com a velocidade errada, e o pulso
        se abre numa cauda oscilatoria atras da frente de onda. O perigo e que
        isso PARECE sinal: reverberacao, camada fina, ringing da fonte. Quem nao
        conhece o efeito interpreta artefato numerico como geologia.

        O controle e o numero de pontos por comprimento de onda:
    """)
    a.eq("G = lambda_min / dh = c_min / (f_max dh)",
         rotulo="pontos por comprimento de onda")
    a.tabela(["ordem espacial", "G minimo recomendado", "comentario"],
             [["O(2)", "~15 a 20", "quase nunca se usa em FWI"],
              ["O(4)", "~8 a 10", "bom equilibrio, padrao do curso"],
              ["O(8)", "~4 a 5", "padrao de producao; malha bem menor"]])

    # curva de dispersao numerica
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    G = np.linspace(2.5, 25, 300)
    kdh = 2 * np.pi / G
    for ordem, cor in zip((2, 4, 8), ["#d1495b", "#1b6ca8", "#2a9d8f"]):
        cc = COEF_D2[ordem]
        N = len(cc) - 1
        # simbolo do operador discreto para a derivada segunda
        simb = -(cc[0] + 2 * sum(cc[j] * np.cos(j * kdh) for j in range(1, N + 1)))
        c_rel = np.sqrt(np.maximum(simb, 0)) / kdh
        ax[0].plot(G, c_rel, color=cor, lw=1.6, label=f"O({ordem})")
    ax[0].axhline(1.0, color="k", lw=0.8, ls="--")
    ax[0].axhspan(0.99, 1.01, color="green", alpha=0.12)
    ax[0].set_xlabel("G = pontos por comprimento de onda")
    ax[0].set_ylabel("c_numerico / c_exato")
    ax[0].set_title("dispersao numerica espacial (faixa verde: erro < 1%)")
    ax[0].set_ylim(0.75, 1.03)
    ax[0].legend(fontsize=8)
    ax[0].invert_xaxis()

    a.aviso("""
        Uma precisao sobre a curva da esquerda: ela e a dispersao do operador
        ESPACIAL isolado (esquema semi-discreto, tempo continuo). A
        discretizacao TEMPORAL acrescenta um erro proprio, e o detalhe
        interessante e que ele tem sinal OPOSTO -- o leap-frog adianta a fase,
        o estencil espacial atrasa, e os dois se cancelam parcialmente.

        Medido para O(4): com G = 6, a curva semi-discreta da 0.9939, enquanto
        o esquema completo da 0.9980 (C = 0.3) ou 1.0055 (C = 0.5). Ou seja, o
        erro real e MENOR que o previsto pela curva espacial sozinha -- ela e
        uma estimativa conservadora, que e o que se quer para dimensionar malha.
    """)

    # demonstracao pratica: mesmo modelo, duas malhas
    a.texto("""
        A figura ao lado mostra o efeito na pratica: o mesmo tiro, no mesmo meio
        homogeneo, simulado com G confortavel e com G apertado.
    """)
    for dh_teste, cor, rot in [(7.0, "#1b6ca8", "G confortavel"),
                               (28.0, "#d1495b", "G apertado")]:
        c0 = 2000.0
        f0 = 12.0
        fmax = 2.5 * f0
        Gv = pontos_por_comprimento_onda(c0, fmax, dh_teste)
        nx = int(2400 / dh_teste)
        nz = int(1200 / dh_teste)
        cfg = Config(dh=dh_teste, dt=0.6 * dt_maximo(c0, dh_teste, 4),
                     nt=int(0.9 / (0.6 * dt_maximo(c0, dh_teste, 4))),
                     ordem=4, n_abs=25, f0=f0)
        cm = np.full((nz, nx), c0, dtype=np.float32)
        s = Acustico2D(cfg, cm.shape)
        geom = Geometria(fontes=np.array([[nz // 2, 4]]),
                         receptores=np.array([[nz // 2, nx - 6]]))
        w = ricker(f0, cfg.dt, cfg.nt)
        d, _ = s.modelar(cm, geom, w)
        tr = d[:, 0] / (np.abs(d[:, 0]).max() + 1e-30)
        t = np.arange(cfg.nt) * cfg.dt
        ax[1].plot(t, tr, color=cor, lw=1.3,
                   label=f"dh = {dh_teste:.0f} m, G = {Gv:.1f} ({rot})")
        print(f"    dh = {dh_teste:5.1f} m -> G = {Gv:5.2f}   "
              f"dt = {cfg.dt*1e3:.3f} ms   nt = {cfg.nt}")
    ax[1].set_xlabel("tempo (s)")
    ax[1].set_ylabel("amplitude normalizada")
    ax[1].set_title("traco a 2.4 km da fonte, meio homogeneo")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    plot.salvar(fig, a, "01_dispersao_numerica", mostrar=False)
    print()
    a.texto("""
        No traco com G apertado a frente de onda chega no tempo certo, mas
        arrasta uma cauda que NAO existe na fisica: o meio e homogeneo, entao so
        poderia haver um pulso. Toda oscilacao depois do pulso e erro numerico.
    """)

    # ==================================================================
    a.secao("Receita para dimensionar uma malha")
    a.texto("""
        Junte as duas restricoes -- dispersao fixa dh, estabilidade fixa dt --
        e a ordem e sempre esta:
    """)
    a.codigo("""
        1. f_max  = 2.5 * f0                      (banda conservadora da Ricker)
        2. dh     = c_min / (G * f_max)           G = 8 (O4) ou 5 (O8)
        3. dt     = 0.9 * C_limite * dh / c_max   (margem de 10%)
        4. nt     = tempo_de_registro / dt
        5. n_abs  >= 0.5 * lambda_max = 0.5 * c_max / f0   (aula 03)
    """, titulo="dimensionamento de malha, na ordem certa")
    print()
    print("    Exemplo trabalhado: c entre 1500 e 4000 m/s, f0 = 10 Hz, O(4), 3 s")
    c_min, c_max, f0, ordem, T = 1500.0, 4000.0, 10.0, 4, 3.0
    f_max = 2.5 * f0
    G = 8.0
    dh = c_min / (G * f_max)
    dt = dt_maximo(c_max, dh, ordem)
    nt = int(T / dt)
    print()
    a.resultado("f_max (conservador)", f"{f_max:.1f}", "Hz")
    a.resultado("lambda_min", f"{c_min/f_max:.1f}", "m")
    a.resultado("dh escolhido (G=8)", f"{dh:.2f}", "m")
    a.resultado("dt maximo (CFL, margem 10%)", f"{dt*1e3:.3f}", "ms")
    a.resultado("nt para 3 s", nt)
    a.resultado("moldura absorvente sugerida", f"{int(0.5*c_max/f0/dh)}", "pontos")
    print()
    a.aviso("""
        Repare no conflito embutido: dh e ditado por c_MIN (o menor comprimento
        de onda) e dt por c_MAX (a onda mais rapida). Um modelo com grande
        contraste de velocidade -- agua a 1500 e sal a 4500, por exemplo -- e
        caro dos dois lados ao mesmo tempo. Essa e a razao pratica de existirem
        malhas variaveis e esquemas implicitos.
    """)

    a.pergunta(
        "Voce ve uma cauda oscilatoria atras da primeira chegada num modelo "
        "homogeneo. Qual a causa mais provavel?",
        ["Reflexao na borda do dominio",
         "Dispersao numerica (G insuficiente)",
         "dt acima do limite CFL",
         "A wavelet tem banda larga demais"],
        1,
        "Num meio homogeneo nao ha o que refletir internamente. Se fosse CFL o "
        "campo teria explodido, nao oscilado suavemente. Cauda dispersiva atras "
        "da frente = G insuficiente. A correcao e reduzir dh OU aumentar a ordem "
        "espacial OU baixar f0.")

    if mostrar_figuras():
        a.secao("Explore: dispersao interativa")
        a.texto("""
            Mova os controles e veja o traco mudar. Aumente dh (reduzindo G) e
            observe a cauda aparecer. Aumente a ordem espacial e veja a mesma
            malha voltar a funcionar. Feche a janela para continuar.
        """)
        from matplotlib.widgets import Slider, RadioButtons
        figi, axi = plt.subplots(figsize=(9.5, 4.5))
        plt.subplots_adjust(bottom=0.32, left=0.3)
        estado = {"dh": 12.0, "ordem": 4, "f0": 12.0}

        def simula():
            dhv, ordv, f0v = estado["dh"], estado["ordem"], estado["f0"]
            c0 = 2000.0
            dtv = 0.6 * dt_maximo(c0, dhv, ordv)
            ntv = int(0.9 / dtv)
            nxv = max(30, int(2400 / dhv))
            nzv = max(20, int(900 / dhv))
            cfgv = Config(dh=dhv, dt=dtv, nt=ntv, ordem=ordv, n_abs=20, f0=f0v)
            cmv = np.full((nzv, nxv), c0, dtype=np.float32)
            sv = Acustico2D(cfgv, cmv.shape)
            gv = Geometria(fontes=np.array([[nzv // 2, 3]]),
                           receptores=np.array([[nzv // 2, nxv - 5]]))
            wv = ricker(f0v, dtv, ntv)
            dv, _ = sv.modelar(cmv, gv, wv)
            tr = dv[:, 0] / (np.abs(dv[:, 0]).max() + 1e-30)
            Gv = pontos_por_comprimento_onda(c0, 2.5 * f0v, dhv)
            return np.arange(ntv) * dtv, tr, Gv

        t0v, tr0, G0 = simula()
        (linha,) = axi.plot(t0v, tr0, lw=1.4, color="#1b6ca8")
        axi.set_xlim(0, 0.9); axi.set_ylim(-1.2, 1.2)
        axi.set_xlabel("tempo (s)"); axi.set_ylabel("amplitude normalizada")
        titulo = axi.set_title(f"G = {G0:.1f}")
        s_dh = Slider(plt.axes([0.35, 0.18, 0.55, 0.03]), "dh (m)", 5.0, 40.0,
                      valinit=12.0)
        s_f0 = Slider(plt.axes([0.35, 0.11, 0.55, 0.03]), "f0 (Hz)", 5.0, 25.0,
                      valinit=12.0)
        radio = RadioButtons(plt.axes([0.04, 0.08, 0.18, 0.18]),
                             ("O(2)", "O(4)", "O(8)"), active=1)

        def atualiza(_=None):
            estado["dh"] = s_dh.val
            estado["f0"] = s_f0.val
            estado["ordem"] = {"O(2)": 2, "O(4)": 4, "O(8)": 8}[radio.value_selected]
            tt, trr, GG = simula()
            linha.set_data(tt, trr)
            titulo.set_text(f"G = {GG:.1f}"
                            + ("   <- dispersao provavel" if GG < 6 else "   (ok)"))
            figi.canvas.draw_idle()

        s_dh.on_changed(atualiza)
        s_f0.on_changed(atualiza)
        radio.on_clicked(atualiza)
        plt.show()

    a.secao("Exercicios")
    a.exercicio(1, """
        Verifique numericamente a ordem de convergencia dos estenciis: calcule o
        erro da derivada segunda de sin(kx) para h, h/2, h/4 e confirme que ele
        cai como h^2, h^4 e h^8.
    """, dica="use COEF_D2 e log2(erro_h / erro_h_meio).")
    a.exercicio(2, """
        Dimensione a malha para: agua 1500 m/s, sal 4500 m/s, f0 = 6 Hz, O(8),
        registro de 6 s, dominio de 12 km x 4 km. Quantos pontos de malha e
        quantos passos de tempo? Estime a memoria de um campo em float32.
    """, dica="nz*nx*4 bytes por campo; a FWI guarda varios.")
    a.exercicio(3, """
        Modifique o codigo desta aula para medir a velocidade de fase NUMERICA:
        propague num meio homogeneo, meca o tempo de chegada em dois receptores
        distantes e compare com c. Repita para G = 4, 6, 8, 12.
    """, dica="use o pico da envoltoria ou o cruzamento por zero como referencia.")

    a.fim([
        "CFL: C = c_max dt/dh <= C_limite. Acima disso o campo explode, sem meio-termo.",
        "Ordem espacial maior permite dh maior, mas aperta o dt.",
        "Dispersao numerica cria cauda oscilatoria que imita geologia. Controle com "
        "G = lambda_min/dh (>=8 para O4, >=5 para O8).",
        "dh vem de c_min; dt vem de c_max. Modelos com contraste alto pagam dos dois lados.",
        "Refinar dh pela metade custa 8x mais em 2D e 16x em 3D.",
    ], proxima="aula03_modelagem_2d.py -- campos de onda, snapshots e bordas absorventes")


if __name__ == "__main__":
    main()
