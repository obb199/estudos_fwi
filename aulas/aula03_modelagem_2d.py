#!/usr/bin/env python3
"""
AULA 03 -- Modelagem 2D: campos de onda, snapshots e bordas absorventes
=======================================================================
Execute:  python aulas/aula03_modelagem_2d.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                               # noqa: E402
import fwikit                                    # noqa: E402
from fwikit.aula import Aula, mostrar_figuras    # noqa: E402
from fwikit import plot                          # noqa: E402
from fwikit.acustico import (Acustico2D, Config, Geometria,   # noqa: E402
                             ricker, dt_maximo, _mascara_sponge)


def modelo_camadas(nz, nx, interfaces, velocidades):
    """Monta um modelo de camadas planas."""
    c = np.zeros((nz, nx), dtype=np.float32)
    limites = [0] + list(interfaces) + [nz]
    for k, v in enumerate(velocidades):
        c[limites[k]:limites[k + 1], :] = v
    return c


def main():
    a = Aula(3, "Modelagem 2D: campos de onda, snapshots e bordas absorventes",
             modulo="Modulo I -- Fisica e numerica da propagacao",
             duracao="70 min", pre_requisitos="aulas 01 e 02",
             objetivos=[
                 "Ler o laco de tempo de um propagador linha a linha",
                 "Interpretar snapshots do campo de onda e identificar cada evento",
                 "Entender por que bordas absorventes sao obrigatorias e como dimensiona-las",
                 "Distinguir superficie livre de borda absorvente e saber quando usar cada uma",
             ])
    a.cabecalho()

    # ==================================================================
    a.secao("O laco de tempo, linha a linha")
    a.texto("""
        Toda a modelagem cabe em cinco linhas. Partindo da equacao acustica
        discretizada com leap-frog no tempo:
    """)
    a.eq("p^(n+1) = 2 p^n - p^(n-1) + c^2 dt^2 [ laplaciano(p^n) + s^n ]",
         rotulo="leap-frog")
    a.codigo("""
        for it in range(nt):
            dados[it] = p_atu[receptores]          # 1. grava o dado
            lap = laplaciano(p_atu)                # 2. operador espacial
            lap[fonte] += wavelet[it] / dh**2      # 3. injeta a fonte
            p_pro = 2*p_atu - p_ant + c2dt2 * lap  # 4. avanca no tempo
            p_pro *= sponge                        # 5. absorve nas bordas
            p_ant, p_atu, p_pro = p_atu, p_pro, p_ant   # rotaciona buffers
    """, titulo="fwikit/acustico.py, metodo _propagar")
    a.texto("""
        Tres detalhes que parecem triviais e nao sao:
    """)
    a.lista([
        "A fonte e dividida por dh^2. A equacao tem s(t) delta(x-xs), e a "
        "discretizacao de uma delta em 2D vale 1/dh^2 na celula. Sem isso, a "
        "amplitude do dado muda quando voce refina a malha -- e voce passa "
        "horas procurando um bug que nao existe.",
        "So TRES campos sao guardados (anterior, atual, proximo) e eles sao "
        "ROTACIONADOS, nao copiados. Copiar arrays a cada passo dobra o tempo "
        "de execucao sem motivo.",
        "O dado e gravado ANTES da atualizacao. Parece irrelevante, mas fixa a "
        "correspondencia de indices entre p^n e d^n -- e disso depende o "
        "gradiente adjunto ser exato (aula 09).",
    ])

    # ==================================================================
    a.secao("Primeiro campo de onda: meio de duas camadas")
    nz, nx, dh = 120, 200, 8.0
    c = modelo_camadas(nz, nx, [55], [1800.0, 2700.0])
    f0 = 14.0
    dt = dt_maximo(c.max(), dh, 4)
    nt = int(1.1 / dt)
    cfg = Config(dh=dh, dt=dt, nt=nt, ordem=4, n_abs=35, f0=f0)
    geom = Geometria(fontes=np.array([[6, nx // 2]]),
                     receptores=np.array([[6, i] for i in range(4, nx - 4, 2)]))
    w = ricker(f0, cfg.dt, cfg.nt)
    solver = Acustico2D(cfg, c.shape)

    print()
    a.resultado("dominio", f"{nx*dh/1000:.2f} x {nz*dh/1000:.2f}", "km")
    a.resultado("malha", f"{nz} x {nx}", "pontos")
    a.resultado("dh / dt", f"{dh:.1f} m / {dt*1e3:.3f} ms")
    a.resultado("nt", nt)
    a.resultado("receptores", geom.nr)

    instantes = [int(f * nt) for f in (0.12, 0.25, 0.42, 0.62)]
    t0 = time.time()
    d, extras = solver.modelar(c, geom, w, snapshots=instantes)
    print()
    a.resultado("tempo de modelagem", f"{time.time()-t0:.2f}", "s")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.2))
    plot.modelo(axes[0, 0], c, dh, "modelo de velocidade", "m/s")
    axes[0, 0].plot(geom.fontes[0, 1] * dh, geom.fontes[0, 0] * dh, "r*",
                    ms=13, mec="k")
    axes[0, 0].plot(geom.receptores[:, 1] * dh, geom.receptores[:, 0] * dh,
                    "wv", ms=2.5)
    for k, it in enumerate(instantes):
        ax = axes.ravel()[k + 1]
        campo = solver.contrair(extras["snapshots"][it])
        plot.perturbacao(ax, campo, dh, f"campo em t = {it*dt*1000:.0f} ms",
                         colorbar=False)
        ax.axhline(55 * dh, color="k", lw=0.8, ls="--", alpha=0.6)
    plot.sismograma(axes[1, 2], d, dt, "sismograma (todos os receptores)",
                    dx_rec=2 * dh)
    fig.tight_layout()
    plot.salvar(fig, a, "01_campos_e_sismograma", mostrar=False)

    a.teoria("O que aparece em cada snapshot", """
        Percorra a figura na ordem e identifique:

        * FRENTE DIRETA -- circulo que sai da fonte. Em meio homogeneo a
          amplitude cai como 1/sqrt(r) em 2D (em 3D seria 1/r).

        * REFLEXAO -- quando a frente cruza a interface (linha tracejada),
          parte da energia volta. A polaridade depende do sinal do contraste
          de impedancia; aqui a camada de baixo e mais rapida, entao a
          reflexao preserva a polaridade.

        * TRANSMISSAO -- o que atravessa, com a frente mais "aberta" porque a
          velocidade la embaixo e maior.

        * ONDA REFRATADA (head wave) -- a frente que corre ao longo da
          interface na velocidade da camada inferior e vaza energia para cima
          num angulo fixo (o angulo critico). E ela que domina os offsets
          longos e carrega a informacao de velocidade de grande escala -- a
          mais valiosa para a FWI.
    """)

    a.pergunta(
        "Qual evento do sismograma carrega mais informacao sobre a VELOCIDADE "
        "de grande escala (os grandes comprimentos de onda do modelo)?",
        ["A onda direta", "As reflexoes de pequeno offset",
         "As refracoes / mergulhos de grande offset", "O ruido de fundo"],
        2,
        "Refracoes e ondas de grande offset percorrem longas distancias "
        "horizontalmente dentro das camadas: o tempo de transito acumula "
        "sensibilidade a velocidade media. Reflexoes de pequeno offset sao "
        "sensiveis ao contraste (alta frequencia), nao a tendencia. "
        "Por isso aquisicoes de offset longo sao tao valorizadas em FWI.")

    a.pausa()

    # ==================================================================
    a.secao("Bordas: o problema e as solucoes")
    a.texto("""
        O computador so tem memoria para um dominio finito, mas a Terra e
        (para efeitos praticos) infinita. Se voce simplesmente truncar o
        dominio, a borda vira um espelho perfeito e o sismograma enche de
        reflexoes que nao existem no dado real.

        Em modelagem isso e feio. Em FWI e pior: o residuo passa a conter
        eventos que a fisica do seu modelo nunca vai explicar, e o gradiente
        tenta explica-los mexendo na velocidade. Voce cria estrutura falsa.
    """)
    a.tabela(
        ["Tecnica", "Como funciona", "Custo", "Qualidade"],
        [["Truncar", "nada", "zero", "inaceitavel"],
         ["Cerjan (sponge)", "multiplica o campo por um taper", "baixo", "razoavel"],
         ["Condicao absorvente", "equacao unidirecional na borda", "baixo", "boa em incidencia normal"],
         ["PML / CPML", "coordenadas complexas; absorve em qualquer angulo", "medio", "excelente"]])
    a.dica("""
        Neste curso o propagador proprio usa CERJAN, por dois motivos didaticos:
        cabe em 10 linhas, e e um operador DIAGONAL -- portanto auto-adjunto.
        Isso preserva a validade do teste de gradiente da aula 09 sem exigir
        que voce derive o adjunto de uma CPML.

        O PyFWI usa CPML, que e o padrao de producao (`inpa['npml']` e
        `inpa['pmlR']`). Quando voce escrever seu proprio codigo para valer,
        CPML e o caminho.
    """)

    a.secao("Experimento: quanto a borda importa")
    a.texto("""
        Mesmo modelo, mesma fonte, variando SO a espessura da moldura
        absorvente. Para medir a reflexao de borda com honestidade precisamos de
        uma REFERENCIA sem borda: rodamos antes a mesma simulacao num dominio
        muito maior, onde nenhuma reflexao de borda chega dentro da janela de
        tempo. A diferenca entre os dois tracos e, por construcao, so artefato.
    """)
    nz2, nx2 = 90, 130
    c2 = np.full((nz2, nx2), 2000.0, dtype=np.float32)
    dh2 = 10.0
    dt2 = dt_maximo(2000.0, dh2, 4)
    nt2 = int(1.4 / dt2)
    lam_max = 2000.0 / 10.0
    print()
    print(f"    lambda na frequencia de pico = {lam_max:.0f} m "
          f"= {lam_max/dh2:.0f} pontos de malha")
    def traco(n_abs, NZ=nz2, NX=nx2):
        cm = np.full((NZ, NX), 2000.0, dtype=np.float32)
        cfg2 = Config(dh=dh2, dt=dt2, nt=nt2, ordem=4, n_abs=n_abs, f0=10.0)
        sv = Acustico2D(cfg2, cm.shape)
        g2 = Geometria(fontes=np.array([[NZ // 2, NX // 2]]),
                       receptores=np.array([[NZ // 2, NX // 2 + 5]]))
        return sv.modelar(cm, g2, ricker(10.0, dt2, nt2))[0][:, 0]

    referencia = traco(30, NZ=400, NX=440)     # dominio grande = sem borda
    amp_direta = float(np.abs(referencia).max())
    i_corte = int(0.40 / dt2)
    print()
    print(f"    amplitude da onda direta (referencia) = {amp_direta:.3e}")
    print()
    print(f"    {'n_abs':<8}{'espessura':<13}{'max|reflexao|':<18}{'em dB'}")
    curvas = []
    for n_abs in [0, 10, 20, 30, 45]:
        tr = traco(n_abs)
        refl = float(np.abs(tr[i_corte:] - referencia[i_corte:]).max())
        db = 20 * np.log10(refl / amp_direta + 1e-30)
        print(f"    {n_abs:<8}{n_abs*dh2:>6.0f} m     "
              f"{refl:<18.4e}{db:>7.1f} dB")
        curvas.append((n_abs, np.arange(nt2) * dt2, tr))
    print()
    a.texto("""
        Leia o resultado com realismo. Com n_abs = 0 a borda e um espelho: a
        reflexao chega a -7 dB da onda direta -- e ela so nao chega a 0 dB porque
        percorreu 900 m a mais e sofreu espalhamento geometrico. Uma moldura de
        Cerjan razoavel leva isso a -16/-18 dB, ou seja, a borda ainda devolve
        uns 13% da amplitude. Isso e MEDIOCRE, e e exatamente por isso que
        codigos de producao usam CPML, que chega a -40/-60 dB. O PyFWI e um
        deles (`inpa['npml']`, `inpa['pmlR']`).

        Note tambem que engrossar a moldura tem retorno decrescente: dobrar
        n_abs de 20 para 45 ganha ~5 dB, nao 20. O taper otimo tambem depende de
        n_abs (forte demais reflete no proprio degrau de absorcao); o `fwikit`
        calibra isso sozinho com fator ~ 0.25/n_abs.
    """)

    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for n_abs, t, tr in curvas:
        ax[0].plot(t, tr / (np.abs(tr).max() + 1e-30), lw=1.1,
                   label=f"n_abs = {n_abs}")
    ax[0].axvline(0.45, color="k", ls="--", lw=0.8)
    ax[0].set_xlabel("tempo (s)"); ax[0].set_ylabel("amplitude normalizada")
    ax[0].set_title("traco: tudo apos a linha tracejada e artefato")
    ax[0].legend(fontsize=7)
    masc = _mascara_sponge(120, 180, 35, 0.25 / 35, False)
    im = ax[1].imshow(masc, cmap="magma", aspect="auto")
    ax[1].set_title("mascara de Cerjan (1 = livre, 0 = absorve)")
    ax[1].grid(False)
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.02)
    fig.tight_layout()
    plot.salvar(fig, a, "02_bordas_absorventes", mostrar=False)

    a.aviso("""
        Regra de dimensionamento: a moldura precisa ter pelo menos MEIO
        comprimento de onda da frequencia mais BAIXA que voce usa -- e a baixa
        frequencia que atravessa a moldura fina.

            n_abs >= 0.5 * c_max / (f_min * dh)

        Como a FWI multiescala (aula 12) comeca justamente nas frequencias
        baixas, dimensione a moldura pela MENOR frequencia do seu plano, nao
        pela frequencia de pico da wavelet.
    """)

    # ==================================================================
    a.secao("Superficie livre: o topo e diferente")
    a.texto("""
        No topo do modelo voce tem uma escolha fisica, nao numerica:

        SUPERFICIE LIVRE (p = 0 em z = 0) -- representa a interface com o ar ou
        com o vacuo. A onda reflete com polaridade INVERTIDA. E o que existe de
        verdade no mar (a superficie da agua) e em terra. Produz as MULTIPLAS
        de superficie e o ghost, que sao sinais reais do dado.

        BORDA ABSORVENTE NO TOPO -- representa um meio que continua para cima.
        Nao e fisico, mas e conveniente: elimina multiplas do sintetico. Muita
        FWI academica roda assim para evitar ter que lidar com multiplas.
    """)
    cfg_sl = Config(dh=dh, dt=dt, nt=nt, ordem=4, n_abs=35,
                    f0=f0, superficie_livre=True)
    s_sl = Acustico2D(cfg_sl, c.shape)
    d_sl, _ = s_sl.modelar(c, geom, w)
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4))
    plot.sismograma(ax[0], d, dt, "topo absorvente", dx_rec=2 * dh)
    plot.sismograma(ax[1], d_sl, dt, "superficie livre (p=0)", dx_rec=2 * dh)
    plot.sismograma(ax[2], d_sl - d, dt, "diferenca = multiplas + ghost",
                    dx_rec=2 * dh)
    fig.tight_layout()
    plot.salvar(fig, a, "03_superficie_livre", mostrar=False)
    a.texto("""
        O terceiro painel isola o que a superficie livre acrescenta. Sao eventos
        de amplitude comparavel as reflexoes primarias -- ignora-los quando o
        dado real os contem e uma das causas mais comuns de FWI que nao converge.
    """)

    a.pergunta(
        "Voce roda FWI com dado real marinho, mas simula com topo absorvente. "
        "O que tende a acontecer?",
        ["Nada; multiplas nao afetam a FWI",
         "O residuo contem multiplas que o modelo nao explica, e a inversao "
         "cria estrutura falsa tentando reproduzi-las",
         "A inversao converge mais rapido",
         "So muda a amplitude, nao a cinematica"],
        1,
        "Qualquer evento presente no dado observado e ausente no sintetico entra "
        "inteiro no residuo. O gradiente entao tenta explicar as multiplas "
        "mexendo na velocidade -- criando artefatos. As saidas sao: modelar a "
        "superficie livre, ou remover as multiplas do dado antes (demultipla).")

    if mostrar_figuras():
        a.secao("Explore: propagacao animada")
        a.texto("""
            Vamos animar o campo de onda. Observe a reflexao na interface, a
            transmissao e a refracao correndo pela interface. Feche a janela para
            continuar.
        """)
        from matplotlib.animation import FuncAnimation
        quadros = list(range(0, nt, max(1, nt // 60)))
        _, ex = solver.modelar(c, geom, w, snapshots=quadros)
        figa, axa = plt.subplots(figsize=(8, 5))
        campo0 = solver.contrair(ex["snapshots"][quadros[0]])
        lim = np.percentile(np.abs(solver.contrair(
            ex["snapshots"][quadros[len(quadros) // 3]])), 99.8)
        ima = axa.imshow(campo0, cmap="RdBu_r", vmin=-lim, vmax=lim,
                         extent=[0, nx * dh, nz * dh, 0], aspect="auto")
        axa.axhline(55 * dh, color="k", lw=1.0, ls="--")
        axa.set_xlabel("x (m)"); axa.set_ylabel("z (m)"); axa.grid(False)
        tit = axa.set_title("t = 0 ms")

        def anima(k):
            ima.set_data(solver.contrair(ex["snapshots"][quadros[k]]))
            tit.set_text(f"t = {quadros[k]*dt*1000:.0f} ms")
            return ima, tit

        _anim = FuncAnimation(figa, anima, frames=len(quadros),
                              interval=60, blit=False, repeat=True)
        plt.show()

    a.secao("Exercicios")
    a.exercicio(1, """
        Adicione uma terceira camada ao modelo e identifique, no sismograma,
        todas as reflexoes primarias. Confira os tempos de chegada em offset
        zero com t = 2 * soma(espessura_i / c_i).
    """)
    a.exercicio(2, """
        Meca a amplitude da onda direta em funcao da distancia e verifique o
        decaimento 1/sqrt(r) esperado em 2D. Por que nao e 1/r?
    """, dica="em 2D a fonte e uma linha infinita perpendicular ao plano.")
    a.exercicio(3, """
        Implemente uma segunda moldura absorvente usando taper LINEAR em vez de
        gaussiano e compare a energia refletida. O que muda e por que a
        suavidade do taper importa?
    """, dica="descontinuidades no coeficiente de absorcao geram reflexao.")

    a.fim([
        "O laco de tempo cabe em 5 linhas; os detalhes (escala 1/dh^2, rotacao de "
        "buffers, indice de gravacao) e que fazem a diferenca.",
        "Snapshots revelam direta, reflexao, transmissao e refracao -- aprenda a "
        "identificar cada uma.",
        "Borda mal dimensionada injeta artefatos no residuo e cria estrutura falsa na FWI.",
        "n_abs >= 0.5 * c_max / (f_min * dh), dimensionada pela MENOR frequencia.",
        "Superficie livre e escolha FISICA (multiplas e ghost), nao numerica.",
    ], proxima="aula04_aquisicao.py -- geometria, sismogramas e o que o dado realmente ve")


if __name__ == "__main__":
    main()
