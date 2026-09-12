#!/usr/bin/env python3
"""
AULA 07 -- O problema inverso: funcao objetivo, nao convexidade e cycle skipping
================================================================================
Execute:  python aulas/aula07_problema_inverso.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                               # noqa: E402
import fwikit                                    # noqa: E402
from fwikit.aula import Aula, mostrar_figuras    # noqa: E402
from fwikit import plot                          # noqa: E402
from fwikit.acustico import (Acustico2D, Config, Geometria,   # noqa: E402
                             ricker, misfit_l2, dt_maximo)


def main():
    a = Aula(7, "O problema inverso: funcao objetivo e cycle skipping",
             modulo="Modulo III -- O problema inverso e o gradiente",
             duracao="70 min", pre_requisitos="aulas 01-06",
             objetivos=[
                 "Escrever a FWI como problema de otimizacao e entender cada simbolo",
                 "Visualizar a funcao objetivo e ver a nao convexidade com os proprios olhos",
                 "Deduzir o criterio de meio ciclo e prever quando havera cycle skipping",
                 "Entender por que a estrategia multiescala funciona",
                 "Avaliar se um modelo inicial e bom o bastante ANTES de gastar horas",
             ])
    a.cabecalho()

    # ==================================================================
    a.secao("A formulacao")
    a.texto("""
        Seja m o vetor de parametros do modelo e F o operador de modelagem
        direta -- no nosso caso, resolver a equacao acustica e amostrar nos
        receptores. Os dados calculados sao:
    """)
    a.eq("d_calc  =  F(m)", rotulo="problema direto")
    a.texto("""
        O vetor residual e a diferenca entre calculado e observado:
    """)
    a.eq("r(m)  =  F(m)  -  d_obs", rotulo="residual")
    a.texto("""
        E a FWI minimiza uma norma desse residual. Com a norma L2 -- de longe a
        mais usada -- o funcional e:
    """)
    a.eq("J(m)  =  (1/2) || F(m) - d_obs ||^2",
         "      =  (1/2) soma_tiros soma_receptores integral_t r^2 dt",
         rotulo="funcional de minimos quadrados")
    a.texto("""
        A solucao iterativa tem sempre a mesma forma, qualquer que seja o
        metodo de otimizacao:
    """)
    a.eq("m_(k+1)  =  m_k  +  alpha_k  d_k", rotulo="atualizacao")
    a.texto("""
        onde d_k e a direcao de busca (construida a partir do gradiente) e
        alpha_k e o comprimento do passo, obtido por busca linear. O Modulo IV e
        inteiro sobre como escolher d_k e alpha_k; o resto deste modulo e sobre
        como obter o gradiente.
    """)
    a.teoria("Por que este problema e dificil", """
        Tres propriedades de J tornam a FWI o que ela e:

        1. NAO LINEAR. F depende de m de forma nao linear (a velocidade aparece
           dentro do operador de onda, e o tempo de transito depende dela de
           maneira integral). Logo J nao e quadratica e nao ha solucao fechada.

        2. NAO CONVEXA. Existem minimos locais -- muitos. Um otimizador local
           converge para o mais proximo do ponto de partida, que nao e
           necessariamente o certo. E o problema central da FWI.

        3. MAL POSTA. Partes do modelo simplesmente nao sao vistas pelos dados
           (zonas de sombra, abaixo da profundidade de penetracao). Nessas
           regioes o gradiente e ~0 e o modelo fica onde estava -- ou pior, e
           preenchido por artefatos vindos da regularizacao.

        A nao convexidade tem um nome proprio quando aparece por diferenca de
        fase: CYCLE SKIPPING. E o assunto do resto desta aula.
    """)

    # ==================================================================
    a.secao("Vendo a funcao objetivo com os proprios olhos")
    a.texto("""
        Vamos fazer o experimento mais simples que revela o problema: um meio
        HOMOGENEO, um tiro, alguns receptores. O modelo tem UM unico parametro,
        a velocidade c. Podemos entao varrer c e desenhar J(c) inteiro -- coisa
        impossivel num problema real com 10^6 parametros.
    """)
    nz, nx, dh = 60, 200, 10.0
    c_verd = 2000.0
    geom = Geometria(fontes=np.array([[30, 10]]),
                     receptores=np.array([[30, i] for i in range(150, 190, 4)]))

    def dados_de(c_val, f0, nt=None):
        dt = dt_maximo(2600.0, dh, 4)
        nt = nt or int(1.6 / dt)
        cfg = Config(dh=dh, dt=dt, nt=nt, ordem=4, n_abs=30, f0=f0)
        s = Acustico2D(cfg, (nz, nx))
        w = ricker(f0, dt, nt)
        d, _ = s.modelar(np.full((nz, nx), c_val, dtype=np.float32), geom, w)
        return d, dt

    velocidades = np.linspace(1500, 2600, 90)
    frequencias = [3.0, 6.0, 12.0, 25.0]
    curvas = {}
    print()
    print("    varrendo J(c) para varias frequencias...")
    for f0 in frequencias:
        d_obs, dt = dados_de(c_verd, f0)
        J = []
        for cv in velocidades:
            d_c, _ = dados_de(cv, f0)
            J.append(misfit_l2(d_c, d_obs, dt)[0])
        J = np.array(J)
        curvas[f0] = J / (J.max() + 1e-30)
        # conta minimos locais
        interior = curvas[f0][1:-1]
        minimos = np.sum((interior < curvas[f0][:-2]) & (interior < curvas[f0][2:]))
        print(f"      f0 = {f0:5.1f} Hz  ->  {minimos:2d} minimos locais na varredura")
    print()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2))
    cores = ["#1b6ca8", "#2a9d8f", "#e9c46a", "#d1495b"]
    for (f0, J), cor in zip(curvas.items(), cores):
        ax[0].plot(velocidades, J, color=cor, lw=1.6, label=f"f0 = {f0:.0f} Hz")
    ax[0].axvline(c_verd, color="k", ls="--", lw=1.0)
    ax[0].text(c_verd + 12, 0.9, "c verdadeiro", fontsize=8)
    ax[0].set_xlabel("velocidade do modelo (m/s)")
    ax[0].set_ylabel("J normalizado")
    ax[0].set_title("funcao objetivo: uma curva por frequencia")
    ax[0].legend(fontsize=8)

    # zoom na alta frequencia
    ax[1].plot(velocidades, curvas[25.0], color="#d1495b", lw=1.8)
    ax[1].axvline(c_verd, color="k", ls="--", lw=1.0)
    ax[1].set_xlabel("velocidade do modelo (m/s)")
    ax[1].set_ylabel("J normalizado")
    ax[1].set_title("f0 = 25 Hz: os minimos locais de perto")
    ax[1].set_xlim(1700, 2400)
    fig.tight_layout()
    plot.salvar(fig, a, "01_funcao_objetivo", mostrar=False)

    a.texto("""
        Olhe a figura com cuidado, porque ela contem a ideia mais importante do
        curso.

        Em BAIXA frequencia a curva e larga e tem um unico minimo: qualquer
        modelo inicial dentro de uma faixa generosa converge para o valor certo.

        Em ALTA frequencia a curva vira um vale estreito cercado de minimos
        LOCAIS. Se o modelo inicial cair em um deles, o otimizador desce para
        esse minimo local e para -- satisfeito, convergido, e errado.
    """)

    # ==================================================================
    a.secao("Cycle skipping: a explicacao fisica")
    a.texto("""
        Por que aparecem minimos locais? A razao e puramente de FASE.

        Considere um unico evento sismico. Se o modelo esta errado, o evento
        sintetico chega deslocado de dt em relacao ao observado. O residuo, para
        um pulso aproximadamente harmonico de periodo T, e governado por esse
        deslocamento:
    """)
    a.lista([
        "|dt| < T/2 -- o sintetico e o observado ainda se sobrepoem no MESMO "
        "ciclo. O residuo aponta na direcao certa e a inversao corrige.",
        "|dt| > T/2 -- o sintetico se alinha com o ciclo VIZINHO do observado. "
        "O residuo fica menor ao empurrar o modelo para o lado ERRADO. A "
        "inversao converge para um minimo local: cycle skipping.",
    ])
    a.eq("|dt|  <  T/2  =  1 / (2 f)", rotulo="criterio de meio ciclo")
    a.texto("""
        Escrito em termos do erro de velocidade num percurso de comprimento L:
    """)
    a.eq("dt  =  L (1/c_ini - 1/c_verd)",
         "criterio:   L |c_verd - c_ini| / (c^2)  <  1 / (2 f)")
    print()
    print("    Erro MAXIMO de velocidade tolerado (sem cycle skipping)")
    print(f"    percurso L = 1500 m, c = 2000 m/s")
    print()
    print(f"    {'f (Hz)':<12}{'T/2 (ms)':<14}{'erro max de c (m/s)':<24}{'em %'}")
    L = 1500.0
    for f in [2, 3, 5, 10, 20, 40]:
        dc = (1.0 / (2 * f)) * c_verd ** 2 / L
        print(f"    {f:<12}{1000/(2*f):<14.1f}{dc:<24.0f}{100*dc/c_verd:.1f}%")
    print()
    a.teoria("A leitura correta desta tabela", """
        A 2 Hz voce pode errar a velocidade em 33% e ainda assim convergir.
        A 40 Hz voce precisa comecar com menos de 2% de erro -- ou seja,
        precisaria ja saber a resposta.

        Isso explica de uma vez:

        * por que baixa frequencia vale ouro em aquisicao sismica, e por que se
          investe tanto em fontes de baixa frequencia;

        * por que a FWI direto em alta frequencia quase sempre falha;

        * por que a estrategia MULTIESCALA (Bunks et al., 1995) funciona:
          inverta primeiro nas frequencias baixas, onde o vale de atracao e
          largo; use o resultado como modelo inicial da proxima banda, que agora
          ja comeca dentro do vale estreito; repita subindo em frequencia.

        Multiescala nao e um truque numerico. E a consequencia direta do
        criterio de meio ciclo.
    """)

    # ==================================================================
    a.secao("Medindo o vale de atracao")
    a.texto("""
        Da para transformar isso em numero. Para cada frequencia, partimos de
        CADA velocidade da varredura, descemos a curva localmente e vemos onde
        paramos. O conjunto de pontos de partida que chega ao minimo CERTO e o
        VALE DE ATRACAO. Sua largura e a tolerancia real da inversao ao erro do
        modelo inicial.

        E podemos comparar com a previsao teorica do criterio de meio ciclo.
    """)

    def destino(J, k):
        """Descida local ingenua a partir do indice k."""
        while 0 < k < len(J) - 1:
            if J[k - 1] < J[k]:
                k -= 1
            elif J[k + 1] < J[k]:
                k += 1
            else:
                break
        return k

    L_medio = float(np.mean([abs(r[1] - geom.fontes[0, 1]) * dh
                             for r in geom.receptores]))
    print()
    print(f"    percurso medio fonte-receptor: L = {L_medio:.0f} m")
    print()
    print(f"    {'f0 (Hz)':<10}{'vale medido (m/s)':<26}{'largura':<12}"
          f"{'previsto (+-)':<16}{'medido (+-)'}")
    valas = {}
    for f0 in frequencias:
        J = curvas[f0]
        acerta = np.array([abs(velocidades[destino(J, k)] - c_verd) < 40
                           for k in range(len(velocidades))])
        if acerta.any():
            # maior intervalo contiguo que contem o c verdadeiro
            k_verd = int(np.argmin(np.abs(velocidades - c_verd)))
            lo = k_verd
            while lo > 0 and acerta[lo - 1]:
                lo -= 1
            hi = k_verd
            while hi < len(acerta) - 1 and acerta[hi + 1]:
                hi += 1
            v_lo, v_hi = velocidades[lo], velocidades[hi]
            largura = v_hi - v_lo
            meio_medido = largura / 2
        else:
            v_lo = v_hi = largura = meio_medido = float("nan")
        previsto = c_verd ** 2 / (2 * f0 * L_medio)
        valas[f0] = (v_lo, v_hi)
        print(f"    {f0:<10.1f}{v_lo:>7.0f} a {v_hi:<15.0f}"
              f"{largura:>7.0f} m/s   {previsto:>10.0f}      {meio_medido:>10.0f}")
    print()
    a.teoria("O criterio de meio ciclo funciona", """
        Compare as duas ultimas colunas. A previsao

            |dc|  <  c^2 / (2 f L)

        deduzida de uma conta de meia linha -- "o erro de tempo nao pode passar
        de meio periodo" -- acerta a ordem de grandeza da largura MEDIDA do vale
        de atracao, e reproduz a dependencia 1/f.

        Isso e valioso na pratica: antes de rodar qualquer inversao, voce pode
        estimar de cabeca qual a frequencia MAIS ALTA com que pode comecar,
        dada a incerteza que voce tem do modelo inicial. Se a sua tomografia tem
        5% de erro e o percurso tipico e 2 km, entao para c = 2000 m/s:

            dc = 0.05 x 2000 = 100 m/s  ->  f_max = c^2/(2 dc L) = 10 Hz

        Comecar acima disso e desperdicio de tempo de maquina.
    """)

    a.secao("Demonstracao: mesmo inicial, destinos diferentes")
    c_ini = 1850.0
    print()
    print(f"    Partindo de c_inicial = {c_ini:.0f} m/s "
          f"({100*abs(c_ini-c_verd)/c_verd:.0f}% de erro):")
    print()
    print(f"    {'f0 (Hz)':<12}{'destino':<16}{'resultado'}")
    for f0 in frequencias:
        J = curvas[f0]
        k = destino(J, int(np.argmin(np.abs(velocidades - c_ini))))
        v = velocidades[k]
        ok = "convergiu" if abs(v - c_verd) < 40 else "CYCLE SKIPPING"
        print(f"    {f0:<12.1f}{v:>7.0f} m/s{'':<6}{ok}")
    print()
    a.texto("""
        Mesmo ponto de partida, mesma fisica, mesmo dado -- e o destino depende
        SO da frequencia. Nao e questao de rodar mais iteracoes nem de escolher
        um otimizador melhor: um otimizador local nao sai de um minimo local.
        Ele esta fazendo exatamente o que deveria.
    """)

    a.pergunta(
        "Sua FWI converge (J para de cair) mas o modelo esta claramente errado. "
        "Qual a PRIMEIRA hipotese a investigar?",
        ["O otimizador precisa de mais iteracoes",
         "Cycle skipping: o modelo inicial estava fora do vale de atracao "
         "da frequencia usada",
         "O gradiente esta implementado errado",
         "Falta regularizacao"],
        1,
        "J parou de cair significa que o otimizador chegou a um minimo -- so que "
        "local. Mais iteracoes nao ajudam. Gradiente errado normalmente impede J "
        "de cair. O teste decisivo: refaca a inversao comecando por uma "
        "frequencia mais baixa. Se o resultado mudar, era cycle skipping.")

    a.pausa()

    # ==================================================================
    a.secao("O papel do modelo inicial")
    a.texto("""
        Combinando tudo: o modelo inicial nao precisa ser bonito nem detalhado.
        Ele precisa de UMA coisa -- estar dentro do vale de atracao da menor
        frequencia com que voce consegue trabalhar. Isso significa acertar os
        GRANDES comprimentos de onda (a tendencia de velocidade), nao os
        detalhes.
    """)
    a.tabela(
        ["Fonte do modelo inicial", "Qualidade tipica", "Comentario"],
        [["Gradiente linear de vel.", "grosseira", "funciona em caso simples"],
         ["Analise de velocidade (NMO)", "boa", "o padrao na industria"],
         ["Tomografia de tempo de transito", "boa", "acerta a tendencia"],
         ["Modelo verdadeiro suavizado", "otima", "SO EXISTE em teste sintetico"],
         ["Migracao + interpretacao", "variavel", "depende do interprete"]])
    a.aviso("""
        Um alerta importante para trabalho sintetico: usar o modelo verdadeiro
        suavizado como inicial -- o que `ModelGenerator(...)(smoothing=1)` faz --
        e um teste GENEROSO. Ele ja comeca dentro do vale de atracao por
        construcao.

        Nao esta errado usa-lo para validar codigo: e exatamente o que voce quer
        ao verificar se o gradiente e a otimizacao funcionam. Mas nao confunda
        isso com avaliar a ROBUSTEZ do metodo. Para isso, degrade o modelo
        inicial de proposito e descubra onde ele quebra. Um experimento que so
        funciona com inicial privilegiado nao diz quase nada sobre dado real.
    """)

    # ==================================================================
    a.secao("Alem do L2: outras funcoes objetivo")
    a.texto("""
        Como o problema do L2 e de FASE, boa parte da pesquisa em FWI consiste
        em trocar o funcional por um que seja menos sensivel a saltos de ciclo.
        Voce vai encontrar estes nomes:
    """)
    a.tabela(
        ["Funcional", "Ideia", "Preco"],
        [["L2 classico", "diferenca amostra a amostra", "cycle skipping"],
         ["Envoltoria", "compara a envoltoria do sinal", "perde resolucao"],
         ["Correlacao cruzada", "penaliza o deslocamento de fase", "precisa janelar"],
         ["Transporte otimo (OT)", "distancia entre 'massas' do sinal", "custo alto"],
         ["Adaptativa (AWI)", "penaliza o filtro de casamento", "mais parametros"]])
    a.dica("""
        Ordem de prioridade na pratica, quando a FWI nao converge:

        1. Baixe a frequencia inicial (multiescala mais agressiva).
        2. Melhore o modelo inicial.
        3. So entao troque o funcional.

        Trocar o funcional e a solucao mais cara e a que mais introduz
        parametros novos para ajustar. Ela resolve casos em que 1 e 2 ja foram
        esgotados -- nao substitui nenhum dos dois.
    """)

    if mostrar_figuras():
        a.secao("Explore: o vale de atracao")
        a.texto("""
            Mova a frequencia e o modelo inicial. O ponto vermelho mostra onde
            uma otimizacao local terminaria partindo dali. Procure a frequencia
            em que o ponto deixa de cair no minimo certo. Feche para continuar.
        """)
        from matplotlib.widgets import Slider
        figi, axi = plt.subplots(figsize=(9.5, 5))
        plt.subplots_adjust(bottom=0.26)
        f_grade = np.array(frequencias)
        (linha,) = axi.plot(velocidades, curvas[6.0], lw=1.8, color="#1b6ca8")
        (pini,) = axi.plot([1700], [0.5], "ks", ms=9, label="inicial")
        (pfim,) = axi.plot([2000], [0.0], "ro", ms=10, label="destino")
        axi.axvline(c_verd, color="k", ls="--", lw=1.0)
        axi.set_xlabel("velocidade (m/s)"); axi.set_ylabel("J normalizado")
        axi.legend(fontsize=8)
        s_f = Slider(plt.axes([0.15, 0.13, 0.72, 0.03]), "f0 (Hz)", 0, 3,
                     valinit=1, valstep=1)
        s_c = Slider(plt.axes([0.15, 0.06, 0.72, 0.03]), "c inicial (m/s)",
                     1520, 2580, valinit=1700)

        def atualiza(_):
            f0 = f_grade[int(s_f.val)]
            J = curvas[f0]
            linha.set_ydata(J)
            k = int(np.argmin(np.abs(velocidades - s_c.val)))
            pini.set_data([velocidades[k]], [J[k]])
            while 0 < k < len(J) - 1:
                if J[k - 1] < J[k]:
                    k -= 1
                elif J[k + 1] < J[k]:
                    k += 1
                else:
                    break
            pfim.set_data([velocidades[k]], [J[k]])
            ok = abs(velocidades[k] - c_verd) < 40
            axi.set_title(f"f0 = {f0:.0f} Hz  ->  destino "
                          f"{velocidades[k]:.0f} m/s  "
                          + ("(CORRETO)" if ok else "(CYCLE SKIPPING)"))
            figi.canvas.draw_idle()

        atualiza(None)
        s_f.on_changed(atualiza)
        s_c.on_changed(atualiza)
        plt.show()

    a.secao("Exercicios")
    a.exercicio(1, """
        Refaca a varredura de J(c) variando o OFFSET dos receptores (aproxime-os
        da fonte). O criterio de meio ciclo prediz que percursos mais curtos
        toleram erro MAIOR de velocidade. Confirme numericamente.
    """, dica="dt = L (1/c_ini - 1/c_verd): L menor, dt menor.")
    a.exercicio(2, """
        Implemente o funcional de ENVOLTORIA (use a transformada de Hilbert de
        scipy.signal) e refaca a varredura a 25 Hz. Os minimos locais somem?
        O que acontece com a largura do minimo global?
    """, dica="scipy.signal.hilbert; compare |env(d_calc)| com |env(d_obs)|.")
    a.exercicio(3, """
        Monte uma varredura em DUAS dimensoes (velocidade de duas camadas) e
        desenhe o mapa de J com contornos. Identifique visualmente os vales.
        Esse e o tipo de figura que vale muito num relatorio.
    """, dica="modelo de 2 camadas, laco duplo, plt.contourf.")

    a.fim([
        "FWI = minimizar J(m) = (1/2)||F(m) - d_obs||^2 iterativamente.",
        "J e nao linear, NAO CONVEXA e mal posta. A nao convexidade e o problema central.",
        "Cycle skipping ocorre quando o erro de tempo passa de meio periodo: |dt| > T/2.",
        "Baixa frequencia = vale de atracao largo. Alta frequencia = resolucao. "
        "Multiescala concilia os dois.",
        "O modelo inicial precisa acertar os GRANDES comprimentos de onda, nao os detalhes.",
    ], proxima="aula08_estado_adjunto.py -- como calcular o gradiente sem forca bruta")


if __name__ == "__main__":
    main()
