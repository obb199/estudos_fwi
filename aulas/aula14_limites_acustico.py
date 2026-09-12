#!/usr/bin/env python3
"""
AULA 14 -- Os limites da aproximacao acustica
==============================================
Execute:  python aulas/aula14_limites_acustico.py

Esta aula fecha o curso olhando para fora: quais hipoteses foram feitas no
caminho, quando cada uma quebra, e o que custa relaxa-la. E teoria e
diagnostico -- nao implementamos nenhuma das extensoes aqui.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                               # noqa: E402
import fwikit                                    # noqa: E402
from fwikit.aula import Aula, mostrar_figuras    # noqa: E402
from fwikit import plot                          # noqa: E402
from fwikit.acustico import ricker, espectro_amplitude  # noqa: E402


def kjartansson_1d(w, dt, L, c0, Q, f_ref=None):
    """
    Propagador analitico de onda plana para um meio homogeneo com Q CONSTANTE,
    segundo Kjartansson (1979). Serve para DEMONSTRAR a fisica da atenuacao --
    nao e um solver de FWI.

    O modelo constant-Q e uma lei de potencia. Definindo

        gamma = (1/pi) arctan(1/Q)

    a velocidade de fase depende da frequencia como

        c(w) = c0 (w / w_ref)^gamma

    e a amplitude decai como

        exp( - w L / (2 Q c(w)) ).

    Repare que dissipacao e dispersao vem JUNTAS: sao consequencia uma da
    outra, impostas pela causalidade (relacoes de Kramers-Kronig). Nao existe
    meio fisicamente admissivel que atenue sem dispersar.
    """
    n = len(w)
    nfft = int(2 ** np.ceil(np.log2(n * 2)))
    W = np.fft.rfft(w, nfft)
    f = np.fft.rfftfreq(nfft, dt)
    omega = 2 * np.pi * f
    f_ref = f_ref or max(f[1], 1.0)
    w_ref = 2 * np.pi * f_ref

    gamma = np.arctan(1.0 / Q) / np.pi
    razao = np.where(omega > 0, omega / w_ref, 1.0)
    c_w = c0 * razao ** gamma                      # dispersao
    atenua = np.exp(-omega * L / (2.0 * Q * c_w))  # dissipacao
    fase = np.exp(-1j * omega * L / c_w)           # propagacao dispersiva
    return np.fft.irfft(W * atenua * fase, nfft)[:n], gamma, c_w, f


def main():
    a = Aula(14, "Os limites da aproximacao acustica",
             modulo="Modulo V -- Onde a fisica adotada deixa de valer",
             duracao="60 min", pre_requisitos="todo o curso",
             objetivos=[
                 "Listar as hipoteses acumuladas no caminho e quando cada uma quebra",
                 "Entender a ligacao causal entre dissipacao e dispersao",
                 "Conhecer o modelo constant-Q de Kjartansson e o que ele implica",
                 "Diagnosticar, a partir do dado, qual fisica o seu problema exige",
             ])
    a.cabecalho()

    # ==================================================================
    a.secao("O inventario de hipoteses")
    a.texto("""
        Vale enxergar de uma vez tudo o que foi assumido ao longo do curso. Cada
        linha e uma decisao que pode estar certa ou errada para o SEU problema.
    """)
    a.tabela(
        ["Hipotese", "Onde entrou", "Quando quebra"],
        [["mu = 0 (sem cisalhamento)", "aula 01", "dado terrestre, contrastes de vs"],
         ["densidade constante", "aula 01", "contrastes fortes de impedancia"],
         ["sem atenuacao (Q infinito)", "aula 01", "meio dissipativo, banda larga"],
         ["isotropia", "aula 01", "folhelhos, fraturas (VTI/TTI)"],
         ["2D", "curso inteiro", "estrutura 3D fora do plano"],
         ["fonte conhecida", "aula 04", "dado real -- a wavelet e estimada"],
         ["ruido gaussiano branco", "aula 07 (L2)", "ruido coerente, multiplas"]])
    a.aviso("""
        Um principio que vale mais que qualquer uma dessas linhas: o que a FWI
        nao consegue explicar pela FISICA, ela explica mexendo no MODELO.

        Nao existe uma valvula de escape. Se o dado contem um efeito que o seu
        operador de modelagem nao reproduz, esse efeito entra inteiro no
        residuo, e o gradiente o converte em estrutura de velocidade. O modelo
        final absorve o erro de fisica -- e parece um modelo, nao um erro.
    """)

    # ==================================================================
    a.secao("Limite 1: densidade")
    a.texto("""
        A equacao que resolvemos supoe densidade constante. A forma geral e:
    """)
    a.eq("(1/K) d2p/dt2  =  div( (1/rho) grad p )  +  s")
    a.texto("""
        Expandindo o divergente aparece um termo extra proporcional a
        grad(1/rho), que so importa onde a densidade varia BRUSCAMENTE. O
        problema conceitual e mais profundo: o que controla a REFLEXAO nao e a
        velocidade, e a IMPEDANCIA Z = rho c.
    """)
    print()
    print("    Coeficiente de reflexao em incidencia normal: R = (Z2-Z1)/(Z2+Z1)")
    print()
    print(f"    {'meio 1':<22}{'meio 2':<22}{'R'}")
    for (c1, r1), (c2, r2), rot in [((2000, 2.0), (2400, 2.0), "so velocidade"),
                                    ((2000, 2.0), (2000, 2.4), "so densidade"),
                                    ((2000, 2.0), (2400, 1.67), "impedancia igual")]:
        Z1, Z2 = c1 * r1, c2 * r2
        R = (Z2 - Z1) / (Z2 + Z1)
        print(f"    c={c1} rho={r1:<12.2f}c={c2} rho={r2:<12.2f}{R:+.4f}   ({rot})")
    print()
    a.texto("""
        Olhe a ultima linha: velocidade 20% maior e densidade 17% menor
        produzem reflexao praticamente NULA. Uma FWI acustica de densidade
        constante interpretaria essa interface como inexistente. E o exemplo
        mais limpo de ambiguidade velocidade-densidade, e a razao pela qual a
        inversao multiparametro sofre de crosstalk.
    """)

    # ==================================================================
    a.secao("Limite 2: elasticidade")
    a.texto("""
        Ja medimos isso na aula 05: no modelo louboutin, a diferenca entre o
        sismograma elastico e o acustico tinha norma comparavel a do proprio
        dado. Sao ondas S e conversoes P-S -- eventos inteiros que nao existem
        no mundo acustico.

        O ponto pratico e que essa energia nao some do dado real. Ou voce a
        modela (FWI elastica), ou a remove no processamento, ou ela vai para o
        residuo.
    """)
    a.tabela(
        ["Extensao", "Parametros", "Custo relativo", "O que resolve"],
        [["Acustica", "vp", "1x", "cinematica P"],
         ["Acustica + densidade", "vp, rho", "1x", "amplitude de reflexao"],
         ["Viscoacustica", "vp, Q", "2-4x", "dissipacao e dispersao"],
         ["Elastica", "vp, vs, rho", "3-5x", "ondas S, conversoes, AVO"],
         ["Viscoelastica", "vp, vs, rho, Qp, Qs", "6-10x", "tudo acima"],
         ["Anisotropica (VTI)", "+ epsilon, delta", "2-3x", "folhelhos"]])

    # ==================================================================
    a.secao("Limite 3: atenuacao e o fator Q")
    a.texto("""
        Meios reais dissipam energia. O fator de qualidade Q parametriza isso:
        e (aproximadamente) o numero de ciclos que uma onda percorre antes de
        sua amplitude cair por um fator e^(-pi). Q alto = pouca perda.
    """)
    a.tabela(["Meio", "Q tipico"],
             [["agua", "> 1000"],
              ["rocha consolidada", "100 - 300"],
              ["sedimento nao consolidado", "20 - 50"],
              ["sedimento raso saturado / gas", "10 - 30"]])
    a.teoria("Dissipacao e dispersao sao inseparaveis", """
        Este e o ponto conceitual mais importante da secao, e o mais
        frequentemente ignorado.

        Um meio que atenua de forma dependente da frequencia e OBRIGADO a ter
        velocidade de fase dependente da frequencia. A razao e CAUSALIDADE: o
        sinal nao pode chegar antes de ser emitido. Formalmente, as partes real
        e imaginaria da resposta de um sistema causal estao ligadas pelas
        relacoes de KRAMERS-KRONIG.

        A consequencia pratica e direta: nao adianta "corrigir a amplitude" de
        um dado atenuado e continuar usando uma equacao sem perdas. Ao atenuar,
        o meio tambem mudou a FASE -- e a FWI e um metodo governado por fase.
        Um operador sem dissipacao esta errado nas duas contas ao mesmo tempo.
    """)
    a.texto("""
        Entre as formas de representar a dissipacao, o modelo de KJARTANSSON
        (1979) e o mais usado quando se quer Q aproximadamente constante na
        banda sismica. Ele parte de uma lei constitutiva de potencia, e o
        expoente se relaciona a Q por:
    """)
    a.eq("gamma  =  (1/pi) arctan(1 / Q)", rotulo="Kjartansson")
    a.eq("c(w)  =  c0 (w / w_ref)^gamma",
         "A(w)  =  exp[ - w L / (2 Q c(w)) ]")
    a.texto("""
        No dominio do tempo, essa dependencia de potencia em frequencia leva a
        operadores de ordem FRACIONARIA -- e por isso a implementacao
        viscoacustica no tempo e substancialmente mais trabalhosa que a
        acustica, e por que ela costuma ser feita com Laplacianos fracionarios
        ou com uma soma de mecanismos de relaxacao (modelo SLS).
    """)

    # ------------------------------------------------------------------
    a.secao("Demonstracao: o que Q faz com um pulso")
    a.texto("""
        Vamos aplicar o propagador analitico de Kjartansson a uma Ricker, num
        meio homogeneo. Isto NAO e um solver de FWI -- e a solucao exata de
        onda plana, que serve para enxergar a fisica isolada.
    """)
    dt = 5e-4
    c0 = 2000.0
    L = 1500.0
    f0 = 20.0
    # o registro precisa conter o tempo de transito L/c0 com folga, senao o
    # pulso e cortado pelo fim da janela (erro facil de cometer e dificil de ver)
    nt = int(1.6 * (L / c0) / dt)
    w0 = ricker(f0, dt, nt)
    t = np.arange(nt) * dt

    print()
    print(f"    meio homogeneo, c0 = {c0:.0f} m/s, distancia L = {L:.0f} m, "
          f"f0 = {f0:.0f} Hz")
    print(f"    tempo de transito sem atenuacao = {L/c0:.3f} s   "
          f"(registro de {nt*dt:.3f} s)")
    print()
    print(f"    {'Q':<10}{'gamma':<12}{'amplitude rel.':<18}"
          f"{'f pico (Hz)':<14}{'atraso (ms)'}")
    curvas = {}
    ref_amp = None
    for Q in [10000, 200, 100, 50, 20]:
        u, gamma, c_w, f = kjartansson_1d(w0, dt, L, c0, Q, f_ref=f0)
        amp = float(np.abs(u).max())
        if ref_amp is None:
            ref_amp = amp
        fk, Sk = espectro_amplitude(u, dt)
        f_pico = fk[np.argmax(Sk)]
        i_ref = int(np.argmax(np.abs(curvas[10000][0]))) if 10000 in curvas \
            else int(np.argmax(np.abs(u)))
        atraso = (int(np.argmax(np.abs(u))) - i_ref) * dt * 1000
        curvas[Q] = (u, gamma, f_pico)
        print(f"    {Q:<10}{gamma:<12.6f}{amp/ref_amp:<18.4f}"
              f"{f_pico:<14.1f}{atraso:+.1f}")
    print()
    a.texto("""
        Tres efeitos aparecem ao mesmo tempo, e nenhum deles existe na equacao
        acustica: a amplitude cai, a frequencia de pico DESCE (as altas
        frequencias sao atenuadas mais), e o pulso chega DEPOIS -- porque a
        velocidade de fase efetiva mudou.
    """)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    cores = plt.cm.viridis(np.linspace(0, 0.85, len(curvas)))
    for (Q, (u, gamma, fp)), cor in zip(curvas.items(), cores):
        rot = "sem atenuacao" if Q > 5000 else f"Q = {Q}"
        ax[0].plot(t, u, color=cor, lw=1.3, label=rot)
        plot.espectro(ax[1], u, dt, rotulo=rot, fmax=80, color=cor, lw=1.3)
    ax[0].set_xlim(L / c0 - 0.10, L / c0 + 0.22)
    ax[0].set_xlabel("tempo (s)"); ax[0].set_ylabel("amplitude")
    ax[0].set_title(f"pulso apos {L:.0f} m")
    ax[0].legend(fontsize=7)
    ax[1].set_title("espectro: as altas frequencias somem")
    ax[1].legend(fontsize=7)

    # velocidade de fase
    ff = np.linspace(1, 80, 300)
    for Q, cor in zip([200, 100, 50, 20], cores[1:]):
        g = np.arctan(1.0 / Q) / np.pi
        ax[2].plot(ff, c0 * (ff / f0) ** g, color=cor, lw=1.4, label=f"Q = {Q}")
    ax[2].axhline(c0, color="k", ls="--", lw=0.9, label="sem atenuacao")
    ax[2].set_xlabel("frequencia (Hz)")
    ax[2].set_ylabel("velocidade de fase (m/s)")
    ax[2].set_title("dispersao de Kjartansson")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    plot.salvar(fig, a, "01_atenuacao_kjartansson", mostrar=False)

    # ------------------------------------------------------------------
    a.secao("Quando ignorar Q provoca cycle skipping")
    a.texto("""
        Aqui as pecas do curso se encaixam. Se voce modela com c0 constante mas
        o dado veio de um meio dispersivo, ha um erro de tempo de transito:
    """)
    a.eq("dt(f)  =  L/c0  -  L/c(f)  =  (L/c0) [ 1 - (f/f_ref)^(-gamma) ]")
    a.texto("""
        E a aula 07 nos deu o criterio que decide se isso importa: o erro de
        tempo nao pode passar de MEIO PERIODO, senao ha cycle skipping.
    """)
    a.eq("|dt(f)|  <  1 / (2 f)", rotulo="criterio de meio ciclo (aula 07)")
    print()
    print(f"    c0 = {c0:.0f} m/s, f_ref = {f0:.0f} Hz")
    print()
    print(f"    {'Q':<8}{'L (m)':<10}{'f (Hz)':<10}{'|dt| (ms)':<14}"
          f"{'T/2 (ms)':<12}{'situacao'}")
    alertas = 0
    for Q in [100, 50, 20, 10]:
        gamma = np.arctan(1.0 / Q) / np.pi
        for L_t in [1000.0, 6000.0]:
            for f_t in [10.0, 60.0]:
                dtt = abs((L_t / c0) * (1 - (f_t / f0) ** (-gamma)))
                meio = 1.0 / (2 * f_t)
                risco = dtt > meio
                alertas += risco
                print(f"    {Q:<8}{L_t:<10.0f}{f_t:<10.0f}{dtt*1000:<14.3f}"
                      f"{meio*1000:<12.1f}"
                      f"{'CYCLE SKIPPING' if risco else 'ok'}")
    print()
    a.dica(f"""
        Leia a tabela como uma ferramenta de decisao, e nao como uma conclusao
        geral: os numeros dependem de c0, de f_ref e da banda.

        O padrao e claro. O risco cresce com percurso LONGO, frequencia ALTA e
        Q BAIXO -- os tres juntos. Em Q = 200 e percursos curtos, a
        aproximacao sem perdas e perfeitamente defensavel. Em Q baixo com
        offsets longos e banda larga, ela deixa de ser.

        E vale lembrar o que a tabela NAO cobre: ela so contabiliza o erro de
        FASE por dispersao. O erro de AMPLITUDE por dissipacao e adicional, e
        afeta diretamente qualquer funcional que compare amplitudes -- como o
        L2 que usamos o curso inteiro.
    """)

    # ==================================================================
    a.secao("Como decidir qual fisica voce precisa")
    a.texto("""
        A decisao nao deve ser por gosto nem por disponibilidade de codigo. Ela
        se faz olhando para o dado e para o residuo.
    """)
    a.lista([
        "OLHE O RESIDUO, nao so o valor de J. Residuo com estrutura COERENTE -- "
        "eventos organizados, nao ruido -- e a assinatura de fisica faltando. "
        "Residuo que parece ruido aleatorio significa que voce chegou ao limite "
        "do dado.",
        "COMPARE ESPECTROS de observado e sintetico em funcao do offset. Se o "
        "sintetico mantem alta frequencia que o dado real ja perdeu em offsets "
        "longos, ha atenuacao nao modelada.",
        "VERIFIQUE AMPLITUDE x OFFSET. Decaimento sistematicamente mais rapido "
        "no dado real que no sintetico aponta para dissipacao ou espalhamento.",
        "PROCURE EVENTOS AUSENTES. Se ha energia no dado real em tempos onde o "
        "sintetico acustico nao produz nada, provavelmente sao ondas S, "
        "conversoes ou multiplas.",
        "TESTE A SENSIBILIDADE. Gere dado sintetico com a fisica mais completa "
        "e inverta com a mais simples. O erro de modelo resultante e a medida "
        "direta do custo da aproximacao -- e o experimento mais informativo "
        "que existe para justificar (ou dispensar) uma extensao.",
    ])
    a.aviso("""
        E uma advertencia sobre o caminho inverso. Adicionar fisica sem
        necessidade tem custo real: mais parametros, mais crosstalk, mais
        minimos locais, mais tempo de maquina, e mais dificuldade em atribuir
        um resultado a uma causa.

        Uma FWI acustica bem feita, com bom modelo inicial e multiescala
        honesta, costuma render mais que uma viscoelastica mal condicionada.
        A pergunta certa nao e "qual a fisica mais completa?", e sim "qual e a
        fisica mais simples que explica o MEU dado?".
    """)

    a.pergunta(
        "Sua FWI acustica converge, mas o residuo final ainda tem eventos "
        "coerentes e bem organizados. O que isso indica?",
        ["Falta mais iteracoes",
         "Fisica faltando no operador de modelagem",
         "O gradiente esta errado",
         "Excesso de regularizacao"],
        1,
        "Residuo COERENTE = estrutura que o operador nao consegue produzir, "
        "qualquer que seja o modelo. Mais iteracoes nao criam fisica nova. "
        "O passo seguinte e identificar QUAL evento e (onda S? multipla? "
        "atenuacao?) e decidir entre modela-lo ou remove-lo do dado.")

    a.secao("Para continuar")
    a.texto("""
        Leituras que cobrem, com profundidade, o que esta aula so mapeou:
    """)
    a.lista([
        "TARANTOLA (1984) -- a formulacao original da FWI acustica.",
        "BUNKS et al. (1995) -- multiescala; a base da aula 12.",
        "VIRIEUX & OPERTO (2009) e VIRIEUX et al. (2017) -- revisoes de "
        "referencia, com o panorama de otimizacao e parametrizacao.",
        "KJARTANSSON (1979) -- o modelo constant-Q desta aula.",
        "CARCIONE (2022), 'Wave Fields in Real Media' -- o tratado sobre meios "
        "anelasticos, anisotropicos e porosos.",
        "PLESSIX (2006) -- o estado adjunto apresentado de forma geral.",
        "ASNAASHARI et al. (2013) -- regularizacao com modelo a priori (aula 13).",
        "LOUBOUTIN et al. (2019) -- Devito, para escrever propagadores e "
        "adjuntos com diferenciacao simbolica.",
    ], marcador="-")

    a.secao("Exercicios")
    a.exercicio(1, """
        Refaca a tabela de cycle skipping para a SUA situacao de interesse:
        escolha c0, f_ref, a faixa de Q e os percursos tipicos do seu problema.
        A partir de que combinacao a aproximacao sem perdas deixa de ser
        defensavel?
    """)
    a.exercicio(2, """
        Use o propagador `kjartansson_1d` desta aula para gerar um "dado
        observado" atenuado num meio homogeneo e ajuste, por minimos quadrados,
        a velocidade c0 de um modelo SEM atenuacao que melhor explique esse
        dado. Quanto o c0 ajustado se desvia do verdadeiro em funcao de Q?
    """, dica="e a versao 1D e barata do experimento que mede o vies que a "
              "aproximacao sem perdas introduz na velocidade.")
    a.exercicio(3, """
        Verifique numericamente as relacoes de Kramers-Kronig no operador de
        Kjartansson: calcule a transformada de Hilbert do logaritmo da
        amplitude e compare com a fase.
    """, dica="scipy.signal.hilbert.")

    a.fim([
        "Toda hipotese nao satisfeita vira estrutura falsa no modelo -- a FWI nao "
        "tem valvula de escape.",
        "Reflexao e governada por IMPEDANCIA: velocidade e densidade sao ambiguas "
        "entre si.",
        "Dissipacao e dispersao sao inseparaveis (causalidade / Kramers-Kronig).",
        "Kjartansson: Q constante via lei de potencia, gamma = arctan(1/Q)/pi; "
        "no tempo leva a operadores de ordem fracionaria.",
        "Ignorar Q pode causar cycle skipping quando percurso longo, frequencia alta "
        "e Q baixo se combinam.",
        "Escolha a fisica mais SIMPLES que explica o seu dado; diagnostique pelo "
        "residuo, nao pelo gosto.",
    ], proxima="FIM DO CURSO -- veja o README para os proximos passos")


if __name__ == "__main__":
    main()
