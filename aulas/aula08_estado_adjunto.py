#!/usr/bin/env python3
"""
AULA 08 -- O metodo do estado adjunto: o gradiente sem forca bruta
===================================================================
Execute:  python aulas/aula08_estado_adjunto.py
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
                             ricker, misfit_l2, gradiente_adjunto, dt_maximo)


def main():
    a = Aula(8, "O metodo do estado adjunto: o gradiente sem forca bruta",
             modulo="Modulo III -- O problema inverso e o gradiente",
             duracao="90 min", pre_requisitos="aulas 01-07",
             objetivos=[
                 "Entender por que diferencas finitas sao inviaveis para o gradiente",
                 "Derivar a equacao adjunta pelo metodo do Lagrangiano, passo a passo",
                 "Reconhecer que o gradiente e uma correlacao cruzada de defasagem zero",
                 "Ler a implementacao linha a linha e saber onde cada termo entra",
                 "Interpretar a forma de um gradiente de FWI e saber o que e artefato",
             ])
    a.cabecalho()

    # ==================================================================
    a.secao("O problema: por que nao usar diferencas finitas")
    a.texto("""
        Queremos dJ/dm, com m tendo N parametros. A abordagem ingenua:
    """)
    a.eq("dJ/dm_i  ~  [ J(m + eps e_i) - J(m) ] / eps", rotulo="forca bruta")
    a.texto("""
        Cada J(...) exige uma modelagem direta completa por tiro. Logo o custo e
        N vezes o custo de uma modelagem, POR ITERACAO. Faca a conta:
    """)
    print()
    print(f"    {'modelo':<20}{'N parametros':<18}{'modelagens/iter':<20}"
          f"{'tempo (a 0.4 s cada)'}")
    for nome, nz, nx in [("100 x 100", 100, 100), ("200 x 400", 200, 400),
                         ("500 x 1000", 500, 1000),
                         ("3D 300^3", 300, 300 * 300)]:
        N = nz * nx
        seg = N * 0.4
        if seg < 3600:
            t = f"{seg/60:.1f} min"
        elif seg < 86400:
            t = f"{seg/3600:.1f} horas"
        elif seg < 86400 * 730:
            t = f"{seg/86400:.1f} dias"
        else:
            t = f"{seg/86400/365:.0f} anos"
        print(f"    {nome:<20}{N:<18,}{N:<20,}{t}")
    print()
    a.texto("""
        Nem em 2D pequeno isso e aceitavel. O metodo do estado adjunto obtem o
        gradiente INTEIRO com DUAS modelagens por tiro -- e o custo nao depende
        de N. E a ideia que viabiliza a FWI.
    """)

    # ==================================================================
    a.secao("A derivacao, passo a passo")
    a.texto("""
        Trabalhamos com o parametro m(x) = 1/c(x)^2, porque a equacao e LINEAR
        nele. O problema direto e:
    """)
    a.eq("m d2p/dt2  -  laplaciano(p)  -  s  =  0", rotulo="restricao")
    a.texto("""
        E o funcional a minimizar:
    """)
    a.eq("J(m) = (1/2) soma_r integral_t [ p(x_r,t) - d_obs(x_r,t) ]^2 dt")
    a.texto("""
        PASSO 1 -- Lagrangiano. A restricao vale para todo x e todo t, entao
        introduzimos um multiplicador de Lagrange lambda(x,t) e integramos:
    """)
    a.eq("L  =  J  +  integral_x integral_t  lambda [ m p_tt - lap(p) - s ]",
         rotulo="Lagrangiano")
    a.texto("""
        Como a restricao e satisfeita pela solucao, L = J e podemos derivar L em
        vez de J -- com a vantagem de podermos ESCOLHER lambda para matar o
        termo dificil.

        PASSO 2 -- Variacao em relacao a p. Aqui esta a jogada. Derivando L em
        relacao a p e integrando por partes DUAS vezes no tempo (o que joga as
        derivadas de p para lambda) e duas vezes no espaco (o laplaciano e
        auto-adjunto), aparece:
    """)
    a.eq("dL/dp  =  [ m lambda_tt - lap(lambda) ]  +  soma_r r(t) delta(x - x_r)")
    a.texto("""
        PASSO 3 -- Escolher lambda. Se exigirmos dL/dp = 0, o termo que dependia
        da variacao de p desaparece. Isso define a EQUACAO ADJUNTA:
    """)
    a.eq("m d2(lambda)/dt2  -  laplaciano(lambda)  =  - soma_r r(t) delta(x - x_r)",
         rotulo="EQUACAO ADJUNTA")
    a.texto("""
        com condicoes FINAIS nulas: lambda = 0 e lambda_t = 0 em t = T.
    """)
    a.teoria("Leia a equacao adjunta com atencao", """
        Ela e a MESMA equacao de onda do problema direto. Mudam duas coisas:

        1. A FONTE nao e mais a wavelet no ponto de tiro. E o RESIDUO -- com o
           sinal trocado -- injetado em TODOS os receptores ao mesmo tempo.
           Cada receptor vira uma fonte.

        2. As condicoes sao FINAIS, nao iniciais. Resolve-se de tras para
           frente no tempo.

        Interpretacao fisica: voce pega o erro que sobrou nos receptores e o
        joga de volta para dentro da Terra, para ver de onde ele veio.

        Como a equacao acustica sem atenuacao e AUTO-ADJUNTA (invariante por
        reversao temporal), na pratica basta rodar o MESMO propagador ao
        contrario. Em meios dissipativos essa simetria se perde e o adjunto
        precisa ser derivado com mais cuidado -- retomamos isso na aula 14.
    """)
    a.texto("""
        PASSO 4 -- Variacao em relacao a m. Com lambda escolhido, o unico termo
        que sobra em dL/dm e:
    """)
    a.eq("dJ/dm(x)  =  integral_t  lambda(x,t)  d2p/dt2(x,t)  dt",
         rotulo="GRADIENTE")
    a.texto("""
        Somando sobre os tiros, e com os pesos de quadratura da malha:
    """)
    a.eq("dJ/dm_i  =  dh^2 dt  soma_tiros soma_n  lambda_i^n  (p_tt)_i^n",
         rotulo="forma discreta")
    a.dica("""
        Repare no que essa formula E: uma CORRELACAO CRUZADA DE DEFASAGEM ZERO
        entre o campo direto (sua segunda derivada temporal) e o campo adjunto,
        ponto a ponto do modelo.

        O gradiente e grande onde os dois campos coincidem no tempo e no espaco.
        E isso tem sentido fisico imediato: sao os pontos onde uma perturbacao
        do modelo teria afetado o dado -- ou seja, os pontos responsaveis pelo
        erro.

        Essa mesma estrutura -- "correlacao entre campo direto e campo
        retropropagado" -- e a condicao de imagem da MIGRACAO RTM. FWI e RTM sao
        parentes proximos: a RTM e essencialmente a primeira iteracao de uma FWI.
    """)
    a.texto("""
        Por fim, para trabalhar em velocidade em vez de vagarosidade ao
        quadrado, basta a regra da cadeia:
    """)
    a.eq("dJ/dc  =  dJ/dm * dm/dc  =  dJ/dm * ( -2 / c^3 )")

    a.pergunta(
        "Qual e o custo de um gradiente de FWI por tiro, pelo estado adjunto?",
        ["1 modelagem", "2 modelagens (direta + adjunta)",
         "N modelagens, N = numero de parametros",
         "depende do numero de receptores"],
        1,
        "Uma modelagem direta (que tambem produz o residuo) e uma adjunta. O "
        "custo independe do numero de parametros E do numero de receptores -- "
        "todos os receptores sao injetados simultaneamente na propagacao "
        "adjunta. E exatamente por isso que a FWI e viavel.")

    a.pausa()

    # ==================================================================
    a.secao("A implementacao, linha a linha")
    a.texto("""
        Veja como cada passo da derivacao vira codigo em `fwikit/acustico.py`.
        Compare com as equacoes acima -- a correspondencia e direta.
    """)
    a.codigo("""
        for s in tiros:
            # (1) DIRETO: resolve a equacao de onda e guarda d2p/dt2
            d_calc, extras = solver.modelar(c, geom, wavelet, i_fonte=s,
                                            guardar_campo=True)

            # (2) RESIDUO e funcional
            Js, r = misfit_l2(d_calc, d_obs[:, :, s], dt)
            J += Js

            # (3) ADJUNTO: mesma equacao, fonte = -residuo nos receptores,
            #     resolvida de tras para frente
            lam = solver.retropropagar(c, geom, r)

            # (4) GRADIENTE: correlacao cruzada de defasagem zero
            g_ext += np.einsum("tzx,tzx->zx", lam, extras["dtt"]) * (dt * dh**2)

        # (5) volta ao dominio fisico -- ADJUNTO da extensao de bordas
        g_m = solver.contrair_adjunto(g_ext)

        # (6) regra da cadeia para velocidade
        g_c = g_m * (-2.0 / c**3)
    """, titulo="fwikit/acustico.py :: gradiente_adjunto()")
    a.aviso("""
        Chame a atencao para o passo (5). O modelo fisico e estendido com uma
        moldura absorvente antes de propagar, e essa extensao REPLICA os pixels
        de borda (`mode='edge'`). Isso e um operador linear E, e o gradiente
        precisa de E^T -- ou seja, a sensibilidade acumulada dentro da moldura
        tem de ser DOBRADA de volta sobre o pixel de borda que a gerou, nao
        simplesmente descartada com um recorte.

        Esquecer isso e um dos erros mais dificeis de achar em FWI: o gradiente
        fica certo no miolo do modelo e erra perto das bordas e da superficie --
        justamente onde ficam fontes e receptores. Na montagem deste curso esse
        detalhe valia 10% de erro na derivada direcional. A aula 09 mostra o
        teste que o revelou.
    """)

    # ==================================================================
    a.secao("Calculando e interpretando um gradiente")
    nz, nx, dh = 90, 160, 10.0
    c_verd = np.full((nz, nx), 2000.0, dtype=np.float32)
    zz, xx = np.mgrid[0:nz, 0:nx]
    c_verd[((zz - 45) ** 2 + (xx - 80) ** 2) < 10 ** 2] = 2300.0
    c_ini = np.full((nz, nx), 2000.0, dtype=np.float32)

    f0 = 10.0
    dt = dt_maximo(2400.0, dh, 4)
    nt = int(1.4 / dt)
    cfg = Config(dh=dh, dt=dt, nt=nt, ordem=4, n_abs=30, f0=f0)
    solver = Acustico2D(cfg, (nz, nx))
    w = ricker(f0, dt, nt)

    n_tiros = 7
    fontes = np.array([[4, ix] for ix in np.linspace(15, nx - 16, n_tiros).astype(int)])
    receptores = np.array([[4, ix] for ix in range(6, nx - 6, 2)])
    geom = Geometria(fontes=fontes, receptores=receptores)

    print()
    print("    gerando dado observado...", end="", flush=True)
    d_obs = np.stack([solver.modelar(c_verd, geom, w, i)[0]
                      for i in range(geom.ns)], axis=-1)
    print(" ok")

    print("    gradiente com 1 tiro...", end="", flush=True)
    t0 = time.time()
    J1, g1, _ = gradiente_adjunto(solver, c_ini, geom, w, d_obs, tiros=[3])
    print(f" {time.time()-t0:.2f} s")
    print(f"    gradiente com {geom.ns} tiros...", end="", flush=True)
    t0 = time.time()
    Jn, gn, res = gradiente_adjunto(solver, c_ini, geom, w, d_obs)
    dur = time.time() - t0
    print(f" {dur:.2f} s")
    print()
    a.resultado("J (todos os tiros)", f"{Jn:.6e}")
    a.resultado("custo por tiro", f"{dur/geom.ns:.2f}", "s (direto + adjunto)")
    a.resultado("|grad| maximo (1 tiro)", f"{np.abs(g1).max():.3e}")
    a.resultado("|grad| maximo (%d tiros)" % geom.ns, f"{np.abs(gn).max():.3e}")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    plot.modelo(ax[0, 0], c_verd, dh, "modelo verdadeiro", "m/s")
    plot.modelo(ax[0, 1], c_ini, dh, "modelo inicial (homogeneo)", "m/s")
    plot.perturbacao(ax[0, 2], c_verd - c_ini, dh,
                     "o que a FWI precisa encontrar", "m/s")
    plot.perturbacao(ax[1, 0], g1, dh, "gradiente: 1 tiro")
    ax[1, 0].plot(fontes[3, 1] * dh, fontes[3, 0] * dh, "k*", ms=12)
    plot.perturbacao(ax[1, 1], gn, dh, f"gradiente: {geom.ns} tiros somados")
    ax[1, 1].plot(fontes[:, 1] * dh, fontes[:, 0] * dh, "k*", ms=8)
    from scipy.ndimage import gaussian_filter
    plot.perturbacao(ax[1, 2], gaussian_filter(gn, 2.0), dh,
                     "gradiente suavizado (sigma = 2)")
    fig.tight_layout()
    plot.salvar(fig, a, "01_gradiente_adjunto", mostrar=False)

    a.teoria("Como ler um gradiente de FWI", """
        Quatro coisas aparecem em praticamente todo gradiente. Saber
        identifica-las evita interpretar artefato como geologia.

        1. O SINAL -- E EM QUAL PARAMETRO. A atualizacao e m <- m - alpha * g,
           e o sentido depende do parametro em que g esta escrito. Em
           VELOCIDADE (g_c = dJ/dc, que e o que estas figuras mostram e o que
           `gradiente_adjunto` devolve por padrao), g_c < 0 faz c AUMENTAR. Em
           VAGAROSIDADE AO QUADRADO (g_m = dJ/dm, a forma em que a derivacao
           acima sai) o sentido se INVERTE: como dm/dc = -2/c^3 < 0, e g_m > 0
           que faz a velocidade aumentar. Confira sempre o sinal contra a
           perturbacao que voce sabe que existe -- e o teste mais barato de
           sanidade.

        2. ARTEFATO DE ALTA AMPLITUDE PERTO DAS FONTES E RECEPTORES. O campo
           tem amplitude enorme ali (esta perto da singularidade da fonte), e a
           correlacao explode. Nao e informacao: e geometria. Se voce nao
           tratar, o primeiro passo da inversao vai gastar tudo mexendo na
           superficie.

        3. O "ARCO" ou "BANANA". Um unico tiro produz uma faixa larga ligando
           fonte e receptor, nao um ponto. E a zona de Fresnel: a onda e sensivel
           a um volume, nao a um raio. Resolucao finita e consequencia direta
           disso.

        4. ILUMINACAO DECRESCENTE COM A PROFUNDIDADE. A amplitude cai com o
           espalhamento geometrico, entao o gradiente e sistematicamente mais
           fraco em profundidade. Sem correcao, a FWI so atualiza a parte rasa.

        Compare os paineis de 1 tiro e de varios: somar tiros e o que faz os
        arcos individuais interferirem construtivamente sobre a anomalia
        verdadeira e destrutivamente no resto. E a cobertura que forma a imagem.
    """)

    # perfil de iluminacao
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    perfil = np.abs(gn).mean(axis=1)
    ax[0].plot(perfil / perfil.max(), np.arange(nz) * dh, lw=1.8)
    ax[0].invert_yaxis()
    ax[0].set_xlabel("|gradiente| medio (normalizado)")
    ax[0].set_ylabel("profundidade (m)")
    ax[0].set_title("iluminacao cai com a profundidade")
    corte = nz // 2
    ax[1].plot(np.arange(nx) * dh, gn[corte] / (np.abs(gn[corte]).max() + 1e-30),
               lw=1.6, label="gradiente")
    ax[1].plot(np.arange(nx) * dh,
               (c_verd - c_ini)[corte] / (np.abs((c_verd - c_ini)[corte]).max() + 1e-30),
               lw=1.6, ls="--", label="perturbacao verdadeira")
    ax[1].set_xlabel("x (m)"); ax[1].set_ylabel("normalizado")
    ax[1].set_title(f"corte horizontal em z = {corte*dh:.0f} m")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    plot.salvar(fig, a, "02_iluminacao_e_corte", mostrar=False)

    razao = float(perfil[5:15].mean() / (perfil[nz - 25:nz - 15].mean() + 1e-30))
    print()
    a.resultado("razao de amplitude raso/profundo no gradiente", f"{razao:.1f}", "x")
    print()
    a.texto(f"""
        Neste modelo pequeno e com fonte rasa a razao fica em torno de
        {razao:.1f}x -- o desequilibrio existe mas ainda e modesto. Em modelos
        realistas, com varios quilometros de profundidade, essa razao passa
        facilmente de uma ordem de grandeza. E ela que justifica o
        PRE-CONDICIONAMENTO da aula 10: sem corrigir, a otimizacao gasta o passo
        inteiro na parte rasa e a parte profunda praticamente nao se move.
    """)

    a.pergunta(
        "O gradiente de um unico tiro mostra arcos largos em vez da anomalia "
        "pontual. Isso e erro?",
        ["Sim, o gradiente esta mal calculado",
         "Nao -- e a zona de Fresnel; um tiro so nao resolve a posicao, "
         "e a soma sobre tiros que localiza",
         "Sim, falta suavizacao",
         "Nao, mas some se voce usar frequencia mais baixa"],
        1,
        "A onda e sensivel a um VOLUME (zona de Fresnel), nao a uma linha. Um "
        "tiro isolado nao consegue distinguir onde ao longo do arco esta o "
        "espalhador. A soma sobre tiros faz os arcos interferirem: construtiva "
        "onde esta a anomalia, destrutiva no resto. Frequencia mais alta afina o "
        "arco (melhor resolucao) mas nao elimina o efeito.")

    if mostrar_figuras():
        a.secao("Explore: a contribuicao de cada tiro")
        a.texto("""
            Veja o gradiente tiro a tiro e a soma acumulada. Observe como cada
            tiro sozinho e ambiguo e como a soma converge para a anomalia.
            Feche para continuar.
        """)
        from matplotlib.widgets import Slider
        gs = []
        acum = np.zeros_like(gn)
        for i in range(geom.ns):
            _, gi, _ = gradiente_adjunto(solver, c_ini, geom, w, d_obs, tiros=[i])
            gs.append(gi)
            acum = acum + gi
        figi, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
        plt.subplots_adjust(bottom=0.2)
        im1 = plot.perturbacao(a1, gs[0], dh, "gradiente do tiro 1")
        im2 = plot.perturbacao(a2, gs[0], dh, "soma acumulada (1 tiro)")
        s_t = Slider(plt.axes([0.15, 0.06, 0.7, 0.03]), "tiros", 1, geom.ns,
                     valinit=1, valstep=1)

        def atualiza(_):
            k = int(s_t.val)
            im1.set_data(gs[k - 1])
            lim1 = np.percentile(np.abs(gs[k - 1]), 99.5)
            im1.set_clim(-lim1, lim1)
            a1.set_title(f"gradiente do tiro {k}")
            soma = sum(gs[:k])
            im2.set_data(soma)
            lim2 = np.percentile(np.abs(soma), 99.5)
            im2.set_clim(-lim2, lim2)
            a2.set_title(f"soma acumulada ({k} tiros)")
            figi.canvas.draw_idle()

        s_t.on_changed(atualiza)
        plt.show()

    a.secao("Exercicios")
    a.exercicio(1, """
        Refaca a derivacao do Lagrangiano em detalhe, escrevendo explicitamente
        as duas integracoes por partes no tempo. Mostre por que as condicoes do
        problema adjunto sao FINAIS e nao iniciais.
    """, dica="os termos de fronteira em t=0 e t=T so se anulam com essa escolha.")
    a.exercicio(2, """
        Calcule o gradiente para UM tiro e UM unico receptor. Voce deve ver um
        arco (a 'banana') ligando fonte e receptor. Varie a posicao do receptor
        e observe o arco se mover.
    """, dica="passe uma geometria com receptores de tamanho 1.")
    a.exercicio(3, """
        Compare o gradiente calculado com f0 = 5 Hz e com f0 = 20 Hz. Qual a
        largura tipica dos arcos em cada caso? Relacione com a zona de Fresnel,
        cuja largura vai como sqrt(lambda * L).
    """)

    a.fim([
        "Gradiente por forca bruta custa N modelagens; pelo adjunto, 2 por tiro.",
        "A equacao adjunta e a MESMA equacao de onda, com -residuo injetado nos "
        "receptores e resolvida de tras para frente.",
        "dJ/dm = integral de lambda * d2p/dt2 -- uma correlacao cruzada de defasagem zero.",
        "O adjunto da extensao de bordas precisa DOBRAR a moldura de volta; "
        "esquecer isso estraga o gradiente perto da superficie.",
        "No gradiente: cheque o sinal, ignore o artefato de fonte/receptor, entenda "
        "os arcos de Fresnel e corrija a iluminacao com profundidade.",
    ], proxima="aula09_verificacao_gradiente.py -- como PROVAR que o gradiente esta certo")


if __name__ == "__main__":
    main()
