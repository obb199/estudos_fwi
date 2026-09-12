#!/usr/bin/env python3
"""
AULA 13 -- Regularizacao: Tikhonov, Variacao Total, vinculos e informacao a priori
==================================================================================
Execute:  python aulas/aula13_regularizacao.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                               # noqa: E402
import fwikit                                    # noqa: E402
from fwikit.aula import Aula                     # noqa: E402
from fwikit import plot                          # noqa: E402
from fwikit.acustico import (Acustico2D, Config, Geometria,   # noqa: E402
                             ricker, misfit_l2, gradiente_adjunto, dt_maximo)
from fwikit.inversao import (lbfgs, limitar, suavizar,         # noqa: E402
                             precondicionar_profundidade,
                             tikhonov, variacao_total, modelo_a_priori)
from fwikit.metricas import erro_relativo_percentual, correlacao  # noqa: E402


def main():
    a = Aula(13, "Regularizacao: Tikhonov, TV, vinculos e informacao a priori",
             modulo="Modulo IV -- Otimizacao, FWI completa e regularizacao",
             duracao="75 min", pre_requisitos="aulas 11 e 12",
             objetivos=[
                 "Entender por que a FWI e mal posta e o que a regularizacao resolve",
                 "Distinguir Tikhonov (suaviza tudo) de TV (preserva interfaces)",
                 "Usar um modelo a priori sem sobrepor o que os dados dizem",
                 "Escolher o peso da regularizacao pela curva em L, e nao no chute",
             ])
    a.cabecalho()

    # ==================================================================
    a.secao("Por que regularizar")
    a.texto("""
        A FWI e mal posta em dois sentidos, e os dois exigem tratamento.

        ESPACO NULO. Existem perturbacoes do modelo que praticamente nao mudam
        os dados: regioes nao iluminadas, estruturas menores que meio
        comprimento de onda, combinacoes de parametros que se cancelam. Nessas
        direcoes J e praticamente plano e o problema nao tem solucao unica.

        AMPLIFICACAO DE RUIDO. Nas direcoes de baixa sensibilidade, um residuo
        pequeno -- inclusive ruido -- exige uma perturbacao GRANDE do modelo
        para ser explicado. Sem controle, a inversao converte ruido em
        estrutura.

        A regularizacao acrescenta um termo que define o que fazer NO ESPACO
        NULO, onde os dados se calam:
    """)
    a.eq("J_total(m)  =  J_dados(m)  +  lambda * R(m)",
         rotulo="funcional regularizado")
    a.aviso("""
        Uma distincao que evita muita confusao: regularizacao NAO melhora o
        ajuste dos dados. Ela sempre PIORA J_dados, por construcao. O que ela
        faz e escolher, entre os muitos modelos que ajustam os dados igualmente
        bem, aquele que e mais plausivel segundo um criterio que VOCE definiu.

        Isso significa que a escolha de R e uma afirmacao sobre a geologia --
        e precisa ser justificada como tal, nao como conveniencia numerica.
    """)

    # ==================================================================
    a.secao("Tikhonov x Variacao Total: a diferenca que importa")
    a.eq("R_tikhonov(m)  =  (1/2) integral | grad m |^2      (norma L2)")
    a.eq("R_TV(m)        =  integral | grad m |               (norma L1)")
    a.texto("""
        A troca de L2 por L1 parece detalhe e nao e. Compare o custo de duas
        maneiras de acomodar uma variacao total de 1:
    """)
    print()
    print("    Um salto de 1 em um so ponto:")
    print("        L2:  1^2              = 1.00")
    print("        L1:  |1|              = 1.00")
    print()
    print("    A mesma variacao espalhada em 10 passos de 0.1:")
    print("        L2:  10 x 0.1^2       = 0.10   <- L2 PREFERE espalhar")
    print("        L1:  10 x |0.1|       = 1.00   <- L1 e indiferente")
    print()
    a.texto("""
        A L2 e barata para muitas variacoes pequenas e cara para um salto
        grande: por isso ela SUAVIZA, borrando interfaces. A L1 cobra o mesmo
        nos dois casos: por isso ela tolera saltos nitidos enquanto ainda pune
        oscilacao de ruido (que tem variacao total alta). Dai a TV
        PRESERVAR INTERFACES.

        Escolha direta: alvo estratificado com contatos nitidos (sal, camadas,
        falhas) pede TV. Alvo com variacao gradual (compactacao, tendencia
        regional) pede Tikhonov.
    """)

    # ==================================================================
    a.secao("Experimento: inversao com dado ruidoso")
    nz, nx, dh = 60, 100, 12.0
    c_verd = np.full((nz, nx), 1900.0, dtype=np.float32)
    c_verd[26:, :] = 2350.0
    c_verd[44:, :] = 2700.0
    zz, xx = np.mgrid[0:nz, 0:nx]
    c_verd[((zz - 35) ** 2 / 6 ** 2 + (xx - 50) ** 2 / 14 ** 2) < 1] = 2550.0
    from scipy.ndimage import gaussian_filter
    c_ini = gaussian_filter(c_verd, 10).astype(np.float32)

    f0 = 8.0
    dt = dt_maximo(2800.0, dh, 4)
    nt = int(1.3 / dt)
    cfg = Config(dh=dh, dt=dt, nt=nt, ordem=4, n_abs=30, f0=f0)
    solver = Acustico2D(cfg, (nz, nx))
    w = ricker(f0, dt, nt)
    geom = Geometria(
        fontes=np.array([[3, ix] for ix in np.linspace(14, nx - 15, 3).astype(int)]),
        receptores=np.array([[3, ix] for ix in range(5, nx - 5, 3)]))
    d_lim = np.stack([solver.modelar(c_verd, geom, w, i)[0]
                      for i in range(geom.ns)], axis=-1)
    rng = np.random.default_rng(7)
    snr_db = 0.0   # ruido com a MESMA energia do sinal
    rms_sinal = float(np.sqrt(np.mean(d_lim ** 2)))
    ruido = rng.normal(0, rms_sinal / 10 ** (snr_db / 20), d_lim.shape)
    d_obs = (d_lim + ruido).astype(np.float32)

    print()
    from fwikit.aula import Aula as _A
    a.resultado("malha", f"{nz} x {nx}")
    a.resultado("tiros / receptores", f"{geom.ns} / {geom.nr}")
    a.resultado("razao sinal/ruido do dado", f"{snr_db:.0f}", "dB")
    a.resultado("erro do modelo inicial",
                f"{erro_relativo_percentual(c_ini, c_verd):.2f}", "%")

    z_mute = int(0.15 * nz)
    caixa = limitar(1700.0, 2900.0)

    def faz_f_e_g(tipo, lam, suav_grad=1.0):
        def f_e_g(c_vec, so_J=False):
            # clipar AQUI tambem: a busca linear avalia J(m + alpha d) ANTES
            # de a projecao ser aplicada, e um passo exagerado pode levar a
            # velocidade a violar o CFL e estourar a simulacao.
            c = np.clip(np.asarray(c_vec, dtype=np.float32),
                        1700.0, 2900.0).reshape(nz, nx)
            if so_J:
                total = 0.0
                for s in range(geom.ns):
                    d, _ = solver.modelar(c, geom, w, s)
                    total += misfit_l2(d, d_obs[:, :, s], dt)[0]
                if tipo == "tikhonov":
                    total += lam * tikhonov(c.astype(np.float64), dh)[0]
                elif tipo == "tv":
                    total += lam * variacao_total(c.astype(np.float64), dh)[0]
                return total
            J, g, _ = gradiente_adjunto(solver, c, geom, w, d_obs, parametro="c")
            g = suavizar(g, suav_grad)
            g[:z_mute, :] = 0.0
            g = precondicionar_profundidade(g, potencia=1.0)
            if tipo == "tikhonov":
                fr, gr = tikhonov(c.astype(np.float64), dh)
                J += lam * fr
                g = g + lam * gr
            elif tipo == "tv":
                fr, gr = variacao_total(c.astype(np.float64), dh)
                J += lam * fr
                g = g + lam * gr
            return J, g.reshape(c_vec.shape)
        return f_e_g

    # escalas de referencia para calibrar lambda
    J0, g0, _ = gradiente_adjunto(solver, c_ini, geom, w, d_obs, parametro="c")
    f_tk = tikhonov(c_ini.astype(np.float64), dh)[0]
    f_tv = variacao_total(c_ini.astype(np.float64), dh)[0]
    lam_tk = 0.02 * J0 / (f_tk + 1e-30)
    lam_tv = 0.02 * J0 / (f_tv + 1e-30)
    print()
    a.resultado("J_dados inicial", f"{J0:.4e}")
    a.resultado("R_tikhonov inicial", f"{f_tk:.4e}")
    a.resultado("R_TV inicial", f"{f_tv:.4e}")
    a.dica("""
        Calibrar lambda: escolha-o de modo que o termo de regularizacao valha
        uma FRACAO conhecida de J_dados no ponto inicial (aqui, 2%). Isso
        torna a escolha independente das unidades e reproduzivel -- muito
        melhor que testar potencias de 10 as cegas.
    """)

    a.texto("""
        O experimento tem quatro variantes, e a quarta existe para revelar algo
        que quase nunca se diz em voz alta. As tres primeiras usam o gradiente
        CRU (sem suavizacao), para que o efeito da regularizacao explicita
        apareca isolado. A quarta nao usa regularizacao explicita nenhuma, mas
        suaviza o gradiente -- uma REGULARIZACAO IMPLICITA.
    """)
    resultados = {}
    print()
    for nome, tipo, lam, suav in [
            ("gradiente cru, sem reg.", None, 0.0, 0.0),
            ("gradiente cru + Tikhonov", "tikhonov", lam_tk, 0.0),
            ("gradiente cru + TV", "tv", lam_tv, 0.0),
            ("so suavizacao do gradiente", None, 0.0, 1.5)]:
        print(f"    rodando: {nome}...", end="", flush=True)
        m_est, hist = lbfgs(faz_f_e_g(tipo, lam, suav),
                            c_ini.astype(np.float64).ravel(), n_iter=18,
                            fracao_passo=0.02, verboso=False,
                            projecao=lambda m: caixa(m))
        c_est = m_est.reshape(nz, nx)
        err = erro_relativo_percentual(c_est, c_verd)
        cor = correlacao(c_est, c_verd)
        resultados[nome] = (c_est, hist, err, cor)
        print(f" erro {err:.2f}%   correlacao {cor:.4f}")
    print()

    a.tabela(["variante", "erro final (%)", "correlacao", "J final / J0"],
             [[n, f"{r[2]:.2f}", f"{r[3]:.4f}", f"{r[1].J[-1]/r[1].J[0]:.4f}"]
              for n, r in resultados.items()]
             + [["(modelo inicial)",
                 f"{erro_relativo_percentual(c_ini, c_verd):.2f}",
                 f"{correlacao(c_ini, c_verd):.4f}", "1.0000"]])

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    vmin, vmax = float(c_verd.min()), float(c_verd.max())
    plot.modelo(ax[0, 0], c_verd, dh, "verdadeiro", "m/s", vmin=vmin, vmax=vmax)
    plot.modelo(ax[0, 1], c_ini, dh, "inicial (suavizado)", "m/s",
                vmin=vmin, vmax=vmax)
    celas = [ax[0, 2], ax[1, 0], ax[1, 1]]
    for k, (nome, r) in enumerate(list(resultados.items())[:3]):
        plot.modelo(celas[k], r[0], dh, f"{nome}\nerro {r[2]:.2f}%", "m/s",
                    vmin=vmin, vmax=vmax)
    kx = nx // 2
    ax[1, 2].plot(c_verd[:, kx], np.arange(nz) * dh, "k-", lw=2,
                  label="verdadeiro")
    ax[1, 2].plot(c_ini[:, kx], np.arange(nz) * dh, "--", color="gray", lw=1.2,
                  label="inicial")
    for nome, r in resultados.items():
        ax[1, 2].plot(r[0][:, kx], np.arange(nz) * dh, lw=1.4, label=nome)
    ax[1, 2].invert_yaxis()
    ax[1, 2].set_xlabel("vp (m/s)"); ax[1, 2].set_ylabel("profundidade (m)")
    ax[1, 2].set_title(f"perfil vertical em x = {kx*dh:.0f} m")
    ax[1, 2].legend(fontsize=7); ax[1, 2].grid(True, alpha=0.3)
    fig.tight_layout()
    plot.salvar(fig, a, "01_regularizacao", mostrar=False)
    nomes = list(resultados)
    e_cru = resultados[nomes[0]][2]
    e_tk = resultados[nomes[1]][2]
    e_tv = resultados[nomes[2]][2]
    e_suav = resultados[nomes[3]][2]
    e_ini = erro_relativo_percentual(c_ini, c_verd)
    melhor_nome = min(resultados, key=lambda k: resultados[k][2])
    reg_ajudou = min(e_tk, e_tv) < e_cru

    a.teoria("O resultado deste experimento -- leia antes de concluir", f"""
        Os numeros, sem maquiagem:

            modelo inicial ................. {e_ini:.2f}%
            gradiente cru, sem reg. ........ {e_cru:.2f}%
            gradiente cru + Tikhonov ....... {e_tk:.2f}%
            gradiente cru + TV ............. {e_tv:.2f}%
            so suavizacao do gradiente ..... {e_suav:.2f}%

        Aqui a regularizacao explicita {"AJUDOU" if reg_ajudou else "PIOROU o resultado"}.
        E vale entender por que, porque e uma situacao comum e mal
        compreendida.

        O modelo inicial ja e suave (foi obtido por filtragem gaussiana do
        verdadeiro), e com 18 iteracoes a inversao nao tem tempo de gerar
        estrutura de alta frequencia. O erro que sobra nao e RUGOSIDADE: e
        falta de CONTRASTE -- os degraus de velocidade ainda nao foram
        recuperados em amplitude. Tikhonov e TV penalizam justamente o
        gradiente espacial, ou seja, penalizam o contraste que ainda falta
        construir. Elas estao puxando na direcao errada para este problema.

        E note a quarta variante: apenas suavizando o gradiente, sem termo
        algum no funcional, chega-se a {e_suav:.2f}% -- o melhor da tabela.
        Uma operacao de uma linha.

        Esse e o ponto central da aula: existe muita REGULARIZACAO IMPLICITA em
        FWI que quase nunca e reconhecida como tal:

          * suavizacao do gradiente (o `g_smooth` do PyFWI);
          * modelo inicial suave -- ele ja restringe onde a solucao pode chegar;
          * estrategia multiescala -- as bandas baixas impoem suavidade;
          * parada antecipada -- poucas iteracoes nao produzem alta frequencia;
          * projecao em caixa.

        Somar regularizacao explicita a tudo isso frequentemente
        SOBRE-regulariza. O sintoma e exatamente o desta tabela: o modelo fica
        mais suave e o erro AUMENTA.
    """)
    a.dica(f"""
        A conclusao pratica NAO e "nao use regularizacao". E:

        1. Meca. Rode com e sem, e compare por uma metrica objetiva. Neste
           caso o melhor foi "{melhor_nome}".

        2. Diagnostique ANTES de escolher R. Se o erro residual e rugosidade,
           Tikhonov/TV ajudam. Se e falta de contraste ou de resolucao, elas
           atrapalham -- o problema e de iluminacao ou de banda, nao de
           suavidade.

        3. Regularizacao explicita ganha importancia quando o dado e muito
           ruidoso E voce roda muitas iteracoes E parte de um modelo que nao e
           suave. Fora dessas condicoes, a regularizacao implicita costuma
           bastar.
    """)
    a.texto("""
        Olhe tambem o perfil vertical da figura: e nele que Tikhonov e TV se
        distinguem uma da outra. A Tikhonov arredonda os degraus; a TV tende a
        manter o degrau e achatar o que esta entre eles.
    """)

    # ==================================================================
    a.secao("A curva em L: escolhendo lambda com criterio")
    a.texto("""
        Se ainda assim voce for usar regularizacao explicita, o peso lambda
        precisa ser escolhido com criterio. Lambda controla o compromisso entre
        ajustar os dados e obedecer ao criterio de regularizacao. A ferramenta
        classica para escolhe-lo e a CURVA EM L: para varios lambdas, ponha J_dados no eixo x e R no eixo y
        (ambos em log). A curva tem forma de L, e o lambda bom fica no
        COTOVELO -- o ponto de maior curvatura.
    """)
    lams = lam_tk * np.array([0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    pontos = []
    print()
    print(f"    {'lambda/lambda0':<18}{'J_dados':<16}{'R_tikhonov':<16}"
          f"{'erro modelo (%)'}")
    for lam in lams:
        m_est, hist = lbfgs(faz_f_e_g("tikhonov" if lam > 0 else None, lam),
                            c_ini.astype(np.float64).ravel(), n_iter=6,
                            fracao_passo=0.02, verboso=False,
                            projecao=lambda m: caixa(m))
        c_est = m_est.reshape(nz, nx)
        Jd = 0.0
        for s in range(geom.ns):
            d, _ = solver.modelar(c_est.astype(np.float32), geom, w, s)
            Jd += misfit_l2(d, d_obs[:, :, s], dt)[0]
        R = tikhonov(c_est, dh)[0]
        err = erro_relativo_percentual(c_est, c_verd)
        pontos.append((lam, Jd, R, err))
        print(f"    {lam/lam_tk if lam_tk else 0:<18.2f}{Jd:<16.5e}"
              f"{R:<16.5e}{err:.2f}")
    print()
    melhor = min(pontos, key=lambda p: p[3])
    print(f"    Menor erro de MODELO com lambda/lambda0 = "
          f"{melhor[0]/lam_tk if lam_tk else 0:.2f}  ({melhor[3]:.2f}%)")
    print()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    Js = [p[1] for p in pontos]
    Rs = [p[2] for p in pontos]
    ax[0].loglog(Js, Rs, "o-", lw=1.6)
    for lam, Jd, R, _ in pontos:
        ax[0].annotate(f"{lam/lam_tk if lam_tk else 0:.1f}", (Jd, R),
                       fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax[0].set_xlabel("J_dados (ajuste)")
    ax[0].set_ylabel("R (regularizacao)")
    ax[0].set_title("curva em L (rotulos = lambda/lambda0)")
    ax[1].semilogx([max(p[0] / lam_tk, 1e-3) if lam_tk else 1e-3 for p in pontos],
                   [p[3] for p in pontos], "o-", lw=1.6)
    ax[1].set_xlabel("lambda / lambda0")
    ax[1].set_ylabel("erro do modelo (%)")
    ax[1].set_title("erro de modelo x lambda")
    fig.tight_layout()
    plot.salvar(fig, a, "02_curva_em_L", mostrar=False)
    a.aviso("""
        Note a honestidade que o segundo painel exige: em problema SINTETICO
        voce conhece o modelo verdadeiro e pode escolher lambda pelo menor erro
        de modelo. Em dado REAL isso e impossivel -- voce so tem J_dados e R.
        Por isso existe a curva em L: ela escolhe lambda usando apenas o que
        estaria disponivel num caso real.

        Se voce reportar resultados escolhendo lambda pelo erro de modelo, diga
        isso explicitamente. Caso contrario o resultado parece melhor do que
        seria na pratica.
    """)

    # ==================================================================
    a.secao("Modelo a priori e vinculos")
    a.texto("""
        Alem de suavidade, ha duas formas mais diretas de injetar conhecimento:
    """)
    a.eq("R_priori(m)  =  (1/2) || W (m - m_priori) ||^2",
         rotulo="Asnaashari et al., 2013")
    a.texto("""
        A matriz de pesos W diz ONDE voce confia no modelo a priori. O uso
        classico: peso alto ao longo de um poco, onde ha perfil sonico medido,
        e peso zero longe dele. Assim voce ancora a inversao no dado de poco
        sem impor nada onde nao tem informacao.

        O vinculo mais simples e mais eficaz de todos e a CAIXA:
    """)
    a.codigo("""
        m = np.clip(m, v_min, v_max)     # apos cada atualizacao
    """, titulo="projecao em caixa")
    a.texto("""
        Parece grosseiro, mas evita o modo de falha mais comum da FWI: um
        passo ruim empurra a velocidade para valores nao fisicos, a simulacao
        fica instavel (CFL!) ou o gradiente vira lixo, e a inversao nao se
        recupera. Use sempre.
    """)
    a.tabela(
        ["Vinculo", "Como", "Quando"],
        [["Caixa (vmin, vmax)", "clip apos cada passo", "sempre"],
         ["Agua fixa", "zerar o gradiente na lamina d'agua", "dado marinho"],
         ["Mascara de profundidade", "zerar abaixo da penetracao", "evita artefato"],
         ["Modelo a priori", "R = ||W(m - m_prior)||^2", "ha poco ou tomografia"],
         ["Relacao entre parametros", "vincular vs e rho a vp", "multiparametro"]])

    a.secao("Como fazer isso no PyFWI")
    a.codigo("""
        inpa['tikhonov'] = {'az': 1.0, 'ax': 1.0, 'lambda_weight': 1e-3}
        inpa['tv']       = {'az': 1.0, 'ax': 1.0, 'lambda_weight': 1e-3}
        inpa['prior_model'] = {...}     # ver Regularization.priori_regularization
        inpa['param_relation'] = {...}  # vinculo entre parametros
    """, titulo="chaves de regularizacao do inpa")
    a.texto("""
        `az` e `ax` permitem anisotropia na regularizacao -- penalizar mais a
        variacao vertical que a horizontal, por exemplo, o que faz sentido em
        meio estratificado. E um recurso subutilizado e muito util: use
        az < ax para favorecer camadas horizontais.
    """)

    a.pergunta(
        "Voce aumenta lambda e J_dados sobe, mas o modelo fica visivelmente "
        "melhor. Isso e contraditorio?",
        ["Sim, algo esta errado",
         "Nao -- parte do ajuste anterior era ajuste ao RUIDO; "
         "regularizar troca ajuste espurio por plausibilidade",
         "Sim, regularizacao nunca deve piorar J_dados",
         "Nao, mas so acontece com TV"],
        1,
        "Regularizacao SEMPRE piora J_dados: e o preco que ela cobra. Quando o "
        "modelo melhora mesmo assim, e porque o ajuste extra que se perdeu era "
        "ajuste ao ruido e a artefatos, nao a sinal. E exatamente para isso "
        "que a regularizacao existe.")

    a.secao("Exercicios")
    a.exercicio(1, """
        Refaca a curva em L para TV em vez de Tikhonov. O cotovelo fica no
        mesmo lugar relativo? Qual das duas e mais sensivel a escolha de lambda?
    """)
    a.exercicio(2, """
        Implemente a regularizacao por modelo a priori com W nao uniforme:
        peso 1 numa coluna (simulando um poco) e 0 no resto. Verifique que a
        inversao fica ancorada ali e livre no resto.
    """, dica="use fwikit.inversao.modelo_a_priori com o argumento `peso`.")
    a.exercicio(3, """
        Use az = 0.2 e ax = 1.0 na Tikhonov (penalizando menos a variacao
        vertical) e veja se o resultado fica mais estratificado. Compare com
        az = ax = 1.
    """)

    a.fim([
        "FWI e mal posta: espaco nulo + amplificacao de ruido. Regularizacao define "
        "o que fazer onde os dados se calam.",
        "Tikhonov (L2) suaviza tudo; TV (L1) preserva interfaces. A escolha e uma "
        "afirmacao sobre a geologia.",
        "Regularizacao SEMPRE piora J_dados. Se ela tambem piorar o MODELO, voce "
        "esta sobre-regularizando.",
        "Muito do que regulariza uma FWI e IMPLICITO: gradiente suavizado, inicial "
        "suave, multiescala, parada antecipada. Contabilize isso antes de somar mais.",
        "Calibre lambda como fracao de J_dados e refine pela curva em L -- nunca "
        "pelo erro de modelo, que em dado real voce nao tem.",
        "Projecao em caixa (vmin, vmax) e o vinculo mais barato e mais eficaz. Use sempre.",
    ], proxima="aula14_limites_acustico.py -- onde a aproximacao acustica deixa de valer")


if __name__ == "__main__":
    main()
