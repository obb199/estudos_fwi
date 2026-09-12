#!/usr/bin/env python3
"""
AULA 11 -- Otimizacao: descida maxima, gradiente conjugado e l-BFGS
===================================================================
Execute:  python aulas/aula11_otimizacao.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                               # noqa: E402
import fwikit                                    # noqa: E402
from fwikit.aula import Aula                     # noqa: E402
from fwikit import plot                          # noqa: E402
from fwikit.acustico import (Acustico2D, Config, Geometria,   # noqa: E402
                             ricker, misfit_l2, gradiente_adjunto, dt_maximo)
from fwikit.inversao import (descida_maxima, gradiente_conjugado,  # noqa: E402
                             lbfgs, limitar, suavizar,
                             precondicionar_profundidade)
from fwikit.metricas import resumo, imprimir_resumo, erro_relativo_percentual  # noqa: E402


def main():
    a = Aula(11, "Otimizacao: descida maxima, gradiente conjugado e l-BFGS",
             modulo="Modulo IV -- Otimizacao, FWI completa e regularizacao",
             duracao="80 min", pre_requisitos="aulas 08-10",
             objetivos=[
                 "Entender por que a FWI nao pode usar metodos de Newton diretos",
                 "Implementar e comparar as tres direcoes de busca classicas",
                 "Entender a busca linear e por que ela domina o custo em FWI",
                 "Escolher o otimizador certo para o seu problema, com criterio",
             ])
    a.cabecalho()

    # ==================================================================
    a.secao("O panorama dos metodos")
    a.texto("""
        Todos os metodos tem a mesma forma. O que muda e como se constroi a
        direcao d_k:
    """)
    a.eq("m_(k+1)  =  m_k  +  alpha_k  d_k")
    a.tabela(
        ["Metodo", "direcao d_k", "memoria", "por que (nao) usar em FWI"],
        [["Newton", "-H^(-1) g", "N^2", "impossivel: H nao cabe na memoria"],
         ["Gauss-Newton", "-(J'J)^(-1) g", "N^2", "idem; usado so em versao truncada"],
         ["Descida maxima", "-g", "N", "barato e robusto, mas lento"],
         ["Grad. conjugado", "-g + beta d_(k-1)", "2N", "bom custo/beneficio"],
         ["l-BFGS", "-H_aprox g", "2mN", "o padrao de mercado"]])
    a.teoria("Por que Newton esta fora", """
        Com N = 10^6 parametros, a Hessiana tem 10^12 entradas: 4 TB em
        float32. Nao se monta, nao se armazena, nao se inverte.

        Mas a Hessiana IMPORTA, e muito. Ela contem duas informacoes
        essenciais:

        * na DIAGONAL, a iluminacao de cada ponto -- por isso a parte rasa
          domina o gradiente bruto (aula 10);

        * FORA DA DIAGONAL, o acoplamento entre parametros -- por isso
          alterar a velocidade num ponto muda o que se deve fazer no ponto
          vizinho.

        Os metodos quase-Newton existem exatamente para capturar parte disso
        sem nunca montar a matriz. O l-BFGS constroi uma aproximacao da
        INVERSA usando apenas os ultimos m pares (s_k, y_k) de variacao de
        modelo e de gradiente. Custo de memoria: 2mN, com m tipicamente 5 a 20.
    """)

    a.secao("A busca linear, onde o tempo de verdade e gasto")
    a.texto("""
        Achar a direcao e metade do trabalho. A outra metade e decidir QUANTO
        andar nela. Em FWI cada tentativa de passo custa uma modelagem direta
        completa de todos os tiros -- entao a qualidade do criterio importa
        tanto quanto a direcao.
    """)
    a.eq("condicao de Armijo:   J(m + alpha d)  <=  J(m) + c1 alpha <g,d>",
         rotulo="descida suficiente")
    a.texto("""
        Duas estrategias, e a segunda e a que se usa na pratica:
    """)
    a.lista([
        "RETROCESSO (backtracking): tente alpha, se falhar reduza pela metade "
        "e repita. Simples, mas gasta 5 a 10 avaliacoes.",
        "AJUSTE PARABOLICO: ja se conhece J(0) e a inclinacao <g,d>. Com UMA "
        "avaliacao extra fica determinada a parabola, cujo minimo da o passo. "
        "Costuma acertar em 2 avaliacoes. E o que `fwikit.inversao` usa.",
    ])
    a.dica("""
        O chute INICIAL do passo tambem importa muito. Em FWI o criterio com
        significado fisico e: "nao altere o modelo em mais de 1-3% de uma vez".
        Isso e mais confiavel que qualquer estimativa puramente matematica,
        porque respeita a escala do problema.
    """)

    # ==================================================================
    a.secao("Comparando os tres em um problema real de FWI")
    nz, nx, dh = 60, 100, 12.0
    c_verd = np.full((nz, nx), 2000.0, dtype=np.float32)
    zz, xx = np.mgrid[0:nz, 0:nx]
    c_verd[((zz - 30) ** 2 / 9 ** 2 + (xx - 50) ** 2 / 16 ** 2) < 1] = 2350.0
    c_ini = np.full((nz, nx), 2000.0, dtype=np.float32)

    f0 = 8.0
    dt = dt_maximo(2400.0, dh, 4)
    nt = int(1.3 / dt)
    cfg = Config(dh=dh, dt=dt, nt=nt, ordem=4, n_abs=30, f0=f0)
    solver = Acustico2D(cfg, (nz, nx))
    w = ricker(f0, dt, nt)
    geom = Geometria(
        fontes=np.array([[3, ix] for ix in np.linspace(10, nx - 11, 5).astype(int)]),
        receptores=np.array([[3, ix] for ix in range(5, nx - 5, 2)]))
    d_obs = np.stack([solver.modelar(c_verd, geom, w, i)[0]
                      for i in range(geom.ns)], axis=-1)
    print()
    a.resultado("malha", f"{nz} x {nx}")
    a.resultado("tiros / receptores", f"{geom.ns} / {geom.nr}")
    a.resultado("f0", f"{f0:.0f}", "Hz")
    a.resultado("erro inicial do modelo",
                f"{erro_relativo_percentual(c_ini, c_verd):.2f}", "%")

    z_mute = int(0.18 * nz)
    contador = {"n": 0}

    def f_e_g(c_vec, so_J=False):
        """Interface exigida pelos otimizadores do fwikit."""
        c = np.asarray(c_vec, dtype=np.float32).reshape(nz, nx)
        if so_J:
            contador["n"] += 1
            total = 0.0
            for s in range(geom.ns):
                d, _ = solver.modelar(c, geom, w, s)
                total += misfit_l2(d, d_obs[:, :, s], dt)[0]
            return total
        contador["n"] += 1
        J, g, _ = gradiente_adjunto(solver, c, geom, w, d_obs, parametro="c")
        g = suavizar(g, 1.5)
        g[:z_mute, :] = 0.0
        g = precondicionar_profundidade(g, potencia=1.0)
        return J, g.reshape(c_vec.shape)

    caixa = limitar(1700.0, 2600.0)
    n_iter = 12
    resultados = {}
    print()
    for nome, otim in [("descida maxima", descida_maxima),
                       ("gradiente conjugado", gradiente_conjugado),
                       ("l-BFGS", lbfgs)]:
        contador["n"] = 0
        print(f"    --- {nome} ---")
        t0 = time.time()
        m_est, hist = otim(f_e_g, c_ini.astype(np.float64).ravel(),
                           n_iter=n_iter, fracao_passo=0.02, verboso=False,
                           projecao=lambda m: caixa(m))
        dur = time.time() - t0
        c_est = m_est.reshape(nz, nx)
        met = resumo(c_est, c_ini, c_verd, nome)
        resultados[nome] = (c_est, hist, met, dur, contador["n"])
        print(f"        J: {hist.J[0]:.4e} -> {hist.J[-1]:.4e}   "
              f"({100*hist.J[-1]/hist.J[0]:.1f}% do inicial)")
        print(f"        erro do modelo: {met['erro_inicial_%']:.2f}% -> "
              f"{met['erro_final_%']:.2f}%   (ganho {met['ganho_%']:+.1f}%)")
        print(f"        {contador['n']} avaliacoes, {dur:.1f} s")
    print()

    a.tabela(
        ["metodo", "J final / J inicial", "erro final (%)", "ganho (%)",
         "avaliacoes"],
        [[nome, f"{r[1].J[-1]/r[1].J[0]:.4f}", f"{r[2]['erro_final_%']:.2f}",
          f"{r[2]['ganho_%']:+.1f}", r[4]]
         for nome, r in resultados.items()])

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    plot.modelo(ax[0, 0], c_verd, dh, "modelo verdadeiro", "m/s",
                vmin=1950, vmax=2400)
    for k, (nome, r) in enumerate(resultados.items()):
        plot.modelo(ax[0, k + 1] if k < 2 else ax[1, 0], r[0], dh,
                    f"{nome}\nerro {r[2]['erro_final_%']:.2f}%", "m/s",
                    vmin=1950, vmax=2400)
    for nome, r in resultados.items():
        ax[1, 1].semilogy(r[1].J_normalizado, "o-", lw=1.5, ms=3.5, label=nome)
    ax[1, 1].set_xlabel("iteracao"); ax[1, 1].set_ylabel("J / J0")
    ax[1, 1].set_title("convergencia por iteracao")
    ax[1, 1].legend(fontsize=8)
    for nome, r in resultados.items():
        n_av = np.linspace(0, r[4], len(r[1].J))
        ax[1, 2].semilogy(n_av, r[1].J_normalizado, "o-", lw=1.5, ms=3.5,
                          label=nome)
    ax[1, 2].set_xlabel("avaliacoes de F (custo real)")
    ax[1, 2].set_ylabel("J / J0")
    ax[1, 2].set_title("convergencia por CUSTO")
    ax[1, 2].legend(fontsize=8)
    fig.tight_layout()
    plot.salvar(fig, a, "01_comparacao_otimizadores", mostrar=False)

    a.aviso("""
        Os dois ultimos paineis contam historias diferentes, e so o segundo
        interessa na pratica.

        Por ITERACAO, o l-BFGS quase sempre vence: cada iteracao dele usa
        informacao de curvatura acumulada.

        Por CUSTO -- numero de modelagens, que e o que consome o seu tempo de
        maquina -- a vantagem diminui, porque as iteracoes do l-BFGS custam
        mais avaliacoes na busca linear.

        Sempre que comparar otimizadores (inclusive num relatorio ou
        dissertacao), mostre o eixo de CUSTO. Comparar por iteracao favorece
        artificialmente os metodos de iteracao cara.
    """)

    # ==================================================================
    a.secao("Como escolher")
    a.tabela(
        ["Situacao", "Escolha", "Motivo"],
        [["Depurando o codigo", "descida maxima", "sem estado, falha de forma previsivel"],
         ["FWI de producao", "l-BFGS", "melhor convergencia por iteracao"],
         ["Memoria apertada", "grad. conjugado", "so guarda a direcao anterior"],
         ["Mudanca de frequencia", "reiniciar o l-BFGS", "a curvatura antiga nao vale mais"],
         ["Gradiente ruidoso (estocastico)", "descida maxima / CG", "l-BFGS se confunde com ruido"]])
    a.dica("""
        Dois detalhes que resolvem a maioria dos problemas praticos:

        1. REINICIE o l-BFGS ao trocar de banda de frequencia na multiescala.
           Os pares (s, y) guardados descrevem a curvatura do problema ANTIGO;
           mantidos, atrapalham em vez de ajudar.

        2. Se a busca linear falhar (alpha = 0), NAO insista com mais
           iteracoes. Isso quase sempre significa que a direcao nao e de
           descida -- gradiente errado, pre-condicionamento exagerado, ou o
           otimizador precisa reiniciar.
    """)

    a.pergunta(
        "Ao passar de 5 Hz para 10 Hz numa FWI multiescala com l-BFGS, "
        "qual o procedimento correto?",
        ["Continuar com o historico acumulado",
         "Reiniciar o l-BFGS (descartar os pares s,y)",
         "Trocar para descida maxima",
         "Reduzir o numero de iteracoes"],
        1,
        "Os pares (s,y) codificam a curvatura de J para a banda antiga. Ao "
        "mudar a frequencia, J muda -- fica mais estreito e com outros minimos. "
        "Reutilizar curvatura antiga produz direcoes ruins. Reinicie e deixe o "
        "l-BFGS reconstruir a aproximacao para a nova banda.")

    a.secao("Exercicios")
    a.exercicio(1, """
        Rode o l-BFGS com memoria 2, 5, 10 e 20 e compare a convergencia POR
        CUSTO. Existe retorno decrescente? A partir de que memoria?
    """, dica="parametro `memoria` em fwikit.inversao.lbfgs.")
    a.exercicio(2, """
        Troque a busca parabolica pela de Armijo puro
        (`busca_linha=busca_linear_armijo`) e compare o numero de avaliacoes
        para atingir o mesmo J. Quanto a busca melhor economiza?
    """)
    a.exercicio(3, """
        Implemente FWI ESTOCASTICA: a cada iteracao sorteie 2 dos 5 tiros para
        calcular o gradiente. Compare a convergencia por custo com a versao que
        usa todos os tiros. Qual compensa?
    """, dica="passe `tiros=` em gradiente_adjunto; o custo por iteracao cai.")

    a.fim([
        "Newton esta fora: a Hessiana tem N^2 entradas. Quase-Newton aproxima H^-1 "
        "com memoria 2mN.",
        "Busca linear domina o custo. Ajuste parabolico acerta em ~2 avaliacoes; "
        "retrocesso puro gasta 5-10.",
        "Passo inicial com significado fisico: nao mexa mais que 1-3% no modelo por vez.",
        "Compare otimizadores por CUSTO (modelagens), nunca so por iteracao.",
        "Reinicie o l-BFGS ao trocar de banda de frequencia.",
    ], proxima="aula12_fwi_multiescala.py -- a FWI completa, do inicio ao fim")


if __name__ == "__main__":
    main()
