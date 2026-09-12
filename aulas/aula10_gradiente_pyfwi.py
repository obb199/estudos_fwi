#!/usr/bin/env python3
"""
AULA 10 -- Gradiente no PyFWI, iluminacao e pre-condicionamento
===============================================================
Execute:  python aulas/aula10_gradiente_pyfwi.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                               # noqa: E402
import fwikit                                    # noqa: E402
from fwikit.aula import Aula                     # noqa: E402
from fwikit import plot                          # noqa: E402
from fwikit.inversao import precondicionar_profundidade, suavizar  # noqa: E402


def main():
    a = Aula(10, "Gradiente no PyFWI, iluminacao e pre-condicionamento",
             modulo="Modulo III -- O problema inverso e o gradiente",
             duracao="70 min", pre_requisitos="aulas 08 e 09",
             objetivos=[
                 "Calcular o gradiente com o PyFWI e entender cada argumento",
                 "Diagnosticar os tres artefatos classicos do gradiente bruto",
                 "Aplicar mascaramento, suavizacao e compensacao de iluminacao",
                 "Entender o que o pre-condicionamento aproxima (e o que ele nao e)",
             ])
    a.cabecalho()

    try:
        import PyFWI.acquisition as acq
        import PyFWI.model_dataset as md
        import PyFWI.wave_propagation as wave
    except Exception as exc:
        a.aviso(f"PyFWI indisponivel ({exc}). Rode a aula 00.")
        return

    # ==================================================================
    a.secao("A API de gradiente do PyFWI")
    a.texto("""
        O PyFWI calcula o gradiente em tres passos. O detalhe que costuma
        travar todo mundo: e preciso pedir checkpointing (`chpr > 0`) ANTES,
        na construcao do propagador -- senao os campos necessarios nao foram
        guardados e o gradiente nao pode ser montado.
    """)
    a.codigo("""
        # chpr > 0 e OBRIGATORIO para gradiente (0 serve so para modelagem)
        W = wave.WavePropagator(inpa, src, rec_loc, (nz, nx),
                                n_well_rec=0, chpr=20, components=0)

        d_est = W.forward_modeling(m0, show=False)       # 1. direto
        res   = {k: d_est[k] - d_obs[k] for k in d_est}  # 2. residuo
        g     = W.gradient(res, parameterization='dv')   # 3. adjunto
        # g e um dicionario com 'vp', 'vs', 'rho'
    """, titulo="o fluxo completo")
    a.tabela(
        ["argumento", "efeito"],
        [["chpr", "percentual de checkpoints; >0 obrigatorio para gradiente"],
         ["parameterization", "'dv' -> vp,vs,rho | outro -> lam,mu,rho"],
         ["inpa['g_smooth']", "sigma da suavizacao gaussiana aplicada ao gradiente"],
         ["inpa['energy_balancing']", "normaliza pela iluminacao (aproxima a Hessiana)"],
         ["inpa['grad_coeff']", "pesos relativos de vp, vs, rho"]])

    # ==================================================================
    a.secao("Calculando")
    inpa = {'ns': 7, 'sdo': 4, 'fdom': 20, 'dh': 7, 'dt': 0.0005, 't': 0.45,
            'npml': 20, 'pmlR': 1e-5, 'pml_dir': 2, 'acq_type': 1,
            'energy_balancing': False, 'g_smooth': 0, 'device': 0}
    M = md.ModelGenerator('louboutin')
    m_verd = M()
    m_ini = M(smoothing=1)
    for mm in (m_verd, m_ini):
        mm['vs'] = np.zeros_like(mm['vs'])          # regime acustico (aula 05)
    nz, nx = m_verd['vp'].shape
    dh = inpa['dh']

    src_loc, rec_loc = acq.surface_seismic(inpa['ns'], dh * 2, dh * nx, dh,
                                           inpa['sdo'])
    src = acq.Source(src_loc, dh, inpa['dt'])
    src.Ricker(inpa['fdom'])

    W = wave.WavePropagator(inpa, src, rec_loc, (nz, nx), n_well_rec=0,
                            chpr=20, components=0)
    print()
    print("    dado observado (modelo verdadeiro)...", end="", flush=True)
    d_obs = W.forward_modeling(m_verd, show=False)
    print(" ok")
    print("    dado calculado (modelo inicial).....", end="", flush=True)
    d_est = W.forward_modeling(m_ini, show=False)
    print(" ok")
    res = {k: d_est[k] - d_obs[k] for k in d_est}
    J = 0.5 * float(sum(np.sum(v ** 2) for v in res.values()))
    print("    gradiente adjunto...................", end="", flush=True)
    t0 = time.time()
    g = W.gradient(res, show=False, parameterization='dv')
    dur = time.time() - t0
    print(f" {dur:.2f} s")
    print()
    a.resultado("J inicial", f"{J:.6e}")
    a.resultado("tempo do gradiente (%d tiros)" % inpa['ns'], f"{dur:.2f}", "s")
    a.resultado("chaves devolvidas", ", ".join(g.keys()))
    for k in g:
        a.resultado(f"|g_{k}| maximo", f"{np.abs(g[k]).max():.3e}")
    print()
    a.texto("""
        O gradiente vem sempre com as tres componentes (vp, vs, rho), mesmo em
        regime acustico. Repare que |g_vs| deu exatamente ZERO: como zeramos vs
        no modelo, mu = 0 e o termo de cisalhamento desaparece identicamente do
        adjunto -- consistente com a prova da aula 05.

        Ja g_rho NAO e zero. Se voce invertesse as tres componentes juntas,
        estaria fazendo inversao multiparametro, com crosstalk entre vp e rho.
        Em FWI acustica de velocidade voce restringe a atualizacao a vp -- e o
        que os parametros k_0 = 1 e k_end = 2 fazem no PyFWI (aula 12).
    """)

    g_vp = g['vp']
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    plot.modelo(ax[0, 0], m_verd['vp'], dh, "vp verdadeiro", "m/s")
    plot.modelo(ax[0, 1], m_ini['vp'], dh, "vp inicial", "m/s")
    plot.perturbacao(ax[0, 2], m_verd['vp'] - m_ini['vp'], dh,
                     "o que falta encontrar", "m/s")
    plot.perturbacao(ax[1, 0], g_vp, dh, "gradiente BRUTO (vp)")
    ax[1, 0].plot(src_loc[:, 0], src_loc[:, 1], "k*", ms=9)

    # ==================================================================
    a.secao("Os tres artefatos do gradiente bruto")
    a.texto("""
        Olhe o painel do gradiente bruto. Ele quase nunca e usado como esta, e
        por tres razoes bem identificadas:
    """)
    perfil = np.abs(g_vp).mean(axis=1)
    raso = float(perfil[:nz // 6].mean())
    fundo = float(perfil[-nz // 3:].mean())
    print()
    a.resultado("amplitude media na faixa rasa", f"{raso:.3e}")
    a.resultado("amplitude media na faixa profunda", f"{fundo:.3e}")
    a.resultado("razao raso / profundo", f"{raso/(fundo+1e-30):.1f}", "x")
    print()
    a.lista([
        "SINGULARIDADE DE FONTE E RECEPTOR. O campo tem amplitude enorme junto "
        "aos pontos de fonte e receptor. A correlacao explode ali e domina o "
        "gradiente inteiro. Nao e informacao sobre o modelo: e geometria.",
        "DESEQUILIBRIO DE ILUMINACAO. A amplitude decai com o espalhamento "
        "geometrico, entao o gradiente e sistematicamente mais fraco em "
        "profundidade -- veja a razao medida acima. Sem correcao, o passo da "
        "otimizacao e gasto quase todo na parte rasa.",
        "CONTEUDO DE ALTA FREQUENCIA. O gradiente bruto carrega oscilacoes na "
        "escala do comprimento de onda. Parte e informacao legitima; parte e "
        "ruido de amostragem de tiros e de borda.",
    ])

    a.teoria("O que o pre-condicionamento realmente e", """
        A atualizacao ideal de Newton seria

            dm = - H^(-1) g

        com H a Hessiana. Em FWI ela e grande demais para ser montada. O
        PRE-CONDICIONADOR P e uma aproximacao barata de H^(-1):

            dm = - P g

        A Hessiana de Gauss-Newton tem na diagonal exatamente a ENERGIA DO
        CAMPO em cada ponto -- que e a iluminacao. Por isso dividir o gradiente
        pela energia do campo direto nao e um truque cosmetico: e uma
        aproximacao diagonal legitima de H^(-1). E o que o PyFWI chama de
        `energy_balancing`.

        O que o pre-condicionamento NAO faz: nao corrige cycle skipping, nao
        inventa informacao onde nao houve iluminacao, e nao substitui um bom
        modelo inicial. Ele so redistribui o passo para as regioes que os dados
        realmente enxergam.
    """)

    # ==================================================================
    a.secao("Aplicando os remedios")
    g_mask = g_vp.copy()
    z_mute = max(3, int(0.16 * nz))   # ~1 comprimento de onda
    g_mask[:z_mute, :] = 0.0                       # mascara a superficie
    g_suave = suavizar(g_mask, 2.0)
    g_prec = precondicionar_profundidade(g_suave, potencia=1.0)

    plot.perturbacao(ax[1, 1], g_suave, dh,
                     f"mascarado (z<{z_mute}) + suavizado")
    plot.perturbacao(ax[1, 2], g_prec, dh, "+ compensacao de profundidade")
    fig.tight_layout()
    plot.salvar(fig, a, "01_gradiente_pyfwi", mostrar=False)

    def correl(x, y):
        x = x - x.mean(); y = y - y.mean()
        return float(np.sum(x * y) / (np.linalg.norm(x) * np.linalg.norm(y)))

    alvo = m_verd['vp'] - m_ini['vp']
    print()
    print(f"    {'versao do gradiente':<38}{'correlacao com o alvo'}")
    print(f"    {'-'*38}{'-'*22}")
    for nome, gg in [("bruto", g_vp), ("mascarado", g_mask),
                     ("mascarado + suavizado", g_suave),
                     ("+ compensacao de profundidade", g_prec)]:
        print(f"    {nome:<38}{correl(-gg, alvo):>+10.4f}")
    print()
    a.texto("""
        Duas leituras importantes, e a segunda costuma ser mal compreendida.

        PRIMEIRA: o sinal importa. A atualizacao e m <- m - alpha*g, entao
        quem tem de se parecer com o alvo e -g, nao g.

        SEGUNDA: os valores ABSOLUTOS de correlacao sao baixos -- da ordem de
        0.05 a 0.09 -- e isso esta CERTO. Um gradiente nao e o modelo. Ele e
        uma direcao de busca: espalhada nos arcos de Fresnel, contaminada por
        iluminacao desigual, e comparada aqui com uma anomalia minuscula (uns
        300 pixels em 10000). Esperar correlacao alta seria esperar que a FWI
        terminasse numa iteracao.

        O que importa e a variacao RELATIVA: o tratamento mais que dobra a
        correlacao. E esse ganho, repetido a cada iteracao, que muda a
        velocidade de convergencia da inversao inteira.
    """)

    # ==================================================================
    a.secao("energy_balancing: deixando o PyFWI fazer")
    a.texto("""
        O PyFWI implementa a compensacao de iluminacao internamente. Compare o
        gradiente com e sem, mudando uma unica chave:
    """)
    inpa_eb = dict(inpa, energy_balancing=True, g_smooth=2)
    W2 = wave.WavePropagator(inpa_eb, src, rec_loc, (nz, nx), n_well_rec=0,
                             chpr=20, components=0)
    W2.forward_modeling(m_ini, show=False)
    g2 = W2.gradient(res, show=False, parameterization='dv')['vp']
    g2_mask = g2.copy()
    g2_mask[:z_mute, :] = 0.0
    print()
    c_manual = correl(-g_prec, alvo)
    c_pyfwi = correl(-g2_mask, alvo)
    c_bruto = correl(-g_vp, alvo)
    a.resultado("correlacao -- gradiente bruto", f"{c_bruto:+.4f}")
    a.resultado("correlacao -- manual (mascara+suav.+prof.)", f"{c_manual:+.4f}")
    a.resultado("correlacao -- energy_balancing + g_smooth", f"{c_pyfwi:+.4f}")
    a.resultado("ganho do melhor tratamento",
                f"{max(c_manual, c_pyfwi)/max(c_bruto, 1e-9):.1f}", "x")
    print()

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    plot.perturbacao(ax[0], g_vp, dh, "bruto")
    plot.perturbacao(ax[1], g_prec, dh, "tratado manualmente")
    plot.perturbacao(ax[2], g2_mask, dh, "energy_balancing + g_smooth (PyFWI)")
    fig.tight_layout()
    plot.salvar(fig, a, "02_precondicionamento", mostrar=False)

    a.dica("""
        Receita pratica de pre-condicionamento, na ordem em que importa:

        1. MASCARE a faixa de fonte/receptor. E o de maior efeito e o mais
           barato. Zere as primeiras celulas em profundidade.

        2. COMPENSE A ILUMINACAO. `energy_balancing=True` no PyFWI, ou divida
           pela energia do campo direto no seu proprio codigo.

        3. SUAVIZE. `g_smooth` entre 1 e 3 celulas. Comece pela frequencia mais
           baixa e reduza a suavizacao conforme sobe em frequencia.

        4. LIMITE o modelo a uma caixa fisica (vmin, vmax). Nao e
           pre-condicionamento, mas evita que um passo ruim leve o modelo para
           valores absurdos.
    """)

    a.pergunta(
        "Seu gradiente so atualiza os primeiros 200 m do modelo. "
        "Qual a causa mais provavel?",
        ["Cycle skipping",
         "Falta de pre-condicionamento: a singularidade de fonte e o "
         "desequilibrio de iluminacao dominam o gradiente",
         "Modelo inicial ruim",
         "A malha e grossa demais"],
        1,
        "Amplitude enorme perto das fontes + decaimento geometrico = gradiente "
        "concentrado no topo. A busca linear entao escolhe um passo pequeno "
        "(limitado pela regiao rasa) e a parte profunda quase nao se move. "
        "Mascarar a superficie e compensar iluminacao resolve. Cycle skipping "
        "daria modelo errado, nao modelo parado.")

    a.secao("Exercicios")
    a.exercicio(1, """
        Varie `g_smooth` de 0 a 6 e meca a correlacao do gradiente com a
        perturbacao verdadeira. Existe um otimo? Ele depende da frequencia?
    """)
    a.exercicio(2, """
        Implemente o pre-condicionador de iluminacao no seu proprio codigo:
        calcule E(x) = integral de p(x,t)^2 dt no campo direto e divida o
        gradiente por (E + eps*max(E)). Compare com energy_balancing do PyFWI.
    """, dica="guarde o campo p com guardar='p' em fwikit.acustico._propagar.")
    a.exercicio(3, """
        Compare o gradiente do PyFWI com o do fwikit para o MESMO modelo e
        geometria (normalize os dois). Eles devem ter a mesma forma. Diferencas
        de escala sao esperadas; diferencas de FORMA nao.
    """, dica="use a correlacao, nao a diferenca absoluta.")

    a.fim([
        "No PyFWI: chpr>0 e obrigatorio para gradiente; a saida e dict com vp, vs, rho.",
        "Gradiente bruto tem tres artefatos: singularidade de fonte/receptor, "
        "desequilibrio de iluminacao e alta frequencia.",
        "Pre-condicionar = aproximar H^(-1). A diagonal da Hessiana de Gauss-Newton "
        "e a energia do campo -- por isso compensar iluminacao e legitimo.",
        "Ordem de importancia: mascarar superficie > compensar iluminacao > "
        "suavizar > limitar em caixa.",
        "Pre-condicionamento nao conserta cycle skipping nem cria informacao onde "
        "nao houve iluminacao.",
    ], proxima="aula11_otimizacao.py -- descida maxima, gradiente conjugado e l-BFGS")


if __name__ == "__main__":
    main()
