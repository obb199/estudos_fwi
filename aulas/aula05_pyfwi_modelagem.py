#!/usr/bin/env python3
"""
AULA 05 -- PyFWI na pratica e a prova numerica do regime acustico
==================================================================
Execute:  python aulas/aula05_pyfwi_modelagem.py
Requer:   OpenCL funcionando (ver aula 00)
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                               # noqa: E402
import fwikit                                    # noqa: E402
from fwikit.aula import Aula                     # noqa: E402
from fwikit import plot                          # noqa: E402


def main():
    a = Aula(5, "PyFWI na pratica e a prova numerica do regime acustico",
             modulo="Modulo II -- PyFWI como ferramenta de producao",
             duracao="70 min", pre_requisitos="aulas 01-04",
             objetivos=[
                 "Montar modelo, aquisicao e propagador no PyFWI",
                 "Entender que o nucleo do PyFWI e ELASTICO e que o caso acustico "
                 "e o limite vs -> 0",
                 "COMPROVAR numericamente que com vs = 0 o tensor de tensoes fica "
                 "isotropico e p = -(sigma_xx + sigma_zz)/2",
                 "Escolher `components` com consciencia do que cada valor devolve",
             ])
    a.cabecalho()

    try:
        import PyFWI.acquisition as acq
        import PyFWI.model_dataset as md
        import PyFWI.wave_propagation as wave
    except Exception as exc:
        a.aviso(f"""
            Nao consegui importar o PyFWI ({exc}).
            Rode a aula 00 para diagnosticar. As aulas do Modulo I e III nao
            dependem do PyFWI e continuam funcionando.
        """)
        return

    # ==================================================================
    a.secao("A anatomia de um script PyFWI")
    a.texto("""
        Todo script de PyFWI tem a mesma estrutura de quatro blocos. Vale
        memorizar, porque a documentacao do pacote e escassa e voce vai montar
        isso de cabeca muitas vezes.
    """)
    a.codigo("""
        # 1. PARAMETROS -- um dicionario unico chamado `inpa`
        inpa = {'ns':1, 'sdo':4, 'fdom':20, 'dh':7, 'dt':0.0005, 't':0.5,
                'npml':20, 'pmlR':1e-5, 'pml_dir':2, 'acq_type':1, 'device':0}

        # 2. MODELO -- dicionario com 'vp', 'vs', 'rho'
        model = md.ModelGenerator('louboutin')()

        # 3. AQUISICAO -- posicoes em METROS, depois a fonte
        src_loc, rec_loc = acq.surface_seismic(ns, rec_dis, offsetx, dh, sdo)
        src = acq.Source(src_loc, dh, dt);  src.Ricker(fdom)

        # 4. PROPAGADOR
        W = wave.WavePropagator(inpa, src, rec_loc, (nz,nx),
                                n_well_rec=0, chpr=0, components=0)
        d = W.forward_modeling(model, show=False)
    """, titulo="o esqueleto que voce vai repetir sempre")
    a.aviso("""
        Tres armadilhas do PyFWI que custam tempo:

        1. As posicoes de fonte e receptor sao dadas em METROS, nao em indices
           de malha. A classe `Source` divide por `dh` internamente.

        2. `inpa['sdo']` e a ordem espacial REAL (4 ou 8), mas internamente o
           codigo faz `self.sdo = sdo/2`. Se voce ler o fonte e se confundir com
           esse fator 2, e por isso.

        3. `inpa['acq_type']` TEM de ser 1 (superficie) ou 2 (crosswell).
           Qualquer outro valor -- 0, por exemplo -- nao levanta erro: o codigo
           simplesmente mapeia os receptores errado, e voce recebe um
           sismograma plausivel e ERRADO. Veja a secao de validacao cruzada no
           fim desta aula: foi exatamente assim que esse problema apareceu.
    """)

    # ==================================================================
    a.secao("Montando a simulacao")
    inpa = {'ns': 1, 'sdo': 4, 'fdom': 20, 'dh': 7, 'dt': 0.0005, 't': 0.5,
            'npml': 20, 'pmlR': 1e-5, 'pml_dir': 2, 'acq_type': 1,
            'energy_balancing': False, 'device': 0}

    modelo = md.ModelGenerator('louboutin')()
    nz, nx = modelo['vp'].shape
    dh = inpa['dh']

    print()
    a.resultado("modelo", f"{nz} x {nx}", "pontos")
    a.resultado("dominio", f"{nx*dh/1000:.2f} x {nz*dh/1000:.2f}", "km")
    for k in ("vp", "vs", "rho"):
        a.resultado(f"{k}: min / max",
                    f"{modelo[k].min():.1f} / {modelo[k].max():.1f}")
    a.resultado("razao vp/vs", f"{(modelo['vp']/modelo['vs']).mean():.4f}")
    print()
    a.texto("""
        Repare na razao vp/vs = 1.732 = sqrt(3). Nao e coincidencia: o PyFWI
        gerou vs a partir de vp assumindo coeficiente de Poisson 0.25, o valor
        classico para rocha consolidada. Esse e o modelo `louboutin`, uma
        inclusao circular rapida num meio homogeneo -- o "hello world" da FWI.

        Note tambem que rho esta em g/cm^3 (~2.2), nao em kg/m^3. O PyFWI e
        consistente internamente, mas se voce misturar unidades ao trazer um
        modelo de fora, a impedancia sai errada e as amplitudes tambem.
    """)

    src_loc, rec_loc = acq.surface_seismic(inpa['ns'], dh * 2, dh * nx, dh,
                                           inpa['sdo'])
    src = acq.Source(src_loc, dh, inpa['dt'])
    src.Ricker(inpa['fdom'])
    a.resultado("fonte em (x, z)", f"{src_loc[0,0]:.1f}, {src_loc[0,1]:.1f}", "m")
    a.resultado("receptores", len(rec_loc))
    a.resultado("wavelet", f"{src.w.size}", "amostras")

    # ==================================================================
    a.secao("O que `components` devolve")
    a.texto("""
        O parametro `components` escolhe QUAL campo o PyFWI grava nos
        receptores. Isso e uma decisao fisica, nao cosmetica:
    """)
    a.tabela(["components", "campos devolvidos", "interpretacao"],
             [["0", "taux, tauz (ambos = media)", "PRESSAO: -(sxx+szz)/2"],
              ["1", "taux", "so a tensao normal em x"],
              ["2", "vx, vz", "velocidade de particula (geofone)"],
              ["3", "taux, tauz, tauxz", "tensor de tensoes completo"],
              ["4", "vx, vz, taux, tauz, tauxz", "tudo"]])
    a.dica("""
        Use components=0 para trabalho acustico (hidrofone mede pressao).
        Use components=2 para imitar geofone (mede velocidade de particula).
        Use components=3 quando quiser INSPECIONAR a fisica -- e o que vamos
        fazer agora.
    """)

    # ==================================================================
    a.secao("A PROVA: o regime acustico e o limite vs -> 0")
    a.texto("""
        Na aula 01 afirmamos que a aproximacao acustica e uma unica hipotese,
        mu = 0, e que dela decorre que o tensor de tensoes fica isotropico:
        sigma_xz = 0 e sigma_xx = sigma_zz. Agora vamos COMPROVAR isso no
        proprio PyFWI, que resolve o sistema elastico completo.

        O experimento: rodar a mesma simulacao duas vezes, mudando so vs.
    """)

    def roda(mod, comps):
        W = wave.WavePropagator(inpa, src, rec_loc, (nz, nx),
                                n_well_rec=0, chpr=0, components=comps)
        return W.forward_modeling(mod, show=False)

    print()
    print("    rodando caso ELASTICO (vs = vp/sqrt(3))...", end="", flush=True)
    t0 = time.time()
    d_el = roda(modelo, 3)
    print(f" {time.time()-t0:.2f} s")

    modelo_ac = {k: v.copy() for k, v in modelo.items()}
    modelo_ac['vs'] = np.zeros_like(modelo['vs'])
    print("    rodando caso ACUSTICO (vs = 0)........", end="", flush=True)
    t0 = time.time()
    d_ac = roda(modelo_ac, 3)
    print(f" {time.time()-t0:.2f} s")

    def nrm(x):
        return float(np.linalg.norm(x))

    print()
    print(f"    {'caso':<12}{'||sxx - szz|| / ||sxx||':<28}{'||sxz|| / ||sxx||'}")
    print(f"    {'-'*12}{'-'*28}{'-'*20}")
    linhas_prova = []
    for nome, d in [("elastico", d_el), ("acustico", d_ac)]:
        dif = nrm(d['taux'] - d['tauz']) / nrm(d['taux'])
        xz = nrm(d['tauxz']) / nrm(d['taux'])
        print(f"    {nome:<12}{dif:<28.4e}{xz:.4e}")
        linhas_prova.append((nome, dif, xz))
    print()

    a.teoria("Leia estes numeros com atencao", """
        No caso ELASTICO:
            sigma_xx e sigma_zz diferem em ~50% -- o tensor NAO e isotropico.
            sigma_xz vale ~35% de sigma_xx -- existe cisalhamento de verdade.

        No caso ACUSTICO (vs = 0):
            sigma_xx - sigma_zz cai para ~1e-7, ou seja, ZERO ate a precisao
            de maquina (float32). O tensor ficou isotropico.
            sigma_xz da EXATAMENTE 0.0 -- nao aproximadamente: exatamente,
            porque mu = 0 multiplica o termo inteiro.

        Isso e a demonstracao numerica do que derivamos na aula 01. Quando o
        tensor e isotropico, um unico escalar o descreve, e esse escalar e a
        pressao:

            p = - (sigma_xx + sigma_zz) / 2 = - sigma_xx = - sigma_zz

        E exatamente isso que o PyFWI devolve quando voce pede components=0.
        Ou seja: o PyFWI nao tem um solver acustico separado. Ele tem um solver
        ELASTICO, e voce obtem o regime acustico zerando vs.
    """)

    dif_dado = nrm(d_el['taux'] - d_ac['taux']) / nrm(d_el['taux'])
    a.resultado("diferenca relativa elastico x acustico no dado",
                f"{100*dif_dado:.1f}", "%")
    print()
    a.texto(f"""
        Leia esse numero como ele merece: a diferenca entre os dois sismogramas
        tem norma de {100*dif_dado:.0f}% da norma do proprio dado elastico. Nao
        e um detalhe de amplitude -- sao as ondas S e as conversoes P-S geradas
        na inclusao, eventos inteiros que simplesmente nao existem no mundo
        acustico.

        A consequencia para a inversao e direta: se o dado observado contem
        esses eventos e o operador de modelagem e acustico, esse residuo inteiro
        entra no gradiente e sai do outro lado como estrutura de velocidade
        FALSA. A FWI nao tem como saber que aquilo era onda S; ela so sabe
        minimizar o residuo mexendo em vp.
    """)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    plot.modelo(ax[0, 0], modelo['vp'], dh, "vp", "m/s")
    plot.modelo(ax[0, 1], modelo['vs'], dh, "vs (caso elastico)", "m/s",
                cmap="magma")
    plot.modelo(ax[0, 2], modelo['rho'], dh, "rho", "g/cm3", cmap="cividis")
    dt = inpa['dt']
    plot.sismograma(ax[1, 0], d_el['taux'], dt, "elastico: sigma_xx")
    plot.sismograma(ax[1, 1], d_ac['taux'], dt, "acustico (vs=0): sigma_xx")
    plot.sismograma(ax[1, 2], d_el['taux'] - d_ac['taux'], dt,
                    "diferenca = ondas S + conversoes")
    fig.tight_layout()
    plot.salvar(fig, a, "01_prova_regime_acustico", mostrar=False)

    a.pergunta(
        "No PyFWI, qual e a maneira correta de rodar uma modelagem ACUSTICA?",
        ["Existe uma classe AcousticPropagator separada",
         "Passar components=0 e pronto",
         "Zerar vs no modelo (e usar components=0 para ler pressao)",
         "Definir inpa['acq_type']=0"],
        2,
        "components=0 so escolhe o que e GRAVADO -- a fisica simulada continua "
        "elastica se vs != 0. O regime acustico exige vs = 0 no MODELO. "
        "Confundir as duas coisas e um erro classico: voce acha que esta "
        "rodando acustico e esta gerando e registrando ondas S."
    )

    a.pausa()

    # ==================================================================
    a.secao("Confronto: PyFWI acustico x nosso propagador")
    a.texto("""
        Duas implementacoes independentes que resolvem a mesma fisica devem
        concordar na CINEMATICA. Amplitudes podem diferir (o PyFWI tem densidade
        variavel e uma normalizacao de fonte propria; o nosso e de densidade
        constante), mas os tempos de chegada tem que bater. Esse tipo de
        confronto cruzado e a forma mais barata de achar um erro grosseiro.
    """)
    from fwikit.acustico import (Acustico2D, Config, Geometria,
                                 ricker as ricker_fwikit)

    d_py = roda(modelo_ac, 0)['taux']
    cfg = Config(dh=dh, dt=dt, nt=d_py.shape[0], ordem=inpa['sdo'],
                 n_abs=inpa['npml'], f0=inpa['fdom'])
    geom = Geometria(
        fontes=np.array([[int(src_loc[0, 1] / dh), int(src_loc[0, 0] / dh)]]),
        receptores=np.array([[int(r[1] / dh), int(r[0] / dh)] for r in rec_loc]))
    solver = Acustico2D(cfg, (nz, nx))
    w = ricker_fwikit(inpa['fdom'], dt, cfg.nt)
    d_meu, _ = solver.modelar(modelo['vp'], geom, w)

    def primeira_chegada(d, dt, frac=0.15):
        """Tempo em que |traco| cruza `frac` do seu proprio maximo."""
        t = np.zeros(d.shape[1])
        for k in range(d.shape[1]):
            amp = np.abs(d[:, k])
            lim = frac * amp.max()
            idx = np.argmax(amp > lim) if (amp > lim).any() else 0
            t[k] = idx * dt
        return t

    fb_py = primeira_chegada(d_py, dt)
    fb_meu = primeira_chegada(d_meu, dt)
    k_apice_py = int(np.argmin(fb_py))
    k_apice_meu = int(np.argmin(fb_meu))
    x_fonte = float(src_loc[0, 0])
    erro_fb = float(np.sqrt(np.mean((fb_py - fb_meu) ** 2)))

    print()
    a.resultado("apice da moveout -- PyFWI",
                f"receptor {k_apice_py} (x = {rec_loc[k_apice_py,0]:.0f} m)")
    a.resultado("apice da moveout -- fwikit",
                f"receptor {k_apice_meu} (x = {rec_loc[k_apice_meu,0]:.0f} m)")
    a.resultado("posicao real da fonte", f"{x_fonte:.0f}", "m")
    a.resultado("erro rms das primeiras chegadas",
                f"{erro_fb*1000:.1f}", "ms")
    print()
    ok_apice = abs(rec_loc[k_apice_py, 0] - x_fonte) <= 2 * dh * 2
    ok_fb = erro_fb * 1000 < 25.0
    if ok_apice and ok_fb:
        print("    -> CINEMATICA CONSISTENTE entre os dois codigos.")
        print("       (o apice cai sobre a fonte nos dois, e as curvas de")
        print("        primeira chegada concordam dentro da tolerancia)")
    else:
        print("    -> DIVERGENCIA. Confira acq_type, dh, dt, ordem e posicoes.")
    print()

    a.dica("""
        Escolhemos comparar PRIMEIRA CHEGADA, e nao a forma do traco, de
        proposito. Os dois codigos resolvem discretizacoes diferentes da mesma
        fisica: o PyFWI e velocity-stress em malha intercalada, densidade
        variavel e CPML; o fwikit e pressao em malha colocalizada, densidade
        constante e Cerjan. Amplitude e fase FINA vao diferir, e isso nao e erro.

        A cinematica, nao. Tempo de transito e propriedade da fisica, nao da
        discretizacao. Se os apices da moveout nao coincidem, ou se as curvas de
        primeira chegada se separam, existe erro de configuracao -- e e ai que
        voce deve olhar primeiro.
    """)
    # --- por que as FORMAS diferem: convencao de fonte -------------------
    a.secao("Por que as formas diferem (e nao e erro de nenhum dos dois)")
    a.texto("""
        A cinematica bate, mas se voce sobrepuser os tracos vai ver que a FORMA
        do pulso difere. Isso tem uma explicacao exata, e ela vem da aula 01.

        O PyFWI resolve o sistema de PRIMEIRA ordem (velocidade-tensao); o
        fwikit resolve a forma de SEGUNDA ordem em pressao. Ao eliminar a
        velocidade de particula para chegar na segunda ordem, o termo-fonte vira
        (1/K) ds/dt. Ou seja: alimentadas com a MESMA Ricker, as duas
        formulacoes tem wavelets EFETIVAS que diferem por uma derivada temporal.

        Da para verificar isso numericamente.
    """)

    def normaliza(x):
        return x / (np.abs(x).max() + 1e-30)

    def correl_traco(a_, b_):
        a_ = a_ - a_.mean(); b_ = b_ - b_.mean()
        return float(np.dot(a_, b_) /
                     (np.linalg.norm(a_) * np.linalg.norm(b_) + 1e-30))

    cs_bruto, cs_deriv = [], []
    for j in range(d_py.shape[1]):
        tp, tm = normaliza(d_py[:, j]), normaliza(d_meu[:, j])
        cs_bruto.append(correl_traco(tp, tm))
        cs_deriv.append(correl_traco(tp, normaliza(np.gradient(tm, dt))))
    print()
    a.resultado("correlacao media, tracos como saem",
                f"{np.mean(cs_bruto):+.4f}")
    a.resultado("correlacao media, derivando o traco do fwikit",
                f"{np.mean(cs_deriv):+.4f}")
    print()
    a.teoria("O que esse salto significa", f"""
        A correlacao sai de {np.mean(cs_bruto):+.2f} para {np.mean(cs_deriv):+.2f}
        aplicando UMA derivada temporal. Nao ha ajuste, nao ha parametro livre:
        e exatamente a relacao prevista entre as duas formulacoes.

        Isso encerra a comparacao de forma satisfatoria. Os dois codigos estao
        certos; o que diferia era a CONVENCAO DE FONTE, nao a fisica.

        Guarde a licao: ao comparar dois codigos de onda, verifique a convencao
        de fonte antes de suspeitar da fisica. As perguntas certas sao: a
        formulacao e de primeira ou de segunda ordem? A fonte entra na pressao,
        na tensao ou na velocidade? Ha divisao por dh^2 (discretizacao da
        delta)? Cada uma dessas escolhas muda a wavelet efetiva, e nenhuma
        delas aparece no sismograma com uma etiqueta.
    """)

    a.teoria("Historia real: como esta comparacao achou um bug", """
        Quando este curso foi montado, esta mesma comparacao falhou: os dois
        codigos discordavam em 159 ms, e o gather do PyFWI nao tinha apice --
        a curva subia monotonicamente do receptor 0 ao 45, como se a fonte
        estivesse fora do arranjo.

        A causa era `inpa['acq_type'] = 0`. O PyFWI aceita esse valor sem
        reclamar, mas so trata 1 (superficie) e 2 (crosswell); com 0 ele
        monta o mapeamento de receptores por outro caminho e devolve um
        sismograma que PARECE razoavel -- ate voce reparar que nao tem apice.

        Duas licoes, e a segunda vale mais que a primeira:

        1. Em aquisicao de superficie, `acq_type` = 1. Sempre.

        2. Um dado sintetico errado quase nunca vem com aviso. Ele vem bonito.
           A unica defesa barata e comparar com uma implementacao independente
           e checar se a FISICA elementar aparece -- neste caso, se o apice da
           moveout esta sobre a fonte. Faca isso ANTES de rodar qualquer
           inversao, nao depois de passar uma semana achando que o gradiente
           esta errado.
    """)

    # ==================================================================
    a.secao("Modelos disponiveis no PyFWI")
    a.texto("""
        O `model_dataset` traz varios modelos sinteticos classicos, todos com a
        mesma interface. Alguns baixam dados da internet na primeira chamada.
    """)
    a.tabela(["nome", "o que e", "tamanho"],
             [["louboutin", "inclusao circular em meio homogeneo", "100 x 100"],
              ["yang", "tres inclusoes (vp, vs, rho separadas)", "100 x 100"],
              ["hu_circles", "circulos concentricos", "100 x 100"],
              ["perturbation_dv", "perturbacao suave de velocidade", "100 x 100"],
              ["hu_laminar / dupuy", "camadas", "variavel"],
              ["marmousi", "Marmousi-2 elastico (baixa da internet)", "grande"]])
    a.codigo("""
        M = md.ModelGenerator('louboutin')
        modelo   = M()               # modelo verdadeiro
        inicial  = M(smoothing=1)    # versao suavizada -> modelo INICIAL da FWI
        monitor  = M(vintage=2)      # versao "monitor" para time-lapse
    """, titulo="gerando verdadeiro e inicial de uma vez")
    a.aviso("""
        `M(smoothing=1)` e a maneira padrao de obter o modelo INICIAL nos
        exemplos do PyFWI. Cuidado: para o modelo `louboutin` ele devolve o
        meio homogeneo puro (sem a inclusao), que e um inicial muito otimista.
        Em dado real voce nunca tem isso. Voltamos a esse ponto na aula 07,
        porque a qualidade do modelo inicial e o que mais determina se a FWI
        converge.
    """)

    a.secao("Exercicios")
    a.exercicio(1, """
        Rode a mesma simulacao com components=2 (vx, vz) e compare com
        components=0. Por que o sismograma de velocidade de particula parece a
        derivada temporal do de pressao?
    """, dica="olhe as equacoes de 1a ordem da aula 01.")
    a.exercicio(2, """
        Varie vs de 0 ate vp/sqrt(3) em cinco passos e meca, a cada passo,
        ||sxx - szz||/||sxx||. Faca o grafico. A transicao e suave ou abrupta?
    """, dica="reaproveite a funcao roda() desta aula.")
    a.exercicio(3, """
        Repita a comparacao PyFWI x fwikit com sdo=8 nos dois codigos e dh maior.
        A concordancia melhora ou piora? Relacione com a aula 02.
    """)

    a.fim([
        "Todo script PyFWI tem 4 blocos: inpa, modelo, aquisicao, propagador.",
        "Posicoes em METROS; inpa['sdo'] e a ordem espacial real.",
        "O nucleo do PyFWI e ELASTICO. Acustico = vs zerado no MODELO, "
        "nao apenas components=0.",
        "Comprovado: com vs=0, sxz = 0 exatamente e sxx = szz ate precisao de "
        "maquina. Pressao = -(sxx+szz)/2 = o que components=0 devolve.",
        "Sempre confronte um codigo novo com outro independente: amplitudes podem "
        "diferir, tempos de chegada nao.",
    ], proxima="aula06_pyfwi_parametros.py -- inpa, PML, checkpointing e custo")


if __name__ == "__main__":
    main()
