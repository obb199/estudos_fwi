#!/usr/bin/env python3
"""
AULA 06 -- PyFWI por dentro: inpa, PML, checkpointing e custo
==============================================================
Execute:  python aulas/aula06_pyfwi_parametros.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                               # noqa: E402
import fwikit                                    # noqa: E402
from fwikit.aula import Aula                     # noqa: E402


def main():
    a = Aula(6, "PyFWI por dentro: inpa, PML, checkpointing e custo",
             modulo="Modulo II -- PyFWI como ferramenta de producao",
             duracao="60 min", pre_requisitos="aula 05",
             objetivos=[
                 "Conhecer todas as chaves do dicionario `inpa` e o que cada uma faz",
                 "Dimensionar a CPML e medir o efeito de npml na reflexao de borda",
                 "Entender o reescalonamento automatico de dt do PyFWI",
                 "Entender checkpointing (chpr): o compromisso memoria x recomputacao",
                 "Medir como o custo escala e planejar um experimento que caiba no seu tempo",
             ])
    a.cabecalho()

    try:
        import PyFWI.acquisition as acq
        import PyFWI.wave_propagation as wave
    except Exception as exc:
        a.aviso(f"PyFWI indisponivel ({exc}). Rode a aula 00.")
        return

    # ==================================================================
    a.secao("Referencia completa do dicionario `inpa`")
    a.texto("""
        O PyFWI concentra quase toda a configuracao num dicionario. A
        documentacao e escassa, entao esta tabela foi levantada lendo o
        codigo-fonte (`wave_propagation.py` e `fwi.py`). Guarde-a.
    """)
    a.tabela(
        ["chave", "obrigatoria", "o que faz"],
        [["dh", "sim", "espacamento da malha (m)"],
         ["dt", "sim", "passo de tempo pedido (veja o reescalonamento abaixo)"],
         ["t", "sim", "duracao do registro (s)"],
         ["fdom", "sim*", "frequencia dominante (usada por voce, na Source)"],
         ["sdo", "nao (=4)", "ordem espacial REAL: 4 ou 8 (omitir equivale a 4)"],
         ["npml", "nao (0)", "espessura da CPML em pontos"],
         ["pmlR", "com npml", "coef. de reflexao teorico da CPML (ex. 1e-5)"],
         ["pml_dir", "com npml", "0 = so z; 1 = so x; 2 = ambas; 3 = ambas sem o topo"],
         ["acq_type", "nao (1)", "1 = superficie, 2 = crosswell"],
         ["device", "nao (0)", "indice do dispositivo OpenCL"],
         ["seimogram_shape", "nao ('2d')", "'2d' -> (nt, nr*ns); '3d' -> (nt, nr, ns)"],
         ["g_smooth", "nao (0)", "sigma da suavizacao gaussiana do gradiente"],
         ["energy_balancing", "nao (False)", "normaliza o gradiente pela iluminacao"],
         ["cost_function_type", "nao ('l2')", "funcional de erro"],
         ["sd", "nao (1.0)", "peso aplicado aos dados CALCULADOS no residuo"],
         ["grad_coeff", "nao ([1,1,1])", "peso relativo de vp, vs, rho no gradiente"],
         ["tv / tikhonov", "nao", "dicionarios de regularizacao (aula 13)"],
         ["prior_model", "nao", "modelo a priori para regularizacao (aula 13)"]])
    a.aviso("""
        Tres detalhes desta tabela que so se descobre lendo o fonte:

        1. `pml_dir` e FACIL de inverter. No codigo, 0 zera a PML em x (sobra
           so z) e 1 zera a PML em z (sobra so x) -- ou seja, o numero indica
           qual direcao voce DESLIGA, nao qual mantem. O valor 3 mantem as duas
           mas remove a PML do TOPO, que e como se modela superficie livre.

        2. `sd` nao e passo de otimizacao: e um peso que multiplica os dados
           CALCULADOS (`prepare_residual(d_est, sd)`), enquanto os observados
           entram com peso 1. Deixe em 1.0 a menos que saiba o que quer.

        3. Omitir `sdo` NAO da ordem 2: o codigo faz `self.sdo = sdo/2` e usa 2
           como padrao interno, o que corresponde a ordem espacial REAL 4.
    """)
    a.aviso("""
        Note o typo no nome da chave: e `seimogram_shape`, nao
        `seismogram_shape`. Escrever certo faz o PyFWI ignorar a chave em
        silencio e usar o padrao '2d'. Erros assim -- chave ignorada sem aviso --
        sao a razao de voce SEMPRE conferir a forma do array devolvido.
    """)

    # ==================================================================
    a.secao("O reescalonamento automatico de dt")
    a.texto("""
        O PyFWI nao usa o dt que voce pede. Nas primeiras linhas do construtor:
    """)
    a.codigo("""
        self.dt_scale = np.ceil(inpa['dt'] / 0.0006)
        self.dt       = inpa['dt'] / self.dt_scale
        self.nt       = int(1 + self.t // self.dt)        # passos INTERNOS
        self.nt_ext   = int(1 + self.t // self.dt_ext)    # amostras de SAIDA
    """, titulo="PyFWI/wave_propagation.py, WavePreparation.__init__")
    a.texto("""
        Ou seja: existe um teto rigido de 0.6 ms no passo interno. Se voce pedir
        dt = 2 ms, o PyFWI subdivide em 4 passos de 0.5 ms e devolve o
        sismograma reamostrado no seu dt. Veja na pratica:
    """)
    print()
    print(f"    {'dt pedido':<14}{'dt_scale':<12}{'dt interno':<14}"
          f"{'passos internos (t=1s)'}")
    for dt_ped in [0.0002, 0.0005, 0.0006, 0.001, 0.002, 0.004]:
        escala = np.ceil(dt_ped / 0.0006)
        dt_int = dt_ped / escala
        print(f"    {dt_ped*1e3:<14.2f}{escala:<12.0f}{dt_int*1e3:<14.4f}"
              f"{int(1 + 1.0 // dt_int)}")
    print()
    a.dica("""
        Consequencias praticas:

        * Pedir dt grande NAO economiza tempo de computacao -- o PyFWI subdivide
          de qualquer jeito. So economiza memoria do sismograma de saida.

        * O teto de 0.6 ms e calibrado para dh na casa de poucos metros. Se voce
          usar dh grande (25-50 m), esse dt fica desnecessariamente pequeno e
          voce paga caro sem ganhar precisao.

        * Se o seu modelo tiver velocidade muito alta E dh pequeno, 0.6 ms pode
          nao ser suficiente e a simulacao fica instavel mesmo assim. Faca a
          conta de CFL da aula 02 por sua conta; o PyFWI nao faz por voce.
    """)

    # ==================================================================
    a.secao("CPML: dimensionando a borda")
    a.texto("""
        O PyFWI usa CPML (Convolutional Perfectly Matched Layer), muito superior
        ao Cerjan que usamos no propagador proprio. Dois parametros controlam:
        `npml` (espessura em pontos) e `pmlR` (o coeficiente de reflexao teorico
        que a camada tenta atingir).

        Vamos medir a reflexao de borda de verdade. Como na aula 03, a
        referencia e um dominio grande onde nenhuma reflexao de borda chega
        dentro da janela de tempo.
    """)
    dh, dt, fdom = 7.0, 0.0005, 20.0
    nz, nx = 100, 100
    c_hom = 2500.0

    def modela(npml, NZ=nz, NX=nx, sdo=4, t_reg=0.35):
        modelo = {'vp': np.full((NZ, NX), c_hom, dtype=np.float32),
                  'vs': np.zeros((NZ, NX), dtype=np.float32),
                  'rho': np.full((NZ, NX), 2.2, dtype=np.float32)}
        inpa = {'ns': 1, 'sdo': sdo, 'fdom': fdom, 'dh': dh, 'dt': dt,
                't': t_reg, 'npml': npml, 'pmlR': 1e-5, 'pml_dir': 2,
                'acq_type': 1, 'energy_balancing': False, 'device': 0}
        src_loc = np.array([[NX * dh / 2, NZ * dh / 2]], dtype=np.float32)
        # o PyFWI deduz `rec_dis` de rec_loc[1]-rec_loc[0]: precisa de >= 2 rec.
        rec_loc = np.array([[NX * dh / 2 + k * 2 * dh, NZ * dh / 2]
                            for k in (3, 4, 5)], dtype=np.float32)
        src = acq.Source(src_loc, dh, dt)
        src.Ricker(fdom)
        W = wave.WavePropagator(inpa, src, rec_loc, (NZ, NX),
                                n_well_rec=0, chpr=0, components=0)
        t0 = time.time()
        d = W.forward_modeling(modelo, show=False)['taux'][:, 0]
        return d, time.time() - t0

    print("    calculando referencia em dominio grande...", end="", flush=True)
    ref, _ = modela(20, NZ=360, NX=360)
    print(" ok")
    amp_dir = float(np.abs(ref).max())
    i_corte = int(0.18 / dt)
    print()
    print(f"    {'npml':<10}{'espessura':<14}{'max|reflexao|':<18}{'em dB'}")
    for npml in [0, 5, 10, 20, 30]:
        d, _ = modela(npml)
        n = min(len(d), len(ref))
        refl = float(np.abs(d[i_corte:n] - ref[i_corte:n]).max())
        db = 20 * np.log10(refl / amp_dir + 1e-30)
        print(f"    {npml:<10}{npml*dh:>6.0f} m     {refl:<18.4e}{db:>7.1f} dB")
    print()
    a.texto("""
        Compare com a tabela da aula 03: o Cerjan chegava a -18 dB com 45
        pontos. A CPML faz melhor com muito menos. E por isso que ela e o padrao
        de producao, apesar de ser bem mais trabalhosa de implementar (e de
        adjuntar -- lembre que no nosso propagador escolhemos Cerjan justamente
        para manter o operador auto-adjunto).
    """)
    a.dica("""
        Regra pratica para npml no PyFWI: 10 a 20 pontos resolvem a maioria dos
        casos. Como na aula 03, dimensione pela MENOR frequencia que voce vai
        usar na FWI multiescala, nao pela frequencia de pico.

        `pmlR` entre 1e-4 e 1e-6 e a faixa usual. Valores mais extremos nao
        melhoram -- o limite passa a ser a discretizacao, nao a teoria.
    """)

    # ==================================================================
    a.secao("Ordem espacial: o que se ganha e o que se paga")
    print()
    print(f"    {'sdo':<8}{'tempo (s)':<14}{'obs'}")
    for sdo in [4, 8]:
        _, dur = modela(20, sdo=sdo)
        print(f"    {sdo:<8}{dur:<14.3f}{'estencil de %d pontos' % (sdo+1)}")
    print()
    a.texto("""
        O custo por passo cresce pouco com a ordem, porque o gargalo e acesso a
        memoria, nao aritmetica. O ganho real de sdo=8 nao esta no tempo por
        passo: esta em poder usar dh maior (G ~ 5 em vez de ~8, aula 02), o que
        reduz o numero de pontos em 2D pelo quadrado da razao. E ai a economia e
        grande.
    """)

    # ==================================================================
    a.secao("Checkpointing: o compromisso central da FWI")
    a.texto("""
        Aqui esta um dos pontos mais importantes de engenharia da FWI, e vale
        entender ANTES de escrever qualquer codigo proprio.

        Para calcular o gradiente pelo estado adjunto (aula 08) e preciso
        correlacionar, em cada ponto e em cada instante, o campo DIRETO com o
        campo ADJUNTO. Mas o campo direto avanca no tempo e o adjunto retrocede.
        Voce nunca tem os dois ao mesmo tempo sem pagar por isso.
    """)
    a.tabela(
        ["Estrategia", "Memoria", "Computacao", "Quando usar"],
        [["Guardar tudo", "nt x nz x nx", "1 modelagem", "2D pequeno (o que o fwikit faz)"],
         ["Recomputar", "1 campo", "~nt modelagens", "nunca, na pratica"],
         ["Checkpointing", "n_chk x nz x nx", "~2 modelagens", "o padrao"],
         ["Reversao de campo", "contorno", "2 modelagens", "quando nao ha atenuacao"]])
    a.texto("""
        Checkpointing e o meio-termo: voce guarda o campo completo so em alguns
        instantes (os checkpoints) e, durante a retropropagacao, reconstroi os
        instantes intermediarios repropagando a partir do checkpoint mais
        proximo. No PyFWI isso e o parametro `chpr`, em PORCENTAGEM:
    """)
    a.codigo("""
        chp = int(chpr * self.nt / 100)
        self.chp = np.linspace(0, self.nt-1, chp, dtype=np.int32)
    """, titulo="PyFWI/wave_propagation.py")
    nt_ex = 700
    nz_ex, nx_ex = 200, 400
    print()
    print(f"    Exemplo: nt = {nt_ex}, modelo {nz_ex} x {nx_ex}, float32")
    print()
    print(f"    {'chpr (%)':<12}{'checkpoints':<16}{'memoria':<16}{'vs guardar tudo'}")
    mem_total = nt_ex * nz_ex * nx_ex * 4 / 1e9
    for chpr in [0, 1, 5, 10, 25, 100]:
        n_chk = max(2, int(chpr * nt_ex / 100)) if chpr else 2
        mem = n_chk * nz_ex * nx_ex * 4 / 1e9
        print(f"    {chpr:<12}{n_chk:<16}{mem*1000:>8.1f} MB      "
              f"{100*mem/mem_total:>6.2f}%")
    print(f"\n    (guardar todos os {nt_ex} campos custaria "
          f"{mem_total*1000:.0f} MB por tiro)")
    print()
    a.aviso("""
        Em 3D esse calculo deixa de ser academico. Um modelo 500 x 500 x 300 com
        nt = 5000 exigiria 1.5 TB por tiro para guardar tudo. Nenhuma maquina
        faz isso. Por isso todo codigo de FWI 3D de producao usa checkpointing
        (ou reversao de campo a partir do contorno).

        Detalhe que quase ninguem menciona: a reversao de campo exige que o
        operador seja reversivel no tempo. Sem atenuacao, e. COM atenuacao, nao
        -- reverter um meio dissipativo amplifica ruido exponencialmente. Quem
        trabalha com fisica dissipativa e obrigado a usar checkpointing.
    """)

    # ==================================================================
    a.secao("Quanto custa uma FWI: a conta que voce precisa fazer")
    a.texto("""
        Antes de planejar qualquer experimento, faca esta conta. Ela evita
        descobrir na sexta-feira que a rodada leva tres semanas.
    """)
    a.eq("custo ~ n_iter x n_freq x n_tiros x 2 x custo_modelagem",
         rotulo="regra de bolso")
    a.texto("""
        O fator 2 e direto + adjunto. Se a busca linear fizer 2 avaliacoes
        extras por iteracao, multiplique por ~2 de novo.
    """)
    _, dur_1 = modela(20)
    print()
    a.resultado("custo de 1 modelagem (medido agora)", f"{dur_1:.3f}", "s")
    for n_tiros, n_iter, n_freq in [(10, 20, 3), (50, 30, 4), (200, 40, 5)]:
        total = n_iter * n_freq * n_tiros * 2 * 2 * dur_1
        print(f"    {n_tiros:>3} tiros x {n_iter:>2} iter x {n_freq} freq  ->  "
              f"{total/60:>8.1f} min   ({total/3600:.2f} h)")
    print()
    a.dica("""
        Duas alavancas mudam essa conta por ordens de grandeza:

        * PARALELISMO POR TIRO. Os tiros sao independentes: o gradiente e uma
          SOMA sobre eles. E paralelismo trivial, e e a primeira coisa a fazer.

        * SUBCONJUNTO ALEATORIO DE TIROS por iteracao (estocastico). Em vez dos
          200 tiros, use 20 sorteados a cada iteracao. O gradiente fica ruidoso,
          mas a direcao media esta certa e o custo cai 10x. Funciona
          surpreendentemente bem.
    """)

    a.pergunta(
        "Voce pede inpa['dt'] = 0.002 s. O que o PyFWI faz?",
        ["Usa 0.002 s e pode ficar instavel",
         "Recusa e levanta erro",
         "Subdivide internamente em 4 passos de 0.0005 s e devolve o dado "
         "amostrado em 0.002 s",
         "Ajusta dh automaticamente"],
        2,
        "dt_scale = ceil(0.002/0.0006) = 4, entao o passo interno vira 0.0005 s. "
        "Voce nao economiza computacao pedindo dt grande -- so reduz o tamanho "
        "do sismograma de saida. E o PyFWI nao verifica CFL contra o SEU c_max: "
        "essa conta continua sendo sua.")

    a.secao("Exercicios")
    a.exercicio(1, """
        Meca o tempo de modelagem em funcao de nz*nx (use modelos de 50x50 a
        300x300) e confirme que o custo escala linearmente com o numero de
        pontos. Depois inclua a dependencia com nt.
    """)
    a.exercicio(2, """
        Varie `pmlR` de 1e-2 a 1e-8 com npml fixo em 15 e meca a reflexao de
        borda. Existe um valor otimo ou a curva satura? Compare com o que voce
        viu no Cerjan da aula 03.
    """)
    a.exercicio(3, """
        Com `seimogram_shape='3d'` e ns=5, confira a forma do array devolvido.
        Depois escreva a chave ERRADA (`seismogram_shape`) e veja o que acontece.
        Como voce detectaria esse erro num script grande?
    """, dica="a resposta e: sempre imprima d.shape depois da modelagem.")

    a.fim([
        "`inpa` concentra a configuracao; a tabela desta aula e a referencia que "
        "falta na documentacao.",
        "PyFWI reescalona dt para no maximo 0.6 ms internamente -- pedir dt grande "
        "nao economiza computacao.",
        "CPML com npml de 10-20 pontos supera folgadamente o Cerjan com 45.",
        "Checkpointing (chpr) troca memoria por recomputacao; e obrigatorio em 3D.",
        "Custo ~ n_iter x n_freq x n_tiros x 4 x custo_modelagem. Faca a conta ANTES.",
    ], proxima="aula07_problema_inverso.py -- a funcao objetivo e o cycle skipping")


if __name__ == "__main__":
    main()
