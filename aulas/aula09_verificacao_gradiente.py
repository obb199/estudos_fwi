#!/usr/bin/env python3
"""
AULA 09 -- Verificacao do gradiente: como PROVAR que esta certo
===============================================================
Execute:  python aulas/aula09_verificacao_gradiente.py
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
from fwikit.verificacao import (teste_diferencas_finitas,     # noqa: E402
                                teste_derivada_direcional,
                                teste_taylor, direcao_suave, veredito)


def main():
    a = Aula(9, "Verificacao do gradiente: como PROVAR que esta certo",
             modulo="Modulo III -- O problema inverso e o gradiente",
             duracao="70 min", pre_requisitos="aula 08",
             objetivos=[
                 "Aplicar os tres testes de gradiente e saber o que cada um detecta",
                 "Interpretar as ordens de convergencia do teste de Taylor",
                 "Diagnosticar a CAUSA de uma reprovacao a partir do padrao do erro",
                 "Saber que tolerancia e razoavel e o que e limite de precisao numerica",
             ])
    a.cabecalho()

    # ==================================================================
    a.secao("Por que este e o teste mais importante do seu codigo")
    a.texto("""
        Um gradiente errado nao gera erro nem trava o programa. Ele gera uma
        inversao que roda ate o fim, produz uma curva de convergencia
        decrescente e devolve um modelo plausivel -- e errado.

        Nenhum dos sintomas normais aparece. Voce so descobre quando alguem
        tenta reproduzir, ou quando o resultado nao bate com dado independente.
        Por isso a verificacao de gradiente e uma etapa OBRIGATORIA, e nao um
        extra: e o unico teste que separa "o codigo roda" de "o codigo esta
        certo".
    """)
    a.teoria("Os tres testes", """
        1. DIFERENCAS FINITAS PONTUAIS
           Compara dJ/dm_i em alguns pixels. Localiza ONDE o erro esta --
           util para descobrir se o problema e no miolo ou nas bordas.

        2. DERIVADA DIRECIONAL
           Compara <g,h> com a diferenca centrada de J ao longo de h.
           Testa o gradiente INTEIRO num numero so. E o teste do dia a dia:
           custa 2 modelagens por alpha.

        3. TESTE DE TAYLOR
           Verifica as ORDENS de convergencia:
               E0 = |J(m+ah) - J(m)|             -> deve cair como O(a)
               E1 = |J(m+ah) - J(m) - a<g,h>|    -> deve cair como O(a^2)
           E o mais rigoroso, e o que se apresenta em trabalho academico.
           E1 so e de segunda ordem se o gradiente estiver EXATO.
    """)

    # ==================================================================
    a.secao("Montando o caso de teste")
    a.texto("""
        Um caso de teste bom e PEQUENO (para os testes rodarem em segundos),
        mas nao degenerado. Use uma perturbacao real e cobertura razoavel.
    """)
    nz, nx, dh = 60, 90, 10.0
    c_verd = np.full((nz, nx), 2000.0, dtype=np.float32)
    zz, xx = np.mgrid[0:nz, 0:nx]
    c_verd[((zz - 30) ** 2 + (xx - 45) ** 2) < 8 ** 2] = 2300.0
    c_ini = np.full((nz, nx), 2000.0, dtype=np.float32)

    f0 = 12.0
    dt = dt_maximo(2400.0, dh, 4)
    nt = int(0.9 / dt)
    cfg = Config(dh=dh, dt=dt, nt=nt, ordem=4, n_abs=30, f0=f0)
    solver = Acustico2D(cfg, (nz, nx))
    w = ricker(f0, dt, nt)
    geom = Geometria(fontes=np.array([[3, 20], [3, 70]]),
                     receptores=np.array([[3, i] for i in range(4, 88, 2)]))
    d_obs = np.stack([solver.modelar(c_verd, geom, w, i)[0]
                      for i in range(geom.ns)], axis=-1)

    # J como funcao de m = 1/c^2 -- esta e a funcao que os testes derivam
    def J_de_m(m):
        c = (1.0 / np.sqrt(m)).astype(np.float32)
        total = 0.0
        for s in range(geom.ns):
            d, _ = solver.modelar(c, geom, w, s)
            total += misfit_l2(d, d_obs[:, :, s], dt)[0]
        return total

    m0 = (1.0 / c_ini ** 2).astype(np.float64)
    J0, g_m, _ = gradiente_adjunto(solver, c_ini, geom, w, d_obs, parametro="m")
    print()
    a.resultado("malha", f"{nz} x {nx}")
    a.resultado("tiros / receptores", f"{geom.ns} / {geom.nr}")
    a.resultado("J(m0)", f"{J0:.6e}")
    a.resultado("|g| maximo", f"{np.abs(g_m).max():.4e}")
    a.aviso("""
        Detalhe que economiza horas: faca os testes no parametro em que o
        gradiente foi DERIVADO -- aqui m = 1/c^2 -- e nao em velocidade. Se
        voce testar em c sem aplicar a regra da cadeia, o teste reprova e voce
        vai caçar um bug que nao existe.
    """)

    # ==================================================================
    a.secao("Teste 1: diferencas finitas pontuais")
    a.texto("""
        A escolha do epsilon e da posicao dos pontos nao e detalhe. O epsilon
        precisa ser grande o bastante para que J(m+eps) - J(m-eps) saia do ruido
        de float32, e pequeno o bastante para continuar no regime linear.
        Usamos aqui uma fracao do valor tipico de m.

        Escolhemos tambem, de proposito, quatro pontos de gradiente FORTE e dois
        de gradiente FRACO -- para voce ver o teste funcionar e ver onde ele
        perde sentido.
    """)
    ordenados = np.dstack(np.unravel_index(np.argsort(-np.abs(g_m).ravel()),
                                           g_m.shape))[0]
    fortes = [tuple(int(v) for v in ordenados[k]) for k in (0, 40, 120, 400)]
    fracos = [tuple(int(v) for v in ordenados[k]) for k in (-1, -200)]
    pontos = fortes + fracos
    eps = 1e-2 * float(np.mean(m0))
    print()
    a.resultado("epsilon usado", f"{eps:.3e}",
                f"({100*eps/np.mean(m0):.1f}% de m tipico)")
    linhas = teste_diferencas_finitas(J_de_m, m0, g_m, pontos, eps=eps)
    print()
    print(f"    {'ponto':<12}{'|g| relativo':<15}{'adjunto':<15}"
          f"{'dif. finitas':<15}{'razao':<10}")
    print(f"    {'-'*12}{'-'*15}{'-'*15}{'-'*15}{'-'*10}")
    pico = float(np.abs(g_m).max())
    for L in linhas:
        rel = abs(L["adjunto"]) / pico
        marca = "" if rel > 1e-2 else "   <- gradiente fraco"
        print(f"    {str(L['ponto']):<12}{rel:<15.2e}{L['adjunto']:<15.5e}"
              f"{L['dif_finitas']:<15.5e}{L['razao']:<10.4f}{marca}")
    print()
    a.texto("""
        Nos pontos de gradiente forte a razao fica proxima de 1: o gradiente
        esta correto ponto a ponto. Nos pontos marcados como fracos a razao
        degenera -- e ali numerador e denominador sao ambos da ordem do ruido
        de float32, entao a razao entre eles nao significa nada.

        Isso NAO e erro de gradiente: e o limite de precisao do proprio teste.
        Por isso o teste pontual serve para LOCALIZAR problemas, nunca para
        aprovar ou reprovar um gradiente. Quem decide isso sao os testes 2 e 3,
        que envolvem o campo inteiro e nao dependem de um pixel isolado.
    """)

    # ==================================================================
    a.secao("Teste 2: derivada direcional")
    a.texto("""
        Agora o gradiente inteiro de uma vez. A direcao `h` importa: uma direcao
        ALEATORIA e quase ortogonal ao gradiente (que e suave), o que deixa
        <g,h> pequeno e o teste dominado pela curvatura. Use uma direcao suave --
        o proprio gradiente normalizado.
    """)
    h = direcao_suave(g_m, m0, amplitude=1e-2)
    gh, linhas_dir = teste_derivada_direcional(J_de_m, m0, g_m, h,
                                               alphas=(0.4, 0.2, 0.1, 0.05,
                                                       0.02, 0.01))
    print()
    print(f"    <g, h> analitico = {gh:.8e}")
    print()
    print(f"    {'alpha':<12}{'numerico (centrado)':<26}{'razao':<12}")
    print(f"    {'-'*12}{'-'*26}{'-'*12}")
    for L in linhas_dir:
        print(f"    {L['alpha']:<12.4g}{L['numerico']:<26.8e}{L['razao']:<12.6f}")
    ok, msg = veredito(linhas_dir, tol=0.02)
    print()
    print(f"    {msg}")
    print()
    a.dica("""
        Como LER esta tabela -- e aqui esta o valor do teste:

        * Razao ~ 1 e ESTAVEL em todos os alphas -> gradiente correto.

        * Razao ESTAVEL mas diferente de 1 (por exemplo 0.91 em todos) ->
          erro SISTEMATICO. Procure: fator de escala esquecido, sinal trocado,
          peso de quadratura (dt, dh^2) faltando, ou um termo do adjunto
          ausente. Um valor estavel e otima noticia: o bug e determinstico.

        * Razao INSTAVEL, variando com alpha -> ruido numerico. Reduza a faixa
          de alpha, aumente a perturbacao, ou use precisao dupla.
    """)

    # ==================================================================
    a.secao("Teste 3: teste de Taylor")
    a.texto("""
        O teste definitivo. Ele nao compara valores: compara TAXAS DE
        CONVERGENCIA, o que o torna insensivel a erros de escala do proprio
        teste.
    """)
    _, gh2, linhas_t = teste_taylor(J_de_m, m0, g_m, h,
                                    alphas=(0.4, 0.2, 0.1, 0.05, 0.025),
                                    J0=J0)
    print()
    print(f"    {'alpha':<10}{'E0':<15}{'E1':<15}{'ordem E0':<12}{'ordem E1':<12}")
    print(f"    {'-'*10}{'-'*15}{'-'*15}{'-'*12}{'-'*12}")
    for L in linhas_t:
        o0 = "--" if not np.isfinite(L["ordem_E0"]) else f"{L['ordem_E0']:.2f}"
        o1 = "--" if not np.isfinite(L["ordem_E1"]) else f"{L['ordem_E1']:.2f}"
        print(f"    {L['alpha']:<10.4g}{L['E0']:<15.5e}{L['E1']:<15.5e}"
              f"{o0:<12}{o1:<12}")
    print()
    ordens1 = [L["ordem_E1"] for L in linhas_t if np.isfinite(L["ordem_E1"])]
    ordem_media = float(np.mean(ordens1)) if ordens1 else float("nan")
    a.resultado("ordem media de E1 (esperado ~2)", f"{ordem_media:.2f}")
    print()
    a.texto("""
        E0 caindo como O(a) confirma que J e diferenciavel e que a perturbacao
        esta no regime linear. E1 caindo como O(a^2) confirma que o gradiente e
        EXATO -- porque so entao o termo de primeira ordem e cancelado
        perfeitamente e sobra a curvatura.

        Se E1 cair como O(a) em vez de O(a^2), o gradiente esta errado: sobrou
        termo de primeira ordem. Esse e o diagnostico mais informativo de todos.
    """)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    al = np.array([L["alpha"] for L in linhas_t])
    E0 = np.array([L["E0"] for L in linhas_t])
    E1 = np.array([L["E1"] for L in linhas_t])
    ax[0].loglog(al, E0, "o-", lw=1.6, label="E0 (medido)")
    ax[0].loglog(al, E1, "s-", lw=1.6, label="E1 (medido)")
    ax[0].loglog(al, E0[0] * (al / al[0]), "k--", lw=1.0, label="O(a)")
    ax[0].loglog(al, E1[0] * (al / al[0]) ** 2, "k:", lw=1.2, label="O(a^2)")
    ax[0].set_xlabel("alpha"); ax[0].set_ylabel("erro")
    ax[0].set_title("teste de Taylor")
    ax[0].legend(fontsize=8)
    ax[0].invert_xaxis()

    razoes = np.array([L["razao"] for L in linhas_dir])
    alphas_d = np.array([L["alpha"] for L in linhas_dir])
    ax[1].semilogx(alphas_d, razoes, "o-", lw=1.6)
    ax[1].axhline(1.0, color="k", ls="--", lw=1.0)
    ax[1].axhspan(0.98, 1.02, color="green", alpha=0.15)
    ax[1].set_xlabel("alpha")
    ax[1].set_ylabel("<g,h> analitico / numerico")
    ax[1].set_title("derivada direcional (faixa verde: +-2%)")
    ax[1].invert_xaxis()
    fig.tight_layout()
    plot.salvar(fig, a, "01_testes_de_gradiente", mostrar=False)

    # ==================================================================
    a.secao("Estudo de caso: o bug que este teste encontrou")
    a.texto("""
        Vale contar um caso real, porque ele mostra como os tres testes se
        complementam. Este mesmo propagador, quando foi escrito para o curso,
        REPROVOU no teste da derivada direcional:
    """)
    a.codigo("""
        alpha      numerico (centrado)   razao
        0.4        2.41766014e-06        0.911797
        0.2        2.41807837e-06        0.911639
        0.1        2.41746544e-06        0.911871
        0.05       2.41784379e-06        0.911728
        0.02       2.41945963e-06        0.911119
        0.01       2.41738744e-06        0.911900
    """, titulo="resultado REPROVADO, antes da correcao")
    a.texto("""
        Note o padrao: 0.9118 em TODOS os alphas, ao longo de uma faixa de 40x.
        Perfeitamente estavel. Isso descarta ruido numerico e aponta para erro
        sistematico -- um termo faltando, nao um problema de precisao.

        O teste PONTUAL localizou a regiao: as razoes davam ~0.99 no miolo do
        modelo e caiam para ~0.72 perto da superficie. O erro estava nas bordas.

        A causa: antes de propagar, o modelo e estendido com uma moldura
        absorvente usando `mode='edge'`, que REPLICA cada pixel de borda por
        toda a moldura. Isso e um operador linear E. O gradiente estava sendo
        levado de volta ao dominio fisico com um simples RECORTE -- mas o
        adjunto de E nao e recortar, e SOMAR a moldura de volta sobre o pixel
        de borda que a originou.
    """)
    a.codigo("""
        # ERRADO -- descarta a sensibilidade acumulada na moldura
        g_m = g_ext[topo:topo+nz, n:n+nx]

        # CERTO -- E^T: dobra a moldura de volta sobre as bordas
        g[topo]            += g[:topo].sum(axis=0)
        g[topo+nz-1]       += g[topo+nz:].sum(axis=0)
        g[:, n]            += g[:, :n].sum(axis=1)
        g[:, n+nx-1]       += g[:, n+nx:].sum(axis=1)
    """, titulo="fwikit/acustico.py :: contrair_adjunto()")
    a.texto("""
        Com a correcao, a razao passou de 0.9118 para 1.0001. O gradiente estava
        errado em quase 9% -- o suficiente para degradar a convergencia sem
        nunca impedir a inversao de rodar. Sem o teste, isso teria passado.
    """)
    a.dica("""
        A licao generalizavel: TODA operacao que voce aplica ao modelo antes de
        propagar precisa do seu adjunto no caminho de volta do gradiente.
        Extensao de bordas, interpolacao entre malhas, suavizacao,
        reparametrizacao, mudanca de unidade. Se voce esqueceu de uma, o teste
        de gradiente acusa -- e e o unico que acusa.
    """)

    a.pergunta(
        "Seu teste de Taylor da ordem 1 para E0 e ordem 1 para E1. "
        "O que isso significa?",
        ["O gradiente esta correto",
         "O gradiente esta errado -- sobrou termo de primeira ordem",
         "A perturbacao alpha esta grande demais",
         "O problema nao e diferenciavel"],
        1,
        "E1 remove o termo de primeira ordem usando o gradiente. Se E1 ainda "
        "cai como O(a), o termo nao foi cancelado: o gradiente nao e a "
        "derivada correta. Se alpha fosse grande demais, as ordens estariam "
        "erradas de forma INSTAVEL e melhorariam ao reduzir alpha.")

    a.secao("Exercicios")
    a.exercicio(1, """
        Introduza um bug DE PROPOSITO: multiplique o gradiente por 1.5, ou
        troque o sinal do residuo no adjunto. Rode os tres testes e observe como
        cada um denuncia o problema (e qual denuncia mais claramente).
    """)
    a.exercicio(2, """
        Rode o teste de Taylor com h ALEATORIO em vez de suave. Explique por que
        as ordens ficam piores mesmo com o gradiente correto.
    """, dica="calcule <g,h> nos dois casos e compare com E0.")
    a.exercicio(3, """
        Refaca o teste com o solver em float64 (mude os dtype em
        fwikit/acustico.py). Ate que alpha as ordens continuam limpas? Esse e o
        piso de precisao do teste.
    """)

    a.fim([
        "Gradiente errado nao gera erro: gera resultado plausivel e errado. "
        "Verificar e obrigatorio.",
        "Tres testes: pontual (localiza), direcional (diagnostica), Taylor (prova).",
        "Razao estavel != 1 -> erro sistematico. Razao instavel -> ruido numerico.",
        "E1 ~ O(a^2) e a prova de que o gradiente e exato.",
        "Toda operacao aplicada ao modelo antes de propagar precisa do adjunto "
        "no caminho de volta.",
    ], proxima="aula10_gradiente_pyfwi.py -- gradiente no PyFWI, iluminacao e pre-condicionamento")


if __name__ == "__main__":
    main()
