#!/usr/bin/env python3
"""
AULA 00 -- Ambiente, diagnostico e por que cada peca existe
===========================================================
Execute:  python aulas/aula00_ambiente.py
"""
import os
import platform
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fwikit                                    # noqa: E402  (aplica o compat)
from fwikit.aula import Aula                     # noqa: E402


def main():
    a = Aula(0, "Ambiente, diagnostico e o mapa do curso",
             modulo="Modulo 0 -- Preparacao",
             duracao="20 min",
             objetivos=[
                 "Verificar que PyFWI, OpenCL e o stack cientifico estao funcionando",
                 "Entender por que o PyFWI precisa de uma camada de compatibilidade",
                 "Conhecer a arquitetura do curso e o papel de cada ferramenta",
             ])
    a.cabecalho()

    # ------------------------------------------------------------------
    a.secao("O mapa: o que e FWI, em um paragrafo")
    a.texto("""
        Full-Waveform Inversion e um problema de otimizacao. Voce tem um dado
        sismico observado d_obs. Voce tem um simulador F que, dado um modelo de
        subsuperficie m, produz um sismograma sintetico F(m). A FWI procura o
        modelo m que faz F(m) ficar o mais parecido possivel com d_obs -- usando
        a FORMA DE ONDA inteira, nao so o tempo de chegada.
    """)
    a.eq("m* = argmin_m  J(m),      J(m) = (1/2) || F(m) - d_obs ||^2", rotulo="FWI")
    a.texto("""
        Simples de escrever, brutal de resolver. Tres dificuldades definem o
        campo inteiro, e o curso e organizado em torno delas:
    """)
    a.lista([
        "F e CARO. Cada avaliacao e uma simulacao completa da equacao da onda. "
        "-> Modulos I e II: como simular bem e rapido.",
        "J e NAO CONVEXO. Tem minimos locais por toda parte (cycle skipping). "
        "-> Modulo III: por que, e como escapar.",
        "m tem 10^4 a 10^8 incognitas. Nao da para montar a Hessiana, nem para "
        "calcular o gradiente por forca bruta. -> Modulo III (estado adjunto) e "
        "Modulo IV (otimizacao).",
    ])

    a.teoria("A ideia central que faz a FWI ser viavel", """
        Calcular dJ/dm por diferencas finitas exigiria UMA simulacao por
        parametro do modelo. Com 10^6 parametros, e 10^6 simulacoes por
        iteracao: impossivel.

        O metodo do ESTADO ADJUNTO obtem o gradiente INTEIRO com apenas DUAS
        simulacoes por tiro: uma direta e uma adjunta. O custo passa a ser
        independente do numero de parametros.

        Sem essa ideia nao existiria FWI. Ela e o assunto da aula 08.
    """)

    # ------------------------------------------------------------------
    a.secao("Diagnostico do ambiente")
    print()
    a.resultado("Python", platform.python_version())
    a.resultado("Sistema", f"{platform.system()} {platform.release()}")
    a.resultado("Executavel", sys.executable)
    print()

    estado = {}
    for nome, mod in [("numpy", "numpy"), ("scipy", "scipy"),
                      ("matplotlib", "matplotlib"), ("pyopencl", "pyopencl"),
                      ("PyFWI", "PyFWI"), ("h5py", "h5py"), ("segyio", "segyio")]:
        try:
            m = __import__(mod)
            v = getattr(m, "__version__", getattr(m, "VERSION_TEXT", "?"))
            estado[nome] = str(v)
            a.resultado(nome, v)
        except Exception as exc:
            estado[nome] = None
            a.resultado(nome, f"AUSENTE ({type(exc).__name__})")

    # ------------------------------------------------------------------
    a.secao("A camada de compatibilidade (e por que ela existe)")
    a.texto("""
        O PyFWI 0.1.10 foi publicado junto com o artigo do autor e congelado no
        tempo. Ele importa tres simbolos que ja nao existem no numpy 2.x e no
        scipy >= 1.12. Sem remendo, `import PyFWI.fwi` falha na hora.
    """)
    a.codigo("""
        # PyFWI/seismic_io.py
        from numpy.lib.function_base import kaiser      # removido no numpy 2.0
        # PyFWI/fwi.py
        from scipy.optimize.optimize import MemoizeJac  # removido no scipy 1.12
        from scipy.optimize.lbfgsb import fmin_l_bfgs_b # namespace depreciado
    """, titulo="o que quebra")
    a.texto("""
        A alternativa seria rebaixar o ambiente inteiro para numpy 1.x -- o que
        te prende no passado. Preferimos reconstruir os simbolos ausentes antes
        do PyFWI ser importado. E o que `fwikit/compat.py` faz, e por isso todas
        as aulas comecam com `import fwikit`.
    """)
    relatorio = fwikit._compat.aplicar()
    print()
    if relatorio:
        print("    Remendos aplicados nesta sessao:")
        for r in relatorio:
            print(f"      - {r}")
    else:
        print("    Nenhum remendo necessario neste ambiente.")
    print()
    a.dica("""
        Esta situacao e a regra, nao a excecao, em geofisica computacional.
        Codigo academico publicado com um artigo raramente recebe manutencao.
        Saber ler um ImportError, achar onde o simbolo foi parar e remendar sem
        quebrar o resto e parte do oficio.
    """)

    # ------------------------------------------------------------------
    a.secao("OpenCL: onde o PyFWI realmente roda")
    a.texto("""
        O nucleo do PyFWI nao e Python: sao kernels OpenCL (arquivos .cl
        instalados junto com o pacote). O Python so monta os buffers e despacha.
        Por isso a modelagem e rapida -- e por isso o PyFWI exige um dispositivo
        OpenCL funcionando.
    """)
    try:
        import pyopencl as cl
        plataformas = cl.get_platforms()
        print()
        for p in plataformas:
            print(f"    Plataforma: {p.name}  ({p.version})")
            for d in p.get_devices():
                tipo = cl.device_type.to_string(d.type)
                mem = d.global_mem_size / 1e9
                print(f"      -> {d.name}  [{tipo}]  {mem:.1f} GB  "
                      f"{d.max_compute_units} unidades")
        if not plataformas:
            raise RuntimeError("nenhuma plataforma OpenCL")
        print()
        a.dica("""
            Sem GPU? Instale o POCL (`pip install pocl-binary-distribution` ou o
            pacote `pocl-opencl-icd` da sua distribuicao) para ter um dispositivo
            OpenCL em CPU. Tudo no curso funciona -- so mais devagar. As aulas dos
            Modulos I e V e as aulas 07-09, 11 e 13 usam o propagador proprio em
            NumPy, que nao precisa de OpenCL nenhum.
        """)
    except Exception as exc:
        print()
        a.aviso(f"""
            OpenCL indisponivel ({exc}).

            As aulas que usam o PyFWI (05, 06, 10, 12) nao vao rodar, mas as
            demais (01-04, 07-09, 11, 13 e 14) usam o propagador proprio em NumPy
            e funcionam normalmente. Para habilitar: instale um ICD OpenCL (POCL para CPU,
            ou o driver da sua GPU).
        """)

    # ------------------------------------------------------------------
    a.secao("Teste de fumaca: uma onda de verdade")
    a.texto("""
        Vamos propagar uma onda num meio homogeneo com o propagador proprio do
        curso (`fwikit.acustico`) so para confirmar que tudo funciona.
    """)
    import numpy as np
    from fwikit.acustico import Acustico2D, Config, Geometria, ricker, dt_maximo

    cfg = Config(dh=10.0, dt=1.0e-3, nt=500, ordem=4, n_abs=30, f0=12.0)
    c = np.full((70, 100), 2000.0, dtype=np.float32)
    geom = Geometria(fontes=np.array([[35, 50]]),
                     receptores=np.array([[5, i] for i in range(5, 95, 3)]))
    w = ricker(cfg.f0, cfg.dt, cfg.nt)
    solver = Acustico2D(cfg, c.shape)
    import time
    t0 = time.time()
    d, _ = solver.modelar(c, geom, w)
    dtempo = time.time() - t0
    print()
    a.resultado("dt usado", f"{cfg.dt*1e3:.3f}", "ms")
    a.resultado("dt maximo estavel (CFL)", f"{dt_maximo(c.max(), cfg.dh, cfg.ordem)*1e3:.3f}", "ms")
    a.resultado("passos de tempo", cfg.nt)
    a.resultado("tempo de execucao", f"{dtempo:.2f}", "s")
    a.resultado("sismograma", f"{d.shape[0]} x {d.shape[1]}", "(nt x nr)")
    a.resultado("campo finito (sem NaN)", "SIM" if np.isfinite(d).all() else "NAO -- INSTAVEL")
    a.resultado("amplitude maxima", f"{np.abs(d).max():.3e}")

    a.pausa()

    # ------------------------------------------------------------------
    a.secao("Como o curso esta organizado")
    a.tabela(
        ["Modulo", "Aulas", "Assunto"],
        [["I", "01-04", "Fisica e numerica da propagacao"],
         ["II", "05-06", "PyFWI como ferramenta de producao"],
         ["III", "07-10", "O problema inverso e o gradiente adjunto"],
         ["IV", "11-13", "Otimizacao, FWI completa e regularizacao"],
         ["V", "14", "Limites da aproximacao acustica"]])
    a.texto("""
        Duas ferramentas convivem no curso, de proposito:

        `fwikit.acustico` -- propagador proprio, ~480 linhas de NumPy (metade
        delas comentario), sem GPU.
        Transparente: voce ve cada termo. E com ele que derivamos e verificamos
        o gradiente adjunto.

        `PyFWI` -- implementacao de producao em OpenCL, elastica, com CPML e
        checkpointing. Rapida, mas caixa-preta. Serve para rodar FWI de verdade
        e como referencia para validar o seu proprio codigo.

        Aprender so o primeiro te deixa lento. Aprender so o segundo te deixa
        sem saber o que fazer quando o resultado sai errado.
    """)

    a.pergunta(
        "Por que o metodo do estado adjunto e indispensavel em FWI?",
        ["Porque e mais preciso que diferencas finitas",
         "Porque da o gradiente inteiro com 2 simulacoes, "
         "independente do numero de parametros",
         "Porque garante convergencia para o minimo global",
         "Porque permite usar GPU"],
        1,
        "O custo do gradiente por diferencas finitas cresce com o numero de "
        "parametros (uma simulacao por parametro). O adjunto custa 2 simulacoes "
        "por tiro, sempre. Ele nao e mais preciso nem resolve nao convexidade -- "
        "so torna o calculo viavel.")

    a.fim([
        "FWI = minimizar a diferenca entre sismograma observado e simulado.",
        "As tres dificuldades: F e caro, J e nao convexo, m e gigante.",
        "O estado adjunto e o que torna o gradiente calculavel.",
        "PyFWI precisa de compat (numpy 2 / scipy 1.12+) e de OpenCL.",
        "O curso usa um propagador transparente para entender, e o PyFWI para produzir.",
    ], proxima="aula01_regime_acustico.py -- de onde vem a equacao que vamos resolver")


if __name__ == "__main__":
    main()
