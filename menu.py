#!/usr/bin/env python3
"""
Menu do Curso de FWI com PyFWI
===============================
    python menu.py              # menu interativo
    python menu.py 7            # roda direto a aula 07
    python menu.py todas        # roda todas em modo rapido (teste de fumaca)
    python menu.py --listar     # so lista
"""
import os
import subprocess
import sys

RAIZ = os.path.dirname(os.path.abspath(__file__))
AULAS_DIR = os.path.join(RAIZ, "aulas")

MODULOS = [
    ("Modulo 0 -- Preparacao", [0]),
    ("Modulo I -- Fisica e numerica da propagacao", [1, 2, 3, 4]),
    ("Modulo II -- PyFWI como ferramenta de producao", [5, 6]),
    ("Modulo III -- O problema inverso e o gradiente", [7, 8, 9, 10]),
    ("Modulo IV -- Otimizacao, FWI completa e regularizacao", [11, 12, 13]),
    ("Modulo V -- Onde a fisica adotada deixa de valer", [14]),
]

TITULOS = {
    0: ("Ambiente, diagnostico e o mapa do curso", "20 min", ""),
    1: ("O regime acustico: de onde vem a equacao", "60 min", ""),
    2: ("Diferencas finitas: estabilidade e dispersao", "60 min", ""),
    3: ("Modelagem 2D: campos, snapshots e bordas", "70 min", ""),
    4: ("Aquisicao: geometria e sismogramas", "60 min", ""),
    5: ("PyFWI e a prova do regime acustico", "70 min", "OpenCL"),
    6: ("PyFWI por dentro: inpa, PML, checkpointing", "60 min", "OpenCL"),
    7: ("Funcao objetivo e cycle skipping", "70 min", ""),
    8: ("O metodo do estado adjunto", "90 min", ""),
    9: ("Verificacao do gradiente", "70 min", ""),
    10: ("Gradiente no PyFWI e pre-condicionamento", "70 min", "OpenCL"),
    11: ("Otimizacao: descida maxima, CG e l-BFGS", "80 min", ""),
    12: ("A FWI completa: multiescala", "90 min", "OpenCL, LENTA"),
    13: ("Regularizacao: Tikhonov, TV e vinculos", "75 min", ""),
    14: ("Os limites da aproximacao acustica", "60 min", ""),
}

C = {"az": "\033[36m", "vd": "\033[32m", "am": "\033[33m",
     "ng": "\033[1m", "fr": "\033[2m", "z": "\033[0m"}
if not sys.stdout.isatty():
    C = {k: "" for k in C}


def arquivo_da_aula(n):
    for f in sorted(os.listdir(AULAS_DIR)):
        if f.startswith(f"aula{n:02d}_") and f.endswith(".py"):
            return os.path.join(AULAS_DIR, f)
    return None


def listar():
    print()
    print(C["az"] + "=" * 78 + C["z"])
    print(C["az"] + C["ng"] + "  CURSO DE FULL-WAVEFORM INVERSION COM PyFWI" + C["z"])
    print(C["az"] + "=" * 78 + C["z"])
    for modulo, nums in MODULOS:
        print()
        print(C["az"] + f"  {modulo}" + C["z"])
        for n in nums:
            titulo, dur, obs = TITULOS[n]
            marca = C["am"] + f" [{obs}]" + C["z"] if obs else ""
            existe = "" if arquivo_da_aula(n) else C["am"] + "  (ausente)" + C["z"]
            print(f"    {C['ng']}{n:2d}{C['z']}  {titulo:<46} "
                  f"{C['fr']}{dur:>7}{C['z']}{marca}{existe}")
    print()
    print(C["fr"] + "  PDF teorico: curso_fwi_completo.pdf   "
          "Figuras: saidas/aulaXX/" + C["z"])
    print(C["fr"] + "  Cada aula aceita --rapido (sem pausas) "
          "e --sem-figuras (nao abre janelas)" + C["z"])
    print()


def rodar(n, extra=None):
    caminho = arquivo_da_aula(n)
    if not caminho:
        print(f"  Aula {n:02d} nao encontrada em {AULAS_DIR}")
        return 1
    cmd = [sys.executable, caminho] + (extra or [])
    print(C["vd"] + f"\n  >>> {os.path.basename(caminho)}\n" + C["z"])
    return subprocess.call(cmd, cwd=RAIZ)


def main():
    args = sys.argv[1:]
    if args and args[0] in ("--listar", "-l", "lista"):
        listar()
        return
    if args and args[0] == "todas":
        listar()
        print(C["am"] + "  Rodando TODAS as aulas em modo rapido "
              "(pode levar 20+ minutos)\n" + C["z"])
        falhas = []
        for _, nums in MODULOS:
            for n in nums:
                cod = rodar(n, ["--rapido", "--sem-figuras"])
                if cod != 0:
                    falhas.append(n)
        print()
        if falhas:
            print(C["am"] + f"  Aulas com erro: {falhas}" + C["z"])
        else:
            print(C["vd"] + "  Todas as aulas rodaram sem erro." + C["z"])
        return
    if args and args[0].isdigit():
        rodar(int(args[0]), args[1:])
        return

    while True:
        listar()
        try:
            esc = input(C["ng"] + "  Numero da aula (ou 'q' para sair): " + C["z"])
        except (EOFError, KeyboardInterrupt):
            print()
            return
        esc = esc.strip().lower()
        if esc in ("q", "sair", "quit", "exit", ""):
            return
        if esc.isdigit() and int(esc) in TITULOS:
            rodar(int(esc))
            try:
                input(C["fr"] + "\n  [ ENTER para voltar ao menu ] " + C["z"])
            except (EOFError, KeyboardInterrupt):
                return
        else:
            print(C["am"] + "  Numero invalido." + C["z"])


if __name__ == "__main__":
    main()
