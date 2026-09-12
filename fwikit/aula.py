"""
fwikit.aula
===========
Mini-framework didatico para o terminal: secoes, caixas de teoria, equacoes,
pausas, quizzes com correcao e exercicios.

Cada aula do curso e um script Python normal. Este modulo so cuida da
apresentacao, para o codigo da aula ficar limpo e voce enxergar a FISICA.

MODOS DE EXECUCAO
-----------------
    python aulas/aula03_....py              # interativo (pausas + quiz)
    python aulas/aula03_....py --rapido     # sem pausas, sem quiz, sem janelas
    python aulas/aula03_....py --sem-figuras# nao abre janelas (salva PNG)
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import textwrap

# --------------------------------------------------------------------------
# Cores ANSI (desligadas automaticamente se a saida nao for um terminal)
# --------------------------------------------------------------------------
_TTY = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(codigo: str) -> str:
    return codigo if _TTY else ""


RESET = _c("\033[0m")
NEGRITO = _c("\033[1m")
FRACO = _c("\033[2m")
ITALICO = _c("\033[3m")
CIANO = _c("\033[36m")
AZUL = _c("\033[34m")
VERDE = _c("\033[32m")
AMARELO = _c("\033[33m")
VERMELHO = _c("\033[31m")
MAGENTA = _c("\033[35m")


def _largura() -> int:
    return min(shutil.get_terminal_size((88, 24)).columns, 92)


def _visivel(s: str) -> int:
    """Comprimento ignorando codigos ANSI."""
    return len(re.sub(r"\033\[[0-9;]*m", "", s))


# --------------------------------------------------------------------------
# Flags globais de execucao
# --------------------------------------------------------------------------
def modo_rapido() -> bool:
    return "--rapido" in sys.argv or os.environ.get("FWI_RAPIDO") == "1"


def mostrar_figuras() -> bool:
    if "--sem-figuras" in sys.argv or modo_rapido():
        return False
    if os.environ.get("FWI_SEM_FIGURAS") == "1":
        return False
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class Aula:
    """Conduz uma aula no terminal."""

    def __init__(self, numero: int, titulo: str, modulo: str = "",
                 objetivos: list[str] | None = None,
                 pre_requisitos: str = "", duracao: str = ""):
        self.numero = numero
        self.titulo = titulo
        self.modulo = modulo
        self.objetivos = objetivos or []
        self.pre_requisitos = pre_requisitos
        self.duracao = duracao
        self.acertos = 0
        self.total_perguntas = 0
        self._n_secao = 0
        self.dir_saida = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "saidas", f"aula{numero:02d}")
        os.makedirs(self.dir_saida, exist_ok=True)

    # ---------------------------------------------------------------- layout
    def cabecalho(self) -> None:
        L = _largura()
        print()
        print(CIANO + "=" * L + RESET)
        if self.modulo:
            print(CIANO + FRACO + f"  {self.modulo.upper()}" + RESET)
        print(CIANO + NEGRITO + f"  AULA {self.numero:02d} -- {self.titulo}" + RESET)
        print(CIANO + "=" * L + RESET)
        if self.duracao:
            print(FRACO + f"  Duracao estimada: {self.duracao}" + RESET)
        if self.pre_requisitos:
            print(FRACO + f"  Pre-requisitos:   {self.pre_requisitos}" + RESET)
        if self.objetivos:
            print()
            print(NEGRITO + "  Ao final desta aula voce sera capaz de:" + RESET)
            for o in self.objetivos:
                for i, linha in enumerate(textwrap.wrap(o, L - 8)):
                    print(f"    {'*' if i == 0 else ' '} {linha}")
        print(CIANO + "-" * L + RESET)

    def secao(self, titulo: str) -> None:
        self._n_secao += 1
        L = _largura()
        print()
        print(AZUL + NEGRITO + f"  {self.numero}.{self._n_secao}  {titulo}" + RESET)
        print(AZUL + "  " + "-" * (L - 4) + RESET)

    def texto(self, s: str) -> None:
        L = _largura()
        for paragrafo in textwrap.dedent(s).strip().split("\n\n"):
            limpo = " ".join(paragrafo.split())
            print(textwrap.fill(limpo, L - 6, initial_indent="  ",
                                subsequent_indent="  "))
            print()

    def lista(self, itens: list[str], marcador: str = "*") -> None:
        L = _largura()
        for it in itens:
            linhas = textwrap.wrap(" ".join(it.split()), L - 10)
            for i, linha in enumerate(linhas):
                print(f"    {marcador if i == 0 else ' '} {linha}")
        print()

    def _caixa(self, titulo: str, corpo: str, cor: str, simbolo: str) -> None:
        L = _largura()
        largura_interna = L - 6
        print(cor + "  +" + "-" * (largura_interna) + "+" + RESET)
        cab = f" {simbolo} {titulo} "
        print(cor + "  |" + NEGRITO + cab.ljust(largura_interna) + RESET
              + cor + "|" + RESET)
        print(cor + "  +" + "-" * (largura_interna) + "+" + RESET)
        for paragrafo in textwrap.dedent(corpo).strip().split("\n\n"):
            linhas_brutas = paragrafo.split("\n")
            # mantem quebras manuais em blocos que parecem equacoes/codigo
            eh_bloco = any(l.startswith("    ") for l in linhas_brutas)
            if eh_bloco:
                linhas = [l.rstrip() for l in linhas_brutas]
            else:
                linhas = textwrap.wrap(" ".join(paragrafo.split()),
                                       largura_interna - 2)
            for linha in linhas:
                print(cor + "  |" + RESET + " " + linha.ljust(largura_interna - 1)
                      + cor + "|" + RESET)
            print(cor + "  |" + " " * largura_interna + "|" + RESET)
        print(cor + "  +" + "-" * (largura_interna) + "+" + RESET)
        print()

    def teoria(self, titulo: str, corpo: str) -> None:
        self._caixa(titulo, corpo, MAGENTA, "TEORIA")

    def dica(self, corpo: str, titulo: str = "NA PRATICA") -> None:
        self._caixa(titulo, corpo, VERDE, ">>")

    def aviso(self, corpo: str, titulo: str = "ATENCAO") -> None:
        self._caixa(titulo, corpo, AMARELO, "!!")

    def dissertacao(self, corpo: str) -> None:
        """Caixa que conecta o conteudo ao projeto de mestrado (Kjartansson)."""
        self._caixa("CONEXAO COM SUA DISSERTACAO", corpo, CIANO, "->")

    def eq(self, *linhas: str, rotulo: str = "") -> None:
        """Imprime uma equacao centralizada em texto."""
        L = _largura()
        print()
        for linha in linhas:
            pad = max(0, (L - _visivel(linha)) // 2)
            print(" " * pad + NEGRITO + linha + RESET)
        if rotulo:
            print(" " * max(0, L - len(rotulo) - 4) + FRACO + f"({rotulo})" + RESET)
        print()

    def codigo(self, s: str, titulo: str = "") -> None:
        L = _largura()
        if titulo:
            print(FRACO + f"  # {titulo}" + RESET)
        print(FRACO + "  " + "." * (L - 4) + RESET)
        for linha in textwrap.dedent(s).strip("\n").split("\n"):
            print(FRACO + "  | " + RESET + AMARELO + linha + RESET)
        print(FRACO + "  " + "." * (L - 4) + RESET)
        print()

    def resultado(self, rotulo: str, valor, unidade: str = "") -> None:
        print(f"    {rotulo:.<46} {NEGRITO}{valor}{RESET} {unidade}")

    def tabela(self, cabecalho: list[str], linhas: list[list], larguras=None) -> None:
        larguras = larguras or [max(len(str(cabecalho[i])),
                                    *(len(str(l[i])) for l in linhas)) + 2
                                for i in range(len(cabecalho))]
        sep = "  +" + "+".join("-" * w for w in larguras) + "+"
        print(sep)
        print("  |" + "|".join(NEGRITO + str(c).center(w) + RESET
                               for c, w in zip(cabecalho, larguras)) + "|")
        print(sep)
        for linha in linhas:
            print("  |" + "|".join(str(c).center(w)
                                   for c, w in zip(linha, larguras)) + "|")
        print(sep)
        print()

    # ------------------------------------------------------------ interacao
    def pausa(self, msg: str = "Pressione ENTER para continuar") -> None:
        if modo_rapido():
            return
        try:
            input(FRACO + f"\n  [ {msg} ] " + RESET)
        except (EOFError, KeyboardInterrupt):
            print()

    def pergunta(self, enunciado: str, alternativas: list[str],
                 correta: int, explicacao: str) -> None:
        """Quiz de multipla escolha. `correta` e o indice 0-based."""
        self.total_perguntas += 1
        L = _largura()
        print()
        print(AMARELO + NEGRITO + f"  ? QUIZ {self.total_perguntas}" + RESET)
        for linha in textwrap.wrap(" ".join(enunciado.split()), L - 6):
            print("  " + linha)
        print()
        for i, alt in enumerate(alternativas):
            print(f"    ({chr(97+i)}) {alt}")
        print()

        if modo_rapido():
            print(VERDE + f"    Resposta: ({chr(97+correta)})" + RESET)
            self.acertos += 1
        else:
            resp = None
            while resp is None:
                try:
                    bruto = input(FRACO + "    Sua resposta [a/b/c/...]: " + RESET)
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                bruto = bruto.strip().lower()
                if bruto and bruto[0] in [chr(97 + i) for i in range(len(alternativas))]:
                    resp = ord(bruto[0]) - 97
                else:
                    print(FRACO + "    (digite uma das letras)" + RESET)
            if resp == correta:
                self.acertos += 1
                print(VERDE + NEGRITO + "    CORRETO!" + RESET)
            elif resp is not None:
                print(VERMELHO + NEGRITO
                      + f"    Nao. A resposta e ({chr(97+correta)})." + RESET)

        for linha in textwrap.wrap(" ".join(explicacao.split()), L - 8):
            print(FRACO + "    " + linha + RESET)
        print()

    def exercicio(self, numero: int, enunciado: str, dica: str = "",
                  arquivo: str = "") -> None:
        L = _largura()
        print()
        print(VERDE + NEGRITO + f"  >> EXERCICIO {self.numero}.{numero}" + RESET)
        for linha in textwrap.wrap(" ".join(enunciado.split()), L - 6):
            print("  " + linha)
        if dica:
            print()
            for i, linha in enumerate(textwrap.wrap("Dica: " + " ".join(dica.split()),
                                                    L - 8)):
                print(FRACO + "    " + linha + RESET)
        if arquivo:
            print(FRACO + f"    Esqueleto: {arquivo}" + RESET)
        print()

    # ------------------------------------------------------------ encerramento
    def fim(self, resumo: list[str], proxima: str = "") -> None:
        L = _largura()
        print()
        print(CIANO + "=" * L + RESET)
        print(CIANO + NEGRITO + f"  RESUMO DA AULA {self.numero:02d}" + RESET)
        print(CIANO + "-" * L + RESET)
        for r in resumo:
            for i, linha in enumerate(textwrap.wrap(" ".join(r.split()), L - 8)):
                print(f"    {'*' if i == 0 else ' '} {linha}")
        if self.total_perguntas:
            print()
            pct = 100 * self.acertos / self.total_perguntas
            cor = VERDE if pct >= 70 else AMARELO
            print(cor + f"    Quiz: {self.acertos}/{self.total_perguntas} "
                        f"({pct:.0f}%)" + RESET)
        print()
        print(FRACO + f"    Figuras salvas em: {self.dir_saida}" + RESET)
        if proxima:
            print(NEGRITO + f"    Proxima: {proxima}" + RESET)
        print(CIANO + "=" * L + RESET)
        print()
