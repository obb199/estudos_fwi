# Curso de Full-Waveform Inversion com PyFWI

Curso completo de FWI — teoria e prática juntas — do problema direto ao
problema inverso, com código executável e verificação numérica.

São **15 aulas interativas** em Python, um **PDF teórico de 56 páginas**, e uma
caixa de ferramentas (`fwikit`) com um propagador acústico próprio cujo
gradiente adjunto é **verificado contra diferenças finitas**.

Todos os números, tabelas e figuras do PDF foram gerados pelo código deste
repositório. Nenhum foi copiado de livro.

---

## Comece por aqui

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python menu.py           # menu interativo
.venv/bin/python menu.py 0         # aula 00: diagnóstico do ambiente
```

O PDF está em [`curso_fwi_completo.pdf`](curso_fwi_completo.pdf).

Cada aula aceita `--rapido` (sem pausas nem quiz) e `--sem-figuras` (não abre
janelas, só salva PNG).

---

## As aulas

| # | Aula | Duração | Requer |
|---|------|---------|--------|
| **Módulo 0 — Preparação** |
| 00 | Ambiente, diagnóstico e o mapa do curso | 20 min | |
| **Módulo I — Física e numérica da propagação** |
| 01 | O regime acústico: de onde vem a equação | 60 min | |
| 02 | Diferenças finitas: estabilidade e dispersão | 60 min | |
| 03 | Modelagem 2D: campos, snapshots e bordas | 70 min | |
| 04 | Aquisição: geometria e sismogramas | 60 min | |
| **Módulo II — PyFWI como ferramenta de produção** |
| 05 | PyFWI e a prova numérica do regime acústico | 70 min | OpenCL |
| 06 | PyFWI por dentro: `inpa`, PML, checkpointing | 60 min | OpenCL |
| **Módulo III — O problema inverso e o gradiente** |
| 07 | Função objetivo e cycle skipping | 70 min | |
| 08 | O método do estado adjunto | 90 min | |
| 09 | Verificação do gradiente | 70 min | |
| 10 | Gradiente no PyFWI e pré-condicionamento | 70 min | OpenCL |
| **Módulo IV — Otimização, FWI completa e regularização** |
| 11 | Otimização: descida máxima, CG e l-BFGS | 80 min | |
| 12 | A FWI completa: multiescala | 90 min | OpenCL, **lenta** |
| 13 | Regularização: Tikhonov, TV e vínculos | 75 min | |
| **Módulo V — Os limites** |
| 14 | Os limites da aproximação acústica | 60 min | |

Cada aula é um script independente: roda, mede, desenha, e faz perguntas. As
aulas 01, 02, 03, 04, 07 e 08 abrem janelas interativas com controles
deslizantes quando há display disponível.

---

## As duas ferramentas, e por que são duas

**`fwikit.acustico`** — propagador acústico 2D escrito do zero em NumPy, 476
linhas (com mais comentário que código), sem GPU. Transparente: cada termo está visível. É com ele que o
gradiente adjunto é derivado e *verificado*.

**PyFWI** — implementação de produção em OpenCL: elástica, com CPML e
checkpointing. Rápida e caixa-preta. Serve para rodar inversões de verdade e
como referência independente.

Aprender só o primeiro deixa você lento. Aprender só o segundo deixa você sem
saber o que fazer quando o resultado sai errado.

---

## Quatro coisas que este curso descobriu rodando o código

**1. O núcleo do PyFWI é elástico; o caso acústico é o limite `vs = 0`.**
Não existe solver acústico separado. A aula 05 comprova: com `vs = 0`,
`‖σxx − σzz‖/‖σxx‖ = 4×10⁻⁷` e `σxz = 0` *exatamente*. A pressão que
`components=0` devolve é `−(σxx+σzz)/2`.

**2. `inpa['acq_type']` precisa ser 1 (superfície) ou 2 (crosswell).**
O valor 0 não levanta erro: mapeia os receptores errado e devolve um
sismograma plausível e errado — sem ápice na *moveout*. Encontrado por
validação cruzada contra o `fwikit` (aula 05).

**3. O adjunto da extensão de bordas é somar, não recortar.**
O modelo é estendido com `mode='edge'` antes de propagar, replicando os pixels
de borda. O gradiente precisa de `Eᵀ`: a sensibilidade acumulada na moldura
tem de ser **dobrada de volta** sobre a borda. Esquecer isso valia **9% de
erro** na derivada direcional — corrigido, a razão passou de 0,9118 para
1,0001 (aula 09).

**4. Os dois códigos discordavam na forma do pulso — e ambos estavam certos.**
Ao passar da formulação de 1ª ordem (velocidade–tensão, PyFWI) para a de 2ª
ordem (pressão, `fwikit`), o termo-fonte vira `(1/K)·∂s/∂t`. As wavelets
efetivas diferem por **uma derivada temporal**. A correlação entre os traços
sai de **+0,12 para +0,85** aplicando essa derivada, sem nenhum parâmetro de
ajuste (aulas 01 e 05).

---

## Verificação

```bash
.venv/bin/python testes/verificar_fisica.py
```

29 verificações que confrontam cada fórmula do curso com sua derivação
analítica ou com uma referência independente — não testam se o código roda,
testam se ele está **certo**:

| Bloco | O que é verificado |
|---|---|
| 1 | Coeficientes de diferenças finitas anulam os momentos de Taylor até a ordem esperada |
| 2 | `C_limite = 2/√(ndim·S)` bate com von Neumann analítico (0,7071 / 0,6124 / 0,5546) |
| 3 | A simulação é estável abaixo do limite CFL e estoura acima |
| 4 | Ricker: pico em `f0`, média zero, `t0 = 1/f0` trunca < 0,1% da amplitude |
| 5 | Reduções elásticas: `vp = √(K/ρ)` com μ=0; Poisson 0,25 → `vp/vs = √3` |
| 6 | **Gradiente adjunto**: derivada direcional (erro 0,02%) e teste de Taylor (ordem de `E₁` = 2,00) |
| 7 | Gradientes de Tikhonov, TV e modelo a priori contra diferenças finitas |
| 8 | Kjartansson: `1/(2Q)` aproxima `tan(πγ/2)`; Q ciclos levam a amplitude a `e^-π` |
| 9 | Distância crítica ≠ distância de cruzamento na refração |

O gradiente adjunto é o item que mais importa, e passa nos três testes padrão:
razão 0,9999–1,0006 nas diferenças finitas pontuais, erro máximo de 0,02% na
derivada direcional (estável em 40× de faixa em α), e ordem 2,00 no teste de
Taylor.

---

## Estrutura

```
.
├── curso_fwi_completo.pdf     # o PDF teórico (56 páginas)
├── menu.py                    # lançador interativo
├── requirements.txt
├── aulas/                     # 15 aulas, uma por arquivo
├── fwikit/
│   ├── compat.py              # remendos para PyFWI em numpy 2 / scipy 1.17
│   ├── aula.py                # framework didático de terminal
│   ├── plot.py                # visualização padronizada
│   ├── acustico.py            # propagador 2D + estado adjunto (do zero)
│   ├── inversao.py            # otimizadores, busca linear, regularização
│   ├── verificacao.py         # testes de gradiente
│   └── metricas.py            # erro relativo, correlação, R²
├── testes/
│   └── verificar_fisica.py    # 29 verificações de física e matemática
├── docs/curso_fwi.tex         # fonte LaTeX do PDF
└── saidas/aulaXX/             # figuras geradas pelas aulas
```

---

## Compatibilidade: por que existe `fwikit/compat.py`

O PyFWI 0.1.10 foi publicado junto com o artigo do autor e congelado no tempo.
Ele importa três símbolos que já não existem:

```python
from numpy.lib.function_base import kaiser       # removido no numpy 2.0
from scipy.optimize.optimize import MemoizeJac   # removido no scipy 1.12
from scipy.optimize.lbfgsb import fmin_l_bfgs_b  # namespace depreciado
```

A alternativa seria rebaixar o ambiente inteiro para numpy 1.x. Preferimos
reconstruir os símbolos ausentes antes do PyFWI ser importado — é o que
`import fwikit` faz automaticamente. Não é preciso fixar versões antigas.

---

## Requisitos

- Python 3.9+
- numpy, scipy, matplotlib
- PyFWI 0.1.10, segyio, h5py
- **OpenCL** para o núcleo do PyFWI (aulas 05, 06, 10, 12). Com GPU, o driver
  já fornece o ICD. Sem GPU, instale um runtime de CPU
  (`pip install pocl-binary-distribution` ou `apt install pocl-opencl-icd`).
  As aulas do Módulo I, III (07–09), IV (11, 13) e V usam o propagador próprio
  e **não precisam de OpenCL**.
- LaTeX com XeLaTeX, apenas se quiser recompilar o PDF:
  `cd docs && xelatex curso_fwi.tex` (duas vezes).

Ambiente em que tudo foi executado e validado: Python 3.11, numpy 2.4,
scipy 1.17, PyFWI 0.1.10, pyopencl 2026.1 sobre NVIDIA RTX 3060.

---

## Referências

Tarantola (1984) · Bunks et al. (1995) · Plessix (2006) ·
Virieux & Operto (2009) · Asnaashari et al. (2013) ·
Virieux et al. (2017) · Kjartansson (1979) · Carcione (2022) ·
Mardan et al. (2023, PyFWI) · Louboutin et al. (2019, Devito)

A lista completa está no Apêndice C do PDF.
