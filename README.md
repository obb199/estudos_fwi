# Curso de Full-Waveform Inversion com PyFWI

Um curso completo de **Full-Waveform Inversion (FWI)**, do problema direto ao
problema inverso, em que teoria e prática andam juntas. A equação da onda é
derivada, discretizada, programada e verificada; depois o mesmo raciocínio é
levado ao PyFWI, uma implementação de produção em GPU, para rodar inversões de
verdade.

O repositório reúne três coisas que se complementam:

- **15 aulas executáveis em Python**, conduzidas no terminal, com teoria,
  experimentos numéricos, figuras, quizzes e exercícios;
- **um PDF teórico de 61 páginas** ([`curso_fwi_completo.pdf`](curso_fwi_completo.pdf)),
  que acompanha as aulas capítulo a capítulo;
- **a teoria completa em três níveis de profundidade**, três volumes
  independentes com os mesmos doze capítulos — do introdutório qualitativo ao
  avançado ([veja abaixo](#a-coleção-em-três-níveis));
- **a caixa de ferramentas `fwikit`**, com um propagador acústico 2D escrito do
  zero em NumPy, o gradiente pelo método do estado adjunto, otimizadores,
  regularização e testes de gradiente.

Todos os números, tabelas e figuras dos PDFs são produzidos pelo código deste
repositório, e uma bateria de 34 verificações confronta as fórmulas do curso
com suas derivações analíticas e com medidas feitas no próprio propagador.

---

## Sumário

- [Para quem é](#para-quem-é)
- [Começo rápido](#começo-rápido)
- [Como uma aula funciona](#como-uma-aula-funciona)
- [Mapa do curso](#mapa-do-curso)
- [O PDF teórico](#o-pdf-teórico)
- [A coleção em três níveis](#a-coleção-em-três-níveis)
- [As duas ferramentas](#as-duas-ferramentas)
- [`fwikit` por dentro](#fwikit-por-dentro)
- [Verificação numérica](#verificação-numérica)
- [Tempo de execução](#tempo-de-execução)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Requisitos e instalação](#requisitos-e-instalação)
- [Recompilando o PDF](#recompilando-o-pdf)
- [Referências](#referências)

---

## Para quem é

O curso foi pensado para quem vai **trabalhar** com FWI (em pesquisa, numa
dissertação ou em aplicação) e precisa entender o método o bastante para
modificá-lo: trocar a física do operador, reescrever o adjunto, diagnosticar
uma inversão que não converge.

Ajuda chegar com:

- cálculo diferencial e integral e álgebra linear básica;
- noções de equações diferenciais parciais e de ondas;
- Python e NumPy no nível de ler e alterar scripts;
- alguma familiaridade com dado sísmico é bem-vinda, mas não obrigatória: a
  aula 04 apresenta o vocabulário de aquisição do zero.

---

## Começo rápido

```bash
git clone https://github.com/obb199/estudos_fwi.git
cd estudos_fwi

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python menu.py            # menu interativo com todas as aulas
.venv/bin/python menu.py 0          # aula 00: diagnóstico do ambiente
.venv/bin/python testes/verificar_fisica.py   # 34 verificações, poucos segundos
```

A aula 00 confere numpy, scipy, matplotlib, PyFWI e OpenCL, aplica a camada de
compatibilidade do PyFWI e roda um teste de fumaça do propagador. Se o OpenCL
não estiver disponível, as aulas 01–04, 07–09, 11, 13 e 14 continuam
funcionando normalmente.

O lançador também aceita:

```bash
.venv/bin/python menu.py --listar   # só lista as aulas
.venv/bin/python menu.py 7          # roda direto a aula 07
.venv/bin/python menu.py todas      # roda todas em modo rápido (teste de fumaça)
```

---

## Como uma aula funciona

Cada aula é um script independente em [`aulas/`](aulas/) e segue sempre a
mesma estrutura:

1. **Cabeçalho** com módulo, duração estimada, pré-requisitos e objetivos.
2. **Seções** que alternam texto, equações, caixas de teoria (`TEORIA`),
   recomendações práticas (`NA PRÁTICA`) e alertas (`ATENÇÃO`).
3. **Experimentos numéricos** que rodam na hora e imprimem resultados e
   tabelas medidas, não copiadas.
4. **Figuras** salvas sempre em `saidas/aulaXX/` como PNG.
5. **Quizzes** de múltipla escolha com gabarito e explicação.
6. **Exercícios** com dicas, para continuar sozinho.
7. **Resumo final** e indicação da próxima aula.

Modos de execução:

| Comando / variável | Efeito |
|---|---|
| `python aulas/aulaXX_*.py` | modo interativo: pausas, quizzes e janelas |
| `--rapido` ou `FWI_RAPIDO=1` | sem pausas, quizzes respondidos automaticamente, sem janelas |
| `--sem-figuras` ou `FWI_SEM_FIGURAS=1` | não abre janelas, só salva os PNG |
| `NO_COLOR=1` | desliga as cores ANSI do terminal |

As aulas **01, 02, 03, 04, 07 e 08** abrem janelas interativas com controles
deslizantes (wavelet de Ricker, dispersão numérica, propagação animada, tiros
de uma aquisição, vale de atração, contribuição de cada tiro ao gradiente)
quando há um display disponível.

---

## Mapa do curso

| # | Aula | O que se aprende | Estudo | Requer |
|---|---|---|---|---|
| **Módulo 0 — Preparação** |||||
| 00 | Ambiente, diagnóstico e o mapa do curso | Diagnóstico do ambiente, por que o PyFWI precisa de uma camada de compatibilidade, as três dificuldades que definem a FWI | 20 min | |
| **Módulo I — Física e numérica da propagação** |||||
| 01 | O regime acústico | Da elastodinâmica em velocidade–tensão à equação acústica; o que a hipótese μ = 0 preserva e descarta; pressão como −(σxx+σzz)/2; o parâmetro m = 1/c²; a wavelet de Ricker, sua banda e o limite de resolução | 60 min | |
| 02 | Diferenças finitas | Estênceis de ordem 2, 4 e 8; condição CFL por von Neumann; dispersão numérica espacial e temporal; pontos por comprimento de onda; receita para dimensionar malha e passo de tempo | 60 min | |
| 03 | Modelagem 2D | O laço de tempo linha a linha; snapshots do campo; bordas absorventes medidas contra uma referência sem borda; superfície livre, múltiplas e *ghost* | 70 min | |
| 04 | Aquisição | Organizações do dado; anatomia de um *shot gather* (direta, refrações, reflexões, difrações); distância crítica e de cruzamento; abertura × profundidade; *aliasing* espacial; ruído; geometrias de superfície, VSP e *crosswell* | 60 min | |
| **Módulo II — PyFWI como ferramenta de produção** |||||
| 05 | PyFWI e a prova do regime acústico | Anatomia de um script PyFWI; o parâmetro `components`; prova numérica de que `vs = 0` torna o tensor de tensões isotrópico; validação cruzada da cinemática contra o `fwikit`; convenções de fonte e de sinal | 70 min | OpenCL |
| 06 | PyFWI por dentro | Referência do dicionário `inpa`; reescalonamento automático de `dt`; CPML medida; custo da ordem espacial; *checkpointing* e memória; a conta de custo de uma FWI | 60 min | OpenCL |
| **Módulo III — O problema inverso e o gradiente** |||||
| 07 | Função objetivo e *cycle skipping* | J(c) varrida em quatro frequências; o critério de meio ciclo; vale de atração medido contra o previsto; o papel do modelo inicial; funcionais além do L2 | 70 min | |
| 08 | O método do estado adjunto | Por que diferenças finitas são inviáveis; derivação pelo Lagrangiano; o gradiente como correlação cruzada de defasagem zero; implementação linha a linha; como ler um gradiente | 90 min | |
| 09 | Verificação do gradiente | Diferenças finitas pontuais, derivada direcional e teste de Taylor; como diagnosticar a causa de uma reprovação pelo padrão do erro | 70 min | |
| 10 | Gradiente no PyFWI e pré-condicionamento | API de gradiente do PyFWI; os artefatos do gradiente bruto; máscara, suavização e compensação de iluminação; o pseudo-Hessiano | 70 min | OpenCL |
| **Módulo IV — Otimização, FWI completa e regularização** |||||
| 11 | Otimização | Descida máxima, gradiente conjugado e l-BFGS; busca linear de Armijo e parabólica; comparação medida por iteração e por custo | 80 min | |
| 12 | A FWI completa: multiescala | Estratégia de Bunks et al. no PyFWI; monoescala × multiescala com o mesmo custo; estudo de robustez ao modelo inicial; como reportar uma FWI | 90 min | OpenCL, execução longa |
| 13 | Regularização | Espaço nulo e amplificação de ruído; Tikhonov × Variação Total com dado ruidoso; regularização implícita; curva em L; modelo a priori e vínculos | 75 min | |
| **Módulo V — Os limites** |||||
| 14 | Os limites da aproximação acústica | Inventário de hipóteses; impedância e ambiguidade velocidade–densidade; elasticidade; atenuação e o modelo constant-Q de Kjartansson; quando ignorar Q causa *cycle skipping*; como escolher a física | 60 min | |

A coluna "Estudo" é o tempo sugerido para ler, rodar e responder. O tempo de
máquina está em [Tempo de execução](#tempo-de-execução).

---

## O PDF teórico

[`curso_fwi_completo.pdf`](curso_fwi_completo.pdf) tem 61 páginas e acompanha
as aulas capítulo a capítulo. Cada capítulo indica as aulas correspondentes,
e as tabelas e figuras vêm das execuções do código.

| Parte | Capítulo | Aulas |
|---|---|---|
| — | Como usar este material | — |
| I — Física e numérica da propagação | 1. O que é FWI, e por que é difícil | 00 |
| | 2. O regime acústico | 01 |
| | 3. Diferenças finitas: estabilidade e dispersão | 02 |
| | 4. Bordas, superfície livre e aquisição | 03, 04 |
| II — PyFWI | 5. PyFWI na prática | 05 |
| | 6. PyFWI por dentro: parâmetros e custo | 06 |
| III — O problema inverso | 7. Função objetivo e *cycle skipping* | 07 |
| | 8. O método do estado adjunto | 08, 10 |
| | 9. Verificação do gradiente | 09 |
| IV — Otimização, FWI completa e regularização | 10. Otimização | 11 |
| | 11. A FWI completa: multiescala | 12 |
| | 12. Regularização | 13 |
| V — Os limites | 13. Os limites da aproximação acústica | 14 |
| Apêndices | A. Checklist de diagnóstico · B. Notação · C. Referências | — |

O texto usa quatro tipos de caixa: **teoria** (o conceito), **prática** (o que
fazer), **atenção** (o detalhe que muda o resultado) e **armadilha** (o erro que
custa tempo). O apêndice A organiza os sintomas mais comuns de uma FWI que dá
errado (J não cai, J cai mas o modelo está errado, só a parte rasa é atualizada,
o campo estoura, cauda oscilatória) com as causas a investigar em cada caso.

---

## A coleção em três níveis

Além do PDF que acompanha as aulas, o repositório traz **a mesma teoria escrita
três vezes**, em profundidade crescente. Não são um resumo e dois
aprofundamentos: são três passagens completas pelo assunto, com **os mesmos
doze capítulos, na mesma ordem, tratando dos mesmos tópicos**. O que cresce é a
complexidade e a completude, nunca o escopo.

| Volume | Nível | Páginas | O que faz |
|---|---|---|---|
| [`fwi_nivel1_introdutorio.pdf`](fwi_nivel1_introdutorio.pdf) | Introdutório | 59 | *O quê* e *por quê*. Ideias, figuras, analogias e conclusões. As fórmulas aparecem, mas são **lidas em palavras**, não derivadas. |
| [`fwi_nivel2_intermediario.pdf`](fwi_nivel2_intermediario.pdf) | Intermediário | 67 | *Como se faz*. Todas as deduções, as versões discretas, os algoritmos, as contas de dimensionamento e exercícios resolvidos. |
| [`fwi_nivel3_avancado.pdf`](fwi_nivel3_avancado.pdf) | Avançado | 67 | *Por que funciona e quando falha*. Espaços de funções, teoria de espalhamento, estrutura da Hessiana, multiparâmetro, funcionais alternativos e inferência. |

Os doze capítulos, idênticos nos três volumes:

| # | Capítulo | # | Capítulo |
|---|---|---|---|
| 1 | O que é FWI | 7 | Verificação do gradiente |
| 2 | Do meio elástico à equação acústica | 8 | Otimização e a Hessiana |
| 3 | Diferenças finitas: estabilidade e dispersão | 9 | A FWI completa: fluxo de trabalho |
| 4 | Bordas, superfície livre e aquisição | 10 | Regularização e informação a priori |
| 5 | Função objetivo e *cycle skipping* | 11 | Os limites da física adotada |
| 6 | O método do estado adjunto | 12 | Diagnóstico |

Isso permite três modos de leitura: **em profundidade crescente** (volume 1
inteiro, depois o 2, depois o 3); **por tópico** (o capítulo 6 dos três
volumes em sequência, por exemplo); ou **por necessidade** (o volume 2 como
texto principal, descendo ao 1 quando uma ideia não fizer sentido e subindo ao
3 quando precisar do detalhe fino). Marcas ao longo do texto apontam o caminho
entre os níveis.

**Toda ferramenta matemática é apresentada antes de ser aplicada.** Onde o
texto vai usar multiplicadores de Lagrange, análise de von Neumann,
transformada de Hilbert, espaços de Hilbert ou inferência bayesiana, uma caixa
*Antes de usar* introduz a ferramenta, fixa a notação e destaca exatamente a
propriedade que será explorada adiante — separando aprender a ferramenta de
aprender a aplicação.

**Os resultados foram conferidos contra a literatura primária.** Fórmulas,
convenções e dados bibliográficos foram confrontados com as fontes originais —
Kjartansson (1979) para constant-Q, Wu & Toksöz (1987) e Sirgue & Pratt (2004)
para a cobertura de número de onda, Tarantola (1986) e Operto *et al.* (2013)
para os padrões de radiação, Tromp *et al.* (2005) para as fontes adjuntas e os
núcleos *banana-doughnut*, Warner & Guasch (2016) para a AWI, Komatitsch &
Martin (2007) para a CPML, Métivier *et al.* (2013) para o Newton truncado e
Schäfer *et al.* (2014) para a transformação 2,5D. As afirmações quantitativas
próprias (limites CFL, banda da Ricker, dispersão do esquema completo, energia
discreta do *leap-frog*, padrões de radiação, respostas dos exercícios) foram
verificadas numericamente.

Os três volumes compartilham o estilo `docs/fwiestilo.sty` e usam o mesmo
conjunto de caixas: **a ideia** (a abertura de cada capítulo em uma frase),
**antes de usar** (o preparo matemático), **teoria**, **prática**, **atenção**,
**armadilha**, **derivação**, **exercícios** e **o que fica** (o fechamento).

---

## As duas ferramentas

O curso usa duas implementações de propósito.

**`fwikit.acustico`** é um propagador acústico 2D escrito do zero em NumPy,
com cerca de 520 linhas, das quais mais da metade é documentação. Não usa GPU
e não pretende ser rápido: é transparente, cada termo da equação está à vista,
e é com ele que o gradiente adjunto é derivado, implementado e verificado.

**PyFWI** ([Mardan et al., 2023](#referências)) é uma implementação de
produção em OpenCL: elástica, em velocidade–tensão com malha intercalada, com
CPML e *checkpointing*. É rápida e mais opaca. No curso ela serve para rodar
inversões de verdade e como referência independente para validar o `fwikit`.
O regime acústico no PyFWI é obtido zerando `vs` no modelo.

Aprender só a primeira deixa você lento; só a segunda deixa você sem saber o
que fazer quando o resultado sai errado.

---

## `fwikit` por dentro

Importar `fwikit` já aplica a camada de compatibilidade do PyFWI e silencia os
avisos do pyopencl.

| Módulo | Conteúdo |
|---|---|
| [`acustico.py`](fwikit/acustico.py) | `Config`, `Geometria`, `Acustico2D` (modelagem, campos, retropropagação), `ricker`, `espectro_amplitude`, `largura_banda`, `cfl`, `cfl_limite`, `dt_maximo`, `pontos_por_comprimento_onda`, `misfit_l2`, `gradiente_adjunto` |
| [`inversao.py`](fwikit/inversao.py) | `descida_maxima`, `gradiente_conjugado` (Polak–Ribière com reinício), `lbfgs` (dois laços), buscas lineares de Armijo e parabólica, passo inicial por escala, projeção em caixa, suavização e compensação de profundidade, `tikhonov`, `variacao_total`, `modelo_a_priori` |
| [`verificacao.py`](fwikit/verificacao.py) | testes de gradiente: diferenças finitas pontuais, derivada direcional, teste de Taylor, direção suave e veredito automático |
| [`metricas.py`](fwikit/metricas.py) | erro relativo do modelo e dos dados, ganho sobre o modelo inicial, correlação, R², RMS e resumo |
| [`plot.py`](fwikit/plot.py) | convenções visuais únicas: modelos, perturbações e gradientes, sismogramas, traços e espectros |
| [`aula.py`](fwikit/aula.py) | o mini-framework das aulas no terminal: seções, caixas, equações, tabelas, quizzes, exercícios |
| [`compat.py`](fwikit/compat.py) | reconstrução dos símbolos que o PyFWI 0.1.10 importa e que não existem mais no numpy 2 e no scipy recente |

### A formulação do propagador

- **Equação:** acústica com densidade constante, escrita na vagarosidade ao
  quadrado m = 1/c², em que ela é linear:
  `m ∂²p/∂t² − ∇²p = s(t) δ(x − xs)`.
- **Tempo:** diferença centrada de 2ª ordem (*leap-frog*).
- **Espaço:** diferenças centradas de ordem 2, 4 ou 8. Limite CFL em 2D:
  0,7071 (O2), 0,6124 (O4) e 0,5546 (O8); `dt_maximo` aplica margem de 10%.
- **Fonte:** delta discreta com peso 1/Δh², de modo que a amplitude não depende
  do refinamento da malha.
- **Bordas:** camada absorvente de Cerjan com intensidade calibrada
  (0,25/n_abs). É um operador diagonal e portanto auto-adjunto, o que mantém o
  gradiente exato.
- **Topo:** absorvente ou superfície livre com p = 0 exatamente em z = 0, pelo
  método das imagens.
- **Adjunto:** o mesmo laço de tempo, com o resíduo de sinal trocado injetado
  nos receptores e invertido no tempo. O gradiente é a correlação de defasagem
  zero `∂J/∂m = Δh² Δt Σ λ ∂²p/∂t²`, levado ao domínio físico pelo adjunto da
  extensão de bordas e convertido para velocidade por `∂m/∂c = −2/c³`.
- **Memória:** o campo direto é guardado inteiro, sem *checkpointing* —
  adequado aos modelos 2D pequenos do curso.

### Exemplo: modelar, calcular o gradiente e inverter

```python
import numpy as np
import fwikit
from fwikit.acustico import (Acustico2D, Config, Geometria, ricker, dt_maximo,
                             misfit_l2, gradiente_adjunto)
from fwikit.inversao import lbfgs, limitar, suavizar
from fwikit.metricas import resumo, imprimir_resumo

# modelo verdadeiro: inclusão rápida num meio homogêneo
nz, nx, dh = 60, 100, 10.0
c_verd = np.full((nz, nx), 2000.0, dtype=np.float32)
zz, xx = np.mgrid[0:nz, 0:nx]
c_verd[(zz - 30) ** 2 + (xx - 50) ** 2 < 8 ** 2] = 2300.0
c_ini = np.full((nz, nx), 2000.0, dtype=np.float32)

# malha, fonte e aquisição de superfície (posições em índices [iz, ix])
dt = dt_maximo(2300.0, dh, ordem=4)
cfg = Config(dh=dh, dt=dt, nt=int(1.0 / dt), ordem=4, n_abs=30, f0=10.0)
solver = Acustico2D(cfg, (nz, nx))
w = ricker(cfg.f0, dt, cfg.nt)
geom = Geometria(fontes=np.array([[3, ix] for ix in (20, 50, 80)]),
                 receptores=np.array([[3, ix] for ix in range(5, 95, 2)]))

# dado "observado", com forma (nt, nr, ns)
d_obs = np.stack([solver.modelar(c_verd, geom, w, i)[0]
                  for i in range(geom.ns)], axis=-1)

# funcional e gradiente no formato que os otimizadores esperam
def f_e_g(c_vec, so_J=False):
    c = np.asarray(c_vec, dtype=np.float32).reshape(nz, nx)
    if so_J:
        return sum(misfit_l2(solver.modelar(c, geom, w, s)[0],
                             d_obs[:, :, s], dt)[0] for s in range(geom.ns))
    J, g, _ = gradiente_adjunto(solver, c, geom, w, d_obs)   # dJ/dc
    g = suavizar(g, 1.5)
    g[:8, :] = 0.0                                          # máscara da superfície
    return J, g.reshape(c_vec.shape)

c_est, hist = lbfgs(f_e_g, c_ini.astype(np.float64).ravel(), n_iter=8,
                    projecao=limitar(1800.0, 2600.0), verboso=False)
imprimir_resumo(resumo(c_est.reshape(nz, nx), c_ini, c_verd))
```

Roda em poucos segundos numa CPU comum. J cai para cerca de 9% do valor
inicial e o erro do modelo, de 2,7% para 1,8%.

---

## Verificação numérica

```bash
.venv/bin/python testes/verificar_fisica.py
```

[`testes/verificar_fisica.py`](testes/verificar_fisica.py) não testa se o
código roda; testa se ele está **certo**. São 34 verificações em 11 blocos,
cada uma confrontando uma fórmula do curso com sua derivação analítica ou com
uma medida independente. Terminam em poucos segundos e devem ser rodadas
depois de qualquer alteração em `fwikit/`.

| Bloco | O que é verificado | Checks |
|---|---|---|
| 1 | Coeficientes de diferenças finitas anulam os momentos de Taylor até a ordem esperada | 3 |
| 2 | `C_limite = 2/√(ndim·S)` coincide com o máximo do símbolo de von Neumann, buscado numericamente em todo k·Δh | 3 |
| 3 | A simulação é estável abaixo do limite CFL e estoura acima | 4 |
| 4 | Ricker: pico do espectro em f0, média nula, atraso t0 = 1/f0 trunca menos de 0,1% da amplitude | 6 |
| 5 | Reduções elásticas: vp = √(K/ρ) só com μ = 0; Poisson 0,25 dá vp/vs = √3 | 2 |
| 6 | **Gradiente adjunto**: derivada direcional (erro de 0,02%) e teste de Taylor (ordem de E₁ = 2,00) | 2 |
| 7 | Gradientes de Tikhonov, Variação Total e modelo a priori contra diferenças finitas | 3 |
| 8 | Kjartansson: 1/(2Q) aproxima tan(πγ/2); Q ciclos levam a amplitude a e^−π | 5 |
| 9 | Distância crítica e distância de cruzamento da refração são grandezas distintas | 1 |
| 10 | Velocidade de fase **medida no propagador** coincide com a relação de dispersão do esquema completo (O4 e O8 perto do CFL); a curva só espacial subestima o erro | 3 |
| 11 | Superfície livre com p = 0 em z = 0, medida pelo tempo do *ghost* contra a fonte-imagem (O4 e O8) | 2 |

A aula 09 aprofunda o bloco 6 com os três testes padrão de gradiente:
diferenças finitas pontuais (razão entre 0,9999 e 1,0006 nos pontos de
gradiente forte), derivada direcional ao longo de 40× de faixa em α e teste de
Taylor.

---

## Tempo de execução

Tempos com `--rapido` no ambiente de referência (CPU comum para o `fwikit`,
NVIDIA RTX 3060 para o PyFWI):

| Aula | Tempo | Aula | Tempo |
|---|---|---|---|
| 00 | ~1 s | 08 | ~4 s |
| 01 | ~1 s | 09 | ~3 s |
| 02 | ~4 s | 10 | ~25 s |
| 03 | ~3 s | 11 | ~35 s |
| 04 | ~5 s | 12 | ~10 min |
| 05 | ~5 s | 13 | ~80 s |
| 06 | ~6 s | 14 | ~1 s |
| 07 | ~40 s | `verificar_fisica.py` | ~5 s |

Sem GPU, as aulas do PyFWI rodam num dispositivo OpenCL de CPU (POCL), mais
devagar. A aula 12 executa duas inversões completas e é, de longe, a mais
longa.

---

## Estrutura do repositório

```
.
├── README.md
├── requirements.txt
├── menu.py                       # lançador interativo das aulas
├── curso_fwi_completo.pdf        # o PDF que acompanha as aulas (61 páginas)
├── fwi_nivel1_introdutorio.pdf   # coleção em três níveis: introdutório
├── fwi_nivel2_intermediario.pdf  #                          intermediário
├── fwi_nivel3_avancado.pdf       #                          avançado
├── aulas/
│   ├── aula00_ambiente.py
│   ├── aula01_regime_acustico.py
│   ├── aula02_diferencas_finitas.py
│   ├── aula03_modelagem_2d.py
│   ├── aula04_aquisicao.py
│   ├── aula05_pyfwi_modelagem.py
│   ├── aula06_pyfwi_parametros.py
│   ├── aula07_problema_inverso.py
│   ├── aula08_estado_adjunto.py
│   ├── aula09_verificacao_gradiente.py
│   ├── aula10_gradiente_pyfwi.py
│   ├── aula11_otimizacao.py
│   ├── aula12_fwi_multiescala.py
│   ├── aula13_regularizacao.py
│   └── aula14_limites_acustico.py
├── fwikit/
│   ├── __init__.py               # aplica o compat ao importar
│   ├── acustico.py               # propagador 2D + estado adjunto
│   ├── inversao.py               # otimizadores, busca linear, regularização
│   ├── verificacao.py            # testes de gradiente
│   ├── metricas.py               # métricas de modelo e de dados
│   ├── plot.py                   # visualização padronizada
│   ├── aula.py                   # framework didático de terminal
│   └── compat.py                 # compatibilidade do PyFWI com numpy 2 / scipy recente
├── testes/
│   └── verificar_fisica.py       # 34 verificações de física e matemática
├── docs/
│   ├── fwiestilo.sty             # estilo compartilhado pelos três volumes
│   ├── curso_fwi.tex             # fonte do PDF que acompanha as aulas
│   ├── fwi_nivel1_introdutorio.tex
│   ├── fwi_nivel2_intermediario.tex
│   └── fwi_nivel3_avancado.tex
└── saidas/
    └── aulaXX/                   # figuras PNG geradas pelas aulas
```

As figuras em `saidas/` são versionadas porque o PDF as inclui diretamente.
Rodar uma aula regenera as suas.

---

## Requisitos e instalação

**Python 3.9 ou mais recente**, com as dependências de
[`requirements.txt`](requirements.txt):

- `numpy`, `scipy`, `matplotlib`;
- `PyFWI==0.1.10`, com `segyio`, `h5py` e `hdf5storage`;
- `pyopencl`.

### OpenCL

O núcleo do PyFWI são kernels OpenCL, exigidos pelas aulas **05, 06, 10 e 12**.

- **Com GPU** (NVIDIA, AMD ou Intel), o driver já fornece o ICD; basta o
  `pyopencl`.
- **Sem GPU**, instale um runtime de CPU:
  `pip install pocl-binary-distribution` ou, no Debian/Ubuntu,
  `sudo apt install pocl-opencl-icd`.

As demais aulas usam só o `fwikit` e não precisam de OpenCL.

### Compatibilidade do PyFWI

O PyFWI 0.1.10 importa três símbolos de namespaces que não existem mais ou
estão depreciados:

```python
from numpy.lib.function_base import kaiser       # removido no numpy 2.0
from scipy.optimize.optimize import MemoizeJac   # removido no scipy 1.12
from scipy.optimize.lbfgsb import fmin_l_bfgs_b  # namespace depreciado
```

Em vez de fixar versões antigas, [`fwikit/compat.py`](fwikit/compat.py)
reconstrói esses símbolos antes de o PyFWI ser importado. Basta `import fwikit`
no início do script — todas as aulas já fazem isso. A aula 00 mostra quais
remendos foram aplicados no seu ambiente.

### Ambiente de referência

Tudo foi executado e validado com Python 3.11, numpy 2.4, scipy 1.17,
matplotlib 3.11, PyFWI 0.1.10 e pyopencl 2026.1 sobre uma NVIDIA RTX 3060.

---

## Recompilando os PDFs

É preciso XeLaTeX. Os fontes ficam em `docs/`, as figuras vêm de `saidas/` e os
PDFs versionados ficam na raiz:

```bash
cd docs
for f in curso_fwi fwi_nivel1_introdutorio \
         fwi_nivel2_intermediario fwi_nivel3_avancado; do
    xelatex -interaction=nonstopmode "$f.tex"   # duas vezes, pelo sumário
    xelatex -interaction=nonstopmode "$f.tex"   # e pelas referências cruzadas
done
cp curso_fwi.pdf ../curso_fwi_completo.pdf
cp fwi_nivel*.pdf ..
```

Os três volumes compartilham `docs/fwiestilo.sty`: mexer nele exige recompilar
os três. Se você alterar o código de uma aula, rode-a antes de recompilar, para
que as figuras dos PDFs reflitam a mudança.

---

## Referências

- Tarantola, A. (1984). *Inversion of seismic reflection data in the acoustic approximation.* Geophysics, 49(8).
- Cerjan, C. et al. (1985). *A nonreflecting boundary condition for discrete acoustic and elastic wave equations.* Geophysics, 50(4).
- Kjartansson, E. (1979). *Constant-Q wave propagation and attenuation.* Journal of Geophysical Research, 84.
- Bunks, C. et al. (1995). *Multiscale seismic waveform inversion.* Geophysics, 60(5).
- Shin, C., Jang, S., Min, D.-J. (2001). *Improved amplitude preservation for prestack depth migration by inverse scattering theory.* Geophysical Prospecting, 49(5).
- Plessix, R.-E. (2006). *A review of the adjoint-state method for computing the gradient of a functional with geophysical applications.* Geophysical Journal International, 167(2).
- Nocedal, J., Wright, S. J. (2006). *Numerical Optimization*, 2ª ed. Springer.
- Virieux, J., Operto, S. (2009). *An overview of full-waveform inversion in exploration geophysics.* Geophysics, 74(6).
- Asnaashari, A. et al. (2013). *Regularized seismic full waveform inversion with prior model information.* Geophysics, 78(2).
- Virieux, J. et al. (2017). *An Introduction to Full Waveform Inversion.* Society of Exploration Geophysicists.
- Louboutin, M. et al. (2019). *Devito (v3.1.0): an embedded domain-specific language for finite differences and geophysical exploration.* Geoscientific Model Development, 12(3).
- Carcione, J. M. (2022). *Wave Fields in Real Media*, 4ª ed. Elsevier.
- Mardan, A., Giroux, B., Fabien-Ouellet, G. (2023). *PyFWI: A Python package for full-waveform inversion and reservoir monitoring.* SoftwareX.

As referências completas, com páginas, estão no apêndice C do PDF.
