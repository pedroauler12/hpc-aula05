# Análise individual: por que o corpus não escala como o pi_mpi

Raphael Silva · Ponderada da Aula 5 (HPC) · cluster de 4 nós com SLURM, MPI e Dask

## 1. Resumo

O mesmo cluster produziu duas curvas de speedup muito diferentes. Com 32 processos, o `pi_mpi` da Aula 3 chega a **21,57×** (eficiência de 67%). O pipeline TF-IDF em Dask, sobre 300.000 resenhas, chega a **1,21×** (eficiência de 4%). O corpus fica abaixo do caso de 1 worker com 2, 4 e 8 workers (0,58×, 0,72× e 0,81×) e só o alcança com 16 (1,01×, valor dentro do ruído entre rodadas, seção 8).

A tese é que a distância entre as curvas não tem causa única. O `pi_mpi` é uma boa régua do cálculo puro: cada processo sorteia pontos sozinho e só comunica no fim. O pipeline roda no mesmo hardware, mas move dados, e isso aparece de duas formas:

1. **Já com 1 worker, 62% do tempo não é cálculo local** (leitura, contagem de documentos por termo, estatísticas finais). Essa fração limita o speedup a cerca de 1,6× se nada mais piorasse.
2. **De 1 para 2 workers, essas etapas ficam mais lentas**, no mesmo nó, onde a rede não participa: `df` vai de 7,3 s para 16,3 s e `stats` de 7,4 s para 23,0 s. A explicação mais plausível é o custo de serializar e redistribuir dados entre processos, que o caso de 1 worker evita. Isso é uma inferência por eliminação, não uma medição (seção 6.1).

Sobre a fração serial: o valor efetivo em 32 workers é **f ≈ 0,82** (Karp-Flatt; o ajuste de Amdahl a p ≥ 2 dá 0,88), e ele inclui o overhead de distribuir. A fração do tempo de 1 worker que não é cálculo local (0,62) é uma cota superior da parte intrinsecamente serial. No `pi_mpi`, f ≈ 0,013. O modelo de Amdahl com f fixo não descreve a curva do corpus (seção 7).

O texto segue a ordem: método (2), a figura (3), o que o hardware entrega (4), o corpus por dentro (5), cada gargalo com o número que o sustenta (6), Amdahl (7), limites (8) e conclusão (9). As inconsistências encontradas nos arquivos do repositório estão no Apêndice A.

## 2. Plataforma e método

| Item | Valor |
|---|---|
| Nós de cálculo | 4 (c1 a c4) mais um master |
| CPU por nó | Intel Core i3-13100T: 4 cores × 2 threads = 8 CPUs lógicas (16 cores físicos e 32 lógicas no total) |
| Memória por nó | 7,6 GB (cerca de 4,1 GB livres) |
| Rede | Gigabit Ethernet (switch isolado, segundo o README) |
| Armazenamento | NFS exportado pelo master; `/tmp` dos nós fica em RAM (nós rodam da RAM, segundo o `job_hello.sbatch`) |
| Software | Rocky Linux 9.8, OpenHPC, Warewulf 4, SLURM 25.11.4, `gnu15` + `openmpi5`; Python 3.11.16, dask/distributed 2026.8.0, pandas 3.0.5, pyarrow 25.0.0, scikit-learn 1.9.1, numpy 2.4.6 |

Até 16 processos há um por core físico; com 32, cada core executa dois (hyperthreading, SMT).

**pi_mpi** (Monte Carlo, 4 bilhões de pontos, xorshift64 com semente por rank). O programa faz um `MPI_Barrier`, um `MPI_Bcast` do N e dois `MPI_Reduce` (a soma dos acertos e o máximo de `t_calc`). Mede `t_total`, `t_calc` (só o laço, máximo entre ranks) e define `t_serial = t_total − t_calc`, que inclui o `Bcast` e os `Reduce`. Portanto **não é serial puro**: é uma cota superior do que não é sorteio.

**Pipeline do corpus** (`pipeline/pipeline.py`). Leitura, tokenização, stopwords, tf e tfidf fazem `persist()` seguido de `wait()`; `df` e `stats` usam `compute()`. `t_calc = tokeniza + stopwords + tf + tfidf` (etapas que operam partição a partição, ditas "estreitas") e `t_serial = t_total − t_calc`, isto é, leitura, `df` e `stats`. As etapas `df` e `stats` são "largas": precisam combinar resultados de todas as partições (shuffle ou redução). Neste texto, `t_serial` é o nome da coluna dos CSVs; ele **contém trabalho paralelo** (por exemplo, o `Counter` de cada partição em `df`), e só a fusão final é serial de fato. O `t_total` começa depois do `wait_for_workers`; a subida do cluster (`t_sobe`) fica fora. O tempo de construir o IDF no driver (cerca de 0,04 s, constante) fica dentro do `t_total` e fora de qualquer etapa; a mesma diferença confirma que o cronômetro das etapas fecha com o total.

**Estatística.** O `pi_mpi` usa o menor de 2 rodadas; o corpus usa a mediana de 3, com rodadas intercaladas entre as configurações e encadeadas por `--dependency=afterany`. Os critérios diferem; no `pi_mpi` as duas rodadas ficam a no máximo 2% uma da outra, então a assimetria não muda a leitura.

**Posicionamento.** `pi_mpi`: 1, 2 e 4 processos em 1 nó; 8 em 2 nós; 16 em 4 nós (4 por nó, `--hint=nomultithread`); 32 em 4 nós, 8 por nó com SMT. Corpus: configurações (nós × workers por nó) 1×1, 1×2, 1×4, 2×4, 4×4 e 4×8, sempre com **128 partições** (tamanho fixo, escalabilidade forte) e uma thread por worker. Dois detalhes que o leitor precisa ter em mente: os workers foram lançados com `--cpu-bind=none` (mapeamento em cores não fixado), e o **scheduler e o cliente rodam no primeiro nó**, junto com os workers. Com 32 workers, c1 executa 8 workers mais scheduler e cliente em 8 CPUs lógicas.

**Ambiente.** Foram necessárias correções que afetam a medição: sem `RealMemory` no `slurm.conf` o SLURM assumia 1 MB por nó e o Dask limitava cada worker a 128 KiB (com `StreamBufferFullError`); a correção foi `RealMemory=7000` e `--memory-limit 0` (a diferença entre `ambiente/sobe_dask_aula04.sh` e `pipeline/sobe_dask.sh`). `/opt` é somente leitura nos nós, então o código vai em `/home`; o pacote Spark conflita com o pandas 3; o ambiente não tem matplotlib. `confere_nos.sh` confere versões iguais nos nós e `hello_dask.py` imprime `CHECKPOINT OK` quando as tarefas se distribuem (o repositório não guarda o log dessa execução).

## 3. A figura: três curvas

![Speedup: corpus TF-IDF em Dask vs pi_mpi vs linear ideal](../resultados/curvas.png)

| p | Linear ideal | pi_mpi | Eficiência | Corpus | Eficiência | pi_mpi ÷ corpus |
|---|---|---|---|---|---|---|
| 1 | 1 | 1,00 | 100% | 1,00 | 100% | 1,0 |
| 2 | 2 | 1,99 | 100% | 0,58 | 29% | 3,4 |
| 4 | 4 | 3,69 | 92% | 0,72 | 18% | 5,1 |
| 8 | 8 | 7,33 | 92% | 0,81 | 10% | 9,0 |
| 16 | 16 | 14,78 | 92% | 1,01 | 6% | 14,6 |
| 32 | 32 | 21,57 | 67% | 1,21 | 4% | 17,8 |

A distância absoluta entre as curvas (1,42; 2,97; 6,51; 13,76; 20,36) só cresce. A figura tem escala linear em y, o que deixa a curva do corpus colada em 1; a tabela de eficiência acima é a leitura mais fiel dela. Três trechos:

- **De 1 a 16**, o `pi_mpi` acompanha a linear ideal (92%) e o corpus fica entre 0,58 e 1,01.
- **O corpus está abaixo de 1 em p = 2, 4 e 8**, com o pior ponto em p = 2. Esse mínimo é a pista principal (seção 6.1).
- **De 16 para 32**, as duas curvas ganham menos que o dobro: 1,46× no `pi_mpi` e 1,20× no corpus, coincidindo com o uso das segundas threads dos cores.

## 4. O que o hardware entrega

Os microbenchmarks da Parte 1 calibram o que esperar de rede e de cálculo antes de olhar o corpus.

**Rede (ping-pong, jobs 46 e 47).** As colunas são tempos de **ida e volta**.

| Tamanho | Mesmo nó | Nós diferentes | Razão | Vazão entre nós |
|---|---|---|---|---|
| 1 B | 0,24 µs | 235,37 µs | 980,7× | ~0,0 MB/s |
| 1 KiB | 0,36 µs | 221,33 µs | 614,8× | 9,3 MB/s |
| 1 MiB | 125,24 µs | 19.560,92 µs | 156,2× | 107,2 MB/s |

A latência de ida entre nós é, portanto, de cerca de 110 a 118 µs, e a vazão de 107,2 MB/s é cerca de 86% dos 125 MB/s de um gigabit. Esses são os limites de qualquer dado que cruze o cabo; o NFS do master partilha o mesmo cabo, mas o ping-pong não mediu o NFS em si.

**Cálculo (`soma_reduce`, N = 2·10⁹).** Tempos do rank 0: 0,478 s (1 processo), 0,478 s (2), 0,241 s (4), 0,129 s (8) e 0,033 s (32). Cada processo faz 1/p do trabalho, então o ideal em 4 processos seria 0,12 s; o tempo observado é o **dobro** do ideal em todo p ≥ 2 (eficiência de 50% em p = 4, 46% em p = 8, 45% em p = 32). O padrão é o de ranks dividindo cores (irmãos SMT): o job não fixou o posicionamento (não usa `--hint=nomultithread`) e o mapeamento não foi verificado. O speedup de 14,5× em 32 processos é, portanto, reflexo do posicionamento, e não um teto do hardware. Isso também vale como aviso: o desempenho neste cluster depende muito de onde os processos caem.

**Colaterais.** `hello_mpi` (job 45) e a distribuição do job 168 confirmaram 2 ranks por nó (8 processos) e 8 por nó (32). O `MPI_Allreduce` do job 167 fez os 4 ranks imprimirem o mesmo total, `2000000001000000000`, igual a N(N+1)/2.

## 5. O corpus por dentro

### Escolha e dados

O corpus é o **Amazon Polarity** (primeiras 300.000 resenhas do arquivo `train-00000-of-00004`; título e conteúdo em inglês; licença Apache 2.0), convertido por `prepara_dataset.py` em 32 arquivos Parquet (83 MB; o original tem 259,8 MB). O tamanho foi limitado pela memória (cerca de 4,1 GB livres por nó). A escolha veio de uma calibração com 1 worker:

| | AG News | Amazon |
|---|---|---|
| Documentos | 120.000 | 300.000 |
| Vocabulário | 61.488 | 196.361 |
| `t_total` | 11,91 s | 27,83 s |
| `t_serial` | 6,28 s (53%) | 17,52 s (63%) |
| `tfidf` | 3,46 s | 1,48 s |
| Subida do cluster | 3,3 s | 4,0 s |

Duas leituras da calibração. Primeiro, com 12 s de execução o AG News deixaria a subida do cluster e o ruído pesando mais que as etapas. Segundo, a fração que não é cálculo local **cresceu** de 53% para 63% quando o corpus foi de 120 mil para 300 mil documentos (são corpora diferentes, então é um indício, não um experimento controlado). O `tfidf` do AG News (3,46 s com menos documentos) é maior que o do Amazon, o que aponta para uma versão anterior do envio do IDF naquela calibração.

### Validade dos dados

Os 18 jobs da série (`logs_serie/corpus-149` a `166`) produzem o mesmo vocabulário (196.361 termos), o mesmo top TF-IDF (`book` = 12501,7; `great` = 7942,8; `movie` = 7406,0) e a mesma distribuição de tamanhos, para qualquer número de workers. O resultado é, portanto, o mesmo em todas as configurações; só o tempo muda. Nenhum log registra aviso, erro ou reinício de worker.

### Anatomia do caso de 1 worker

| Etapa | Tempo (s) | % do `t_total` | Natureza |
|---|---|---|---|
| leitura | 2,25 | 8% | I/O + decodificação |
| tokeniza | 4,72 | 17% | estreita |
| stopwords | 2,31 | 8% | estreita |
| tf | 1,72 | 6% | estreita |
| df | 7,31 | 27% | larga |
| tfidf | 1,48 | 5% | estreita |
| stats | 7,35 | 27% | larga (redução) |

`t_serial` = 16,97 s, **62%** do `t_total` (27,34 s); `t_calc` = 10,21 s. Se a parte que não é cálculo fosse fixa e o cálculo escalasse perfeitamente, o teto seria 1/0,62 ≈ 1,6×.

### Todas as etapas por número de workers (mediana, s)

| workers | leitura | tokeniza | stopwords | tf | df | tfidf | stats | `t_sobe` |
|---|---|---|---|---|---|---|---|---|
| 1 | 2,25 | 4,72 | 2,31 | 1,72 | 7,31 | 1,48 | 7,35 | 2,93 |
| 2 | 1,72 | 2,52 | 1,19 | 1,28 | 16,33 | 0,94 | 23,01 | 2,47 |
| 4 | 1,67 | 1,52 | 1,01 | 0,74 | 12,57 | 0,78 | 19,42 | 2,89 |
| 8 | 2,71 | 1,24 | 0,62 | 0,67 | 10,90 | 0,72 | 16,18 | 32,90 |
| 16 | 4,56 | 0,37 | 0,22 | 0,32 | 7,73 | 0,72 | 12,22 | 36,94 |
| 32 | 2,13 | 0,30 | 0,20 | 0,21 | 7,11 | 1,31 | 11,18 | 32,30 |

As etapas estreitas escalam: de 1 a 16 workers, `tokeniza` ganha 12,6×, `stopwords` 10,3× e `tf` 5,5×. O `tfidf` é a exceção: cai de 1,48 s para 0,72 s e não desce mais. O `t_calc` inteiro vai de 10,21 s para 1,65 s em 16 (6,2×, eficiência de 39%).

O `t_serial` absoluto é 16,97 s (1 worker), 41,10 s (2), 33,72 s (4), 30,23 s (8), 24,93 s (16) e 20,49 s (32). Ele salta 2,4× de 1 para 2 workers e depois escala só parcialmente.

### Quem pesa em 32 workers

| Etapa | Tempo (s) | % do `t_total` (22,51 s) |
|---|---|---|
| stats | 11,18 | 49,7% |
| df | 7,11 | 31,6% |
| leitura | 2,13 | 9,5% |
| tfidf | 1,31 | 5,8% |
| tokeniza + stopwords + tf | 0,71 | 3,2% |

Com 32 workers, **81% do tempo está em `df` e `stats`**, e o cálculo estreito (tokeniza, stopwords, tf) é 3%. É esse quadro, e não o cálculo, que explica o 1,21×. A etapa `stats` (soma de TF-IDF por partição mais o histograma de tamanhos) é a maior e a menos discutida nos dados do repositório; ela é, junto com `df`, o alvo natural de qualquer otimização.

## 6. Os gargalos, um a um

Cada gargalo segue o roteiro: onde aparece, o número que o sustenta e o peso na curva. A fração serial tem sua seção própria, a 7.

### 6.1 Shuffle e serialização

**Onde aparece.** Em `df` (`flatten().frequencies()`) e `stats` (`reduction`), as etapas em que cada partição gera um dicionário parcial (de até 196 mil termos, o vocabulário total) que precisa ser combinado.

**Números.**

- De 1 para 2 workers, no mesmo nó, `df` vai de 7,31 s para 16,33 s (2,2×) e `stats` de 7,35 s para 23,01 s (3,1×). Sem rede envolvida, o que sobra é o custo de serializar os dicionários e movê-los entre processos.
- `df` + `stats` são 54% do `t_total` com 1 worker e 83%, 85%, 81%, 74% e 81% com 2, 4, 8, 16 e 32.
- A variação entre rodadas em p = 2 vem de `df` (17,5 / 13,1 / 16,3 s) e `stats` (23,5 / 20,8 / 23,0 s), não da leitura, o que combina com um custo de comunicação entre processos que varia.
- **Indício indireto.** Um comentário em `pipeline.py` registra, para a etapa `tfidf`, que passar o IDF como argumento das tarefas custava 175 s (como `Future`) ou 7 s com 111 mil termos (e travava com 300 mil documentos), contra 0,17 s com `client.run`. Não há log dessas execuções, as condições não estão registradas, o 0,17 s é menor que o `tfidf` medido na série (0,72 a 1,48 s) e o episódio trata de uma etapa estreita. Serve como sinal de que o custo de transportar objetos Python em Dask é alto, não como prova sobre `df` e `stats`.

**Peso.** É a maior parte da distância entre as curvas (81% do tempo em 32 workers).

**Ressalva.** Shuffle, serialização e transporte não foram medidos isoladamente: estão dentro de `df` e `stats`. O argumento é por eliminação (mesmo nó, etapas largas, sem rede). Falta também explicar por que o caso de 1 worker não paga: a hipótese é que, com um único processo, os dados permanecem no mesmo processo e não há o que serializar; não foi verificada. O pipeline está na versão 1, com `frequencies` e `reduction` nos parâmetros padrão do Dask.

### 6.2 I/O no NFS

**Onde aparece.** Em `leitura`, que lê 83 MB de Parquet do NFS do master. A etapa inclui também o reparticionamento de 32 arquivos em 128 partições, a conversão para bag e a decodificação, então não é I/O puro.

**Números.**

- A 107,2 MB/s, 83 MB levariam cerca de 0,8 s; a leitura mediana fica entre 1,67 s e 4,56 s, de 2 a 6 vezes isso. O excesso não é só banda.
- A sequência por workers é 2,25 / 1,72 / 1,67 / 2,71 / 4,56 / 2,13 s: não é monótona. Sobe de 4 para 8 (+62%) e para 16 workers, onde chega a **17%** do `t_total`, e volta a 2,13 s em 32.
- Em p = 16 a leitura varia entre rodadas de 4,56 s, 2,12 s e 6,44 s, e essa variação (amplitude de 4,3 s) acompanha a de todo o `t_total` do ponto (amplitude de 4,2 s). É um comportamento compatível com vários nós disputando o mesmo servidor, o que é uma hipótese, já que não medimos o servidor.

**Peso.** Moderado: de 8% do tempo com 1 worker a 17% em 16. Explica a instabilidade do ponto de 16 workers mais do que a forma geral da curva.

**Ligação com a Parte 1.** A resposta 4.4 (item 5) calcula o que aconteceria se cada rank do `pi_mpi` lesse 200 MB do mesmo servidor: cerca de 1,9 s independentes de p, portanto tempo serial. Tratando esses 1,9 s como acréscimo fixo a f (método aproximado), f sobe de 1,3% para cerca de 6,7%, o teto de 79 para cerca de 15 e as previsões de Amdahl passam a S(16) ≈ 8,0 e S(32) ≈ 10,4, contra 14,8 e 21,6 medidos sem essa leitura. O texto da 4.4 chega a 6,6% e 10,5× usando 1866 ms (200 MB ÷ 107,2 MB/s).

### 6.3 Rede gigabit

**Onde aparece.** Só com mais de um nó: a partir de 8 workers.

**Números.**

- Cruzar para o segundo nó (4 para 8 workers) tem efeitos claros em duas medidas: a leitura sobe de 1,67 s para 2,71 s e `t_sobe` vai de 2,9 s para 32,9 s.
- Em `df` e `stats`, porém, o tempo continua caindo (12,57 para 10,90 s e 19,42 para 16,18 s, quedas de 13% e 17%). Como o número de workers dobrou, a queda é bem menor que os cerca de 50% que um escalamento sem penalidade daria, então **pode haver uma penalidade de rede mascarada pelo paralelismo**. Não há medição de tráfego para separá-las.
- Uma ordem de grandeza ajuda: um dicionário de 196 mil termos por partição, a poucas dezenas de bytes por entrada, ocupa alguns MB; a 107 MB/s, mover o conjunto de dicionários de 128 partições custa da ordem de segundos, o que é compatível com a fração dos tempos de `df` e `stats` que sobrevive após o salto de 1 para 2 workers. Isso é estimativa, não medição.
- `t_sobe` é 2,5 a 3,1 s com 1 nó e 32 a 37 s com 2 ou 4 nós, sem diferença entre 2 e 4 nós nem entre 4 e 8 workers por nó. Um platô desse tipo sugere uma constante (por exemplo, espera de conexão ou timeout), não uma limitação de banda. A causa não foi verificada e o tempo fica fora do `t_total`.

**Peso.** Secundário frente à serialização (o salto 1→2 acontece dentro de um nó), mas provavelmente presente a partir de 8 workers. É o próximo limite se a serialização for reduzida.

### 6.4 Hyperthreading a partir de 16 workers

**Onde aparece.** De 16 para 32 workers, quando cada core passa a ter dois.

**Números.**

- No `pi_mpi`, dobrar os processos rende 1,46× (2,36 s para 1,62 s), com a eficiência caindo de 92% para 67%. O ganho típico de SMT é de 10% a 50%, não 100%.
- No corpus, o `t_total` vai de 26,98 s para 22,51 s (1,20×). Esse número depende do ponto de 16 workers, que é ruidoso: com o menor tempo de p = 16 (23,67 s), o ganho seria 1,05×.
- Pelas medianas das etapas, a maior parte do ganho vem da leitura (−2,4 s), e não do cálculo. As três etapas estreitas sem o `tfidf` somam 0,91 s em 16 workers e 0,71 s em 32 (1,28×).
- O `t_calc` sobe de 1,65 s para 2,01 s; isso vem do `tfidf` (0,72 s para 1,31 s, +0,59 s), compensado em parte pelas outras três etapas (−0,20 s). Uma hipótese é o custo de entregar o IDF a 32 workers em vez de 16 (`client.run` alcança todos); não foi testada.
- Um confundidor do ponto 32: c1 roda 8 workers mais scheduler e cliente em 8 CPUs lógicas, o que pode atrasar esse nó.

**Peso.** No `pi_mpi`, o SMT é o que separa a curva da linear em 32 (sem ele, 14,78 em 16 processos). No corpus, o efeito é pequeno e misturado com ruído; a explicação por SMT é plausível, mas não isolada.

## 7. Fração serial e a lei de Amdahl

A lei diz que S(p) = 1 / (f + (1 − f)/p), com teto 1/f. O fator de Karp-Flatt, e(p) = (1/S − 1/p) / (1 − 1/p), é a fração serial efetiva medida em cada ponto: se ele fosse constante, f seria fixa.

### 7.1 pi_mpi

O ajuste de mínimos quadrados de 1/S contra 1/p dá **f = 0,0127** (pontos 1 a 16) e **f = 0,0136** (todos), com tetos de 79 e 74. Com f = 0,0127, o modelo prevê S(16) = 13,44 e S(32) = 22,96, contra 14,78 e 21,57 medidos (−9% e +6%).

O bom encaixe aparente esconde um problema. A eficiência é 99,7% em p = 2 e cai para 92,3% em p = 4, com `t_serial` igual a zero e tudo no mesmo nó; depois fica em 92% até 16. Amdahl com f fixo prevê uma queda gradual, não um degrau. O f de 1,3% é, portanto, sobretudo a assinatura desse degrau entre 2 e 4 processos, e não de uma fração serial no código (o `t_serial` medido é no máximo 0,009 s em 34,87 s). Causas possíveis são a frequência do processador (turbo e limite de potência do i3-13100T) e a contenção de memória ou cache; nenhuma foi medida. O `t_calc` é o máximo entre ranks, então desbalanceamento não explica o resíduo. O f = 0,0127 a 0,0136 serve como resumo descritivo do `pi_mpi`, não como medida de serialização.

### 7.2 Corpus

| Método | f | Observação |
|---|---|---|
| `t_serial / t_total` com 1 worker | 0,62 | cota superior da parte serial; contém trabalho paralelo de `df` e `stats` |
| Ajuste de Amdahl, todos os pontos | 1,14 (truncado em 1) | degenera |
| Ajuste de Amdahl, p ≥ 2 | 0,88 | intercepto livre; ver abaixo |
| Karp-Flatt em p = 2, 4, 8, 16, 32 | 2,45; 1,51; 1,26; 0,99; 0,82 | fração efetiva com overhead |

O que essas linhas dizem:

1. **O modelo de Amdahl não se ajusta ao corpus.** O ajuste com f fixo falha de dois modos: usando todos os pontos o intercepto passa de 1, e nos pontos p ≥ 2 o 0,88 é o intercepto de uma reta livre, que prevê S(32) = 1,06 e S(2) = 1,06 (o medido em p = 2 é 0,58) e nem satisfaz S(1) = 1. Impondo S(1) = 1 o ajuste também dá f > 1. O teto de 1,13× ali não é uma previsão do modelo, é 1/f de um ajuste que não descreve os dados. O speedup medido de 1,21× está acima dele, o que mostra o desencaixe, não coerência.
2. **A razão é que o custo não paralelo não é constante.** O `t_serial` salta de 16,97 s para 41,10 s ao passar de 1 para 2 workers e depois cai para 20,49 s em 32; o modelo supõe um valor fixo.
3. **O Karp-Flatt é a estimativa mais informativa.** Ele decresce de 2,45 (maior que 1, impossível como fração, sinal de que o tempo aumentou em vez de diminuir) até 0,82 em 32 workers. A queda é em parte aritmética (S < 1 em p = 2 a 8) e indica que o custo extra não é um custo fixo estrito: há um salto ao sair do caso de 1 worker e, depois, escala parcial (o `t_total` cai de 47,2 s para 22,5 s de p = 2 a 32, e `df` + `stats` caem de 39,3 s para 18,3 s).

**Estimativa final.** A fração serial efetiva do pipeline com 32 workers é **f ≈ 0,82** (Karp-Flatt), próxima do 0,88 do ajuste; ambos incluem o overhead de distribuir. A fração intrinsecamente serial é **no máximo 0,62** (o que não é cálculo local no caso de 1 worker, que já contém trabalho paralelo) e, de fato, menor que isso; ela não foi isolada. A fusão final das reduções é serial de fato, mas roda como tarefa em um worker e, no driver, só resta o IDF (cerca de 0,04 s). O resultado é que, **por construção do pipeline, um terço a dois terços do problema escala mal**, e o custo de distribuir faz o resto. Comparado ao `pi_mpi` (f ≈ 0,013, já discutido), a fração efetiva do corpus é da ordem de 50 a 65 vezes maior (0,62 ÷ 0,0127 e 0,82 ÷ 0,0127, tomando o mesmo f de 1 a 16).

A parte que é de cálculo escala de forma razoável (6,2× com 16 workers, 39%). O gargalo não é o pipeline ser inerentemente serial, e sim o trabalho que escala (10,2 s de 27,3 s) ser pequeno perto do que move dados.

## 8. Robustez e limites

**Variação entre rodadas** (dispersão = (máx − mín) ÷ mediana):

| workers | `t_total` por rodada (s) | Dispersão |
|---|---|---|
| 1 | 27,03 / 27,34 / 27,42 | 1,4% |
| 2 | 48,63 / 41,45 / 47,16 | 15,2% |
| 4 | 37,75 / 36,87 / 40,69 | 10,1% |
| 8 | 32,18 / 33,93 / 33,58 | 5,2% |
| 16 | 26,98 / 23,67 / 27,90 | 15,7% |
| 32 | 22,65 / 22,50 / 22,51 | 0,6% |

Os extremos são estáveis; 2 e 16 workers variam muito. Em p = 16, o speedup é 1,01 pela mediana e 1,14 pelo menor tempo. Portanto "o corpus só alcança o caso de 1 worker em 16 workers" depende do critério estatístico, e o 1,01 não é distinguível de 1. O resultado de 32 workers (1,21× pela mediana, 1,20× pelo mínimo) é robusto.

**Sensibilidade ao posicionamento.** A série de 09/09 do `pi_mpi` (`speedup_reconstruido_0909.csv` e `speedup_0909_concorrente.csv`) foi descartada por rodar jobs concorrentes e sem posição controlada, mas é um contraste útil: 8 processos deram 5,56× em 1 nó (6,26 s), 7,33× em 2 nós (série final, 4,76 s) e 7,48× em 4 nós com 2 por nó (4,66 s); 16 processos em 2 nós deram 10,92×, contra 14,78× em 4 nós. O mesmo número de processos rende diferente conforme se divide o core (SMT) ou se espalha por nós, o que combina com o que o `soma_reduce` mostrou na seção 4.

**Limites.**

- Não há baseline serial puro (sem Dask); o "1 worker" inclui o overhead do Dask. Isso afeta a curva do corpus, e com um baseline mais rápido o speedup seria menor.
- Um corpus, um cluster e uma única versão do pipeline. A calibração indica que a fração não-cálculo cresce com o tamanho do corpus (53% para 63%), o que, se confirmado, **desfavorece** a ideia de que um corpus maior diluiria o custo de distribuir (Gustafson); não foi testado com controle.
- Não foram medidos memória, tráfego de rede, tamanho das mensagens serializadas nem o shuffle isolado. As atribuições das seções 6.1 e 6.3 são por eliminação e as das 6.2 e 6.4 incluem hipóteses, marcadas como tais.
- O `pi_mpi` é um laço sem tráfego de memória; ele calibra o cálculo puro, mas não limita o que cargas Python com alocação, scheduler e cliente dividindo um nó conseguem.
- A partição de 128 partições em 32 workers dá 4 por worker, o que pode causar efeitos de cauda (não medidos).

## 9. Conclusão

O `pi_mpi` e o corpus rodaram no mesmo cluster e diferem no que importa para escalar. O primeiro calcula em cada processo e comunica um número: fica em 92% de eficiência até 16 processos (com um degrau entre 2 e 4 que não expliquei) e cai para 67% com as segundas threads dos cores. O segundo já gasta 62% do tempo fora do cálculo local com 1 worker; a distribuição piora essa parte de 1 para 2 workers e só a recupera parcialmente até 32.

Por peso, o que separa as curvas é: a serialização e o shuffle nas etapas `df` e `stats` (81% do tempo em 32 workers), a fração não paralela de base, a leitura no NFS (8% a 17% do tempo), a rede (presente a partir de 8 workers e na subida do cluster) e o hyperthreading (que decide o último ponto do `pi_mpi` e é menos claro no corpus). A fração serial efetiva do pipeline é f ≈ 0,8 e a intrínseca é, no máximo, 0,62, contra 0,013 (descritivo) no `pi_mpi`. O modelo de Amdahl com f fixo não descreve o corpus, porque o custo não paralelo salta ao entrar no regime distribuído.

Próximos passos, pelo que a evidência sustenta:

1. **Medir antes de otimizar.** Usar o dashboard do Dask (porta 8787) para separar shuffle, serialização e transporte em `df` e `stats`, e registrar tráfego de rede e `t_sobe` por nó.
2. **Atacar `df` e `stats`.** Testar `split_every` e `reduction` com parâmetros diferentes do padrão (o Dask já combina em árvore) e reduzir o que cada partição envia (por exemplo, filtrar termos raros antes de combinar).
3. **Isolar a leitura.** Copiar o corpus para o disco local dos nós ou usar cache por nó, e repetir a série.
4. **Controlar SMT e posicionamento.** Repetir 32 workers sem SMT (mais nós, se houver) e com `--cpu-bind`, e tirar scheduler e cliente dos nós de trabalho.
5. **Corpus e baseline.** Medir um corpus de tamanho diferente com as mesmas condições e acrescentar um baseline serial sem Dask.

## 10. Reprodução

```bash
# Gráfico das três curvas (na raiz do repositório)
python3 pipeline/plota_curvas.py resultados/speedup_corpus.csv resultados/speedup.csv

# Speedup, Karp-Flatt, medianas por etapa e ajustes de Amdahl (1-16 e 1-32) do corpus
python3 pipeline/analisa_corpus.py resultados/corpus_bruto.csv resultados/speedup_corpus.csv

# Ajuste de Amdahl do pi_mpi (grava speedup.png no diretório atual)
cd parte1/codigo_mpi && python3 analisa_speedup.py ../resultados/speedup.csv
```

O ajuste a p ≥ 2 (f = 0,8845), os speedups por etapa e as decomposições desta análise foram calculados à parte com `numpy` sobre `resultados/corpus_bruto.csv`: `np.polyfit(1/p, 1/S, 1)` sobre os pontos indicados, com S = mediana de `t_total` de 1 worker ÷ mediana de `t_total` de p workers. Dados brutos: `resultados/corpus_bruto.csv` (18 execuções, jobs 149 a 166 em `logs_serie/`), `resultados/speedup_bruto.csv` (2 rodadas, jobs 134 a 145), microbenchmarks em `parte1/resultados/` e calibração em `calibracao/`.

## Apêndice A. Inconsistências encontradas no repositório

- **Três versões do `soma_reduce`.** Jobs 48 a 52: só o rank 0 imprime "N =". Jobs 53 e 62: todos os ranks imprimem "N =", e só o rank 0 imprime o total (`soma-62.out` não tem linha de Allreduce). Job 167: todos imprimem o total (Allreduce). O arquivo `tabelas-blocos-2-e-3.md` diz que o job 62 já tinha o Allreduce, o que contradiz `soma-62.out`. O tempo do laço é o mesmo nas três versões.
- **`t_serial` do `pi_mpi`.** A resposta 4.4 traz 0,0040 / 0,0069 / 0,0074 s (8, 16, 32 processos: menor `t_serial` por rodada); `speedup.csv` traz 0,0047 / 0,0085 / 0,0078 (linha de menor `t_total`). A conclusão (cresce ao cruzar nós) vale nos dois casos, mas em `speedup.csv` o valor de 32 fica abaixo do de 16.
- **Previsão de Amdahl da 4.4.** O texto dá S(32) = 22,95 com f = 0,013; o recálculo dá 22,96 (f = 0,0127) e 22,51 (f = 0,0136). O `analisa_speedup.py` ajusta todos os pontos: f = 0,014, teto 73,6 e S(32) = 22,52.
- **Correção de NFS da 4.4.** Esta análise usa f + 1,9/34,87 = 6,7% e S(32) ≈ 10,4; a 4.4 usa 1866 ms e chega a 6,6% e 10,5×.
- **Comentário do `pipeline.py`** cita um teste com 111 mil termos que não corresponde a nenhum vocabulário medido e não tem log em `calibracao/`.
- **Arquivo ausente.** O `relatorio-aula03.pdf`, citado em `tabelas-blocos-2-e-3.md`, não está no repositório.
- **Duplicatas.** `speedup.csv`, `speedup_bruto.csv` e `speedup_pi_mpi.png` existem em `resultados/` e em `parte1/resultados/` (conteúdos idênticos).
