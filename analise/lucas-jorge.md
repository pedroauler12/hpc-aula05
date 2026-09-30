# Análise Individual: Lucas Ramenzoni Jorge

HPC, Inteli 2026.2 (Prof. João Luisi). Atividade Ponderada da Aula 5.  
Os dados quantitativos apresentados são provenientes das medições consolidadas pelo grupo e registrados na pasta `../resultados/`. A interpretação analítica, a avaliação teórica e a discussão dos gargalos apresentadas a seguir refletem minha análise individual.

---

## 1. Resumo Executivo

* **Corpus de Dados:** Processamento de 300.000 resenhas textuais extraídas do dataset *Amazon Polarity*.
* **Pipeline de PLN:** Implementado em Dask com as etapas sequenciais de leitura, tokenização, remoção de stopwords, Term Frequency (TF), Document Frequency (DF), TF-IDF e estatísticas globais do vocabulário.
* **Estatística da Medição:** Para o pipeline sobre o corpus, cada ponto experimental foi medido em 3 rodadas, utilizando-se a mediana dos tempos. Para a aplicação `pi_mpi` da Aula 3, utilizou-se o menor tempo obtido entre 2 rodadas com posição de processos rigidamente controlada.
* **Comportamento do Escalamento:** O speedup do pipeline sobre o corpus permaneceu **abaixo de 1,00× nas configurações de 2 a 8 workers** (0,58×, 0,72× e 0,81×) e atingiu um teto modesto de **1,21× com 32 workers**. Em contrapartida, a aplicação CPU-bound `pi_mpi` alcançou **21,57× de speedup** sob a mesma infraestrutura.
* **Gargalo Principal:** As etapas estreitas e highly paralelizáveis (tokenização, stopwords, TF e TF-IDF) escalam de forma eficiente. Contudo, as etapas de agregação global (`df` e `stats`) pioram acentuadamente logo na transição de 1 para 2 workers dentro do mesmo nó físico, demonstrando que o gargalo primário reside no custo de **serialização e transferência de dicionários de alto vocabulário (~196 mil termos)** e no gerenciamento do framework, e não na capacidade da rede física.
* **Modelagem Teórica:** Para o `pi_mpi`, a Lei de Amdahl ajusta-se com fração serial de $f \approx 1{,}3\%$ e teto teórico $S_{\max} \approx 78{,}5\times$. Para o pipeline sobre o corpus, o ajuste clássico de Amdahl degenera ($f = 1{,}0$) devido aos valores de speedup inferiores a 1. A métrica de Karp-Flatt $e(p)$ decresce de 2,45 (2 workers) para 0,82 (32 workers), provando que a limitação não decorre de uma fração serial fixa, mas de um overhead constante de comunicação/serialização introduzido na saída do modo uniprocesso.

---

## 2. Metodologia e Infraestrutura

### 2.1 Especificações do Cluster
* **Nós Computacionais:** 1 nó Master e 4 nós de computação (`c1` a `c4`) rodando Rocky Linux 9.8 via OpenHPC, Warewulf 4 e SLURM 25.11.4.
* **Processadores por Nó:** Intel Core i3-13100T contendo 4 núcleos físicos e 2 threads por núcleo via SMT/Hyperthreading, totalizando 8 CPUs lógicas por nó.
* **Capacidade Total do Cluster:** 16 núcleos físicos e 32 CPUs lógicas nos 4 nós computacionais.
* **Memória RAM:** 7.627 MB totais por nó, com o sistema operacional mantido em RAM via vfs-image, resultando em ~4,1 GB livres por nó.
* **Rede:** Gigabit Ethernet em switch isolado, apresentando latência RTT de ping-pong de 235,37 µs (1 B) e vazão útil de 107,2 MB/s para blocos de 1 MiB (~85,8% do limite teórico da interface).

### 2.2 Configuração dos Experimentos
* **Aplicação `pi_mpi`:** Submetida via SLURM com alocação explícita (`--hint=nomultithread` até 16 processos para alocar 1 processo por núcleo físico; e `--ntasks-per-core=2` em 32 processos para ativar SMT).
* **Pipeline de PLN em Dask:** Executado com 4 workers de 1 thread por nó para contornar a contenção do Global Interpreter Lock (GIL) do CPython no processamento textual.
* **Particionamento:** O corpus foi fixado em 128 partições para todas as configurações de escala.
* **Distribuição do Vetor IDF:** O dicionário contendo os pesos IDF (~196 mil termos) foi injetado nos workers utilizando `client.run`, reduzindo o tempo dessa distribuição de vários minutos para 0,17 s.

---

## 3. Resultados Experimentais

### 3.1 Tabela Comparativa de Speedup e Eficiência

A tabela abaixo compara o desempenho do pipeline sobre o corpus (mediana de 3 rodadas) contra a aplicação `pi_mpi` (menor tempo de 2 rodadas):

| Workers / Procs | Nós | Corpus $t_{\text{total}}$ (s) | Speedup Corpus | Eficiência Corpus | Speedup `pi_mpi` | Eficiência `pi_mpi` | Karp-Flatt $e(p)$ Corpus |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 1 | 27,34 | 1,00× | 100% | 1,00× | 100% | — |
| **2** | 1 | 47,16 | 0,58× | 29% | 1,99× | 100% | 2,45 |
| **4** | 1 | 37,75 | 0,72× | 18% | 3,69× | 92% | 1,51 |
| **8** | 2 | 33,58 | 0,81× | 10% | 7,33× | 92% | 1,26 |
| **16** | 4 | 26,98 | 1,01× | 6% | 14,78× | 92% | 0,99 |
| **32** | 4 | 22,51 | 1,21× | 4% | 21,57× | 67% | 0,82 |

![Gráfico Comparativo de Speedup](../resultados/curvas.png)

### 3.2 Descomposição do Tempo de Execução por Etapa (Mediana em Segundos)

| Workers | Leitura (NFS) | Tokenização | Stopwords | TF | **DF (Larga)** | TF-IDF | **Stats (Larga)** | Subida Cluster ($t_{\text{sobe}}$) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 2,25 s | 4,72 s | 2,31 s | 1,72 s | **7,31 s** | 1,48 s | **7,35 s** | 2,93 s |
| **2** | 1,72 s | 2,52 s | 1,19 s | 1,28 s | **16,33 s** | 0,94 s | **23,01 s** | 2,47 s |
| **4** | 1,67 s | 1,52 s | 1,01 s | 0,74 s | **12,57 s** | 0,78 s | **19,42 s** | 2,89 s |
| **8** | 2,71 s | 1,24 s | 0,62 s | 0,67 s | **10,90 s** | 0,72 s | **16,18 s** | 32,90 s |
| **16** | 4,56 s | 0,37 s | 0,22 s | 0,32 s | **7,73 s** | 0,72 s | **12,22 s** | 36,94 s |
| **32** | 2,13 s | 0,30 s | 0,20 s | 0,21 s | **7,11 s** | 1,31 s | **11,18 s** | 32,30 s |

---

## 4. Discussão dos Gargalos de Desempenho

A acentuada divergência entre as curvas do `pi_mpi` e do pipeline decorre da diferença fundamental em seus perfis computacionais. O `pi_mpi` é um código numérico CPU-bound que realiza pouquíssima comunicação intermediária e nenhuma leitura de disco durante o cálculo. Por sua vez, o pipeline de PLN realiza I/O distribuído, movimentação de dados e consolidação global de vocabulários.

### 4.1 Divergência de Desempenho

* **Comportamento das Etapas Estreitas:** As etapas puramente locais de transformação de texto escalam com o aumento de workers. A tokenização cai de 4,72 s (1 worker) para 0,30 s (32 workers), a remoção de stopwords reduz de 2,31 s para 0,20 s e o cálculo de TF reduz de 1,72 s para 0,21 s. O tempo computacional paralelizável combinado ($t_{\text{calc}}$) cai de 10,21 s para 2,01 s.
* **Dominância da Parcela Não Escalável:** No entanto, $t_{\text{calc}}$ representa apenas 37,3% do tempo total com 1 worker (10,21 s de 27,34 s) e cai para 8,9% com 32 workers (2,01 s de 22,51 s). A maior parte do tempo de execução é consumida pela fração $t_{\text{serial}}$ (I/O, `df` e `stats`), que salta de 62,1% com 1 worker para mais de 80% a partir de 2 workers.

### 4.2 Shuffle e Serialização (O Gargalo Dominante)

* **Onde Aparece na Curva:** Nas etapas de agregação global `df` (contagem de frequência de documentos) e `stats` (soma de TF-IDF por termo).
* **Número que Prova a Natureza do Custo:** Já com 1 worker, `df` (7,31 s) e `stats` (7,35 s) somam 14,66 s (53,6% do tempo total). Ao passar de **1 para 2 workers no mesmo nó**, o tempo da etapa `df` sobe de 7,31 s para **16,33 s** e a etapa `stats` dispara de 7,35 s para **23,01 s**, elevando a soma dessas duas etapas para 39,34 s (83,4% de $t_{\text{total}}$).
* **Análise do Gargalo:** Como essa piora substancial ocorre dentro de um único nó computacional, exclui-se a rede física como causa. O custo é gerado pela **serialização/deserialização de objetos em CPython (pickle)** para movimentação de dicionários intermediários contendo ~196 mil termos únicos entre os processos do Dask e pela coordenação da agregação.
* **Reforço com a Adição de Nós:** Ao cruzar de 1 para 2 nós (4 → 8 workers), as etapas `df` e `stats` **melhoram** (12,57 s → 10,90 s e 19,42 s → 16,18 s). Isso confirma que o barramento de rede não é a causa do achatamento, mas sim o custo fixo de estruturar a redução entre processos Python.

### 4.3 I/O no Sistema de Arquivos Compartilhado (NFS)

* **Onde Aparece na Curva:** Na etapa inicial de leitura dos arquivos Parquet.
* **Número que Prova:** Os 83 MB do corpus levariam cerca de **0,77 s** para serem transmitidos no cabo Gigabit do nó master (83 MB ÷ 107,2 MB/s medidos no ping-pong). Contudo, a etapa de leitura mede entre **1,67 s e 4,56 s** (chegando a 4,56 s com 16 workers, onde respondeu por 16,9% do tempo total).
* **Análise do Gargalo:** O tempo não diminui com mais workers porque todas as instâncias de leitura concorrem pela mesma interface de rede e pelo mesmo disco do nó master onde o NFS está hospedado. Trata-se de um gargalo secundário que compõe a parcela não escalável.

### 4.4 Infraestrutura de Rede Gigabit

* **Valores Medidos:** Latência RTT de 235,37 µs e largura de banda útil de 107,2 MB/s.
* **Impacto no Pipeline:** Na volumetria testada (83 MB), a rede não atingiu saturação física. Isso é evidenciado pelo fato de que o tempo total **melhora** ao passar de 4 workers em 1 nó (37,75 s) para 8 workers em 2 nós (33,58 s). O único ponto em que se nota efeito de transferência na rede ocorre na etapa `tfidf` com 32 workers, onde o tempo sobe de 0,72 s para 1,31 s devido ao envio concorrente do vetor de IDF para 32 trabalhadores.

### 4.5 Hyperthreading (SMT) a partir de 16 Workers

* **No `pi_mpi`:** Mantém eficiência de 92% até 16 processos (1 processo por núcleo físico) e cai para **67% em 32 processos**, fornecendo ganho de apenas **1,46×** na passagem de 16 para 32 processos (speedup de 14,78× para 21,57×).
* **No Pipeline sobre o Corpus:** Ao passar de 16 para 32 workers, o speedup total sobe ligeiramente de 1,01× para 1,21×. No entanto, as etapas computacionais paralelizáveis ($t_{\text{calc}}$) **pioram de 1,65 s para 2,01 s**.
* **Análise do Gargalo:** A divisão de unidades de execução e estruturas de cache por duas vCPUs lógicas no mesmo núcleo físico prejudica o desempenho do código Python no processamento de strings (operação com acesso intensivo à RAM e desvios de fluxo), demonstrando baixa aderência ao Hyperthreading.

### 4.6 Subida do Cluster ($t_{\text{sobe}}$)

* **Comportamento Registrado:** O tempo para que o Dask suba e disponibilize todos os workers no cluster varia de **2,47 s a 2,93 s em 1 nó** e salta para **32,30 s a 36,94 s com 2 ou 4 nós**.
* **Análise:** Essa latência decorre da inicialização dos interpretadores e do carregamento dos pacotes Python mantidos no NFS ao cruzar a rede entre nós. Esse tempo foi medido à parte e mantido fora do tempo útil de processamento ($t_{\text{total}}$).

---

## 5. Estimativa da Fração Serial e Análise Teórica

A Lei de Amdahl estabelece o speedup limite por:
$$S(p) = \frac{1}{f + \frac{1-f}{p}}$$

### 5.1 Aplicação `pi_mpi`
* **Ajuste Numérico (1 a 16 Processos):** O ajuste de mínimos quadrados sobre $1/S$ em função de $1/p$ resulta em $f \approx 1{,}27\%$, projetando um teto teórico assintótico de $S_{\max} = \frac{1}{0{,}0127} \approx 78{,}5\times$.
* **Ajuste Completo (1 a 32 Processos):** Considerando todos os pontos, obtém-se $f \approx 1{,}36\%$ ($S_{\max} \approx 73{,}6\times$).
* **Acurácia do Modelo:** Para 32 processos, a Lei de Amdahl com $f=1{,}27\%$ prevê $S(32) = 22{,}95$. O valor medido foi de **21,57** (diferença de 6,0%). A ligeira discrepância ocorre porque a lei assume processadores homogêneos reais, enquanto as últimas 16 instâncias compartilham núcleos via SMT.

### 5.2 Pipeline em Dask sobre o Corpus
* **Degeneração da Regressão Clássica:** Como o speedup medido no corpus é inferior a 1,00× nas configurações de 2 a 8 workers, o ajuste linear de Amdahl gera uma reta com intercepto superior a 1, que resulta em **$f = 1{,}00$ (truncado)**. Essa degeneração demonstra que a curva não é adequadamente descrita por uma fração serial fixa.
* **Estimativa por Decomposição com 1 Worker:** A proporção da parcela não paralelizável com 1 worker é $t_{\text{serial}} / t_{\text{total}} = 16{,}97 / 27{,}34 = 62{,}1\%$. Se essa fatia fosse constante, o teto teórico de speedup seria $S_{\max} = \frac{1}{0{,}621} \approx 1{,}61\times$. O valor medido em 32 workers foi de 1,21×.
* **Análise pela Métrica de Karp-Flatt ($e(p) = \frac{\frac{1}{S(p)} - \frac{1}{p}}{1 - \frac{1}{p}}$):**
  * $e(2) = 2{,}45$
  * $e(4) = 1{,}51$
  * $e(8) = 1{,}26$
  * $e(16) = 0{,}99$
  * $e(32) = 0{,}82$
* **Interpretação Teórica:** A Lei de Amdahl exige um valor de $e(p)$ estritamente constante. O fato de $e(p)$ ser superior a 1 e **estritamente decrescente** indica a presença de um **custo fixo de comunicação e serialização pago imediatamente ao sair de 1 worker**. Esse overhead de infraestrutura é diluído à medida que a quantidade de trabalhadores $p$ aumenta, confirmando que o gargalo é imposto pelo mecanismo de agregação de dicionários do framework e não por uma fração serial intrínseca do algoritmo.

---

## 6. Escopo, Limitações e Trabalhos Não Realizados

Para garantir a transparência da análise, é essencial explicitar os limites metodológicos do estudo:

* **Versão Baseline do Pipeline (v1):** As operações de redução `df` e `stats` foram executadas utilizando as funções padrão `frequencies` e `reduction` do Dask. Uma abordagem alternativa com reduções em árvore distribuída (versão v2) não foi medida neste experimento.
* **Inexistência de Baseline Serial Puro:** O tempo de referência de 1 worker foi medido executando o próprio código Dask configurado para 1 worker. Não foi avaliado um script Python puramente imperativo sem a biblioteca Dask, de modo que os custos do próprio framework estão embutidos em $t_{\text{total}}(1)$.
* **Carga de Trabalho Fixa:** A avaliação limitou-se a um volume fixo de 300.000 documentos do *Amazon Polarity*. O efeito da variação do tamanho do dataset no escalamento (análise de Gustafson) não foi objeto deste estudo.

---

## 7. Conclusão

O pipeline de PLN em Dask obteve um speedup máximo de **1,21× com 32 workers**, enquanto o benchmark numérico `pi_mpi` em C/MPI alcançou **21,57×** na mesma infraestrutura. O fraco escalamento da aplicação sobre o corpus não decorre de gargalos no switch Gigabit ou no servidor NFS, mas sim do elevado overhead de **serialização e consolidação de dicionários Python de grande porte** nas etapas de `df` e estatísticas do vocabulário, que consomem mais de 80% do tempo de execução distribuída. A degeneração da regressão de Amdahl e a tendência decrescente da métrica de Karp-Flatt comprovam que o fator limitante consiste em um custo de movimentação e agregação de dados entre instâncias, o qual exige estratégias de redução mais eficientes para ser mitigado.

---

## 8. Instruções de Reprodução

1. **Configuração do SLURM:** Incluir `RealMemory=7000` na definição dos nós em `/etc/slurm/slurm.conf` e executar `scontrol reconfigure`.
2. **Execução do `pi_mpi` (Parte 1):**
```bash
cd parte1/codigo_mpi
module load gnu15 openmpi5 && make
./speedup_2rodadas.sh
python3 analisa_speedup.py
```
3. **Execução do Pipeline em Dask (Parte 2):**
```bash
cd pipeline
./executa_serie.sh
python3 gera_speedup_corpus.py
```
4. **Verificação dos Resultados:**
   * Dados brutos do corpus: `resultados/corpus_bruto.csv`
   * Resumo das medianas: `resultados/speedup_corpus.csv`
   * Gráfico comparativo: `resultados/curvas.png`
