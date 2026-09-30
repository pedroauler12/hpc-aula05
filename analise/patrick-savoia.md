 # Análise individual: Patrick Savoia

HPC, Inteli 2026.2, Prof. João Luisi. Ponderada da Aula 5, item individual da Parte 2.
Os números foram medidos pelo grupo e estão em `../resultados/`. Os cálculos derivados (decomposição da distância, cenários, Karp-Flatt por etapa) e a interpretação são meus.

## Resumo

- **O que se mediu:** o speedup de um pipeline TF-IDF em Dask (300.000 resenhas do Amazon Polarity) em seis configurações (de 1 a 32 workers), comparado ao `pi_mpi` da Aula 3 no mesmo cluster.
- **Resultado:** o corpus perde para 1 worker em 2, 4 e 8 workers (S < 1) e chega a **S(32) = 1,21**; o `pi_mpi` chega a **21,57**. Com 32 workers o pipeline usa **26× mais tempo de CPU** que com 1 worker para ficar 1,21× mais rápido.
- **Para onde vai a distância:** com 32 workers, o tempo medido (22,5 s) fica 21,7 s acima do ideal T₁/32 (0,85 s). Desse excesso, **83% está em duas etapas**, `stats` (51%) e `df` (32%); a leitura no NFS responde por 10%, o envio do IDF por 6% e as etapas estreitas por só 2%.
- **Causa:** as agregações ficam **mais lentas** ao passar de 1 para 2 workers **no mesmo nó** (`stats`: 7,3 → 23,0 s). Isso aponta para a serialização de dicionários entre processos, e não para a rede.
- **Fração serial (Amdahl):** estimo **f ≈ 0,62** para o pipeline (teto de ~1,6×), contra f ≈ 0,013 do `pi_mpi`. O ajuste clássico degenera (f = 1). Por isso cruzei a estimativa com Karp-Flatt, com o Karp-Flatt de cada etapa e com um cenário hipotético sem a penalidade de serialização: todos ficam na mesma faixa.

## 1. Como li os números

**Cluster:** master + 4 nós (c1 a c4), cada um com um Intel Core i3-13100T (4 cores físicos × 2 threads = 8 CPUs lógicas) e ~4,1 GB de RAM livres. No total são **16 cores físicos e 32 CPUs lógicas**, ligados por Ethernet gigabit, com NFS servido pelo master. A escada de configurações foi 1, 2 e 4 workers em 1 nó; 8 em 2 nós; 16 em 4 nós (1 worker por core físico); 32 em 4 nós (2 workers por core, SMT). Cada worker é um processo com 1 thread, e as 128 partições são as mesmas em todas as configurações (problema de tamanho fixo, *strong scaling*).

**Definições usadas no CSV:** `t_calc` é a soma das etapas estreitas (tokeniza, stopwords, tf, tfidf); `t_serial` = `t_total` − `t_calc`, ou seja, leitura, `df`, `stats` e o que houver entre as etapas. O tempo conta só depois de os workers estarem prontos; o tempo de subir o cluster (`t_sobe`) fica numa coluna própria.

**Dois cuidados antes de interpretar:**

1. **A cronometragem por etapa cobre quase tudo.** A soma das 7 etapas difere do `t_total` em 0,05 s a 0,83 s (no máximo 3%, com 16 workers). Portanto, dá para atribuir o tempo às etapas sem uma parcela grande "sem dono".
2. **O ruído entre rodadas é baixo onde importa.** Coeficiente de variação do `t_total` nas 3 rodadas:

| workers | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| CV | 0,6% | 6,8% | 4,2% | 2,3% | 6,9% | 0,3% |

   Os pontos extremos da curva (1 e 32) são os mais estáveis, e o speedup final de 1,21 não depende do acaso. Onde há mais ruído é com 2 e com 16 workers. No de 16 workers, o ruído vem quase todo da leitura (seção 4.3).

**Estatísticas diferentes nas duas curvas:** o `pi_mpi` usa o **menor** tempo de 2 rodadas (roteiro da Aula 3) e o corpus usa a **mediana** de 3 (Parte 2). Recalculei o corpus com o menor tempo de cada ponto: S = 1; 0,65; 0,73; 0,84; 1,14; 1,20. O formato da curva não muda.

## 2. Gráfico com as três curvas

![Speedup: corpus TF-IDF, pi_mpi e linear ideal](../resultados/curvas.png)

| p | nós | T corpus (s) | S corpus | efic. | T pi_mpi (s) | S pi_mpi | efic. | CPU·s do corpus (p × T) |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 27,34 | 1,00 | 100% | 34,87 | 1,00 | 100% | 27 |
| 2 | 1 | 47,16 | 0,58 | 29% | 17,48 | 1,99 | 100% | 94 |
| 4 | 1 | 37,75 | 0,72 | 18% | 9,45 | 3,69 | 92% | 151 |
| 8 | 2 | 33,58 | 0,81 | 10% | 4,76 | 7,33 | 92% | 269 |
| 16 | 4 | 26,98 | 1,01 | 6% | 2,36 | 14,78 | 92% | 432 |
| 32 | 4 | 22,51 | 1,21 | 4% | 1,62 | 21,57 | 67% | 720 |

A última coluna traduz a eficiência em custo: rodar com 32 workers consome **720 worker·s** para fazer o que 1 worker faz em 27. No `pi_mpi`, o custo total quase não muda até 16 processos (eficiência de 92%). Para um usuário do cluster, a pergunta prática deixa de ser "quantos workers usar" e passa a ser "compensa distribuir?". Para este pipeline, na v1, a resposta é não.

### Speedup de cada etapa

T₁ da etapa / Tₚ da etapa (medianas, em segundos na primeira coluna):

| etapa | tipo | T₁ (s) | p=2 | p=4 | p=8 | p=16 | p=32 |
|---|---|---|---|---|---|---|---|
| tokeniza | estreita | 4,72 | 1,87 | 3,11 | 3,81 | 12,8 | 15,7 |
| stopwords | estreita | 2,31 | 1,94 | 2,29 | 3,73 | 10,5 | 11,6 |
| tf | estreita | 1,72 | 1,34 | 2,32 | 2,57 | 5,38 | 8,19 |
| tfidf | estreita + envio do IDF | 1,48 | 1,57 | 1,90 | 2,06 | 2,06 | **1,13** |
| leitura | I/O no NFS | 2,25 | 1,31 | 1,35 | **0,83** | **0,49** | 1,06 |
| df | larga (`frequencies`) | 7,31 | **0,45** | 0,58 | 0,67 | 0,95 | 1,03 |
| stats | redução | 7,35 | **0,32** | 0,38 | 0,45 | 0,60 | 0,66 |

As etapas se dividem em dois grupos. As que processam **cada documento sozinho** (tokeniza, stopwords, tf) escalam como o `pi_mpi`. As que precisam **juntar informação de todos os documentos** (`df`, `stats`) nunca ficam mais rápidas que com 1 worker.

## 3. Decomposição da distância: quanto cada etapa custa

Para saber qual gargalo pesa mais, comparei cada etapa com o tempo que ela teria se escalasse perfeitamente (T₁ da etapa / p). A diferença é o **excesso** daquela etapa.

| etapa | excesso com p = 2 (s) | % | excesso com p = 8 (s) | % | excesso com p = 32 (s) | % |
|---|---|---|---|---|---|---|
| stats | 19,34 | 58% | 15,26 | 51% | 10,95 | 51% |
| df | 12,67 | 38% | 9,98 | 34% | 6,88 | 32% |
| leitura | 0,59 | 2% | 2,43 | 8% | 2,06 | 10% |
| tfidf | 0,20 | 1% | 0,54 | 2% | 1,26 | 6% |
| tokeniza + stopwords + tf | 0,62 | 2% | 1,43 | 5% | 0,43 | 2% |
| entre etapas | 0,08 | 0% | 0,51 | 2% | 0,07 | 0% |
| **total do excesso** | **33,49** | | **30,16** | | **21,66** | |

(Ideal: T₁/p = 13,67 s, 3,42 s e 0,85 s.)

Três leituras:
1. **As agregações são o problema em qualquer p:** respondem por 83% a 96% do excesso.
2. **O peso da leitura cresce quando o job ocupa mais de um nó** (de 2% para 8-10%). É o NFS entrando na conta.
3. **O `tfidf` é o único excesso que cresce de 8 para 32** (0,54 → 1,26 s). É o sintoma de um custo proporcional ao número de workers (seção 4.5).

## 4. Os gargalos pedidos

### 4.1 Shuffle e serialização (o dominante)

**Onde na curva:** na queda de 1,00 para 0,58 entre 1 e 2 workers, e no patamar baixo até 8 workers.

**Número que prova:** ao dobrar de 1 para 2 workers sem sair de c1, `df` vai de 7,31 para 16,33 s (+123%) e `stats` de 7,35 para 23,01 s (+213%). As etapas estreitas caem pela metade no mesmo passo. Sem sair do nó, a rede física não participa. O que muda de 1 para 2 é a existência de **outro processo**.

**Mecanismo:** `frequencies` (no `df`) e `reduction` (no `stats`) combinam resultados parciais das 128 partições em uma árvore. Com 1 worker, os parciais já estão na memória do único processo que vai combiná-los. Com 2 ou mais, parte deles é um `dict` Python com dezenas de milhares de termos que precisa ser serializado (pickle), enviado e desserializado. Isso é trabalho de CPU em Python puro, feito por workers de 1 thread. No fim do `df`, o dicionário completo, com **196.361 termos**, ainda volta ao cliente.

`stats` sofre mais que `df` (+213% contra +123%), e há uma explicação coerente com o código: `stats` faz **duas** agregações (a soma do TF-IDF por termo, com valores `float`, e o histograma de tamanhos), enquanto `df` faz a contagem mais o `count()`. É um palpite a partir do código, não uma medição separada.

Depois do salto, as duas etapas melhoram devagar com mais workers (`stats`: 23,0 → 19,4 → 16,2 → 12,2 → 11,2 s). O custo de serializar também se divide: cada worker serializa pedaços menores ao mesmo tempo. A curva volta a subir, mas a partir de um patamar pior que o de 1 worker.

### 4.2 Fração serial

**Onde na curva:** no teto. Mesmo sem comunicação, a curva do corpus não passaria de ~1,6× (seção 5).

**Número que prova:** com 1 worker, leitura + `df` + `stats` = 2,25 + 7,31 + 7,35 = **16,9 s de 27,3 s (62%)**, e nenhuma dessas etapas passa de ~1× com 32 workers.

A diferença para o `pi_mpi` está na estrutura do problema. No `pi_mpi` cada processo calcula a sua fatia sem depender dos outros e, no fim, envia **um número** (`MPI_Reduce`; `t_serial` de 0,007 s com 32 processos). No TF-IDF, o IDF de um termo depende de **todos** os documentos, e a soma do TF-IDF por termo também. Existe um ponto de encontro obrigatório, e o custo dele é proporcional ao tamanho do vocabulário, não ao número de workers.

### 4.3 I/O no NFS

**Onde na curva:** nos pontos com vários nós (8 e 16 workers). A leitura é a única etapa que **piora** ao sair de 1 nó (1,67 → 2,71 s) e só volta ao nível de 1 nó com 32 workers.

**Números:**
- **Piso físico:** o ping-pong da Aula 3 mediu 107 MB/s entre nós. Os 83 MB do corpus saem todos do master pelo mesmo cabo, então custam no mínimo 83 / 107 ≈ **0,8 s**, com qualquer número de workers. Esse piso é serial.
- **Instabilidade:** com 16 workers, as três rodadas da leitura deram 2,1 s, 4,6 s e 6,4 s. É a maior variação relativa da série, e explica quase todo o CV de 6,9% desse ponto. É o padrão esperado quando 4 nós disputam um único servidor NFS.
- **Leitura não é só I/O:** a etapa também decodifica Parquet, reparticiona 32 arquivos em 128 partições e cria 300 mil strings Python. Por isso, mesmo em 1 nó, ela não fica abaixo de ~1,7 s.
- **Efeito no último ponto:** de 16 para 32 workers, o `t_total` caiu 4,5 s, e **2,4 s disso foram a leitura** voltando ao normal (4,56 → 2,13 s). Parte do "ganho" do ponto de 32 é variação de I/O.

**Fora do `t_total`, mas relevante:** a subida do cluster (`t_sobe`) leva ~3 s em 1 nó e **33 a 37 s** quando entra um segundo nó (ou mais), e esse valor quase não depende do número de nós. Um patamar fixo assim sugere algo como importar o ambiente Python pelo NFS nos nós remotos ou um tempo de espera no registro dos workers. Não verifiquei a causa. Se esse tempo contasse, S(32) cairia para (27,34 + 2,93) / (22,51 + 32,30) = **0,55**.

### 4.4 Rede gigabit

**Onde na curva:** na passagem de 4 para 8 workers, quando o job passa de 1 para 2 nós.

**Números:** o ping-pong mostrou latência de 235 µs entre nós contra 0,24 µs no mesmo nó (~1000×) e banda de 107 MB/s. Se a rede fosse o limite, as agregações piorariam ao cruzar para 2 nós. Elas **melhoraram**: `df` 12,57 → 10,90 s e `stats` 19,42 → 16,18 s.

**Leitura:** neste volume de dados, o gargalo das agregações é o custo de CPU de serializar e combinar dicionários, e não o tempo de transmissão. Uma conta de ordem de grandeza confirma: mesmo que cada uma das 128 partições enviasse 1 MB de dicionário parcial, seriam 128 MB, ou ~1,2 s no gigabit, muito menos que os 10 a 16 s que `df` e `stats` levam com 8 workers. A rede pesa onde todo o tráfego converge para **um único ponto**: o servidor NFS (4.3) e o cliente que recebe o vocabulário.

### 4.5 Hyperthreading a partir de 16 workers

**Onde na curva:** entre 16 e 32, quando cada core físico passa a ter 2 workers.

**Números:**
- **`pi_mpi`:** ganho de **1,46×** de 16 para 32 processos (o ideal seria 2×); a eficiência cai de 92% para 67%. Como o `pi_mpi` quase não se comunica, esse é o efeito do SMT medido de forma limpa: duas threads por core rendem ~1,5 core.
- **Etapas estreitas do corpus:** `tokeniza` 1,23×, `stopwords` 1,10× e `tf` 1,52× de 16 para 32. É a mesma faixa do `pi_mpi`, e confirma que o SMT rende menos que um core de verdade também para processamento de texto.
- **`tfidf`:** **piorou** de 0,72 s para 1,31 s. A etapa começa com `client.run(guarda_idf, idf)`, que envia o dicionário IDF (196 mil termos) a **cada** worker: 32 cópias serializadas contra 16. É o único custo do pipeline que cresce com p, e com 32 workers ele supera o ganho do SMT nas outras etapas estreitas (o `t_calc` sobe de 1,65 para 2,01 s).
- **Memória:** com 8 workers por nó, são 8 cópias do IDF e das partições nos ~4,1 GB livres de cada nó. O limite de RAM impede subir muito mais o número de workers por nó.

## 5. Fração serial pela lei de Amdahl

S(p) = 1 / (f + (1 − f)/p), o que é o mesmo que 1/S = f + (1 − f)·(1/p): uma reta em 1/p com intercepto f. O teto é 1/f.

### 5.1 `pi_mpi`: o modelo funciona

O ajuste nos pontos de 1 a 16 (só cores físicos) dá **f = 0,013**, com teto de ~79×. Extrapolando esse f para 32 processos, a fórmula dá 22,95, e a medição ficou em 21,57, 6% abaixo; essa falta é o SMT. Por isso deixei o ponto de 32 fora do ajuste: ele mistura duas coisas diferentes (serialidade e threads que dividem core).

### 5.2 Corpus: o ajuste clássico degenera

Com os 6 pontos, o ajuste por mínimos quadrados dá intercepto acima de 1, truncado em **f = 1**. Isso acontece porque S(2) = 0,58 < 1, e Amdahl não permite S < 1: o modelo supõe que o trabalho é o mesmo com qualquer p e só é dividido. Aqui o trabalho **aumenta** ao sair de 1 worker, porque aparece a serialização (4.1). O fato de o ajuste degenerar já é uma conclusão: **a curva não é limitada só por uma fração serial fixa**.

### 5.3 Minha estimativa: f pela decomposição das etapas

Considero serial a parte do tempo com 1 worker que está em etapas que, na prática, não escalam (speedup ≤ ~1 com 32 workers):

**f ≈ (t_leitura + t_df + t_stats) / t_total = 16,91 / 27,34 = 0,62**, com teto de **1/f ≈ 1,6×**

(o resultado coincide com `t_serial` / `t_total` do CSV: 16,97 / 27,34 = 0,62)

| p | Amdahl (f = 0,62) | medido | medido / previsto |
|---|---|---|---|
| 2 | 1,23 | 0,58 | 47% |
| 4 | 1,40 | 0,72 | 51% |
| 8 | 1,50 | 0,81 | 54% |
| 16 | 1,55 | 1,01 | 65% |
| 32 | 1,58 | 1,21 | 77% |

O medido fica abaixo do previsto em todos os pontos, e a razão medido/previsto **sobe** com p (de 47% para 77%). É o comportamento de um overhead que se divide entre os workers: o Amdahl explica o teto, e a diferença restante é o custo de distribuição.

### 5.4 Três verificações independentes

**(a) Karp-Flatt global.** e(p) = (1/S − 1/p) / (1 − 1/p):

| p | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| e(p) | 2,45 | 1,51 | 1,26 | 0,99 | 0,82 |

Se Amdahl valesse, e(p) seria constante. Ele **passa de 1** (impossível para uma fração; só acontece porque S < 1) e **diminui** com p, o que indica um custo fixo pago ao sair de 1 worker e depois dividido. Com p = 32, e = 0,82: a fração serial de 0,62 mais o overhead que ainda não foi diluído.

**(b) Karp-Flatt por etapa (p = 32).** Aplicando a mesma fórmula a cada etapa isolada:

| etapa | tokeniza | stopwords | tf | tfidf | leitura | df | stats |
|---|---|---|---|---|---|---|---|
| e(32) | 0,03 | 0,06 | 0,09 | 0,88 | 0,95 | 0,97 | 1,54 |

Os dois grupos aparecem nos números. As etapas estreitas se comportam como programas com 3% a 9% de fração serial, a mesma ordem de grandeza do `pi_mpi` (o resíduo vem de dividir 128 tarefas entre 32 workers e do custo do scheduler por tarefa). Leitura, `df` e `stats` ficam perto de 1 ou acima: são, em termos práticos, **seriais**. Esta tabela é a base de somar exatamente essas três etapas na estimativa f = 0,62.

**(c) Cenário sem a penalidade de serialização (hipotético, calculado a partir das medições).** Se `df` e `stats` tivessem mantido o tempo de 1 worker em todas as configurações, sem piorar e sem melhorar, e as outras etapas fossem as medidas:

| p | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| S sem penalidade | 1,22 | 1,34 | 1,29 | 1,26 | 1,45 |
| Amdahl (f = 0,62) | 1,23 | 1,40 | 1,50 | 1,55 | 1,58 |

Tirando a penalidade de serialização, a curva vira praticamente a curva de Amdahl com f = 0,62. O que sobra (1,45 contra 1,58 com 32) é a leitura no NFS e o envio do IDF, os dois custos que crescem com o número de nós e de workers. As três verificações concordam: **f ≈ 0,6 é a fração serial; a distância entre a curva medida e a de Amdahl é overhead de distribuição.**

### 5.5 Por que não usei o ajuste só com p ≥ 2

Ajustei T(p) = T_s + T_p/p deixando de fora o ponto de 1 worker. O f obtido depende dos pontos escolhidos: 0,33 (p de 2 a 32), 0,25 (4 a 32) e 0,14 (8 a 32). Um f que muda com o intervalo não é fração serial: o overhead, que também diminui com p, entra no ajuste como se fosse trabalho paralelo e distorce o resultado. Por isso fico com a estimativa da seção 5.3, que tem significado físico direto e é confirmada pelas três verificações.

## 6. E se? (projeções a partir dos números, não medidas)

**Se as agregações escalassem como a tokenização:** `df` e `stats` teriam o speedup da tokenização (15,7× com 32 workers). Com as demais etapas como medidas, o pipeline daria **S(32) ≈ 5,3** (5,15 s no total). O próximo gargalo seria, então, a leitura (2,13 s) e o `tfidf` (1,31 s), que somariam 67% do tempo. Consertar as agregações não basta para chegar perto do `pi_mpi`; o NFS e o envio do IDF viriam logo em seguida.

**Se o corpus fosse maior (Gustafson):** as etapas estreitas crescem com o número de documentos N. O custo das agregações cresce com o **vocabulário**, que em texto cresce mais devagar que N (lei de Heaps, V ∝ N^β com β < 1). Com mais documentos, a fatia paralela cresceria mais que a serial, e o f cairia. Os dados do grupo só permitem um indício fraco disso: o AG News (120.000 notícias, 61.488 termos) tinha 53% de tempo que não escala na calibração, e o Amazon (300.000 resenhas, 196.361 termos), 62%. Os corpora são diferentes, então isso não testa Heaps. Mostra, porém, que o f depende mais do tamanho do vocabulário que do número de documentos. O limite prático aqui é a memória (~4,1 GB livres por nó), que impede crescer muito o corpus sem mudar a implementação.

## 7. O que aprendi com esta análise

1. **Speedup total esconde a causa; o tempo por etapa mostra.** Olhando só a curva, "o Dask escala mal" parece a conclusão. Separando as etapas, fica claro que três delas escalam de 8× a 16×, e o problema está em duas etapas específicas.
2. **Comparar o mesmo passo em dois cenários isola a causa.** A passagem de 1 para 2 workers no mesmo nó separa serialização de rede. A passagem de 4 para 8, cruzando para o segundo nó, mostra que a rede não piorou as agregações.
3. **Um modelo que falha também informa.** O ajuste de Amdahl dar f = 1 não é erro de conta. Mostra que o pipeline tem um custo que Amdahl não prevê, e o Karp-Flatt decrescente diz de que tipo ele é.
4. **Eficiência é custo.** 4% de eficiência com 32 workers significa ocupar o cluster inteiro para ganhar 21% de tempo. Em um cluster compartilhado, isso tira recursos de outros jobs.
5. **Paralelizar dados não é paralelizar cálculo.** O `pi_mpi` quase não tem dados para mover; o TF-IDF tem um vocabulário inteiro. A abstração do Dask esconde as mensagens, mas o custo de mover os dados continua existindo e reaparece como serialização.

## 8. Limites

- Tudo aqui vale para a **v1** do pipeline, que agrega com `frequencies` e `reduction` sem nenhum ajuste de parâmetro. Nenhuma variante foi medida, então não separo quanto do f = 0,62 é do algoritmo e quanto é da implementação.
- Não houve baseline serial sem Dask. O "1 worker" já inclui scheduler e cliente em processos separados; um script puro provavelmente seria mais rápido, e o speedup real, menor.
- As explicações de mecanismo (tamanho dos dicionários parciais, duas agregações no `stats`, causa do `t_sobe`) são inferências a partir do código e dos tempos; não medi tráfego de rede nem memória.
- Os cenários das seções 5.4(c) e 6 são contas sobre as medições, não execuções.
- Um único corpus, em um único cluster, com 3 rodadas por ponto (suficiente para o formato da curva, conforme o CV da seção 1).

## 9. Conclusão

O pipeline TF-IDF chega a 1,21× com 32 workers, contra 21,6× do `pi_mpi`, e a decomposição mostra por quê. **83% da distância para o ideal está em duas agregações** (`df` e `stats`). Elas têm fração serial alta por natureza (f ≈ 0,62, teto de ~1,6×) e ainda ficam mais lentas ao sair de 1 worker, por causa da serialização de dicionários entre processos, custo que já aparece dentro de um único nó. NFS, rede gigabit e SMT têm efeitos mensuráveis (piso de 0,8 s e variação de 2 a 6 s na leitura, agregações que melhoram ao cruzar a rede, 1,46× de ganho com SMT no `pi_mpi`), mas são secundários. Tirando a penalidade de serialização, a curva coincide com a de Amdahl com f = 0,62. A lição é que, em PLN distribuído, o que limita o speedup é **quanto dado precisa ser juntado**, e não quanto cálculo há para dividir.

## 10. Onde estão os dados

- Rodadas brutas (18 jobs, tempo por etapa): `../resultados/corpus_bruto.csv`
- Medianas no formato da Aula 3: `../resultados/speedup_corpus.csv`
- `pi_mpi`: `../resultados/speedup.csv`; ping-pong: `../parte1/tabelas-blocos-2-e-3.md`
- Logs dos jobs: `../logs_serie/`; comandos para reproduzir: `../README.md`

As contas derivadas desta análise (CV, excesso por etapa, Karp-Flatt por etapa e cenários) saem só desses dois CSVs do corpus, com as fórmulas indicadas em cada seção.
