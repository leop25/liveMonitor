# Previsão Presidencial 2026

Modelo probabilístico para a eleição presidencial brasileira de 2026 (1º turno em 4/out, 2º turno em 25/out).
Ele agrega todas as pesquisas nacionais, corrige o viés de cada instituto, aprende com os erros das pesquisas
em 2006–2022 e simula a eleição inteira milhares de vezes: 1º turno, 2º turno e resultado nas 27 UFs.

```bash
cd eleicoes2026
pip install -r requirements.txt
python prever.py --atualizar      # baixa as pesquisas mais recentes e gera saida/painel.html
```

## Resultado atual (pesquisas até 03/10/2026, finais, 20 mil simulações)

Cenário escolhido: **metade da correção do viés histórico** (explicação abaixo).

| Candidato | 1º turno (válidos) | Intervalo 80% | Vai ao 2º turno | Eleito |
|---|---|---|---|---|
| Lula (PT) | 44,1% | 39,7–48,5% | 85% | **38%** |
| Flávio Bolsonaro (PL) | 45,3% | 40,4–50,1% | 85% | **62%** |
| Augusto Cury (Avante) | 2,9% | 1,1–4,7% | <1% | <1% |
| Renan Santos (Missão) | 3,4% | 1,6–5,2% | <1% | <1% |
| Caiado (PSD) | 3,2% | 1,5–4,9% | <1% | <1% |
| Zema (Novo) | 0,8% | 0,2–1,4% | <1% | <1% |

Chance de 2º turno: 85%. No 1º turno, Lula termina à frente em 42% das simulações. No 2º turno Lula × Flávio,
Lula fica em média com 48,9% dos válidos (faixa de 80%: 43,6–54,0%).

## Dois cenários

| Cenário | 1º turno Lula × Flávio | 2º turno Lula × Flávio | Chance Lula | Chance Flávio |
|---|---|---|---|---|
| **Metade da correção (escolhido)** | 44,1 × 45,3 | 48,9 × 51,1 | **38%** | **62%** |
| Correção histórica inteira | 43,5 × 47,0 | 47,9 × 52,1 | 26% | 74% |
| Sem correção (pesquisas certas em média) | 44,9 × 43,5 | 49,9 × 50,1 | 50% | 50% |

Em todas as cinco eleições de 2006 a 2022, a projeção final subestimou o principal candidato contra o PT no
1º turno (de 1,5 a 7 pontos). Nos confrontos de 2º turno medidos antes do 1º turno, o PT foi superestimado em
3,2 pontos (2018) e 6,1 (2022).

### Escolha do cenário

A favor de o erro ter diminuído:
- os institutos atualizaram as amostras com o Censo 2022 (em 2022 usavam projeções defasadas);
- adotaram modelos de eleitor provável (a Quaest desde 2022/2024) e a Quaest passou, em 2026, a controlar a
  composição política da amostra no desenho, com ponderação por MRP em cerca de 400 subgrupos;
- nas municipais de 2024, as pesquisas divergiram da urna em só 3 das 26 capitais;
- a distância entre institutos presenciais (Datafolha, Quaest) e online (AtlasIntel) na margem Lula–adversário
  caiu de cerca de 5,5 pontos em 2022 para 2,5 pontos em 2026.

Contra:
- em 2022 até os institutos mais precisos (AtlasIntel, MDA) superestimaram a margem de Lula em cerca de 4 pontos;
- a explicação mais aceita (baixa taxa de resposta de eleitores bolsonaristas, decisão no dia e voto útil de
  última hora) não tem correção conhecida, e nenhum instituto brasileiro acompanha os mesmos eleitores antes e
  depois da votação;
- a diferença entre presenciais e online persiste no mesmo sentido.

As evidências sustentam que parte do problema foi corrigida, não todo. Por isso o padrão é `--peso-vies 0.5`.
Os outros cenários ficam no painel e na linha de comando (`--peso-vies 0` ou `--peso-vies 1`).

## Só os institutos mais certeiros (AtlasIntel e MDA)

AtlasIntel e MDA tiveram o menor erro em 2018 e 2022. O pipeline roda também um cenário só com eles
(17 pesquisas de 1º turno e 79 confrontos de 2º turno), corrigido pelo erro histórico **deles**, não pelo
de todos os institutos:

| Cenário | 1º turno Lula × Flávio | 2º turno Lula × Flávio | Chance Lula | Chance Flávio |
|---|---|---|---|---|
| Sem correção | 46,7 × 42,4 | 51,2 × 48,8 | 66% | 34% |
| **Corrigido pelo histórico dos dois** | **46,6 × 44,1** | **49,3 × 50,7** | **50%** | **50%** |

| Erro passado (pesquisa − urna) | PT | Adversário |
|---|---|---|
| MDA 2018, 1º turno | −1,4 | −3,4 |
| MDA 2018, 2º turno medido antes do 1º | +1,3 | – |
| AtlasIntel 2022, 1º turno | +2,0 | −2,0 |
| MDA 2022, 1º turno | −0,1 | −3,5 |
| AtlasIntel 2022, 2º turno medido antes do 1º | +4,5 | – |
| MDA 2022, 2º turno medido antes do 1º | +4,1 | – |

Com esses dois, Lula termina o 1º turno à frente em 66% das simulações e o 2º turno vira empate. A base é
pequena: o histórico dos dois cobre só duas eleições. Por isso o modelo principal continua usando todos os institutos, com peso maior para esses dois
(MDA 2,0; AtlasIntel 1,5). A lista fica em `MELHORES` (`modelo/previsao.py`); para rodar só com eles:
`python prever.py --institutos AtlasIntel,MDA`.

## Como funciona

```
Wikipédia (pesquisas 2010–2026, resultados por UF 2002–2022)
   │  modelo/coleta.py + wikitabela.py   parser de wikitexto com rowspan/colspan
   ▼
Agregador (modelo/agregador.py)
   • passeio aleatório diário por candidato, filtro + suavizador de Kalman
   • efeito fixo por instituto ("house effect") com priori N(0; 2,5²)
   • volatilidade escolhida por máxima verossimilhança
   • nível ancorado nos institutos que mais acertaram em 2010/2018/2022
   ▼
Calibração histórica (modelo/historico.py)
   • backtest do próprio agregador na véspera de 2010, 2018 e 2022 (pesos leave-one-out)
   • + médias finais de 2006 e 2014 → viés e dispersão do erro por papel (PT, adversário, demais)
   • erro dos confrontos de 2º turno medidos antes do 1º turno e correlação com o erro do 1º
   • inclinação de cada UF, persistência AR(1) entre eleições, choques regionais e estaduais
   • efeito "candidato da casa" (7 casos, 2002–2022: 2,1× no estado de origem)
   ▼
Simulação Monte Carlo (modelo/simulacao.py)
   • 1º turno: tendência + deriva até a eleição + erro sistemático (t de Student, 5 g.l.)
   • 2º turno: confrontos diretos + erro correlacionado; sem pesquisa → transferência por afinidade
   • 27 UFs: geografia de 2022 + choques, reajustada para bater o total nacional de cada simulação
   ▼
saida/painel.html (painel interativo) e saida/previsao.json
```

## Noite da eleição: apuração e 2º turno a partir das urnas

```bash
python prever.py --atualizar --apuracao    # baixa pesquisas e a apuração do TSE e refaz tudo
```

`--apuracao` baixa do TSE o arquivo simplificado de cada UF (eleição 6257, Presidente 1º turno;
`resultados.tse.jus.br/oficial/ele2026/6257/dados-simplificados/<uf>/<uf>-c0001-e006257-r.json`).
Antes da eleição o TSE devolve 404 e nada muda. Com apuração disponível (`modelo/pos_primeiro_turno.py`):

1. **Projeção do 1º turno durante a apuração.** A parcial nacional do TSE engana porque a ordem de
   apuração varia entre UFs (em 2022 Bolsonaro liderou as primeiras horas; o Nordeste apura depois).
   A projeção é feita UF a UF: parcial de cada UF, com incerteza que cai conforme as seções são
   totalizadas, ou a previsão pré-eleição onde ainda não há apuração; o total é ponderado pelos votos
   válidos esperados de cada UF.
2. **2º turno a partir das urnas.** Votos reais do 1º turno em cada UF + transferência dos votos dos
   eliminados medida nas pesquisas de 2026 (mesma pesquisa: 1º turno × confronto direto; hoje 64% dos
   votos que trocam de candidato vão para Flávio, em 70 pares de pesquisas) + erro calibrado em
   2018/2022 + choques regionais e estaduais calibrados em 2006–2022. Pesquisas de 2º turno feitas depois
   do 1º turno são agregadas e combinadas por variância inversa.

Teste com o 1º turno real das eleições passadas: a conta acerta o vencedor do 2º turno em 2018 e 2022
e 25–26 das 27 UFs; o erro no total nacional foi de +0,6 (2018) e +2,7 pontos (2022), ambos
superestimando o PT, e esse erro entra na simulação.

O painel ganha a seção "Apuração do 1º turno" no topo e um terceiro modo no mapa ("Depois das urnas").
Se o TSE ficar fora do ar, dá para digitar o resultado em `dados/resultado_1t_2026.csv`
(`uf,pst,Lula,Flávio Bolsonaro,Augusto Cury,Renan Santos,Caiado,Zema,Outros`, votos; `pst` de 0 a 1).

### Ao vivo, de 5 em 5 minutos

```bash
python ao_vivo.py --a-cada 300     # uma rodada a cada 5 min até a apuração chegar a 100%
python ao_vivo.py                  # uma rodada só
```

Cada rodada baixa a apuração, refaz a projeção e o 2º turno, acrescenta um ponto a
`saida/apuracao_historico.csv` e regenera o painel (com o gráfico "projeção × parcial do TSE" ao longo
da noite). Se o TSE não mudou nada desde a rodada anterior, a rodada é pulada.

Para os estados que ainda não apuraram, a projeção parte da previsão pré-eleição e aplica o desvio
observado nos estados já apurados (se Lula supera a previsão onde já se contou, tende a superar onde
falta contar). Num teste com o 1º turno de 2022 apurado com o Nordeste por último, a parcial do TSE
mostrava o candidato de direita à frente até 89% das seções, enquanto a projeção já dava Lula à frente
desde 5% e chegava a 47,3% (real: 48,4%) com metade das seções apuradas.

## Pesquisas fora da Wikipédia

Pesquisas divulgadas e ainda não registradas na Wikipédia podem ser digitadas em
`dados/pesquisas_manuais.csv` (uma linha por cenário, em % do total; no 2º turno, preencha só os dois
candidatos). Quando a Wikipédia registrar a mesma pesquisa (mesmo instituto, data final e turno), vale a
da Wikipédia.

## Opções

```bash
python prever.py --sim 50000          # mais simulações
python prever.py --peso-vies 0        # sem correção do viés histórico
python prever.py --peso-vies 1        # correção histórica inteira
python prever.py --ate 2026-08-15     # "volta no tempo": só pesquisas até a data
python prever.py --institutos AtlasIntel,MDA   # só alguns institutos
python prever.py --apuracao           # usa a apuração do TSE (noite da eleição)
python -m pytest testes               # testes
```

Candidatos, blocos ideológicos e estados de origem ficam em `CONFIG_2026` (`modelo/previsao.py`).

## Dados

- `dados/brutos/pesquisas_*.wiki`: wikitexto das páginas "Opinion polling for the <ano> Brazilian presidential
  election" (Wikipédia em inglês) e, para 2026, também da página "Pesquisas de opinião para a eleição presidencial
  no Brasil em 2026" (Wikipédia em português), que lista institutos a mais (Veritá, Gerp, DataTrends,
  American Analytics, Alfa Inteligência). No 1º turno as duas fontes são unidas por (instituto, data final),
  sem duplicar pesquisas. `--atualizar` baixa as duas de novo.
- `dados/processados/resultados_uf_2002_2022.csv`: votos por candidato e UF (1º e 2º turnos), extraídos das
  páginas "Resultados da eleição presidencial no Brasil em <ano>" (Wikipédia em português, que reproduzem o TSE).
  Os totais nacionais conferem com o TSE a menos de 0,1 ponto.
- 2006 e 2014 não têm página de pesquisas estruturada. Para esses anos o modelo usa a média final divulgada:
  Datafolha 29–30/set/2006 (do gráfico da página da eleição na Wikipédia) e Datafolha/Ibope de 3–4/out/2014,
  em votos válidos, arredondados. São valores aproximados; o encolhimento do viés reduz o peso de cada ano.

## Limitações

- Não usa pesquisas estaduais de presidente: não há compilação delas (nem na Wikipédia), e no Brasil o
  total é nacional, então o ganho seria sobretudo no mapa. O mapa pré-eleição vem da geografia de 2022 mais
  o sorteio de choques; depois do 1º turno, vem das urnas.
- Indecisos são distribuídos na proporção dos votos (conversão para votos válidos). Distribuí-los pela
  rejeição dos candidatos contaria duas vezes o mesmo efeito: o erro histórico usado na correção já
  inclui para onde os indecisos foram em 2006–2022.
- Não modela abstenção diferencial, que em 2022 favoreceu o candidato à direita em alguns estados.
- A correção do viés se baseia em 5 eleições (1º turno) e 2 eleições (confrontos antes do 1º turno).
- A Wikipédia é mantida por voluntários. O parser descarta linhas malformadas, mas pode haver erros de digitação na fonte.
