# Previsão Presidencial 2026

Modelo probabilístico para a eleição presidencial brasileira de 2026 (1º turno em 4/out, 2º turno em 25/out).
Ele agrega todas as pesquisas nacionais, corrige o viés de cada instituto, aprende com os erros das pesquisas
em 2006–2022 e simula a eleição inteira milhares de vezes: 1º turno, 2º turno e resultado nas 27 UFs.

```bash
cd eleicoes2026
pip install -r requirements.txt
python prever.py --atualizar      # baixa as pesquisas mais recentes e gera saida/painel.html
```

## Resultado atual (pesquisas até 27/09/2026, 20 mil simulações)

Cenário escolhido: **metade da correção do viés histórico** (explicação abaixo).

| Candidato | 1º turno (válidos) | Intervalo 80% | Vai ao 2º turno | Eleito |
|---|---|---|---|---|
| Lula (PT) | 43,1% | 38,6–47,7% | 93% | **41%** |
| Flávio Bolsonaro (PL) | 42,8% | 37,7–47,8% | 93% | **59%** |
| Augusto Cury (Avante) | 4,4% | 1,8–7,1% | <1% | <1% |
| Renan Santos (Missão) | 3,8% | 2,2–5,5% | <1% | <1% |
| Caiado (PSD) | 3,7% | 2,2–5,4% | <1% | <1% |
| Zema (Novo) | 1,1% | 0,3–2,0% | <1% | <1% |

Chance de 2º turno: 93%. No 1º turno, Lula termina à frente em 53% das simulações. No 2º turno Lula × Flávio,
Lula fica em média com 49,1% dos válidos (faixa de 80%: 43,8–54,4%).

## Dois cenários

| Cenário | 1º turno Lula × Flávio | 2º turno Lula × Flávio | Chance Lula | Chance Flávio |
|---|---|---|---|---|
| **Metade da correção (escolhido)** | 43,1 × 42,8 | 49,1 × 50,9 | **41%** | **59%** |
| Correção histórica inteira | 42,5 × 44,5 | 48,0 × 52,0 | 30% | 70% |
| Sem correção (pesquisas certas em média) | — | — | 53% | 47% |

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

## Opções

```bash
python prever.py --sim 50000          # mais simulações
python prever.py --peso-vies 0        # sem correção do viés histórico
python prever.py --peso-vies 1        # correção histórica inteira
python prever.py --ate 2026-08-15     # "volta no tempo": só pesquisas até a data
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

- Não usa pesquisas estaduais de presidente; o mapa vem da geografia de 2022 mais o sorteio de choques.
- Não modela abstenção diferencial, que em 2022 favoreceu o candidato à direita em alguns estados.
- A correção do viés se baseia em 5 eleições (1º turno) e 2 eleições (confrontos antes do 1º turno).
- A Wikipédia é mantida por voluntários. O parser descarta linhas malformadas, mas pode haver erros de digitação na fonte.
