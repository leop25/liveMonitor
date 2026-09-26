# Previsão Presidencial 2026

Modelo probabilístico para a eleição presidencial brasileira de 2026 (1º turno em 4/out, 2º turno em 25/out).
Ele agrega todas as pesquisas nacionais, corrige o viés de cada instituto, aprende com os erros das pesquisas
em 2006–2022 e simula a eleição inteira milhares de vezes: 1º turno, 2º turno e resultado nas 27 UFs.

```bash
cd eleicoes2026
pip install -r requirements.txt
python prever.py --atualizar      # baixa as pesquisas mais recentes e gera saida/painel.html
```

## Resultado atual (pesquisas até 24/09/2026, 20 mil simulações)

| Candidato | 1º turno (válidos) | Intervalo 80% | Vai ao 2º turno | Eleito |
|---|---|---|---|---|
| Flávio Bolsonaro (PL) | 44,2% | 39,2–49,3% | 90% | **70%** |
| Lula (PT) | 42,4% | 37,8–47,1% | 90% | **30%** |
| Augusto Cury (Avante) | 4,4% | 1,6–7,3% | <1% | <1% |
| Renan Santos (Missão) | 3,6% | 2,0–5,2% | <1% | <1% |
| Caiado (PSD) | 3,3% | 1,8–4,9% | <1% | <1% |
| Zema (Novo) | 1,1% | 0,3–2,0% | <1% | <1% |

Chance de 2º turno: 90%. No confronto Lula × Flávio, Lula tem em média 48,0% dos válidos.

**A previsão depende muito de uma escolha:** quanto confiar no padrão histórico de erro das pesquisas.

| Correção do viés histórico | Lula | Flávio |
|---|---|---|
| Nenhuma (pesquisas certas em média) | 52% | 48% |
| Metade | 41% | 59% |
| **Padrão do modelo** | **30%** | **70%** |
| 1,5× | 20% | 80% |

Em todas as cinco eleições de 2006 a 2022, a projeção final subestimou o principal candidato contra o PT no
1º turno (de 1,5 a 7 pontos). Nos confrontos de 2º turno medidos antes do 1º turno, o PT foi superestimado em
3,2 pontos (2018) e 6,1 (2022). O modelo aplica esse viés só em parte (média encolhida para zero) e mantém a
incerteza. Se os institutos corrigiram seus métodos depois de 2022, a linha "Nenhuma" fica mais perto da verdade.

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
python prever.py --ate 2026-08-15     # "volta no tempo": só pesquisas até a data
python -m pytest testes               # testes
```

Candidatos, blocos ideológicos e estados de origem ficam em `CONFIG_2026` (`modelo/previsao.py`).

## Dados

- `dados/brutos/pesquisas_*.wiki`: wikitexto das páginas "Opinion polling for the <ano> Brazilian presidential
  election" (Wikipédia em inglês). `--atualizar` baixa de novo a de 2026.
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
