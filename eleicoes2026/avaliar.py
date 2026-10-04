#!/usr/bin/env python3
"""Testa as escolhas do modelo fora da amostra (2010, 2018, 2022) e grava dados/processados/avaliacao.csv.

O painel mostra essa tabela; as escolhas padrão de modelo/previsao.py seguem a variante de menor erro.
"""
from pathlib import Path

import pandas as pd

from modelo.avaliacao import Variante, avaliar, resumir

RAIZ = Path(__file__).resolve().parent

VARIANTES = [
    Variante("Sem correção de viés", peso_vies=0.0),
    Variante("Metade da correção", peso_vies=0.5),
    Variante("Correção inteira", peso_vies=1.0),
    Variante("Correção inteira, sem pesquisas online", peso_vies=1.0, sem_online=True),
    Variante("Correção inteira, sem pesos de qualidade", peso_vies=1.0, usar_pesos=False),
]
ESCOLHIDA = "Metade da correção"

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    d = avaliar(VARIANTES, n_sim=20000)
    r = resumir(d).reset_index()
    r["escolhida"] = r.variante == ESCOLHIDA
    print(d.pivot_table(index="variante", columns="ano", values=["erro_t1_pt", "erro_t1_adv", "erro_t2"],
                        sort=False).round(2).to_string())
    print(r.round(3).to_string(index=False))
    r.to_csv(RAIZ / "dados" / "processados" / "avaliacao.csv", index=False)
    d.to_csv(RAIZ / "dados" / "processados" / "avaliacao_por_ano.csv", index=False)
