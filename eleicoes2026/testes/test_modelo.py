"""Testes rápidos: rode com `python -m pytest testes` a partir de eleicoes2026/."""
import numpy as np
import pandas as pd
import pytest

from modelo import coleta
from modelo.agregador import ajustar_tendencia
from modelo.historico import ELEICOES, calibrar
from modelo.wikitabela import extrair_tabelas, texto_limpo


def test_rowspan_expandido():
    wiki = """{| class="wikitable"
! Instituto !! A !! B
|-
| rowspan="2" |''X''
| 40 || 35
|-
| 41 || 34
|}"""
    g = extrair_tabelas(wiki)[0].grade()
    assert [texto_limpo(c.texto) for c in g[2]] == ["X", "41", "34"]


def test_datas_e_numeros():
    ini, fim = coleta._datas("30 Sep – 1 Oct", 2022)
    assert (ini.month, ini.day, fim.month, fim.day, fim.year) == (9, 30, 10, 1, 2022)
    assert coleta._numero("style=x | '''43,2%'''".split("|")[1]) == 43.2


def test_resultados_batem_com_tse():
    r = coleta.ler_resultados_uf()
    t = r[(r.ano == 2022) & (r.turno == 2)]
    lula = t[t.candidato == "Lula"].votos.sum() / t.votos.sum()
    assert abs(100 * lula - ELEICOES[2022]["r2"]["Lula"]) < 0.05
    assert t.uf.nunique() == 27


def test_pesquisas_2026():
    df = coleta.ler_pesquisas((coleta.BRUTOS / "pesquisas_2026.wiki").read_text(encoding="utf-8"), 2026)
    p1 = coleta.pesquisas_primeiro_turno(df, ["Lula", "Flávio Bolsonaro", "Augusto Cury", "Renan Santos", "Caiado", "Zema"])
    assert len(p1) > 50
    assert p1.amostra.between(500, 30000).all()
    assert (p1[["Lula", "Flávio Bolsonaro"]].sum(axis=1) < 100).all()


def test_agregador_recupera_nivel_e_efeitos():
    rng = np.random.default_rng(0)
    dias = pd.date_range("2026-06-01", periods=90)
    verdade = 40 + np.linspace(0, 3, 90)
    efeitos = {"A": 2.0, "B": -2.0, "C": 0.0}
    linhas = []
    for i in range(0, 90, 2):
        for inst, h in efeitos.items():
            linhas.append({"meio": dias[i], "instituto": inst, "valor": verdade[i] + h + rng.normal(0, 1), "n_efetivo": 2000})
    tr = ajustar_tendencia(pd.DataFrame(linhas), {"A": 1, "B": 1, "C": 1})
    assert abs(tr.media[-1] - verdade[-1]) < 1.0
    assert tr.efeitos["A"] - tr.efeitos["B"] == pytest.approx(4.0, abs=0.8)


def test_calibracao_plausivel():
    c = calibrar()
    assert c.vies_t1["adv"] < 0                  # antipetismo subestimado historicamente
    assert 0.7 < c.lean_coef[1] < 1.1            # geografia persistente
    assert 1.5 < c.multiplicador_casa < 3.5
