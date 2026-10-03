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


def test_le_arquivo_simplificado_do_tse():
    from modelo.pos_primeiro_turno import _ler_json_tse
    d = {"pst": "87,35", "cand": [{"nm": "LULA", "vap": "1000"}, {"nm": "FLÁVIO BOLSONARO", "vap": "900"},
                                  {"nm": "RONALDO CAIADO", "vap": "50"}, {"nm": "FULANO", "vap": "10"}]}
    info = _ler_json_tse(d)
    assert info["pst"] == pytest.approx(0.8735)
    assert info["votos"] == {"Lula": 1000.0, "Flávio Bolsonaro": 900.0, "Caiado": 50.0, "Outros": 10.0}


def test_segundo_turno_a_partir_das_urnas_acerta_2018_e_2022():
    from modelo import pos_primeiro_turno as pp
    calp = pp.calibrar_pos()
    res = coleta.ler_resultados_uf()
    for ano, p, a, r, vence_p in [(2022, "Lula", "Bolsonaro", 0.38, True), (2018, "Haddad", "Bolsonaro", 0.345, False)]:
        r1 = res[(res.ano == ano) & (res.turno == 1)].pivot_table(index="uf", columns="candidato", values="votos").fillna(0)
        tab = pd.DataFrame({p: r1[p], a: r1[a], "Outros": r1.drop(columns=[p, a]).sum(axis=1)})
        pesos = tab.sum(axis=1).to_numpy()
        t1 = np.repeat((tab.to_numpy() / pesos[:, None])[None], 4000, axis=0)
        out = pp.prever_segundo_turno(t1, pesos, list(tab.index), [p, a, "Outros"], p, a, r, 0.07, calp,
                                      np.random.default_rng(0))
        assert (out["vence_principal"].mean() > 0.5) == vence_p


def test_projecao_da_apuracao_corrige_ordem_de_apuracao():
    """Com o Nordeste pouco apurado, a parcial nacional subestima o PT; a projeção por UF não."""
    from modelo import pos_primeiro_turno as pp
    res = coleta.ler_resultados_uf()
    r1 = res[(res.ano == 2022) & (res.turno == 1)].pivot_table(index="uf", columns="candidato", values="votos").fillna(0)
    ne = {"BA", "PE", "CE", "MA", "PI", "PB", "RN", "AL", "SE"}
    apur = pd.DataFrame({"uf": r1.index, "pst": [0.3 if u in ne else 0.9 for u in r1.index],
                         "Lula": r1.Lula, "Flávio Bolsonaro": r1.Bolsonaro,
                         "Outros": r1.drop(columns=["Lula", "Bolsonaro"]).sum(axis=1)})
    apur[["Lula", "Flávio Bolsonaro", "Outros"]] = apur[["Lula", "Flávio Bolsonaro", "Outros"]].mul(apur.pst, axis=0)
    previa = {u: {"t1": {"Lula": 45, "Flávio Bolsonaro": 45, "Outros": 10}, "peso": float(r1.loc[u].sum())} for u in r1.index}
    t1, pesos, ufs, cc = pp.projetar_primeiro_turno(apur.reset_index(drop=True), previa, ["Lula", "Flávio Bolsonaro"],
                                                    np.random.default_rng(0), 2000)
    nac = np.einsum("nsk,s->nk", t1, pesos / pesos.sum()).mean(axis=0) * 100
    parcial = 100 * apur.Lula.sum() / apur[["Lula", "Flávio Bolsonaro", "Outros"]].to_numpy().sum()
    assert abs(nac[0] - 48.43) < abs(parcial - 48.43)
