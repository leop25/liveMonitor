"""Aprendizado com eleições passadas (2006–2022).

Produz os parâmetros que o simulador usa:

1. Erro sistemático das pesquisas: quanto a média das pesquisas da última semana
   errou, por papel do candidato (PT, principal adversário, demais), no 1º turno;
   e quanto os confrontos diretos medidos *antes* do 1º turno erraram o 2º turno.
2. Qualidade de cada instituto: erro das últimas pesquisas de cada instituto
   (2010, 2018, 2022), convertido em peso.
3. Geografia do voto: "inclinação" de cada UF em relação ao país, sua persistência
   entre eleições (regressão AR(1) em logito) e a dispersão/correlação regional
   dos resíduos.
4. Efeito "candidato da casa": quanto um candidato secundário rende a mais no seu
   estado de origem.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from .agregador import ajustar_tendencia, series_primeiro_turno, series_segundo_turno
from .coleta import BRUTOS, ler_pesquisas, ler_resultados_uf, pesquisas_primeiro_turno, pesquisas_segundo_turno

REGIAO = {
    "AC": "N", "AP": "N", "AM": "N", "PA": "N", "RO": "N", "RR": "N", "TO": "N",
    "AL": "NE", "BA": "NE", "CE": "NE", "MA": "NE", "PB": "NE", "PE": "NE", "PI": "NE", "RN": "NE", "SE": "NE",
    "DF": "CO", "GO": "CO", "MT": "CO", "MS": "CO",
    "ES": "SE", "MG": "SE", "RJ": "SE", "SP": "SE",
    "PR": "S", "RS": "S", "SC": "S",
}

# Resultado oficial (TSE, % dos votos válidos) e papéis de cada eleição.
ELEICOES = {
    2006: dict(t1=date(2006, 10, 1), t2=date(2006, 10, 29), pt="Lula", adv="Alckmin",
               r1={"Lula": 48.61, "Alckmin": 41.64, "Heloísa Helena": 6.85, "Outros": 2.90},
               r2={"Lula": 60.83, "Alckmin": 39.17}),
    2010: dict(t1=date(2010, 10, 3), t2=date(2010, 10, 31), pt="Dilma", adv="Serra",
               r1={"Dilma": 46.91, "Serra": 32.61, "Marina Silva": 19.33, "Outros": 1.15},
               r2={"Dilma": 56.05, "Serra": 43.95}),
    2014: dict(t1=date(2014, 10, 5), t2=date(2014, 10, 26), pt="Dilma", adv="Aécio",
               r1={"Dilma": 41.59, "Aécio": 33.55, "Marina Silva": 21.32, "Outros": 3.54},
               r2={"Dilma": 51.64, "Aécio": 48.36}),
    2018: dict(t1=date(2018, 10, 7), t2=date(2018, 10, 28), pt="Haddad", adv="Jair Bolsonaro",
               r1={"Jair Bolsonaro": 46.03, "Haddad": 29.28, "Ciro Gomes": 12.47, "Alckmin": 4.76,
                   "João Amoêdo": 2.50, "Cabo Daciolo": 1.26, "Henrique Meirelles": 1.20,
                   "Marina Silva": 1.00, "Alvaro Dias": 0.80, "Outros": 1.73},
               r2={"Jair Bolsonaro": 55.13, "Haddad": 44.87}),
    2022: dict(t1=date(2022, 10, 2), t2=date(2022, 10, 30), pt="Lula", adv="Jair Bolsonaro",
               r1={"Lula": 48.43, "Jair Bolsonaro": 43.20, "Tebet": 4.16, "Ciro Gomes": 3.04, "Outros": 1.17},
               r2={"Lula": 50.90, "Jair Bolsonaro": 49.10}),
}

# Eleições sem página de pesquisas estruturada: médias finais registradas pela
# imprensa (votos válidos). 2006: Datafolha 29–30/set (Wikipédia pt, gráfico da
# página da eleição). 2014: Datafolha e Ibope de 3–4/out e 24–25/out, conforme
# divulgados (aproximação documentada no README).
FINAIS_MANUAIS = {
    (2006, 1): {"Lula": 51.6, "Alckmin": 36.8, "Heloísa Helena": 8.4, "Outros": 3.2},
    (2014, 1): {"Dilma": 45.0, "Aécio": 26.5, "Marina Silva": 24.0, "Outros": 4.5},
}

PARES_ANTES_T1_MANUAIS: dict[int, float] = {}  # sem série confiável para 2006/2014


def _valido(cands: dict, outros: float) -> dict:
    tot = sum(cands.values()) + (0 if np.isnan(outros) else outros)
    return {k: 100 * v / tot for k, v in cands.items()} if tot > 0 else {}


def _ultimas_por_instituto(df: pd.DataFrame, limite: date, dias: int) -> pd.DataFrame:
    lim = pd.Timestamp(limite)
    janela = df[(df.fim < lim) & (df.fim >= lim - pd.Timedelta(days=dias))]
    return janela.sort_values("fim").groupby("instituto").tail(1)


@dataclass
class Calibracao:
    erros_t1: pd.DataFrame                  # ano, papel, erro (pp de votos válidos)
    erros_h2h: pd.DataFrame                 # ano, erro na parcela do PT no 2º turno (pp)
    erros_institutos: pd.DataFrame          # ano, turno, instituto, rmse
    pesos_institutos: dict[str, float]
    vies_t1: dict[str, float]               # papel -> viés médio (pesquisa - urna)
    dp_t1: dict[str, float]                 # papel -> desvio (RMSE)
    vies_h2h: float
    dp_h2h: float
    rho_t1_t2: float                        # correlação entre erro do 1º turno e do 2º
    lean_coef: tuple[float, float]          # (intercepto, persistência) do AR(1) das UFs
    dp_estado: float                        # resíduo por UF (logito)
    dp_regiao: float                        # componente regional (logito)
    multiplicador_casa: float               # candidato da casa: razão UF/nacional
    resumo: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- pesquisas

def _carregar_pesquisas_historicas() -> dict[int, pd.DataFrame]:
    out = {}
    for ano in (2010, 2018, 2022):
        caminho = BRUTOS / f"pesquisas_{ano}.wiki"
        if caminho.exists():
            out[ano] = ler_pesquisas(caminho.read_text(encoding="utf-8"), ano)
    return out


def _papel(ano: int, cand: str) -> str:
    e = ELEICOES[ano]
    if cand == e["pt"]:
        return "pt"
    if cand == e["adv"]:
        return "adv"
    return "outros"


def erros_primeiro_turno(pesquisas: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Erro da média final (1 pesquisa por instituto, últimos 7 dias) e por instituto."""
    linhas, por_inst = [], []
    for ano, df in pesquisas.items():
        e = ELEICOES[ano]
        t1 = df[df.turno == 1]
        # apenas cenários com o elenco real (o PT e o adversário presentes)
        t1 = t1[t1.candidatos.map(lambda c: e["pt"] in c and e["adv"] in c)]
        finais = _ultimas_por_instituto(t1, e["t1"], 7)
        validos = [_valido(r.candidatos, r.get("outros", np.nan)) for _, r in finais.iterrows()]
        for (_, r), v in zip(finais.iterrows(), validos):
            erros = []
            for cand in (e["pt"], e["adv"]):
                if cand in v:
                    erros.append(v[cand] - e["r1"][cand])
            por_inst.append({"ano": ano, "turno": 1, "instituto": r.instituto,
                             "rmse": float(np.sqrt(np.mean(np.square(erros))))})
        media = pd.DataFrame(validos).mean()
        agreg = {"pt": 0.0, "adv": 0.0, "outros": 0.0}
        for cand, val in media.items():
            agreg[_papel(ano, cand)] += val
        real = {"pt": e["r1"][e["pt"]], "adv": e["r1"][e["adv"]]}
        real["outros"] = 100 - real["pt"] - real["adv"]
        for papel in agreg:
            linhas.append({"ano": ano, "papel": papel, "pesquisa": agreg[papel], "urna": real[papel],
                           "erro": agreg[papel] - real[papel], "fonte": "Wikipédia (média final)",
                           "n_pesquisas": len(finais)})
    for (ano, _), valores in FINAIS_MANUAIS.items():
        e = ELEICOES[ano]
        agreg = {"pt": 0.0, "adv": 0.0, "outros": 0.0}
        for cand, val in valores.items():
            agreg[_papel(ano, cand)] += val
        real = {"pt": e["r1"][e["pt"]], "adv": e["r1"][e["adv"]]}
        real["outros"] = 100 - real["pt"] - real["adv"]
        for papel in agreg:
            linhas.append({"ano": ano, "papel": papel, "pesquisa": agreg[papel], "urna": real[papel],
                           "erro": agreg[papel] - real[papel], "fonte": "registro de imprensa",
                           "n_pesquisas": 1})
    return pd.DataFrame(linhas).sort_values(["ano", "papel"]), pd.DataFrame(por_inst)


def erros_confronto_antes_t1(pesquisas: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Erro, no resultado do 2º turno, dos confrontos diretos medidos nos 10 dias antes do 1º turno.

    É exatamente a situação de hoje: prever o 2º turno com pesquisas feitas antes do 1º.
    Também devolve o erro das pesquisas da última semana do 2º turno por instituto.
    """
    linhas, por_inst = [], []
    for ano, df in pesquisas.items():
        e = ELEICOES[ano]
        par = {e["pt"], e["adv"]}
        t2 = df[(df.turno == 2) & df.candidatos.map(lambda c: set(c) == par)]
        real = e["r2"][e["pt"]]
        antes = _ultimas_por_instituto(t2[t2.fim < pd.Timestamp(e["t1"])], e["t1"], 10)
        if len(antes):
            parcelas = antes.candidatos.map(lambda c: 100 * c[e["pt"]] / (c[e["pt"]] + c[e["adv"]]))
            linhas.append({"ano": ano, "pesquisa": parcelas.mean(), "urna": real,
                           "erro": parcelas.mean() - real, "n_pesquisas": len(antes)})
        finais = _ultimas_por_instituto(t2, e["t2"], 7)
        for _, r in finais.iterrows():
            c = r.candidatos
            p = 100 * c[e["pt"]] / (c[e["pt"]] + c[e["adv"]])
            por_inst.append({"ano": ano, "turno": 2, "instituto": r.instituto, "rmse": abs(p - real)})
    return pd.DataFrame(linhas), pd.DataFrame(por_inst)


def pesos_institutos(erros: pd.DataFrame, k: float = 2.0) -> tuple[dict[str, float], float]:
    """Peso = (erro típico global / erro típico do instituto)², com encolhimento bayesiano."""
    global_mse = float(np.mean(np.square(erros.rmse)))
    pesos = {}
    for inst, g in erros.groupby("instituto"):
        n = len(g)
        mse = (np.sum(np.square(g.rmse)) + k * global_mse) / (n + k)
        pesos[inst] = float(np.clip(global_mse / mse, 0.4, 2.5))
    return pesos, float(np.sqrt(global_mse))


def backtest(pesquisas: dict[int, pd.DataFrame], erros_inst: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Roda o próprio agregador como se fosse a véspera do 1º turno de cada eleição passada.

    Os pesos de qualidade usados no ano Y são calculados sem o ano Y (leave-one-out),
    então o erro medido aqui é um erro fora da amostra do modelo completo — e é ele
    que calibra o erro sistemático da simulação de 2026.
    """
    linhas1, linhas2, trilhas = [], [], []
    for ano, df in pesquisas.items():
        e = ELEICOES[ano]
        pesos, _ = pesos_institutos(erros_inst[erros_inst.ano != ano])
        corte = pd.Timestamp(e["t1"]) - pd.Timedelta(days=1)
        antes = df[df.fim <= corte]
        cands = [c for c in e["r1"] if c != "Outros"]
        p1 = pesquisas_primeiro_turno(antes, cands)
        p1 = p1[p1.meio >= corte - pd.Timedelta(days=150)]
        series = series_primeiro_turno(p1, cands)
        est = {}
        for c, obs in series.items():
            if len(obs) < 3:
                continue
            tr = ajustar_tendencia(obs, pesos, ate=pd.Timestamp(e["t1"]))
            est[c] = tr.em(pd.Timestamp(e["t1"]))[0]
            if c in (e["pt"], e["adv"]):
                trilhas.append(pd.DataFrame({"ano": ano, "candidato": c, "data": tr.datas,
                                             "media": tr.media, "dp": tr.dp}))
        tot = sum(est.values())
        agreg = {"pt": 0.0, "adv": 0.0, "outros": 0.0}
        for c, v in est.items():
            agreg[_papel(ano, c)] += 100 * v / tot
        real = {"pt": e["r1"][e["pt"]], "adv": e["r1"][e["adv"]]}
        real["outros"] = 100 - real["pt"] - real["adv"]
        for papel in agreg:
            linhas1.append({"ano": ano, "papel": papel, "pesquisa": agreg[papel], "urna": real[papel],
                            "erro": agreg[papel] - real[papel], "fonte": "backtest do modelo"})
        p2 = pesquisas_segundo_turno(antes)
        s2 = series_segundo_turno(p2[p2.meio >= corte - pd.Timedelta(days=150)], e["pt"])
        if e["adv"] in s2 and len(s2[e["adv"]]) >= 3:
            tr = ajustar_tendencia(s2[e["adv"]], pesos, ate=pd.Timestamp(e["t2"]))
            m, _ = tr.em(pd.Timestamp(e["t2"]))
            linhas2.append({"ano": ano, "pesquisa": m, "urna": e["r2"][e["pt"]],
                            "erro": m - e["r2"][e["pt"]], "fonte": "backtest do modelo"})
    trilha = pd.concat(trilhas) if trilhas else pd.DataFrame()
    return pd.DataFrame(linhas1), pd.DataFrame(linhas2), trilha


# ----------------------------------------------------------------------------- geografia

def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def inclinacoes_estaduais(res: pd.DataFrame) -> pd.DataFrame:
    """Parcela do PT no 2º turno por UF e sua inclinação (logito UF - logito Brasil)."""
    pt = {2002: "Lula", 2006: "Lula", 2010: "Dilma", 2014: "Dilma", 2018: "Haddad", 2022: "Lula"}
    linhas = []
    for ano, cand in pt.items():
        r = res[(res.ano == ano) & (res.turno == 2)]
        tot = r.groupby("uf").votos.sum()
        v = r[r.candidato == cand].set_index("uf").votos
        nac = v.sum() / tot.sum()
        for uf in tot.index:
            p = v[uf] / tot[uf]
            linhas.append({"ano": ano, "uf": uf, "parcela": p, "validos": tot[uf],
                           "lean": _logit(p) - _logit(nac)})
    return pd.DataFrame(linhas)


def persistencia_geografica(leans: pd.DataFrame) -> dict:
    """AR(1) das inclinações entre eleições consecutivas (2006→…→2022) e resíduos."""
    anos = sorted(leans.ano.unique())
    pares = []
    for a0, a1 in zip(anos[:-1], anos[1:]):
        if a0 < 2006:  # 2002→2006 foi o realinhamento Nordeste/PT; fora do regime atual
            continue
        x = leans[leans.ano == a0].set_index("uf").lean
        y = leans[leans.ano == a1].set_index("uf").lean
        pares.append(pd.DataFrame({"x": x, "y": y, "par": f"{a0}→{a1}"}))
    d = pd.concat(pares).reset_index().rename(columns={"index": "uf"})
    X = np.column_stack([np.ones(len(d)), d.x])
    coef, *_ = np.linalg.lstsq(X, d.y, rcond=None)
    d["res"] = d.y - X @ coef
    d["regiao"] = d.uf.map(REGIAO)
    # decomposição do resíduo: componente regional (por par de eleições) + idiossincrático
    reg = d.groupby(["par", "regiao"]).res.transform("mean")
    dp_reg = float(np.std(d.groupby(["par", "regiao"]).res.mean(), ddof=1))
    dp_est = float(np.std(d.res - reg, ddof=1))
    return {"coef": (float(coef[0]), float(coef[1])), "dp_estado": dp_est, "dp_regiao": dp_reg,
            "rmse_total": float(np.sqrt(np.mean(d.res ** 2))), "dados": d}


def efeito_candidato_casa(res: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Razão (votação na UF de origem / votação nacional) de candidatos secundários."""
    casos = [(2002, "Garotinho", "RJ"), (2002, "Ciro Gomes", "CE"), (2006, "Heloísa", "AL"),
             (2010, "Marina", "AC"), (2014, "Marina", "AC"), (2018, "Ciro", "CE"),
             (2022, "Tebet", "MS")]
    linhas = []
    for ano, cand, uf in casos:
        r = res[(res.ano == ano) & (res.turno == 1)]
        tot = r.groupby("uf").votos.sum()
        v = r[r.candidato == cand].set_index("uf").votos
        nac = v.sum() / tot.sum()
        linhas.append({"ano": ano, "candidato": cand, "uf": uf, "nacional": 100 * nac,
                       "na_uf": 100 * v[uf] / tot[uf], "razao": (v[uf] / tot[uf]) / nac})
    d = pd.DataFrame(linhas)
    return float(np.exp(np.mean(np.log(d.razao)))), d


# ----------------------------------------------------------------------------- tudo junto

def calibrar(encolhimento_vies: float = 2.0) -> Calibracao:
    pesquisas = _carregar_pesquisas_historicas()
    res = ler_resultados_uf()

    e1_simples, inst1 = erros_primeiro_turno(pesquisas)
    e2_simples, inst2 = erros_confronto_antes_t1(pesquisas)
    erros_inst = pd.concat([inst1, inst2])
    pesos, rmse_global = pesos_institutos(erros_inst)

    # Erro fora da amostra do modelo (2010/2018/2022) + registros de imprensa (2006/2014)
    bt1, bt2, trilha = backtest(pesquisas, erros_inst)
    e1 = pd.concat([bt1, e1_simples[e1_simples.fonte == "registro de imprensa"]], ignore_index=True)
    e2 = bt2

    vies_t1, dp_t1 = {}, {}
    for papel, g in e1.groupby("papel"):
        n = len(g)
        # média encolhida para zero: com poucas eleições, não confiar 100% no padrão
        vies_t1[papel] = float(g.erro.mean() * n / (n + encolhimento_vies))
        dp_t1[papel] = float(np.sqrt(np.mean(np.square(g.erro))))
    n2 = len(e2)
    vies_h2h = float(e2.erro.mean() * n2 / (n2 + encolhimento_vies)) if n2 else 0.0
    dp_h2h = float(np.sqrt(np.mean(np.square(e2.erro)))) if n2 else 3.0

    # correlação entre o erro na margem do 1º turno (PT - adv) e o erro do confronto
    m1 = e1.pivot_table(index="ano", columns="papel", values="erro")
    marg = (m1["pt"] - m1["adv"]).rename("m1")
    j = pd.concat([marg, e2.set_index("ano").erro.rename("h2h")], axis=1).dropna()
    rho = float(np.corrcoef(j.m1, j.h2h)[0, 1]) if len(j) >= 3 else 0.5
    rho = float(np.clip(rho * len(j) / (len(j) + 1), 0.0, 0.9))  # encolhe com n pequeno

    leans = inclinacoes_estaduais(res)
    geo = persistencia_geografica(leans)
    mult, casos_casa = efeito_candidato_casa(res)

    return Calibracao(
        erros_t1=e1, erros_h2h=e2, erros_institutos=pd.concat([inst1, inst2]),
        pesos_institutos=pesos, vies_t1=vies_t1, dp_t1=dp_t1, vies_h2h=vies_h2h, dp_h2h=dp_h2h,
        rho_t1_t2=rho, lean_coef=geo["coef"], dp_estado=geo["dp_estado"], dp_regiao=geo["dp_regiao"],
        multiplicador_casa=mult,
        resumo={"rmse_global_institutos": rmse_global, "erros_t1_media_simples": e1_simples,
                "erros_h2h_media_simples": e2_simples, "trilha_backtest": trilha, "leans": leans, "geo": geo["dados"],
                "casos_casa": casos_casa, "resultados_uf": res, "pesquisas": pesquisas},
    )
