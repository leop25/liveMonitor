"""Avaliação fora da amostra das escolhas do modelo (2010, 2018, 2022).

Para cada eleição Y, o modelo completo é rodado como se fosse a véspera do 1º turno de Y,
usando só as pesquisas publicadas até ali e uma calibração que exclui Y (viés, dispersão e
pesos dos institutos vêm das outras eleições). Comparando com a urna, mede-se:

- erro do 1º turno (PT, principal adversário) e do 2º turno (parcela do PT);
- log-loss e Brier da probabilidade de vitória do PT;
- cobertura dos intervalos de 80%.

Cada variante (peso da correção de viés, inclusão de pesquisas online, peso de institutos sem
histórico, ancoragem por qualidade) é avaliada da mesma forma, para que a escolha usada em 2026
seja a que mais acertaria no passado, e não uma opinião.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .agregador import ajustar_tendencia, series_primeiro_turno, series_segundo_turno
from .coleta import pesquisas_primeiro_turno, pesquisas_segundo_turno
from .historico import ELEICOES, Calibracao, calibrar, pesos_institutos
from .simulacao import Configuracao, Entrada, simular_nacional

ANOS = (2010, 2018, 2022)
ONLINE_HIST = {"AtlasIntel", "Revista Fórum/Offerwise"}   # institutos online nas eleições passadas


@dataclass
class Variante:
    nome: str
    peso_vies: float = 0.5
    encolhimento: float = 2.0       # k do encolhimento do viés: média × n/(n+k)
    sem_online: bool = False
    peso_novo: float = 1.0          # peso de instituto sem histórico
    usar_pesos: bool = True         # ancorar/ponderar pela qualidade histórica


def _cal_sem(cal: Calibracao, ano: int, k: float) -> Calibracao:
    """Calibração do erro sistemático excluindo a eleição `ano`."""
    c = copy.copy(cal)
    e1 = cal.erros_t1[cal.erros_t1.ano != ano]
    c.vies_t1, c.dp_t1 = {}, {}
    for papel, g in e1.groupby("papel"):
        n = len(g)
        c.vies_t1[papel] = float(g.erro.mean() * n / (n + k))
        c.dp_t1[papel] = float(np.sqrt(np.mean(np.square(g.erro))))
    e2 = cal.erros_h2h[cal.erros_h2h.ano != ano]
    if len(e2):
        n2 = len(e2)
        c.vies_h2h = float(e2.erro.mean() * n2 / (n2 + k))
        c.dp_h2h = float(max(np.sqrt(np.mean(np.square(e2.erro))), 2.5))
    else:
        c.vies_h2h, c.dp_h2h = 0.0, 4.0
    return c


def _entrada(ano: int, df: pd.DataFrame, pesos: dict, var: Variante) -> tuple[Entrada, list[str]]:
    e = ELEICOES[ano]
    corte = pd.Timestamp(e["t1"]) - pd.Timedelta(days=1)
    antes = df[df.fim <= corte]
    if var.sem_online:
        antes = antes[~antes.instituto.isin(ONLINE_HIST)]
    cands = [c for c in e["r1"] if c != "Outros"]
    p1 = pesquisas_primeiro_turno(antes, cands)
    p1 = p1[p1.meio >= corte - pd.Timedelta(days=150)]
    w = pesos if var.usar_pesos else {}
    m, d = {}, {}
    for c, obs in series_primeiro_turno(p1, cands).items():
        if len(obs) < 3:
            continue
        tr = ajustar_tendencia(obs, w, ate=pd.Timestamp(e["t1"]), peso_padrao=var.peso_novo)
        m[c], d[c] = tr.em(pd.Timestamp(e["t1"]))
    usados = [c for c in cands if c in m]
    tot = sum(m.values())
    m = {c: 100 * v / tot for c, v in m.items()}
    if "Outros" not in m:
        m["Outros"], d["Outros"] = 0.5, 0.5
    t2m, t2d = {}, {}
    p2 = pesquisas_segundo_turno(antes)
    s2 = series_segundo_turno(p2[p2.meio >= corte - pd.Timedelta(days=150)], e["pt"])
    if e["adv"] in s2 and len(s2[e["adv"]]) >= 3:
        tr = ajustar_tendencia(s2[e["adv"]], w, ate=pd.Timestamp(e["t2"]), peso_padrao=var.peso_novo)
        t2m[e["adv"]], t2d[e["adv"]] = tr.em(pd.Timestamp(e["t2"]))
    return Entrada(m, d, t2m, t2d), usados


def avaliar(variantes: list[Variante], n_sim: int = 20000, semente: int = 7) -> pd.DataFrame:
    cal = calibrar()
    pesquisas = cal.resumo["pesquisas"]
    erros_inst = cal.erros_institutos
    linhas = []
    for var in variantes:
        for ano in ANOS:
            e = ELEICOES[ano]
            pesos, _ = pesos_institutos(erros_inst[erros_inst.ano != ano])
            ent, cands = _entrada(ano, pesquisas[ano], pesos, var)
            calY = _cal_sem(cal, ano, var.encolhimento)
            cfg = Configuracao(candidatos=cands, principal=e["pt"], adversario=e["adv"],
                               blocos={e["pt"]: "esquerda", e["adv"]: "direita"}, estado_origem={},
                               n_sim=n_sim, peso_vies=var.peso_vies, semente=semente)
            r = simular_nacional(cfg, ent, calY, np.random.default_rng(semente))
            todos = cands + ["Outros"]
            ip, ia = todos.index(e["pt"]), todos.index(e["adv"])
            t1 = r.t1.to_numpy()
            real_pt, real_adv = e["r1"][e["pt"]], e["r1"][e["adv"]]
            lf = (r.finalistas[:, 0] == ip) & (r.finalistas[:, 1] == ia) & ~r.decidido_t1
            t2 = r.t2_parcela[lf]
            real2 = e["r2"][e["pt"]]
            p_pt = float(np.clip(np.mean(r.vencedor == ip), 0.01, 0.99))
            venceu = real2 > 50
            linhas.append({
                "variante": var.nome, "ano": ano,
                "erro_t1_pt": t1[:, ip].mean() - real_pt, "erro_t1_adv": t1[:, ia].mean() - real_adv,
                "erro_t2": (t2.mean() - real2) if (len(t2) and e["adv"] in ent.t2_media) else np.nan,
                "p_pt": p_pt, "logloss": -np.log(p_pt if venceu else 1 - p_pt),
                "brier": (p_pt - venceu) ** 2,
                "cobre_t1_pt": np.percentile(t1[:, ip], 10) <= real_pt <= np.percentile(t1[:, ip], 90),
                "cobre_t1_adv": np.percentile(t1[:, ia], 10) <= real_adv <= np.percentile(t1[:, ia], 90),
                "cobre_t2": (np.percentile(t2, 10) <= real2 <= np.percentile(t2, 90)) if len(t2) else np.nan,
            })
    return pd.DataFrame(linhas)


def resumir(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("variante", sort=False)
    return pd.DataFrame({
        "rmse_t1_pt": g.erro_t1_pt.apply(lambda x: np.sqrt(np.mean(x ** 2))),
        "rmse_t1_adv": g.erro_t1_adv.apply(lambda x: np.sqrt(np.mean(x ** 2))),
        "rmse_t1_margem": g.apply(lambda x: np.sqrt(np.mean((x.erro_t1_pt - x.erro_t1_adv) ** 2))),
        "rmse_t2": g.erro_t2.apply(lambda x: np.sqrt(np.nanmean(x ** 2))),
        "logloss": g.logloss.mean(), "brier": g.brier.mean(),
        "cobertura_80": g.apply(lambda x: np.nanmean(pd.concat([x.cobre_t1_pt, x.cobre_t1_adv, x.cobre_t2]).astype(float))),
    })
