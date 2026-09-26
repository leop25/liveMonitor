"""Agregador bayesiano de pesquisas: tendência latente + efeito de instituto.

Para cada série (ex.: votos válidos de um candidato no 1º turno, ou a parcela de
Lula num confronto de 2º turno) o modelo é

    y_i = x[t_i] + h[instituto_i] + e_i,        e_i ~ N(0, v_i)
    x[t] = x[t-1] + w_t,                        w_t ~ N(0, q)

- x é a intenção de voto real (passeio aleatório diário) — estimada por um
  suavizador de Kalman (Rauch–Tung–Striebel);
- h é o viés sistemático de cada instituto ("house effect"), com priori
  N(0, τ²) e restrição de soma zero ponderada pela qualidade histórica do
  instituto (os institutos que acertaram mais no passado ancoram o nível);
- v_i combina erro amostral (com efeito de desenho) e erro não amostral,
  dividido pelo peso de qualidade do instituto;
- q (volatilidade diária) é escolhido por máxima verossimilhança.

Os parâmetros são estimados por maximização alternada (tipo EM), o que é rápido,
estável e não exige bibliotecas de inferência bayesiana.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EFEITO_DESENHO = 1.6        # amostras por cotas/painéis têm variância maior que a aleatória simples
ERRO_NAO_AMOSTRAL = 1.2     # pp (desvio-padrão) somado em quadratura
TAU_INSTITUTO = 2.5         # pp: priori do efeito de instituto
GRADE_Q = np.array([0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.3])  # pp²/dia


@dataclass
class Tendencia:
    datas: pd.DatetimeIndex
    media: np.ndarray
    dp: np.ndarray
    efeitos: pd.Series            # por instituto (pp; positivo = superestima)
    q: float
    obs: pd.DataFrame             # observações usadas, com valor ajustado

    def em(self, dia: pd.Timestamp) -> tuple[float, float]:
        """Estimativa e desvio no dia pedido; extrapola como passeio aleatório."""
        ultimo = self.datas[-1]
        if dia <= ultimo:
            i = int(self.datas.get_indexer([dia], method="nearest")[0])
            return float(self.media[i]), float(self.dp[i])
        dias = (dia - ultimo).days
        return float(self.media[-1]), float(np.sqrt(self.dp[-1] ** 2 + self.q * dias))


def _kalman(y, t, v, n_dias, q, x0, p0):
    """Filtro + suavizador RTS numa grade diária. Devolve médias, variâncias e log-verossimilhança."""
    ordem = np.argsort(t, kind="stable")
    y, t, v = y[ordem], t[ordem], v[ordem]
    mf = np.empty(n_dias)
    pf = np.empty(n_dias)
    mp = np.empty(n_dias)
    pp = np.empty(n_dias)
    m, p = x0, p0
    loglik = 0.0
    k = 0
    for d in range(n_dias):
        if d > 0:
            p = p + q
        mp[d], pp[d] = m, p
        while k < len(t) and t[k] == d:
            s = p + v[k]
            inov = y[k] - m
            loglik += -0.5 * (np.log(2 * np.pi * s) + inov ** 2 / s)
            g = p / s
            m = m + g * inov
            p = (1 - g) * p
            k += 1
        mf[d], pf[d] = m, p
    ms = mf.copy()
    ps = pf.copy()
    for d in range(n_dias - 2, -1, -1):
        c = pf[d] / pp[d + 1]
        ms[d] = mf[d] + c * (ms[d + 1] - mp[d + 1])
        ps[d] = pf[d] + c ** 2 * (ps[d + 1] - pp[d + 1])
    return ms, ps, loglik


def ajustar_tendencia(obs: pd.DataFrame, pesos: dict[str, float], ate: pd.Timestamp | None = None,
                      iteracoes: int = 25, peso_padrao: float = 1.0) -> Tendencia:
    """Ajusta a tendência de uma série.

    `obs` precisa das colunas: meio (data), instituto, valor (pp), n_efetivo (tamanho
    de amostra que corresponde ao denominador do valor).
    """
    obs = obs.dropna(subset=["valor"]).copy()
    inicio = obs.meio.min().normalize()
    fim = max(obs.meio.max().normalize(), ate or obs.meio.max().normalize())
    datas = pd.date_range(inicio, fim, freq="D")
    t = ((obs.meio.dt.normalize() - inicio).dt.days).to_numpy()
    y = obs.valor.to_numpy(float)
    n = obs.n_efetivo.fillna(1500).clip(lower=300).to_numpy(float)
    w_inst = obs.instituto.map(pesos).fillna(peso_padrao).to_numpy(float)
    p = np.clip(y, 1, 99)
    v = (EFEITO_DESENHO * p * (100 - p) / n + ERRO_NAO_AMOSTRAL ** 2) / w_inst

    inst = obs.instituto.to_numpy()
    nomes = sorted(set(inst))
    idx = {s: i for i, s in enumerate(nomes)}
    ii = np.array([idx[s] for s in inst])
    # peso de ancoragem: qualidade × volume (limitado) de pesquisas
    contagem = np.bincount(ii, minlength=len(nomes))
    ancora = np.array([pesos.get(s, peso_padrao) for s in nomes]) * np.minimum(contagem, 6) / 6

    h = np.zeros(len(nomes))
    q = 0.02
    x0, p0 = float(np.median(y[: max(3, len(y) // 10)])), 25.0
    for it in range(iteracoes):
        yc = y - h[ii]
        if it % 5 == 0:  # reescolhe q por máxima verossimilhança
            lls = [(_kalman(yc, t, v, len(datas), qq, x0, p0)[2], qq) for qq in GRADE_Q]
            q = max(lls)[1]
        ms, ps, _ = _kalman(yc, t, v, len(datas), q, x0, p0)
        r = y - ms[t]
        num = np.bincount(ii, weights=r / v, minlength=len(nomes))
        den = np.bincount(ii, weights=1 / v, minlength=len(nomes)) + 1 / TAU_INSTITUTO ** 2
        h = num / den
        h = h - np.sum(ancora * h) / np.sum(ancora)
    ms, ps, _ = _kalman(y - h[ii], t, v, len(datas), q, x0, p0)
    # A incerteza do nível de ancoragem (o erro comum a todos os institutos) não entra
    # aqui: é o erro sistemático calibrado nas eleições passadas e somado na simulação.
    obs["ajustado"] = y - h[ii]
    obs["efeito"] = h[ii]
    return Tendencia(datas=datas, media=ms, dp=np.sqrt(ps),
                     efeitos=pd.Series(h, index=nomes).sort_values(), q=float(q), obs=obs)


# ----------------------------------------------------------------------------- séries

def series_primeiro_turno(p1: pd.DataFrame, candidatos: list[str]) -> dict[str, pd.DataFrame]:
    """Converte cada pesquisa em votos válidos (%) por candidato — como o TSE divulga."""
    cols = candidatos + ["Outros"]
    base = p1[cols].fillna(0.0)
    total = base.sum(axis=1)
    fração_valida = total / 100.0
    series = {}
    for c in cols:
        presente = p1[c].notna() if c != "Outros" else pd.Series(True, index=p1.index)
        df = pd.DataFrame({
            "meio": p1.meio, "instituto": p1.instituto,
            "valor": 100 * base[c] / total,
            "n_efetivo": p1.amostra.fillna(2000) * fração_valida.clip(0.5, 1.0),
        })[presente]
        series[c] = df.reset_index(drop=True)
    return series


def series_segundo_turno(p2: pd.DataFrame, principal: str) -> dict[str, pd.DataFrame]:
    """Parcela de `principal` nos votos válidos de cada confronto `principal x rival`."""
    series = {}
    for _, g in p2.groupby(["a", "b"]):
        a, b = g.a.iloc[0], g.b.iloc[0]
        if principal not in (a, b):
            continue
        rival = b if a == principal else a
        vp = np.where(g.a == principal, g.va, g.vb)
        vr = np.where(g.a == principal, g.vb, g.va)
        df = pd.DataFrame({"meio": g.meio.values, "instituto": g.instituto.values,
                           "valor": 100 * vp / (vp + vr),
                           "n_efetivo": g.amostra.fillna(2000).values * (vp + vr) / 100})
        series[rival] = df
    return series
