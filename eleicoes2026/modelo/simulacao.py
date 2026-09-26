"""Simulação Monte Carlo da eleição: 1º turno, 2º turno e mapa por UF.

Cada simulação sorteia, de forma coerente entre si:
  1. a intenção de voto real no dia da eleição (incerteza da tendência + deriva
     até o dia da votação, vinda do agregador);
  2. um erro sistemático das pesquisas com a média e a dispersão observadas nas
     eleições de 2006–2022 (caudas grossas: t de Student com 5 g.l.);
  3. o 2º turno, a partir dos confrontos diretos, com erro correlacionado ao do
     1º turno (se as pesquisas erraram num sentido no 1º, tendem a errar no 2º);
  4. o resultado em cada UF: inclinação histórica da UF (2022, com a persistência
     estimada) + choques regionais e estaduais, reajustados para somar o total
     nacional sorteado.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .historico import REGIAO, Calibracao

GL_T = 5  # graus de liberdade das caudas do erro sistemático


def _t(rng, n, gl=GL_T):
    """t de Student com variância unitária."""
    return rng.standard_t(gl, size=n) / np.sqrt(gl / (gl - 2))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _expit(x):
    return 1 / (1 + np.exp(-x))


@dataclass
class Configuracao:
    candidatos: list[str]
    principal: str                      # candidato do PT (referência dos confrontos)
    adversario: str                     # principal adversário (papel "adv" na calibração)
    blocos: dict[str, str]              # candidato -> bloco ideológico (transferência de votos)
    estado_origem: dict[str, str]       # candidato -> UF de origem
    n_sim: int = 20000
    peso_vies: float = 1.0              # 0 = ignora o viés histórico; 1 = usa o viés encolhido
    semente: int = 2026

    @property
    def blocos_lista(self) -> list[str]:
        return [self.blocos.get(c, "centro") for c in self.candidatos] + ["outros"]


@dataclass
class Entrada:
    t1_media: dict[str, float]          # votos válidos no dia da eleição (%)
    t1_dp: dict[str, float]
    t2_media: dict[str, float]          # rival -> parcela do principal no confronto (%)
    t2_dp: dict[str, float]


@dataclass
class Resultado:
    t1: pd.DataFrame                    # n_sim x candidatos (votos válidos, %)
    finalistas: np.ndarray              # n_sim x 2 (índices)
    decidido_t1: np.ndarray             # bool
    vencedor: np.ndarray                # índice do eleito
    t2_parcela: np.ndarray              # parcela do finalista 0 (principal quando presente)
    estados_t1: dict = field(default_factory=dict)
    estados_t2: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- transferência

def _parcela_por_transferencia(t1: np.ndarray, i: int, j: int, blocos: list[str]) -> np.ndarray:
    """2º turno sem pesquisa do confronto: redistribui votos dos eliminados por afinidade."""
    vi, vj = t1[:, i].copy(), t1[:, j].copy()
    for k in range(t1.shape[1]):
        if k in (i, j):
            continue
        bi, bj, bk = blocos[i], blocos[j], blocos[k]
        if bk == bi and bk != bj:
            fi, fj = 0.65, 0.15
        elif bk == bj and bk != bi:
            fi, fj = 0.15, 0.65
        else:
            fi, fj = 0.40, 0.40
        vi += fi * t1[:, k]
        vj += fj * t1[:, k]
    return 100 * vi / (vi + vj)


# ----------------------------------------------------------------------------- nacional

def simular_nacional(cfg: Configuracao, ent: Entrada, cal: Calibracao, rng) -> Resultado:
    cands = cfg.candidatos + ["Outros"]
    n, k = cfg.n_sim, len(cands)
    ip, ia = cands.index(cfg.principal), cands.index(cfg.adversario)

    # (1) incerteza da tendência + deriva até a eleição
    mu = np.array([ent.t1_media[c] for c in cands])
    dp = np.array([ent.t1_dp[c] for c in cands])
    x = mu + dp * rng.standard_normal((n, k))

    # (2) erro sistemático histórico (pesquisa - urna), por papel; subtraímos para corrigir
    viés = {p: cfg.peso_vies * v for p, v in cal.vies_t1.items()}
    dp_res = {p: np.sqrt(max(cal.dp_t1[p] ** 2 - cal.vies_t1[p] ** 2, 1.0)) for p in cal.dp_t1}
    # erro comum "PT vs. antipetismo" e erro próprio de cada um
    z_eixo = _t(rng, n)
    z_pt, z_adv = _t(rng, n), _t(rng, n)
    rho = -0.35  # erros de PT e adversário tendem a ter sinais opostos (troca de votos)
    e_pt = viés["pt"] + dp_res["pt"] * (np.sqrt(abs(rho)) * z_eixo + np.sqrt(1 - abs(rho)) * z_pt)
    e_adv = viés["adv"] + dp_res["adv"] * (-np.sqrt(abs(rho)) * z_eixo + np.sqrt(1 - abs(rho)) * z_adv)
    x[:, ip] -= e_pt
    x[:, ia] -= e_adv
    # o restante do erro recai sobre os demais, proporcionalmente ao tamanho de cada um
    outros = [i for i in range(k) if i not in (ip, ia)]
    peso_o = np.clip(x[:, outros], 0.05, None)
    peso_o = peso_o / peso_o.sum(axis=1, keepdims=True)
    x[:, outros] += (e_pt + e_adv)[:, None] * peso_o
    x[:, outros] += 0.25 * np.sqrt(np.clip(x[:, outros], 0.05, None)) * rng.standard_normal((n, len(outros)))
    x = np.clip(x, 0.05, None)
    x = 100 * x / x.sum(axis=1, keepdims=True)

    # margem de erro padronizada do 1º turno (positivo = PT foi superestimado)
    marg_t1 = ((e_pt - viés["pt"]) - (e_adv - viés["adv"])) / np.sqrt(dp_res["pt"] ** 2 + dp_res["adv"] ** 2)

    # (3) quem vai ao 2º turno ("Outros" não é candidato)
    reais = x[:, :-1]
    ordem = np.argsort(-reais, axis=1)
    top1, top2 = ordem[:, 0], ordem[:, 1]
    decidido = reais[np.arange(n), top1] > 50.0

    parcela = np.full(n, np.nan)          # parcela do finalista "a" (principal quando presente)
    fa, fb = top1.copy(), top2.copy()
    tem_principal = (fa == ip) | (fb == ip)
    troca = tem_principal & (fb == ip)
    fa[troca], fb[troca] = ip, top1[troca]

    dp_h2h_res = np.sqrt(max(cal.dp_h2h ** 2 - cal.vies_h2h ** 2, 2.5 ** 2))
    z2 = cal.rho_t1_t2 * marg_t1 + np.sqrt(1 - cal.rho_t1_t2 ** 2) * _t(rng, n)
    for j_riv, rival in enumerate(cands[:-1]):
        if rival == cfg.principal:
            continue
        sel = (fa == ip) & (fb == j_riv)
        if not sel.any():
            continue
        if rival in ent.t2_media:
            m = ent.t2_media[rival] + ent.t2_dp[rival] * rng.standard_normal(sel.sum())
            erro = cfg.peso_vies * cal.vies_h2h + dp_h2h_res * z2[sel]
            parcela[sel] = m - erro
        else:
            parcela[sel] = _parcela_por_transferencia(x[sel], ip, j_riv, cfg.blocos_lista) \
                + dp_h2h_res * z2[sel]
    # confrontos sem o principal: transferência por afinidade
    sem = ~tem_principal
    if sem.any():
        par = _parcela_por_transferencia_vetor(x[sem], fa[sem], fb[sem], cfg.blocos_lista)
        parcela[sem] = par + dp_h2h_res * z2[sem]
    parcela = np.clip(parcela, 1, 99)

    vencedor = np.where(decidido, top1, np.where(parcela > 50, fa, fb))
    finalistas = np.column_stack([fa, fb])
    t1 = pd.DataFrame(x, columns=cands)
    return Resultado(t1=t1, finalistas=finalistas, decidido_t1=decidido, vencedor=vencedor,
                     t2_parcela=parcela)


def _parcela_por_transferencia_vetor(t1, fa, fb, blocos):
    out = np.empty(len(fa))
    for idx in range(len(fa)):
        out[idx] = _parcela_por_transferencia(t1[idx:idx + 1], fa[idx], fb[idx], blocos)[0]
    return out


# ----------------------------------------------------------------------------- estados

def _base_estadual(cal: Calibracao):
    """Parcelas de 2022 por UF (1º e 2º turnos) e pesos (votos válidos)."""
    res = cal.resumo["resultados_uf"]
    r1 = res[(res.ano == 2022) & (res.turno == 1)].pivot_table(index="uf", columns="candidato", values="votos")
    r2 = res[(res.ano == 2022) & (res.turno == 2)].pivot_table(index="uf", columns="candidato", values="votos")
    return r1, r2


def _recentrar_logit(base_logit: np.ndarray, pesos: np.ndarray, alvo: np.ndarray, iters: int = 6) -> np.ndarray:
    """Acha deslocamento comum d (por simulação) tal que média ponderada de expit(base+d) = alvo."""
    d = _logit(alvo) - (base_logit * pesos).sum(axis=1) / pesos.sum()
    w = pesos / pesos.sum()
    for _ in range(iters):
        p = _expit(base_logit + d[:, None])
        f = (p * w).sum(axis=1) - alvo
        g = (p * (1 - p) * w).sum(axis=1)
        d -= f / np.maximum(g, 1e-6)
    return _expit(base_logit + d[:, None])


def simular_estados(cfg: Configuracao, cal: Calibracao, res: Resultado, rng) -> None:
    r1, r2 = _base_estadual(cal)
    ufs = list(r1.index)
    regioes = sorted(set(REGIAO.values()))
    reg_idx = np.array([regioes.index(REGIAO[u]) for u in ufs])
    n, S = cfg.n_sim, len(ufs)
    a_coef, b_coef = cal.lean_coef

    def choque():
        return (cal.dp_regiao * rng.standard_normal((n, len(regioes)))[:, reg_idx]
                + cal.dp_estado * rng.standard_normal((n, S)))

    # ---------------- 2º turno: principal x finalista, pela geografia de Lula x Bolsonaro 2022
    tot2 = r2.sum(axis=1).to_numpy()
    p22 = (r2["Lula"] / r2.sum(axis=1)).to_numpy()
    nac22 = r2["Lula"].sum() / r2.to_numpy().sum()
    lean2 = a_coef + b_coef * (_logit(p22) - _logit(nac22))
    alvo = res.t2_parcela / 100
    base = lean2[None, :] + choque()
    est2 = _recentrar_logit(base, tot2, alvo)
    ip = list(res.t1.columns).index(cfg.principal)
    est2[res.finalistas[:, 0] != ip] = np.nan   # geografia só vale para confrontos com o principal
    res.estados_t2 = {"ufs": ufs, "parcela": est2 * 100, "pesos": tot2}

    # ---------------- 1º turno: log-parcelas com inclinações de 2022 e efeito "da casa"
    tot1 = r1.sum(axis=1).to_numpy()
    cands = list(res.t1.columns)
    k = len(cands)
    lean1 = np.zeros((k, S))
    for c, col22 in ((cfg.principal, "Lula"), (cfg.adversario, "Bolsonaro")):
        p = (r1[col22] / r1.sum(axis=1)).to_numpy()
        nac = r1[col22].sum() / r1.to_numpy().sum()
        lean1[cands.index(c)] = b_coef * (np.log(p) - np.log(nac))
    for c, uf in cfg.estado_origem.items():
        if c in cands and uf in ufs:
            lean1[cands.index(c), ufs.index(uf)] += np.log(cal.multiplicador_casa)
    nac1 = res.t1.to_numpy() / 100                                # n x k
    eixo = choque()                                               # eixo PT x antipetismo
    logp = np.log(np.clip(nac1, 1e-4, None))[:, :, None] + lean1[None, :, :]
    logp[:, cands.index(cfg.principal), :] += eixo / 2
    logp[:, cands.index(cfg.adversario), :] -= eixo / 2
    menores = [i for i, c in enumerate(cands) if c not in (cfg.principal, cfg.adversario)]
    logp[:, menores, :] += 0.30 * rng.standard_normal((n, len(menores), S))
    p = np.exp(logp)
    p /= p.sum(axis=1, keepdims=True)
    w = tot1 / tot1.sum()
    for _ in range(4):  # ajuste proporcional iterativo para bater o total nacional sorteado
        nac_sim = (p * w[None, None, :]).sum(axis=2)
        p *= (nac1 / np.maximum(nac_sim, 1e-9))[:, :, None]
        p /= p.sum(axis=1, keepdims=True)
    res.estados_t1 = {"ufs": ufs, "parcelas": p * 100, "candidatos": cands, "pesos": tot1}
