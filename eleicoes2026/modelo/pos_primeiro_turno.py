"""Previsão a partir da apuração do 1º turno (noite de 4/out em diante).

Duas etapas, ambas por simulação:

1. **Projeção do 1º turno durante a apuração.** O TSE divulga o resultado parcial de cada
   UF. A ordem de apuração distorce o total nacional parcial (em 2022 Bolsonaro liderou
   as primeiras horas porque o Nordeste apura depois), então a projeção é feita UF a UF:
   cada UF usa sua parcial, com incerteza que cai conforme as seções são totalizadas, ou
   a previsão pré-eleição quando ainda não apurou nada; o total é ponderado pelo número
   esperado de votos válidos de cada UF.

2. **2º turno a partir das urnas.** Votos reais do 1º turno em cada UF + transferência dos
   votos dos eliminados, medida nas pesquisas de 2026 (mesma pesquisa, 1º turno vs.
   confronto direto) + erro calibrado em 2018/2022 (quanto essa conta errou o 2º turno)
   + choques regionais e estaduais calibrados em 2006–2022. Quando saem pesquisas de 2º
   turno feitas depois do 1º turno, elas são agregadas e combinadas por variância inversa.
"""
from __future__ import annotations

import json
import re
import unicodedata
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .agregador import ajustar_tendencia, series_segundo_turno
from .coleta import BRUTOS, RAIZ, UFS, ler_resultados_uf
from .historico import ELEICOES, REGIAO, _carregar_pesquisas_historicas, _ultimas_por_instituto

# Leiaute de 2026: /dados/<uf>/<uf>-c0001-e<eleição com 6 dígitos>-u.json (não existe mais "dados-simplificados").
# O leiaute antigo (-r.json, até 2022) fica como alternativa. BASE_TSE pode apontar para o simulado oficial:
# https://resultados-sim.tse.jus.br/simulado/simulado2026 (eleição 21270), útil para testar o pipeline.
BASE_TSE = "https://resultados.tse.jus.br/oficial"
URLS_TSE = ["{base}/ele2026/{eleicao}/dados/{uf}/{uf}-c0001-e{ele6}-u.json",
            "{base}/ele2026/{eleicao}/dados-simplificados/{uf}/{uf}-c0001-e{ele6}-r.json"]
ELEICAO_T1 = "6257"     # "Eleição Ordinária Federal - 2026 1º Turno" (Presidente), ver config ele-c.json do TSE
DIR_APURACAO = BRUTOS / "tse_1t_2026"
CSV_MANUAL = RAIZ / "dados" / "resultado_1t_2026.csv"

# Nome na urna (TSE) -> nome no modelo
# Números de urna de 2026, conferidos no arquivo oficial da eleição 6257 (br-c0001-e006257-u.json).
NUMEROS_TSE = {"13": "Lula", "22": "Flávio Bolsonaro", "70": "Augusto Cury", "14": "Renan Santos",
               "55": "Caiado", "30": "Zema"}
NOMES_TSE = [("LULA", "Lula"), ("FLAVIO", "Flávio Bolsonaro"), ("CURY", "Augusto Cury"),
             ("RENAN", "Renan Santos"), ("CAIADO", "Caiado"), ("ZEMA", "Zema")]


def _sem_acento(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().upper()


def _num(x) -> float:
    if x is None:
        return np.nan
    s = str(x).strip()
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


# ----------------------------------------------------------------------------- dados da apuração

def baixar_apuracao(eleicao: str = ELEICAO_T1, destino: Path = DIR_APURACAO, verbose: bool = True,
                    base: str | None = None) -> int:
    """Baixa o arquivo de resultados do TSE de cada UF (e do Brasil). Devolve quantos vieram.

    Um arquivo só é gravado se for um JSON válido; uma falha (404 antes da eleição, rede, TSE fora do
    ar) mantém o último arquivo bom, então o modelo continua com a apuração anterior.
    """
    destino.mkdir(parents=True, exist_ok=True)
    base = base or BASE_TSE
    ok = 0
    for uf in ["br"] + [u.lower() for u in UFS]:
        for modelo_url in URLS_TSE:
            url = modelo_url.format(base=base, eleicao=eleicao, ele6=eleicao.zfill(6), uf=uf)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "eleicoes2026-modelo/1.0"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    corpo = r.read()
                json.loads(corpo)
                (destino / f"{uf}.json").write_bytes(corpo)
                ok += 1
                break
            except Exception:  # noqa: BLE001 - antes da eleição o TSE devolve 404
                continue
    if verbose:
        print(f"  apuração do TSE: {ok} de 28 arquivos" + ("" if ok else " (ainda não publicada)"))
    return ok


def _parcial_oficial(cc: list[str]) -> dict:
    """Parcial nacional como o TSE divulga (arquivo br.json, inclui o exterior)."""
    f = DIR_APURACAO / "br.json"
    try:
        v = _ler_json_tse(json.loads(f.read_text(encoding="utf-8")))["votos"]
    except Exception:  # noqa: BLE001
        return {}
    tot = sum(v.values())
    return {c: float(100 * v.get(c, 0.0) / tot) for c in cc} if tot else {}


def resultado_final_t1(candidatos: list[str], minimo: float = 0.99) -> dict | None:
    """Resultado nacional do 1º turno (% dos válidos) quando a apuração está praticamente completa."""
    f = DIR_APURACAO / "br.json"
    try:
        if _ler_json_tse(json.loads(f.read_text(encoding="utf-8")))["pst"] < minimo:
            return None
    except Exception:  # noqa: BLE001
        return None
    return _parcial_oficial(list(candidatos) + ["Outros"]) or None


def status_apuracao(diretorio: Path | None = None) -> dict | None:
    """Situação dos arquivos do TSE antes de haver votos (para o painel mostrar a apuração "aguardando")."""
    diretorio = diretorio or DIR_APURACAO
    f = diretorio / "br.json"
    if not f.exists():
        return None
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
        s = d.get("s", {})
        return {"arquivos": len(list(diretorio.glob("*.json"))), "secoes": int(s.get("ts") or 0),
                "pst": float(str(s.get("pst", "0")).replace(",", ".")),
                "gerado_tse": f"{d.get('dg', '')} {d.get('hg', '')}".strip()}
    except Exception:  # noqa: BLE001
        return None


def _nome_modelo(numero: str, nome: str) -> str:
    if numero in NUMEROS_TSE:
        return NUMEROS_TSE[numero]
    nome = _sem_acento(nome)
    return next((m for k, m in NOMES_TSE if k in nome), "Outros")


def _ler_json_tse(d: dict) -> dict:
    """% de seções totalizadas e votos válidos por candidato, nos leiautes de 2026 (-u) e 2022 (-r)."""
    votos = {}
    if "carg" in d:   # leiaute 2026: s.pst, candidatos em carg/agr/par/cand, só os "Válido" contam
        cands = [c for a in d["carg"][0].get("agr", []) for p in a.get("par", []) for c in p.get("cand", [])]
        # Descarta só quem está marcado como anulado/nulo. No arquivo oficial antes da apuração o campo
        # "dvt" vem vazio; não dá para exigir "Válido" sem correr o risco de descartar todos os votos.
        invalido = lambda dvt: any(x in str(dvt or "") for x in ("Anulado", "Nulo", "Cassado", "Indeferido"))
        cands = [c for c in cands if not invalido(c.get("dvt"))]
        pst = d.get("s", {}).get("pst")
        nome_de = lambda c: c.get("nmu") or c.get("nm", "")
    else:             # leiaute até 2022
        cands = d.get("cand", [])
        pst = d.get("pst")
        nome_de = lambda c: c.get("nm", "")
    for c in cands:
        v = _num(c.get("vap"))
        if np.isnan(v):
            continue
        chave = _nome_modelo(str(c.get("n", "")), nome_de(c))
        votos[chave] = votos.get(chave, 0.0) + v
    return {"pst": _num(pst) / 100 if pst is not None else np.nan, "votos": votos}


def carregar_apuracao(diretorio: Path | None = None, csv: Path | None = None) -> pd.DataFrame | None:
    """Apuração por UF: colunas uf, pst (0–1) e votos de cada candidato (+ Outros).

    Usa os arquivos do TSE se houver; senão um CSV manual (uf,pst,Lula,Flávio Bolsonaro,…).
    """
    diretorio = DIR_APURACAO if diretorio is None else diretorio
    csv = CSV_MANUAL if csv is None else csv
    linhas = []
    if diretorio.exists():
        for uf in UFS:
            f = diretorio / f"{uf.lower()}.json"
            if f.exists():
                info = _ler_json_tse(json.loads(f.read_text(encoding="utf-8")))
                linhas.append({"uf": uf, "pst": info["pst"], **info["votos"]})
    if not linhas and csv.exists():
        df = pd.read_csv(csv)
        if "pst" not in df:
            df["pst"] = 1.0
        return df
    if not linhas:
        return None
    df = pd.DataFrame(linhas).fillna(0.0)
    return df if df.drop(columns=["uf", "pst"]).to_numpy().sum() > 0 else None


# ----------------------------------------------------------------------------- calibração

@dataclass
class CalibracaoPos:
    vies_urnas: float        # (previsto - real) da conta "1º turno real + transferência das pesquisas", pp
    dp_urnas: float          # erro dessa conta no 2º turno nacional, pp
    dp_estado: float         # resíduo por UF (logito) ao projetar o 2º turno da UF pelo 1º turno
    dp_regiao: float
    vies_pesquisas_t2: float  # erro médio das pesquisas da última semana do 2º turno, pp
    dp_pesquisas_t2: float
    backtest: list


def _transferencia(t1: pd.DataFrame, t2: pd.DataFrame, pt: str, adv: str) -> tuple[float, float, int]:
    """Fração dos votos que mudam de lado do 1º para o 2º turno que vai para `adv`."""
    if t1.empty or t2.empty:
        return np.nan, np.nan, 0
    t2 = t2[t2.candidatos.map(lambda c: set(c) == {pt, adv})]
    if t2.empty:
        return np.nan, np.nan, 0
    m = t1.merge(t2, on=["instituto", "fim"], suffixes=("1", "2"))
    m = m[m.candidatos1.map(lambda c: pt in c and adv in c)]
    if m.empty:
        return np.nan, np.nan, 0
    gp = m.candidatos2.map(lambda c: c[pt]) - m.candidatos1.map(lambda c: c[pt])
    ga = m.candidatos2.map(lambda c: c[adv]) - m.candidatos1.map(lambda c: c[adv])
    ok = (gp + ga) > 0
    r = float(ga[ok].sum() / (gp[ok] + ga[ok]).sum())
    return r, float((ga[ok] / (gp[ok] + ga[ok])).std()), int(ok.sum())


def calibrar_pos(encolhimento: float = 2.0) -> CalibracaoPos:
    pesq = _carregar_pesquisas_historicas()
    res = ler_resultados_uf()

    # (a) 1º turno real + transferência medida nas pesquisas pré-1º turno → erro no 2º turno
    bt = []
    for ano, d in pesq.items():
        e = ELEICOES[ano]
        ini, t1d = pd.Timestamp(e["t1"]) - pd.Timedelta(days=30), pd.Timestamp(e["t1"])
        janela = d[(d.fim >= ini) & (d.fim < t1d)]
        r, _, n = _transferencia(janela[janela.turno == 1], janela[janela.turno == 2], e["pt"], e["adv"])
        if not n:
            continue
        L1, A1 = e["r1"][e["pt"]], e["r1"][e["adv"]]
        prev = L1 + (1 - r) * (100 - L1 - A1)
        bt.append({"ano": ano, "transferencia_adv": r, "previsto": prev, "real": e["r2"][e["pt"]],
                   "erro": prev - e["r2"][e["pt"]]})
    erros = np.array([b["erro"] for b in bt])
    vies = float(erros.mean() * len(erros) / (len(erros) + encolhimento)) if len(erros) else 0.0
    dp = float(max(np.sqrt(np.mean(erros ** 2)), 1.5)) if len(erros) else 2.5

    # (b) 2º turno de cada UF a partir do 1º turno da UF (2006–2022), dado o total nacional
    pt = {2006: ("Lula", "Alckmin"), 2010: ("Dilma", "Serra"), 2014: ("Dilma", "Aécio"),
          2018: ("Haddad", "Bolsonaro"), 2022: ("Lula", "Bolsonaro")}
    resid = []
    for ano, (p, a) in pt.items():
        r1 = res[(res.ano == ano) & (res.turno == 1)].pivot_table(index="uf", columns="candidato", values="votos")
        r2 = res[(res.ano == ano) & (res.turno == 2)].pivot_table(index="uf", columns="candidato", values="votos")
        tot = r1.sum(axis=1)
        L, A = r1[p] / tot, r1[a] / tot
        O = 1 - L - A
        real = r2[p] / r2.sum(axis=1)
        alvo = r2[p].sum() / r2.to_numpy().sum()
        w = (tot / tot.sum()).to_numpy()
        f = 0.5
        for _ in range(60):  # fração dos "outros" que vai ao PT e reproduz o total nacional
            f += (alvo - float(((L + f * O) * w).sum())) / max(float((O * w).sum()), 1e-6)
        prev = np.clip(L + f * O, 0.01, 0.99)
        lg = lambda x: np.log(x / (1 - x))
        e_ = lg(real) - lg(prev)
        resid.append(pd.DataFrame({"ano": ano, "uf": real.index, "res": e_.values}))
    rd = pd.concat(resid)
    rd["regiao"] = rd.uf.map(REGIAO)
    reg = rd.groupby(["ano", "regiao"]).res.transform("mean")
    dp_reg = float(rd.groupby(["ano", "regiao"]).res.mean().std())
    dp_est = float((rd.res - reg).std())

    # (c) pesquisas da última semana do 2º turno (feitas depois do 1º turno)
    ep = []
    for ano, d in pesq.items():
        e = ELEICOES[ano]
        t2 = d[(d.turno == 2) & (d.fim > pd.Timestamp(e["t1"]))]
        t2 = t2[t2.candidatos.map(lambda c: set(c) == {e["pt"], e["adv"]})] if len(t2) else t2
        if t2.empty:
            continue
        fin = _ultimas_por_instituto(t2, e["t2"], 7)
        if len(fin):
            par = fin.candidatos.map(lambda c: 100 * c[e["pt"]] / (c[e["pt"]] + c[e["adv"]]))
            ep.append(par.mean() - e["r2"][e["pt"]])
    ep = np.array(ep)
    vies_p = float(ep.mean() * len(ep) / (len(ep) + encolhimento)) if len(ep) else 0.0
    dp_p = float(max(np.sqrt(np.mean(ep ** 2)), 1.0)) if len(ep) else 2.0
    return CalibracaoPos(vies, dp, dp_est, dp_reg, vies_p, dp_p, bt)


# ----------------------------------------------------------------------------- simulação

def _expit(x):
    return 1 / (1 + np.exp(-x))


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def projetar_primeiro_turno(apur: pd.DataFrame, previa: dict, candidatos: list[str], rng, n: int):
    """Sorteia o resultado final do 1º turno em cada UF a partir da parcial e da previsão prévia.

    `previa`: {uf: {"t1": {cand: %}, "peso": votos válidos esperados}} — da previsão pré-eleição.
    Devolve array n × UF × candidatos (frações dos válidos) e os pesos por UF.
    """
    cands = candidatos + ["Outros"]
    ufs = list(previa)
    S, K = len(ufs), len(cands)
    out = np.empty((n, S, K))
    pesos = np.array([previa[u]["peso"] for u in ufs], float)
    ap = apur.set_index("uf").copy() if apur is not None else pd.DataFrame(columns=["pst"])
    for c in cands:  # candidatos sem voto na UF (ou fora do arquivo) contam zero
        if c not in ap:
            ap[c] = 0.0
    def _prior(uf):
        pr = np.array([previa[uf]["t1"].get(c, 0.0) for c in cands], float)
        return np.clip(pr / pr.sum(), 1e-4, None)

    # Desvio das UFs já apuradas em relação à previsão (log-razão, ponderado pelos votos contados),
    # aplicado às UFs que ainda faltam: se Lula supera a previsão onde já se apurou, tende a superar
    # também onde ainda não se apurou.
    desvio = np.zeros(K)
    peso_desvio = 0.0
    for uf in ufs:
        if uf in ap.index and ap.loc[uf, cands].sum() > 0:
            v = ap.loc[uf, cands].to_numpy(float)
            pst_u = float(ap.loc[uf, "pst"]) if not np.isnan(ap.loc[uf, "pst"]) else 1.0
            w_u = v.sum() * pst_u
            desvio += w_u * (np.log(np.clip(v / v.sum(), 1e-4, None)) - np.log(_prior(uf)))
            peso_desvio += w_u
    if peso_desvio > 0:
        desvio /= peso_desvio
        frac = peso_desvio / max(sum(previa[u]["peso"] for u in ufs), 1.0)   # fração do país já contada
        desvio *= frac / (frac + 0.05)                                       # encolhe no início da noite

    for s, uf in enumerate(ufs):
        prior = _prior(uf) * np.exp(desvio)
        prior = prior / prior.sum()
        if uf in ap.index and ap.loc[uf, cands].sum() > 0:
            v = ap.loc[uf, cands].to_numpy(float)
            obs = np.clip(v / v.sum(), 1e-4, None)
            pst = float(np.clip(ap.loc[uf, "pst"] if not np.isnan(ap.loc[uf, "pst"]) else 1.0, 0, 1))
            # parcial tende à final; a ordem de apuração dentro da UF ainda pode mover alguns pontos
            base = pst * obs + (1 - pst) * (0.5 * obs + 0.5 * prior)
            dp = 0.20 * np.sqrt(1 - pst)          # logito
            pesos[s] = max(pesos[s], v.sum() / max(pst, 0.01)) if pst > 0.5 else pesos[s]
        else:
            base, dp = prior, 0.12                 # sem apuração: previsão da UF ajustada pelo desvio
        z = np.log(base)[None, :] + dp * rng.standard_normal((n, K))
        p = np.exp(z)
        out[:, s, :] = p / p.sum(axis=1, keepdims=True)
    return out, pesos, ufs, cands


def prever_segundo_turno(t1_uf: np.ndarray, pesos: np.ndarray, ufs: list[str], cands: list[str],
                         principal: str, adversario: str, transf: float, transf_dp: float,
                         calp: CalibracaoPos, rng, pesquisas_t2: tuple[float, float] | None = None,
                         peso_vies: float = 1.0) -> dict:
    """2º turno principal × adversário a partir do 1º turno (sorteado ou real) por UF."""
    n, S, K = t1_uf.shape
    ip, ia = cands.index(principal), cands.index(adversario)
    w = pesos / pesos.sum()
    L, A = t1_uf[:, :, ip], t1_uf[:, :, ia]
    O = 1 - L - A
    nac_t1 = np.einsum("nsk,s->nk", t1_uf, w)
    decidido = nac_t1[:, :-1].max(axis=1) > 0.5
    vencedor_t1 = nac_t1[:, :-1].argmax(axis=1)

    # transferência (com incerteza) e erro histórico dessa conta
    r = np.clip(transf + transf_dp * rng.standard_normal(n), 0.05, 0.95)
    r_uf = np.clip(r[:, None] + 0.05 * rng.standard_normal((n, S)), 0.02, 0.98)
    base = L + (1 - r_uf) * O                                     # parcela do principal, por UF
    nac_base = (base * w).sum(axis=1) * 100
    nac_urnas = nac_base - peso_vies * calp.vies_urnas + calp.dp_urnas * rng.standard_normal(n)
    if pesquisas_t2 is not None:  # combina com as pesquisas feitas depois do 1º turno
        m_p, dp_p = pesquisas_t2
        m_p = m_p - peso_vies * calp.vies_pesquisas_t2
        dp_p = np.sqrt(dp_p ** 2 + calp.dp_pesquisas_t2 ** 2)
        w_u, w_p = 1 / calp.dp_urnas ** 2, 1 / dp_p ** 2
        media = (w_u * (nac_base - peso_vies * calp.vies_urnas) + w_p * m_p) / (w_u + w_p)
        nac = media + np.sqrt(1 / (w_u + w_p)) * rng.standard_normal(n)
    else:
        nac = nac_urnas
    nac = np.clip(nac, 1, 99)

    # UFs: geografia do próprio 1º turno + choques, recentrada no total nacional sorteado
    regioes = sorted(set(REGIAO.values()))
    ri = np.array([regioes.index(REGIAO[u]) for u in ufs])
    lg = _logit(base) + calp.dp_regiao * rng.standard_normal((n, len(regioes)))[:, ri] \
        + calp.dp_estado * rng.standard_normal((n, S))
    d = _logit(nac / 100) - (lg * w).sum(axis=1)
    for _ in range(6):
        p = _expit(lg + d[:, None])
        d -= ((p * w).sum(axis=1) - nac / 100) / np.maximum((p * (1 - p) * w).sum(axis=1), 1e-6)
    uf2 = _expit(lg + d[:, None]) * 100

    vence_principal = np.where(decidido, vencedor_t1 == ip, nac > 50)
    return {"nac_t1": nac_t1, "decidido": decidido, "vencedor_t1": vencedor_t1, "t2": nac, "t2_uf": uf2,
            "vence_principal": vence_principal, "transf": r}


# ----------------------------------------------------------------------------- pipeline

def executar(saida_pre: dict, p1: pd.DataFrame, p2: pd.DataFrame, brutas: pd.DataFrame, cfg_d: dict,
             data_t1: pd.Timestamp, data_t2: pd.Timestamp, pesos_inst: dict, n: int = 20000,
             semente: int = 2026, peso_vies: float = 1.0, apur: pd.DataFrame | None = None) -> dict | None:
    """Roda a previsão pós-1º turno se houver apuração. `saida_pre` é a previsão pré-eleição."""
    apur = carregar_apuracao() if apur is None else apur
    if apur is None:
        return None
    P, A = cfg_d["principal"], cfg_d["adversario"]
    cands = cfg_d["candidatos"]
    rng = np.random.default_rng(semente + 7)
    calp = calibrar_pos()

    previa = {e["uf"]: {"t1": e["t1"], "peso": e["eleitores_validos_2022"]} for e in saida_pre["estados"]}
    t1_uf, pesos, ufs, cc = projetar_primeiro_turno(apur, previa, cands, rng, n)

    # transferência medida nas pesquisas de 2026 (últimos 30 dias antes do 1º turno)
    jan = brutas[(brutas.fim >= data_t1 - pd.Timedelta(days=30)) & (brutas.fim < data_t1)]
    r, r_dp, n_pares = _transferencia(jan[jan.turno == 1], jan[jan.turno == 2], P, A)
    r_dp = float(np.sqrt((r_dp / np.sqrt(max(n_pares, 1))) ** 2 + 0.06 ** 2))  # + variação entre eleições

    # pesquisas de 2º turno feitas depois do 1º turno
    pos = p2[p2.meio > data_t1]
    agreg = None
    if len(pos):
        s2 = series_segundo_turno(pos, P).get(A)
        if s2 is not None and len(s2) >= 2:
            tr = ajustar_tendencia(s2, pesos_inst, ate=data_t2)
            agreg = tr.em(data_t2)

    res = prever_segundo_turno(t1_uf, pesos, ufs, cc, P, A, r, r_dp, calp, rng, agreg, peso_vies)
    ap = apur.set_index("uf")
    pst_total = float((ap.reindex(ufs).pst.fillna(0).to_numpy() * pesos).sum() / pesos.sum())
    votos_br = ap.reindex(ufs).reindex(columns=cc).fillna(0).sum()
    nac_t1 = res["nac_t1"] * 100
    ip, ia = cc.index(P), cc.index(A)
    ha_t2 = ~res["decidido"]
    estados = []
    for s, uf in enumerate(ufs):
        u2 = res["t2_uf"][ha_t2, s]
        estados.append({"uf": uf, "pst": float(ap.pst.get(uf, 0.0)) if uf in ap.index else 0.0,
                        "t1": {c: float(np.mean(t1_uf[:, s, k]) * 100) for k, c in enumerate(cc)},
                        "t2_media": float(u2.mean()) if len(u2) else None,
                        "t2_p10": float(np.percentile(u2, 10)) if len(u2) else None,
                        "t2_p90": float(np.percentile(u2, 90)) if len(u2) else None,
                        "t2_p_principal": float(np.mean(u2 > 50)) if len(u2) else None})
    t2 = res["t2"][ha_t2]
    return {
        "apurado": pst_total, "apurado_ufs": int((ap.reindex(ufs).pst.fillna(0) >= 0.999).sum()),
        "parcial_br": _parcial_oficial(cc) or ({c: float(100 * v / votos_br.sum()) for c, v in votos_br.items()}
                                               if votos_br.sum() else {}),
        "t1": {c: {"media": float(nac_t1[:, k].mean()), "p10": float(np.percentile(nac_t1[:, k], 10)),
                   "p90": float(np.percentile(nac_t1[:, k], 90)),
                   "primeiro": float(np.mean(nac_t1[:, :-1].argmax(axis=1) == k))}
               for k, c in enumerate(cc)},
        "p_decidido_t1": float(res["decidido"].mean()),
        "p_vence_t1": {c: float(np.mean(res["decidido"] & (res["vencedor_t1"] == k))) for k, c in enumerate(cc[:-1])},
        "t2": {"media": float(t2.mean()) if len(t2) else None,
               "p10": float(np.percentile(t2, 10)) if len(t2) else None,
               "p90": float(np.percentile(t2, 90)) if len(t2) else None,
               "hist": np.histogram(t2, bins=np.arange(30, 70.5, 0.5))[0].tolist() if len(t2) else []},
        "p_principal": float(res["vence_principal"].mean()),
        "transferencia": {"adversario": r, "dp": r_dp, "n_pesquisas": n_pares},
        "pesquisas_pos_t1": {"n": int(len(pos[(pos.a == A) | (pos.b == A)])),
                             "media": agreg[0] if agreg else None, "dp": agreg[1] if agreg else None},
        "calibracao": {"vies_urnas": calp.vies_urnas, "dp_urnas": calp.dp_urnas, "dp_estado": calp.dp_estado,
                       "dp_regiao": calp.dp_regiao, "backtest": calp.backtest},
        "estados": estados,
    }
