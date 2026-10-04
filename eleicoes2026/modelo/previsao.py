"""Pipeline completo da previsão de 2026."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import coleta, pos_primeiro_turno
from .agregador import ajustar_tendencia, series_primeiro_turno, series_segundo_turno
from .historico import REGIAO, calibrar, vies_de_institutos
from .simulacao import Configuracao, Entrada, simular_estados, simular_nacional

BRASILIA = timezone(timedelta(hours=-3))
DATA_T1 = pd.Timestamp("2026-10-04")
DATA_T2 = pd.Timestamp("2026-10-25")
INICIO_SERIE = pd.Timestamp("2026-01-15")
# Metade do viés histórico. O teste fora da amostra em 2010/2018/2022 favorece a correção inteira, mas
# ele só enxerga o regime antigo de métodos. Nas municipais de 2024 (já com Censo 2022 e eleitor
# provável), o exagero da vantagem da esquerda caiu para ~1,9 ponto na margem, contra ~6,9 nas
# presidenciais de 2006–2022: o viés diminuiu, sem sumir (dados/processados/erros_municipais_2024.csv).
PESO_VIES_PADRAO = 0.5
# Institutos com menor erro em 2018 e 2022 (ver historico.pesos_institutos).
MELHORES = ["AtlasIntel", "MDA"]

# Modo de coleta de cada instituto (fontes: Exame, Correio Braziliense, InfoMoney e relatórios dos
# institutos; ver README). "desconhecido" = o método não aparece nas divulgações consultadas.
MODO_COLETA = {
    "AtlasIntel": "online", "Palver": "online",
    "Real Time Big Data": "misto",          # telefone + abordagens digitais
    "Datafolha": "presencial", "Quaest": "presencial", "Ipec": "presencial", "Paraná Pesquisas": "presencial",
    "MDA": "presencial", "Vox Brasil": "presencial",
    "Futura": "telefone", "Gerp": "telefone", "Nexus": "telefone", "Ideia": "telefone",
    "PoderData": "telefone", "DataTrends": "telefone", "Ipespe": "telefone", "FSB": "telefone",
}
# Pesquisas online entram: no teste fora da amostra, excluí-las piorou o erro de 2022
# (AtlasIntel foi a mais precisa). `--sem-online` as exclui.
EXCLUIR_MODOS_PADRAO: tuple[str, ...] = ()

CONFIG_2026 = dict(
    candidatos=["Lula", "Flávio Bolsonaro", "Augusto Cury", "Renan Santos", "Caiado", "Zema"],
    principal="Lula",
    adversario="Flávio Bolsonaro",
    partidos={"Lula": "PT", "Flávio Bolsonaro": "PL", "Augusto Cury": "Avante", "Renan Santos": "Missão",
              "Caiado": "PSD", "Zema": "Novo", "Outros": "—"},
    blocos={"Lula": "esquerda", "Flávio Bolsonaro": "direita", "Caiado": "direita", "Zema": "direita",
            "Renan Santos": "direita", "Augusto Cury": "centro"},
    estado_origem={"Caiado": "GO", "Zema": "MG", "Renan Santos": "SP", "Augusto Cury": "SP"},
)


def _resumo_trilha(tr, desde: pd.Timestamp) -> list[dict]:
    m = tr.datas >= desde
    return [{"d": d.strftime("%Y-%m-%d"), "m": round(float(a), 2), "s": round(float(b), 2)}
            for d, a, b in zip(tr.datas[m], tr.media[m], tr.dp[m])]


def _pontos(obs: pd.DataFrame, desde: pd.Timestamp) -> list[dict]:
    o = obs[obs.meio >= desde]
    return [{"d": r.meio.strftime("%Y-%m-%d"), "v": round(float(r.valor), 2), "i": r.instituto,
             "a": round(float(r.ajustado), 2)} for r in o.itertuples()]


def executar(n_sim: int = 20000, peso_vies: float = PESO_VIES_PADRAO, atualizar: bool = False, semente: int = 2026,
             hoje: date | None = None, verbose: bool = True, institutos: list[str] | None = None,
             vies_proprio: bool = False, cenario_melhores: bool = True, apuracao: bool = False,
             excluir_modos: tuple[str, ...] = EXCLUIR_MODOS_PADRAO) -> dict:
    """Roda a previsão completa.

    institutos: restringe as pesquisas a esses institutos.
    vies_proprio: com `institutos`, corrige pelo erro histórico só deles (não o de todos).
    cenario_melhores: acrescenta à saída o cenário só com os institutos historicamente mais precisos.
    apuracao: baixa a apuração do 1º turno do TSE; com apuração disponível (TSE ou CSV manual),
        acrescenta a previsão pós-1º turno em saida["pos_t1"].
    """
    log = print if verbose else (lambda *a, **k: None)
    if apuracao and (not institutos or excluir_modos):
        log("• Baixando a apuração do 1º turno (TSE)…")
        pos_primeiro_turno.baixar_apuracao(verbose=verbose)
    if atualizar:
        log("• Atualizando pesquisas na Wikipédia…")
        coleta.atualizar_fontes()

    cfg_d = CONFIG_2026
    cands = cfg_d["candidatos"]

    log("• Lendo pesquisas de 2026…")
    brutas = coleta.ler_pesquisas((coleta.BRUTOS / "pesquisas_2026.wiki").read_text(encoding="utf-8"), 2026)
    manuais = coleta.ler_pesquisas_manuais()
    if len(manuais):  # divulgadas e ainda fora da Wikipédia; não duplica o que a Wikipédia já tem
        chaves = set(zip(brutas.instituto, brutas.fim.dt.normalize(), brutas.turno))
        novas = manuais[[(i, f.normalize(), t) not in chaves for i, f, t in zip(manuais.instituto, manuais.fim, manuais.turno)]]
        brutas = pd.concat([brutas, novas], ignore_index=True)
        log(f"  + {len(novas)} cenários de pesquisas digitadas à mão (dados/pesquisas_manuais.csv)")
    if hoje is not None:  # permite "voltar no tempo" (backtest) cortando pesquisas futuras
        brutas = brutas[brutas.fim <= pd.Timestamp(hoje)]
    p1 = coleta.pesquisas_primeiro_turno(brutas, cands)
    caminho_pt = coleta.BRUTOS / "pesquisas_2026_pt.wiki"
    if caminho_pt.exists():  # a página em português lista institutos que a inglesa não traz
        brutas_pt = coleta.ler_pesquisas(caminho_pt.read_text(encoding="utf-8"), 2026)
        brutas_pt = brutas_pt[brutas_pt.fim <= pd.Timestamp(hoje or date.today())]
        antes = len(p1)
        p1 = coleta.unir_fontes(p1, coleta.pesquisas_primeiro_turno(brutas_pt, cands))
        log(f"  + {len(p1) - antes} pesquisas de 1º turno só na Wikipédia em português")
        brutas = pd.concat([brutas, brutas_pt[brutas_pt.turno == 1]])
    p1 = p1[p1.meio >= INICIO_SERIE].reset_index(drop=True)
    p2 = coleta.pesquisas_segundo_turno(brutas)
    p2 = p2[p2.meio >= INICIO_SERIE].reset_index(drop=True)
    excluidos = []
    if excluir_modos and not institutos:  # exclui institutos por modo de coleta (ex.: online)
        presentes = set(p1.instituto) | set(p2.instituto)
        excluidos = sorted(i for i in presentes if MODO_COLETA.get(i, "desconhecido") in excluir_modos)
        institutos = sorted(presentes - set(excluidos))
        vies_proprio = True   # corrige pelo erro histórico dos institutos que ficaram, não de todos
        brutas = brutas[~brutas.instituto.isin(excluidos)]   # também fora da transferência de votos
        log(f"  sem pesquisas {'/'.join(excluir_modos)}: fora {', '.join(excluidos)}")
    if institutos:  # previsão restrita a alguns institutos (ex.: os mais precisos no passado)
        p1 = p1[p1.instituto.isin(institutos)].reset_index(drop=True)
        p2 = p2[p2.instituto.isin(institutos)].reset_index(drop=True)
        if not excluidos:
            log(f"  apenas: {', '.join(institutos)}")
    log(f"  {len(p1)} pesquisas de 1º turno e {len(p2)} confrontos de 2º turno desde {INICIO_SERIE:%d/%m/%Y}")

    log("• Calibrando com eleições de 2006–2022 (backtest, qualidade dos institutos, geografia)…")
    cal = calibrar()
    erros_escolhidos = None
    if institutos and vies_proprio:
        cal, erros_escolhidos = vies_de_institutos(cal, institutos)

    log("• Agregando pesquisas (Kalman + efeito de instituto)…")
    tend1 = {}
    for c, obs in series_primeiro_turno(p1, cands).items():
        tend1[c] = ajustar_tendencia(obs, cal.pesos_institutos, ate=DATA_T1)
    t1_media, t1_dp = {}, {}
    for c, tr in tend1.items():
        t1_media[c], t1_dp[c] = tr.em(DATA_T1)
    soma = sum(t1_media.values())
    t1_media = {c: 100 * v / soma for c, v in t1_media.items()}

    tend2 = {}
    for rival, obs in series_segundo_turno(p2, cfg_d["principal"]).items():
        if rival not in cands or len(obs) < 4:
            continue
        tend2[rival] = ajustar_tendencia(obs, cal.pesos_institutos, ate=DATA_T2)
    t2_media = {r: tr.em(DATA_T2)[0] for r, tr in tend2.items()}
    t2_dp = {r: tr.em(DATA_T2)[1] for r, tr in tend2.items()}

    cfg = Configuracao(candidatos=cands, principal=cfg_d["principal"], adversario=cfg_d["adversario"],
                       blocos=cfg_d["blocos"], estado_origem=cfg_d["estado_origem"], n_sim=n_sim,
                       peso_vies=peso_vies, semente=semente)
    ent = Entrada(t1_media, t1_dp, t2_media, t2_dp)

    log(f"• Simulando {n_sim:,} eleições…".replace(",", "."))
    rng = np.random.default_rng(semente)
    res = simular_nacional(cfg, ent, cal, rng)
    simular_estados(cfg, cal, res, rng)

    # sensibilidade ao viés histórico
    sens = []
    for pv in (0.0, 0.5, 1.0, 1.5):
        if pv == peso_vies:  # o cenário escolhido usa a própria simulação principal
            r2 = res
        else:
            c2 = Configuracao(**{**cfg.__dict__, "peso_vies": pv, "n_sim": min(n_sim, 10000)})
            r2 = simular_nacional(c2, ent, cal, np.random.default_rng(semente + 1))
        todos = cands + ["Outros"]
        ip_, ia_ = todos.index(cfg.principal), todos.index(cfg.adversario)
        lf = (r2.finalistas[:, 0] == ip_) & (r2.finalistas[:, 1] == ia_) & ~r2.decidido_t1
        sens.append({"peso_vies": pv, "vitoria": {c: float(np.mean(r2.vencedor == i)) for i, c in enumerate(todos[:-1])},
                     "decidido_t1": float(np.mean(r2.decidido_t1)),
                     "t1": {c: float(r2.t1[c].mean()) for c in todos},
                     "t2_principal": float(np.mean(r2.t2_parcela[lf])) if lf.any() else None,
                     "lider_t1_principal": float(np.mean(r2.t1.to_numpy()[:, :-1].argmax(axis=1) == ip_))})

    saida = _consolidar(cfg, cfg_d, res, ent, cal, tend1, tend2, p1, p2, sens)
    saida["gerado_em"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    saida["ultima_pesquisa"] = max(p1.fim.max(), p2.fim.max()).strftime("%Y-%m-%d")
    if not institutos or excluidos:
        pos = pos_primeiro_turno.executar(saida, p1, p2, brutas, cfg_d, DATA_T1, DATA_T2, cal.pesos_institutos,
                                          n=n_sim, semente=semente, peso_vies=peso_vies)
        if pos is not None:
            log(f"• Apuração do 1º turno: {100 * pos['apurado']:.1f}% · {cfg_d['principal']} eleito em "
                f"{100 * pos['p_principal']:.0f}% das simulações")
            saida["pos_t1"] = pos
        elif DATA_T1 <= pd.Timestamp(hoje or datetime.now(BRASILIA).date()) < DATA_T2:
            st = pos_primeiro_turno.status_apuracao()
            saida["apuracao_espera"] = {**(st or {"arquivos": 0}),
                                        "verificado_em": datetime.now(BRASILIA).strftime("%H:%M")}
    if erros_escolhidos is not None:
        saida["erros_institutos_escolhidos"] = erros_escolhidos.round(2).to_dict("records")
    mun = coleta.PROCESSADOS / "erros_municipais_2024.csv"
    if mun.exists():
        saida["evidencia_2024"] = pd.read_csv(mun).to_dict("records")
    aval = coleta.PROCESSADOS / "avaliacao.csv"
    if aval.exists():
        saida["avaliacao"] = pd.read_csv(aval).to_dict("records")
    saida["modo_coleta"] = {"excluidos": excluidos, "modos_excluidos": list(excluir_modos) if excluidos else [],
                            "classificacao": {i: MODO_COLETA.get(i, "desconhecido")
                                              for i in sorted(set(p1.instituto) | set(excluidos))}}
    if cenario_melhores and not institutos:
        log(f"• Cenário só com os institutos mais precisos ({', '.join(MELHORES)})…")
        sub = executar(n_sim=min(n_sim, 20000), peso_vies=1.0, hoje=hoje, verbose=False, semente=semente,
                       institutos=MELHORES, vies_proprio=True, cenario_melhores=False)
        saida["cenario_melhores"] = _resumo_cenario(sub, cfg_d)
    return saida


def _resumo_cenario(sub: dict, cfg_d: dict) -> dict:
    """Resumo compacto de uma previsão restrita a alguns institutos, para o painel."""
    P, A = cfg_d["principal"], cfg_d["adversario"]
    lf = next((c for c in sub["confrontos"] if c["a"] == P and c["b"] == A), None)
    ultimas = {}
    for inst in MELHORES:
        t1 = {c: [p for p in sub["pesquisas_t1"].get(c, []) if p["i"] == inst] for c in (P, A)}
        t2 = [p for p in sub["pesquisas_t2"].get(A, []) if p["i"] == inst]
        if t1[P] and t1[A]:
            ultimas[inst] = {"data": t1[P][-1]["d"], "t1_p": t1[P][-1]["v"], "t1_a": t1[A][-1]["v"],
                             "t2_p": t2[-1]["v"] if t2 else None, "t2_data": t2[-1]["d"] if t2 else None}
    sens = {s["peso_vies"]: s for s in sub["sensibilidade"]}
    return {
        "institutos": MELHORES, "n_t1": sub["n_pesquisas_t1"], "n_t2": sub["n_confrontos_t2"],
        "ultimas": ultimas, "erros_historicos": sub.get("erros_institutos_escolhidos", []),
        "vies": {"pt": sub["calibracao"]["vies_t1"]["pt"], "adv": sub["calibracao"]["vies_t1"]["adv"],
                 "h2h": sub["calibracao"]["vies_h2h"]},
        "com_correcao": {"t1": {c["nome"]: c["t1_media"] for c in sub["candidatos"]},
                         "t2_principal": lf["a_media"] if lf else None,
                         "t2_p10": lf["a_p10"] if lf else None, "t2_p90": lf["a_p90"] if lf else None,
                         "vitoria": {c["nome"]: c["p_vitoria"] for c in sub["candidatos"]},
                         "lider_t1_principal": next(c["p_primeiro_lugar_t1"] for c in sub["candidatos"] if c["nome"] == P)},
        "sem_correcao": {"t1": sens[0.0]["t1"], "t2_principal": sens[0.0]["t2_principal"],
                         "vitoria": sens[0.0]["vitoria"]},
    }


def _pct(a, q):
    return float(np.nanpercentile(a, q))


def _consolidar(cfg, cfg_d, res, ent, cal, tend1, tend2, p1, p2, sens) -> dict:
    todos = cfg.candidatos + ["Outros"]
    n = cfg.n_sim
    t1 = res.t1.to_numpy()
    ip = todos.index(cfg.principal)

    candidatos = []
    for i, c in enumerate(cfg.candidatos):
        no_t2 = (res.finalistas == i).any(axis=1) & ~res.decidido_t1
        candidatos.append({
            "nome": c, "partido": cfg_d["partidos"].get(c, ""), "bloco": cfg.blocos.get(c, ""),
            "t1_media": float(t1[:, i].mean()), "t1_p10": _pct(t1[:, i], 10), "t1_p90": _pct(t1[:, i], 90),
            "t1_p025": _pct(t1[:, i], 2.5), "t1_p975": _pct(t1[:, i], 97.5),
            "agregado_bruto": float(ent.t1_media[c]),
            "p_vitoria": float(np.mean(res.vencedor == i)),
            "p_segundo_turno": float(np.mean(no_t2)),
            "p_vence_t1": float(np.mean(res.decidido_t1 & (res.vencedor == i))),
            "p_primeiro_lugar_t1": float(np.mean(t1[:, :-1].argmax(axis=1) == i)),
            "hist_t1": np.histogram(t1[:, i], bins=np.arange(0, 70.5, 0.5))[0].tolist(),
        })

    # confrontos de 2º turno
    confrontos = []
    for a in range(len(cfg.candidatos)):
        for b in range(a + 1, len(cfg.candidatos)):
            sel = (~res.decidido_t1) & (((res.finalistas[:, 0] == a) & (res.finalistas[:, 1] == b)) |
                                        ((res.finalistas[:, 0] == b) & (res.finalistas[:, 1] == a)))
            if sel.sum() < max(20, n * 0.001):
                continue
            ganha_a = np.mean(res.vencedor[sel] == a)
            par = np.where(res.finalistas[sel, 0] == a, res.t2_parcela[sel], 100 - res.t2_parcela[sel])
            confrontos.append({"a": todos[a], "b": todos[b], "prob": float(sel.mean()),
                               "p_a_vence": float(ganha_a), "a_media": float(par.mean()),
                               "a_p10": _pct(par, 10), "a_p90": _pct(par, 90),
                               "hist": np.histogram(par, bins=np.arange(30, 70.5, 0.5))[0].tolist()})
    confrontos.sort(key=lambda d: -d["prob"])

    # estados
    e1, e2 = res.estados_t1, res.estados_t2
    ufs = e1["ufs"]
    estados = []
    for s, uf in enumerate(ufs):
        p = e1["parcelas"][:, :, s]
        lider = p[:, :-1].argmax(axis=1)
        par2 = e2["parcela"][:, s]
        ok = ~np.isnan(par2)
        adv = todos.index(cfg.adversario)
        so_lf = ok & (res.finalistas[:, 1] == adv)
        estados.append({
            "uf": uf, "regiao": REGIAO[uf], "eleitores_validos_2022": float(e1["pesos"][s]),
            "t1": {c: round(float(p[:, i].mean()), 2) for i, c in enumerate(todos)},
            "t1_lider": {c: round(float(np.mean(lider == i)), 4) for i, c in enumerate(cfg.candidatos)},
            "t2_lula_media": float(np.nanmean(par2[so_lf])) if so_lf.any() else None,
            "t2_lula_p10": _pct(par2[so_lf], 10) if so_lf.any() else None,
            "t2_lula_p90": _pct(par2[so_lf], 90) if so_lf.any() else None,
            "t2_p_lula": float(np.mean(par2[so_lf] > 50)) if so_lf.any() else None,
        })

    # tendência e pesquisas para os gráficos
    desde = pd.Timestamp("2026-03-01")
    trilhas1 = {c: _resumo_trilha(tr, desde) for c, tr in tend1.items()}
    pontos1 = {c: _pontos(tr.obs, desde) for c, tr in tend1.items() if c in cfg.candidatos}
    trilhas2 = {r: _resumo_trilha(tr, desde) for r, tr in tend2.items()}
    pontos2 = {r: _pontos(tr.obs, desde) for r, tr in tend2.items()}
    efeitos = {}
    for c in (cfg.principal, cfg.adversario):
        for inst, h in tend1[c].efeitos.items():
            efeitos.setdefault(inst, {})[c] = round(float(h), 2)
    for inst, h in tend2[cfg.adversario].efeitos.items():
        efeitos.setdefault(inst, {})["t2"] = round(float(h), 2)
    contagem = p1.instituto.value_counts().to_dict()
    institutos = [{"instituto": k, "n": int(contagem.get(k, 0)),
                   "peso": round(cal.pesos_institutos.get(k, 1.0), 2),
                   "tem_historico": k in cal.pesos_institutos, **v} for k, v in efeitos.items()]
    institutos.sort(key=lambda d: -d["n"])

    geo = cal.resumo["geo"]
    calib = {
        "erros_t1": cal.erros_t1.round(2).to_dict("records"),
        "erros_h2h": cal.erros_h2h.round(2).to_dict("records"),
        "erros_t1_media_simples": cal.resumo["erros_t1_media_simples"].round(2).to_dict("records"),
        "vies_t1": cal.vies_t1, "dp_t1": cal.dp_t1, "vies_h2h": cal.vies_h2h, "dp_h2h": cal.dp_h2h,
        "rho_t1_t2": cal.rho_t1_t2, "persistencia_uf": cal.lean_coef[1], "dp_estado": cal.dp_estado,
        "dp_regiao": cal.dp_regiao, "multiplicador_casa": cal.multiplicador_casa,
        "casos_casa": cal.resumo["casos_casa"].round(2).to_dict("records"),
        "geo_pontos": geo[["uf", "x", "y", "par"]].round(3).to_dict("records"),
    }

    return {
        "eleicao": {"t1": DATA_T1.strftime("%Y-%m-%d"), "t2": DATA_T2.strftime("%Y-%m-%d")},
        "n_sim": n, "peso_vies": cfg.peso_vies,
        "p_segundo_turno": float(1 - res.decidido_t1.mean()),
        "candidatos": candidatos, "confrontos": confrontos, "estados": estados,
        "tendencia_t1": trilhas1, "pesquisas_t1": pontos1,
        "tendencia_t2": trilhas2, "pesquisas_t2": pontos2,
        "t2_agregado": {r: {"media": ent.t2_media[r], "dp": ent.t2_dp[r]} for r in ent.t2_media},
        "institutos": institutos, "calibracao": calib, "sensibilidade": sens,
        "n_pesquisas_t1": int(len(p1)), "n_confrontos_t2": int(len(p2)),
    }


def salvar_json(saida: dict, caminho) -> None:
    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (pd.Timestamp, date)):
            return o.isoformat()
        raise TypeError(type(o))
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(saida, f, ensure_ascii=False, default=conv)
