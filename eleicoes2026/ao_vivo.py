#!/usr/bin/env python3
"""Previsão ao vivo durante a apuração do 1º turno.

Cada rodada: baixa a apuração do TSE (27 UFs), refaz a projeção do 1º turno e a previsão do
2º turno a partir das urnas, guarda o ponto no histórico da noite (saida/apuracao_historico.csv)
e regenera o painel (saida/painel.html).

Uso:
    python ao_vivo.py                 # uma rodada
    python ao_vivo.py --a-cada 300    # uma rodada a cada 5 minutos, até a apuração terminar
    python ao_vivo.py --forcar        # roda mesmo que o TSE não tenha mudado nada
    python ao_vivo.py --simulado      # testa o pipeline no simulado oficial do TSE (saída em saida_simulado/)
"""
from __future__ import annotations

import argparse
import hashlib
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import json

import pandas as pd

from modelo import pos_primeiro_turno
from modelo.previsao import executar, salvar_json
from modelo.relatorio import gerar_painel

RAIZ = Path(__file__).resolve().parent
SAIDA = RAIZ / "saida"
HISTORICO = SAIDA / "apuracao_historico.csv"
MARCA = SAIDA / ".apuracao_hash"
BRASILIA = timezone(timedelta(hours=-3))


def _hash_apuracao() -> str:
    h = hashlib.sha256()
    for f in sorted(pos_primeiro_turno.DIR_APURACAO.glob("*.json")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


SIMULADO = {"base": "https://resultados-sim.tse.jus.br/simulado/simulado2026", "eleicao": "21270"}


def usar_simulado() -> None:
    """Aponta tudo para o simulado oficial do TSE, com saída e arquivos separados dos reais."""
    global SAIDA, HISTORICO, MARCA
    pos_primeiro_turno.BASE_TSE = SIMULADO["base"]
    pos_primeiro_turno.ELEICAO_T1 = SIMULADO["eleicao"]
    pos_primeiro_turno.DIR_APURACAO = RAIZ / "dados" / "brutos" / "tse_simulado"
    SAIDA = RAIZ / "saida_simulado"
    HISTORICO, MARCA = SAIDA / "apuracao_historico.csv", SAIDA / ".apuracao_hash"


def _nomes_do_simulado() -> None:
    """No simulado os candidatos se chamam 'CANDIDATO 9995'; atribui os nomes reais pela ordem de votos."""
    f = pos_primeiro_turno.DIR_APURACAO / "br.json"
    if not f.exists():
        return
    d = json.loads(f.read_text(encoding="utf-8"))
    cands = [c for a in d["carg"][0]["agr"] for p in a["par"] for c in p["cand"]
             if str(c.get("dvt", "")).startswith("Válido")]
    cands.sort(key=lambda c: -int(c.get("vap") or 0))
    nomes = ["Lula", "Flávio Bolsonaro", "Augusto Cury", "Renan Santos", "Caiado", "Zema"]
    pos_primeiro_turno.NUMEROS_TSE = {str(c["n"]): nm for c, nm in zip(cands, nomes)}


def rodada(n_sim: int = 10000, forcar: bool = False, simulado: bool = False) -> dict | None:
    agora = datetime.now(BRASILIA)
    baixados = pos_primeiro_turno.baixar_apuracao(eleicao=pos_primeiro_turno.ELEICAO_T1,
                                                  destino=pos_primeiro_turno.DIR_APURACAO, verbose=False)
    if simulado:
        _nomes_do_simulado()
    if not baixados and not pos_primeiro_turno.CSV_MANUAL.exists():
        print(f"[{agora:%H:%M}] TSE ainda não publicou a apuração")
        return None
    marca = _hash_apuracao()
    if not forcar and MARCA.exists() and MARCA.read_text() == marca:
        print(f"[{agora:%H:%M}] apuração sem mudança desde a última rodada")
        return None

    s = executar(n_sim=n_sim, verbose=False)
    q = s.get("pos_t1")
    if q is None:
        salvar_json(s, SAIDA / "previsao.json")   # painel mostra a apuração "aguardando votos"
        gerar_painel(s, SAIDA)
        print(f"[{agora:%H:%M}] arquivos do TSE sem votos ainda")
        return None

    P, A = s["candidatos"][0]["nome"], s["candidatos"][1]["nome"]
    ponto = {
        "hora": agora.strftime("%Y-%m-%d %H:%M"), "apurado": round(100 * q["apurado"], 2),
        "parcial_lula": round(q["parcial_br"].get(P, float("nan")), 2),
        "parcial_flavio": round(q["parcial_br"].get(A, float("nan")), 2),
        "proj_lula": round(q["t1"][P]["media"], 2), "proj_flavio": round(q["t1"][A]["media"], 2),
        "p_segundo_turno": round(1 - q["p_decidido_t1"], 4),
        "p_lula_vence_t1": round(q["p_vence_t1"].get(P, 0), 4),
        "p_flavio_vence_t1": round(q["p_vence_t1"].get(A, 0), 4),
        "t2_lula": round(q["t2"]["media"], 2) if q["t2"]["media"] is not None else None,
        "p_lula_eleito": round(q["p_principal"], 4),
    }
    hist = pd.read_csv(HISTORICO) if HISTORICO.exists() else pd.DataFrame()
    hist = pd.concat([hist, pd.DataFrame([ponto])], ignore_index=True)
    SAIDA.mkdir(exist_ok=True)
    hist.to_csv(HISTORICO, index=False)
    q["historico"] = hist.where(pd.notna(hist), None).to_dict("records")

    salvar_json(s, SAIDA / "previsao.json")
    gerar_painel(s, SAIDA)
    MARCA.write_text(marca)

    print(f"[{agora:%H:%M}] apurado {ponto['apurado']:.1f}% · parcial {P} {ponto['parcial_lula']:.1f} × "
          f"{ponto['parcial_flavio']:.1f} · projeção {ponto['proj_lula']:.1f} × {ponto['proj_flavio']:.1f} · "
          f"2º turno {100 * ponto['p_segundo_turno']:.0f}% · {P} eleito {100 * ponto['p_lula_eleito']:.0f}%")
    return ponto


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a-cada", type=int, default=0, help="segundos entre rodadas (0 = uma rodada só)")
    ap.add_argument("--sim", type=int, default=10000)
    ap.add_argument("--forcar", action="store_true")
    ap.add_argument("--simulado", action="store_true", help="usa o simulado oficial do TSE (teste)")
    args = ap.parse_args()
    if args.simulado:
        usar_simulado()
    while True:
        ponto = rodada(args.sim, args.forcar, args.simulado)
        if not args.a_cada or (ponto and ponto["apurado"] >= 100):
            break
        time.sleep(args.a_cada)


if __name__ == "__main__":
    main()
