#!/usr/bin/env python3
"""Previsão da eleição presidencial de 2026.

Uso:
    python prever.py                 # usa as pesquisas em cache (dados/brutos)
    python prever.py --atualizar     # baixa as pesquisas mais recentes antes
    python prever.py --sim 50000 --peso-vies 1   # correção histórica inteira
    python prever.py --ate 2026-09-01   # "volta no tempo": só pesquisas até a data
    python prever.py --institutos AtlasIntel,MDA   # só alguns institutos
    python prever.py --atualizar --apuracao        # noite da eleição: usa a apuração do TSE
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from modelo.previsao import PESO_VIES_PADRAO, executar, salvar_json
from modelo.relatorio import gerar_painel

RAIZ = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atualizar", action="store_true", help="baixa as pesquisas mais recentes da Wikipédia")
    ap.add_argument("--apuracao", action="store_true",
                    help="baixa a apuração do 1º turno do TSE e projeta o resultado e o 2º turno a partir das urnas")
    ap.add_argument("--sim", type=int, default=20000, help="número de simulações (padrão: 20000)")
    ap.add_argument("--peso-vies", type=float, default=PESO_VIES_PADRAO,
                    help="quanto do viés histórico das pesquisas aplicar (0 = nenhum, 0,5 = padrão, 1 = inteiro)")
    ap.add_argument("--semente", type=int, default=2026)
    ap.add_argument("--ate", type=date.fromisoformat, default=None, help="usa só pesquisas até esta data (AAAA-MM-DD)")
    ap.add_argument("--institutos", type=lambda t: [x.strip() for x in t.split(",") if x.strip()], default=None,
                    help="usa só estes institutos, separados por vírgula (ex.: AtlasIntel,MDA)")
    ap.add_argument("--saida", type=Path, default=RAIZ / "saida")
    args = ap.parse_args()

    s = executar(n_sim=args.sim, peso_vies=args.peso_vies, atualizar=args.atualizar,
                 semente=args.semente, hoje=args.ate, institutos=args.institutos,
                 apuracao=args.apuracao)
    args.saida.mkdir(parents=True, exist_ok=True)
    salvar_json(s, args.saida / "previsao.json")
    html, _ = gerar_painel(s, args.saida)

    print("\n══ Previsão presidencial 2026 ═══════════════════════════════════")
    print(f"  Pesquisas até {s['ultima_pesquisa']} · {s['n_sim']:,} simulações · peso do viés {s['peso_vies']}".replace(",", "."))
    print(f"\n  {'Candidato':<18}{'1º turno':>9}{'  (80%)':>14}{'2º turno':>10}{'Eleito':>9}")
    for c in s["candidatos"]:
        print(f"  {c['nome']:<18}{c['t1_media']:>8.1f}%  {c['t1_p10']:>5.1f}–{c['t1_p90']:<5.1f}"
              f"{100 * c['p_segundo_turno']:>9.0f}%{100 * c['p_vitoria']:>8.0f}%")
    print(f"\n  Chance de 2º turno: {100 * s['p_segundo_turno']:.0f}%")
    for c in s["confrontos"][:3]:
        print(f"  {c['a']} × {c['b']}: ocorre em {100 * c['prob']:.1f}% · {c['a']} vence {100 * c['p_a_vence']:.0f}%"
              f" · {c['a']} com {c['a_media']:.1f}% dos válidos")
    print("\n  Sensibilidade ao viés histórico (chance de vitória):")
    for x in s["sensibilidade"]:
        print("   peso {:.1f}: ".format(x["peso_vies"]) +
              " · ".join(f"{k} {100 * v:.0f}%" for k, v in x["vitoria"].items() if v > 0.005))
    if "pos_t1" in s:
        q = s["pos_t1"]
        print(f"\n  ── Apuração do 1º turno: {100 * q['apurado']:.1f}% das seções ──")
        for c, v in q["t1"].items():
            if c != "Outros":
                print(f"   {c:<18}{v['media']:5.1f}%  ({v['p10']:.1f}–{v['p90']:.1f})")
        print(f"   Decidido no 1º turno: {100 * q['p_decidido_t1']:.0f}%")
        if q["t2"]["media"] is not None:
            print(f"   2º turno: Lula {q['t2']['media']:.1f}% ({q['t2']['p10']:.1f}–{q['t2']['p90']:.1f})"
                  f" · Lula eleito {100 * q['p_principal']:.0f}%")
    print(f"\n  Painel: {html}\n  Dados:  {args.saida / 'previsao.json'}")


if __name__ == "__main__":
    main()
