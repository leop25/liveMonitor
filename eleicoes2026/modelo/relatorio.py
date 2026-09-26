"""Gera o painel HTML (arquivo único, sem dependências além de fontes do Google)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TEMPLATE = Path(__file__).with_name("painel_template.html")


def _json(saida: dict) -> str:
    def conv(o):
        if isinstance(o, np.floating):
            return round(float(o), 4)
        if isinstance(o, np.integer):
            return int(o)
        raise TypeError(type(o))
    texto = json.dumps(saida, ensure_ascii=False, default=conv, allow_nan=False)
    return texto.replace("</", "<\\/")


def gerar_painel(saida: dict, destino: Path) -> tuple[Path, Path]:
    """Escreve `painel.html` (documento completo) e `painel_fragmento.html` (para publicar)."""
    frag = TEMPLATE.read_text(encoding="utf-8").replace("__DADOS__", _json(_limpar(saida)))
    destino.mkdir(parents=True, exist_ok=True)
    completo = ('<!doctype html>\n<html lang="pt-BR">\n<head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
                '</head>\n<body>\n' + frag + "\n</body>\n</html>\n")
    p1 = destino / "painel.html"
    p2 = destino / "painel_fragmento.html"
    p1.write_text(completo, encoding="utf-8")
    p2.write_text(frag, encoding="utf-8")
    return p1, p2


def _limpar(o):
    """Troca NaN/inf por None (JSON válido) recursivamente."""
    if isinstance(o, dict):
        return {k: _limpar(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_limpar(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    return o
