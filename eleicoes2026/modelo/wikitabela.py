"""Parser mínimo de tabelas em wikitexto (MediaWiki) com suporte a rowspan/colspan.

A Wikipédia é a fonte mais completa e aberta de pesquisas eleitorais brasileiras
compiladas. Este módulo converte o wikitexto bruto em grades (listas de linhas)
já com células mescladas expandidas, prontas para interpretação semântica.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_ATTR_RE = re.compile(r"(rowspan|colspan)\s*=\s*\"?(\d+)\"?", re.I)


@dataclass
class Celula:
    texto: str
    cabecalho: bool
    rowspan: int = 1
    colspan: int = 1
    bruto: str = ""


@dataclass
class Tabela:
    secao: list[str]                      # pilha de títulos de seção acima da tabela
    linhas: list[list[Celula]] = field(default_factory=list)

    def grade(self) -> list[list[Celula | None]]:
        """Expande rowspan/colspan, devolvendo uma grade retangular."""
        grade: list[list[Celula | None]] = []
        pendentes: dict[int, tuple[Celula, int]] = {}  # coluna -> (célula, linhas restantes)
        for linha in self.linhas:
            nova: list[Celula | None] = []
            col = 0
            fila = list(linha)
            while fila or any(c >= col for c in pendentes):
                if col in pendentes:
                    cel, resta = pendentes[col]
                    nova.append(cel)
                    if resta <= 1:
                        del pendentes[col]
                    else:
                        pendentes[col] = (cel, resta - 1)
                    col += 1
                    continue
                if not fila:
                    # buracos antes de colunas pendentes mais à direita
                    if any(c > col for c in pendentes):
                        nova.append(None)
                        col += 1
                        continue
                    break
                cel = fila.pop(0)
                for _ in range(cel.colspan):
                    nova.append(cel)
                    if cel.rowspan > 1:
                        pendentes[col] = (cel, cel.rowspan - 1)
                    col += 1
            grade.append(nova)
        largura = max((len(l) for l in grade), default=0)
        for l in grade:
            l.extend([None] * (largura - len(l)))
        return grade


def _dividir_fora_de_colchetes(s: str, sep: str) -> list[str]:
    """Divide `s` por `sep` ignorando ocorrências dentro de [[...]] e {{...}}."""
    partes, atual, prof, i = [], [], 0, 0
    while i < len(s):
        dois = s[i:i + 2]
        if dois in ("[[", "{{"):
            prof += 1
            atual.append(dois)
            i += 2
            continue
        if dois in ("]]", "}}"):
            prof = max(0, prof - 1)
            atual.append(dois)
            i += 2
            continue
        if prof == 0 and s.startswith(sep, i):
            partes.append("".join(atual))
            atual = []
            i += len(sep)
            continue
        atual.append(s[i])
        i += 1
    partes.append("".join(atual))
    return partes


def _celula(bruto: str, cabecalho: bool) -> Celula:
    # {{left}}/{{center}} expandem para 'style=... |' e, portanto, carregam o separador.
    bruto = re.sub(r"\{\{\s*(left|center|right)\s*\}\}", r'style="text-align:\1" |', bruto, count=1)
    partes = _dividir_fora_de_colchetes(bruto, "|")
    if len(partes) >= 2 and not partes[0].strip().startswith(("[[", "{{", "'")) and (
        "=" in partes[0] or partes[0].strip() == ""
    ):
        atributos, texto = partes[0], "|".join(partes[1:])
    else:
        atributos, texto = "", bruto
    rowspan = colspan = 1
    for nome, val in _ATTR_RE.findall(atributos):
        if nome.lower() == "rowspan":
            rowspan = int(val)
        else:
            colspan = int(val)
    return Celula(texto=texto.strip(), cabecalho=cabecalho, rowspan=rowspan, colspan=colspan, bruto=bruto)


def extrair_tabelas(wikitexto: str) -> list[Tabela]:
    """Extrai todas as tabelas de nível superior do wikitexto."""
    tabelas: list[Tabela] = []
    secao: list[str] = []
    atual: Tabela | None = None
    linha: list[Celula] | None = None
    profundidade = 0
    ultima: Celula | None = None

    for raw in wikitexto.splitlines():
        s = raw.strip()
        m = re.match(r"^(={2,6})\s*(.*?)\s*\1\s*$", s)
        if m and atual is None:
            nivel = len(m.group(1))
            secao = secao[: nivel - 2] + [m.group(2)]
            continue
        if s.startswith("{|"):
            profundidade += 1
            if profundidade == 1:
                atual = Tabela(secao=list(secao))
                linha = None
            continue
        if atual is None:
            continue
        if s.startswith("|}"):
            profundidade -= 1
            if profundidade == 0:
                if linha:
                    atual.linhas.append(linha)
                tabelas.append(atual)
                atual, linha, ultima = None, None, None
            continue
        if profundidade > 1:
            continue
        if s.startswith("|-"):
            if linha:
                atual.linhas.append(linha)
            linha = []
            ultima = None
            continue
        if s.startswith("|+"):
            continue
        if s.startswith("!") or s.startswith("|"):
            if linha is None:
                linha = []
            cab = s.startswith("!")
            corpo = s[1:]
            sep = "!!" if cab else "||"
            pedacos = _dividir_fora_de_colchetes(corpo, sep)
            if cab and len(pedacos) == 1:
                pedacos = _dividir_fora_de_colchetes(corpo, "||")
            for p in pedacos:
                ultima = _celula(p, cab)
                linha.append(ultima)
            continue
        # continuação de célula em múltiplas linhas
        if ultima is not None:
            ultima.texto = (ultima.texto + " " + s).strip()
    return tabelas


_LINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]*))?\]\]")


def texto_limpo(s: str) -> str:
    """Remove marcação wiki comum, preservando o texto visível."""
    s = re.sub(r"<ref[^>]*/>", "", s)
    s = re.sub(r"<ref[^>]*>.*?</ref>", "", s, flags=re.S)
    s = re.sub(r"<ref[^>]*>.*", "", s, flags=re.S)
    s = re.sub(r"\{\{\s*Deprecated archive\s*\|[^{}]*?title=([^{}|]*)[^{}]*\}\}", r"\1", s, flags=re.I)
    s = re.sub(r"\{\{\s*usurped\s*\|\s*1=(.*)\}\}", r"\1", s, flags=re.I | re.S)
    s = re.sub(r"\{\{\s*(?:efn|Efn|refn|sfn|Webarchive|webarchive)[^{}]*\}\}", "", s)
    s = re.sub(r"\{\{\s*(?:small|nowrap|left|center)\s*\|([^{}]*)\}\}", r"\1", s, flags=re.I)
    s = re.sub(r"\{\{\s*(?:left|center)\s*\}\}", "", s, flags=re.I)
    s = re.sub(r"\{\{\s*(?:n/a|NA|N/A|na|—|-)\s*(?:\|[^}]*)?\}\}", "", s)
    s = re.sub(r"\{\{[^{}]*\}\}", "", s)
    s = re.sub(r"\[\[(?:File|Ficheiro|Image|Arquivo):[^\]]*\]\]", "", s, flags=re.I)
    s = _LINK_RE.sub(lambda m: m.group(2) or m.group(1), s)
    s = re.sub(r"\[https?://\S+\s+([^\]]*)\]", r"\1", s)
    s = re.sub(r"\[https?://\S+\]", "", s)
    s = re.sub(r"<br\s*/?>", " ", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    s = s.replace("'''", "").replace("''", "").replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", s).strip()


def primeiro_link_pessoa(s: str) -> str | None:
    """Primeiro [[link]] do cabeçalho que não seja arquivo nem partido (em {{small}})."""
    s = re.sub(r"\{\{\s*small\s*\|.*?\}\}", "", s, flags=re.I | re.S)
    s = re.sub(r"\[\[(?:File|Image|Ficheiro|Arquivo):[^\]]*\]\]", "", s, flags=re.I)
    m = _LINK_RE.search(s)
    return m.group(1).strip() if m else None
