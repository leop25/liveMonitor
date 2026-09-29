"""Coleta e interpretação de pesquisas eleitorais e resultados históricos.

Fontes:
- Pesquisas: páginas "Opinion polling for the <ano> Brazilian presidential election"
  da Wikipédia em inglês (wikitexto), que compilam todas as pesquisas registradas.
- Resultados por UF (2002–2022): páginas "Resultados da eleição presidencial no
  Brasil em <ano>" da Wikipédia em português, que reproduzem a totalização do TSE.

Os arquivos brutos ficam em dados/brutos para que o modelo rode offline e seja
reprodutível; `atualizar_fontes()` baixa versões novas.
"""
from __future__ import annotations

import re
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .wikitabela import extrair_tabelas, primeiro_link_pessoa, texto_limpo

RAIZ = Path(__file__).resolve().parent.parent
BRUTOS = RAIZ / "dados" / "brutos"
PROCESSADOS = RAIZ / "dados" / "processados"

UFS = ["AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS", "MG", "PA",
       "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO"]
NOME_UF = {
    "Acre": "AC", "Alagoas": "AL", "Amapá": "AP", "Amazonas": "AM", "Bahia": "BA", "Ceará": "CE",
    "Distrito Federal": "DF", "Espírito Santo": "ES", "Goiás": "GO", "Maranhão": "MA",
    "Mato Grosso": "MT", "Mato Grosso do Sul": "MS", "Minas Gerais": "MG", "Pará": "PA",
    "Paraíba": "PB", "Paraná": "PR", "Pernambuco": "PE", "Piauí": "PI", "Rio de Janeiro": "RJ",
    "Rio Grande do Norte": "RN", "Rio Grande do Sul": "RS", "Rondônia": "RO", "Roraima": "RR",
    "Santa Catarina": "SC", "São Paulo": "SP", "Sergipe": "SE", "Tocantins": "TO",
}

# Nome canônico dos candidatos a partir do alvo do link na Wikipédia.
CANONICO = {
    "Luiz Inácio Lula da Silva": "Lula",
    "Flávio Bolsonaro": "Flávio Bolsonaro",
    "Ronaldo Caiado": "Caiado",
    "Romeu Zema": "Zema",
    "Renan Santos": "Renan Santos",
    "Augusto Cury": "Augusto Cury",
    "Jair Bolsonaro": "Jair Bolsonaro",
    "Fernando Haddad": "Haddad",
    "Ciro Gomes": "Ciro Gomes",
    "Simone Tebet": "Tebet",
    "Geraldo Alckmin": "Alckmin",
    "Marina Silva": "Marina Silva",
    "Dilma Rousseff": "Dilma",
    "José Serra": "Serra",
    "Tarcísio de Freitas": "Tarcísio",
    "Michelle Bolsonaro": "Michelle Bolsonaro",
    "Ratinho Júnior": "Ratinho Jr.",
    "Eduardo Leite": "Eduardo Leite",
    "Pablo Marçal": "Pablo Marçal",
}

APELIDOS = {"Bolsonaro": "Jair Bolsonaro", "Ciro": "Ciro Gomes", "Marina": "Marina Silva",
            "Lula": "Lula", "Haddad": "Haddad", "Alckmin": "Alckmin", "Serra": "Serra", "Dilma": "Dilma"}

MESES = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
MESES.update({"fev": 2, "abr": 4, "mai": 5, "ago": 8, "set": 9, "out": 10, "dez": 12})

URL_PESQUISAS = "https://en.wikipedia.org/w/index.php?title=Opinion_polling_for_the_{ano}_Brazilian_presidential_election&action=raw"
URL_PESQUISAS_PT = "https://pt.wikipedia.org/w/index.php?title=Pesquisas_de_opini%C3%A3o_para_a_elei%C3%A7%C3%A3o_presidencial_no_Brasil_em_{ano}&action=raw"
URL_RESULTADOS = "https://pt.wikipedia.org/wiki/Resultados_da_elei%C3%A7%C3%A3o_presidencial_no_Brasil_em_{ano}"
URL_RESULTADOS_2006 = "https://pt.wikipedia.org/wiki/Elei%C3%A7%C3%A3o_presidencial_no_Brasil_em_2006"


# ----------------------------------------------------------------------------- download

def _baixar(url: str, destino: Path) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": "eleicoes2026-modelo/1.0 (pesquisa acadêmica)"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            conteudo = r.read()
    except Exception as e:  # noqa: BLE001 - relatar e seguir com o cache local
        print(f"  ! falha ao baixar {url}: {e}")
        return False
    if len(conteudo) < 5000:
        print(f"  ! resposta curta demais de {url}; mantendo cache")
        return False
    destino.write_bytes(conteudo)
    return True


def atualizar_fontes(anos_pesquisas=(2026,), anos_resultados=()) -> None:
    """Baixa novamente as páginas-fonte (por padrão, apenas as pesquisas de 2026)."""
    BRUTOS.mkdir(parents=True, exist_ok=True)
    for ano in anos_pesquisas:
        ok = _baixar(URL_PESQUISAS.format(ano=ano), BRUTOS / f"pesquisas_{ano}.wiki")
        print(f"  pesquisas {ano}: {'atualizado' if ok else 'cache'}")
        if ano == 2026:
            ok = _baixar(URL_PESQUISAS_PT.format(ano=ano), BRUTOS / f"pesquisas_{ano}_pt.wiki")
            print(f"  pesquisas {ano} (Wikipédia em português): {'atualizado' if ok else 'cache'}")
    for ano in anos_resultados:
        url = URL_RESULTADOS_2006 if ano == 2006 else URL_RESULTADOS.format(ano=ano)
        ok = _baixar(url, BRUTOS / f"res_{ano}.html")
        print(f"  resultados {ano}: {'atualizado' if ok else 'cache'}")


# ----------------------------------------------------------------------------- pesquisas

def _numero(txt: str) -> float:
    t = texto_limpo(txt).replace("%", "").replace("pp", "").strip()
    t = t.replace("±", "").strip()
    if not t or t in {"–", "-", "—", "_", "N/A", "(N/A)", "n/a"}:
        return np.nan
    t = t.replace(",", ".") if re.fullmatch(r"\d+,\d+", t) else t.replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", t)
    return float(m.group()) if m else np.nan


def _datas(txt: str, ano_padrao: int) -> tuple[date | None, date | None]:
    t = texto_limpo(txt).lower().replace("–", "-").replace("—", "-")
    anos = [int(a) for a in re.findall(r"\b(20\d\d)\b", t)]
    ano = anos[-1] if anos else ano_padrao
    t = re.sub(r"\b20\d\d\b", " ", t)
    tokens = re.findall(r"(\d{1,2})(?:\s*([a-zç]{3,}))?", t)
    dias = []
    mes_corrente = None
    for d, m in reversed(tokens):
        if m and m[:3] in MESES:
            mes_corrente = MESES[m[:3]]
        if mes_corrente is None:
            continue
        dias.append((int(d), mes_corrente))
    if not dias:
        return None, None
    dias.reverse()

    def mk(d, m):
        try:
            return date(ano, m, d)
        except ValueError:
            return None
    fim = mk(*dias[-1])
    ini = mk(*dias[0])
    if ini and fim and ini > fim:  # período que cruza a virada do ano
        ini = date(ano - 1, ini.month, ini.day)
    return ini, fim


def _classificar_coluna(cabecalhos: list[str], links: list[str | None]) -> tuple[str, str | None]:
    rotulo = " ".join(cabecalhos).lower()
    pessoa = next((l for l in links if l), None)
    if pessoa and not any(p in pessoa for p in ("Party", "Partido", "Federation", "Coalition")):
        return "candidato", CANONICO.get(pessoa, pessoa)
    if pessoa:  # coluna por partido: o nome do candidato vem dentro da célula
        return "partido", None
    for chave, termos in [
        ("instituto", ("pollster", "polling firm", "publisher", "instituto", "contratante")),
        ("data", ("period", "date", "fieldwork", "administered", "data(s)", "data de")),
        ("amostra", ("sample", "amostra")),
        ("margem", ("margin", "margem")),
        ("vantagem", ("lead",)),
        ("indecisos", ("blank", "undec", "abst", "null", "none", "indecis", "branco", "nulo")),
        ("outros", ("others", "outros", "not affiliated")),
        ("link", ("link", "ref")),
    ]:
        if any(t in rotulo for t in termos):
            return chave, None
    return "?", None


def ler_pesquisas(wikitexto: str, ano_padrao: int) -> pd.DataFrame:
    """Converte a página de pesquisas em DataFrame (uma linha por cenário pesquisado)."""
    registros = []
    for t_idx, tab in enumerate(extrair_tabelas(wikitexto)):
        secoes = " / ".join(tab.secao).lower()
        if "aggregat" in secoes or "see also" in secoes:
            continue
        turno = 2 if ("second round" in secoes or "segundo turno" in secoes) else 1
        if turno == 2 and "other" in secoes.split(" / ")[-1]:
            continue
        ano_secao = [int(a) for a in re.findall(r"\b(20\d\d)\b", secoes)]
        ano_tab = ano_secao[-1] if ano_secao else ano_padrao
        grade = tab.grade()
        if not grade:
            continue
        # linhas de cabeçalho: prefixo de linhas só com células de cabeçalho ou vazias
        n_cab = 0
        for linha in grade:
            cels = [c for c in linha if c is not None]
            if cels and all(c.cabecalho or texto_limpo(c.texto) == "" for c in cels):
                n_cab += 1
            else:
                break
        if n_cab == 0:
            continue
        largura = len(grade[0])
        tipos = []
        for j in range(largura):
            cabs = [grade[i][j] for i in range(n_cab) if grade[i][j] is not None]
            vistos, textos, links = set(), [], []
            for c in cabs:
                if id(c) in vistos:
                    continue
                vistos.add(id(c))
                textos.append(texto_limpo(c.texto))
                links.append(primeiro_link_pessoa(c.texto))
            tipos.append(_classificar_coluna(textos, links))
        if not any(t == "candidato" for t, _ in tipos):
            continue
        if not any(t == "instituto" for t, _ in tipos):
            continue
        fim_anterior = None
        for i_lin, linha in enumerate(grade[n_cab:]):
            reg = {"turno": turno, "tabela": t_idx, "secao": tab.secao[-1] if tab.secao else ""}
            cands = {}
            for (tipo, nome), cel in zip(tipos, linha):
                if cel is None:
                    continue
                if tipo == "instituto":
                    reg["instituto"] = texto_limpo(cel.texto)
                elif tipo == "data":
                    reg["data_txt"] = texto_limpo(cel.texto)
                    reg["inicio"], reg["fim"] = _datas(cel.texto, ano_tab)
                elif tipo == "amostra":
                    digitos = re.sub(r"\D", "", texto_limpo(cel.texto))
                    reg["amostra"] = float(digitos) if digitos else np.nan
                elif tipo == "candidato":
                    cands[nome] = _numero(cel.texto)
                elif tipo == "partido":
                    m = re.search(r"\(([^()]+)\)", texto_limpo(cel.texto))
                    if m:
                        quem = APELIDOS.get(m.group(1).strip(), m.group(1).strip())
                        cands[quem] = _numero(re.sub(r"\(.*?\)", "", cel.texto))
                elif tipo in ("outros", "indecisos"):
                    reg[tipo] = np.nansum([reg.get(tipo, np.nan), _numero(cel.texto)]) \
                        if not np.isnan(_numero(cel.texto)) else reg.get(tipo, np.nan)
            inst = reg.get("instituto", "")
            if not inst or len(inst) > 40 or re.match(r"^[\d#%]", inst) or re.search(r"result|resultado|election|eleição|valid votes|total votes", inst, re.I):
                continue
            if reg.get("fim") is None:
                continue
            # tabelas em ordem cronológica reversa sem o ano na data: ajusta a virada do ano
            if not re.search(r"20\d\d", reg.get("data_txt", "")) and fim_anterior is not None:
                while reg["fim"] > fim_anterior + pd.Timedelta(days=60).to_pytimedelta():
                    reg["fim"] = reg["fim"].replace(year=reg["fim"].year - 1)
                    if reg.get("inicio"):
                        reg["inicio"] = reg["inicio"].replace(year=reg["inicio"].year - 1)
            fim_anterior = reg["fim"]
            cands = {k: v for k, v in cands.items() if not np.isnan(v)}
            if len(cands) < 2:
                continue
            reg["candidatos"] = cands
            registros.append(reg)
    df = pd.DataFrame(registros)
    if df.empty:
        return df
    df["instituto"] = df["instituto"].map(normalizar_instituto)
    if "amostra" not in df:
        df["amostra"] = np.nan
    df["fim"] = pd.to_datetime(df["fim"])
    df["inicio"] = pd.to_datetime(df["inicio"]).fillna(df["fim"])
    df["meio"] = df["inicio"] + (df["fim"] - df["inicio"]) / 2
    return df.reset_index(drop=True)


def normalizar_instituto(nome: str) -> str:
    n = re.sub(r"\s+", " ", nome).strip()
    n = re.sub(r"\(.*?\)", "", n).strip()
    base = n.lower()
    regras = [
        ("atlas", "AtlasIntel"), ("datafolha", "Datafolha"), ("quaest", "Quaest"),
        ("ipec", "Ipec"), ("ibope", "Ipec"), ("poderdata", "PoderData"),
        ("paraná pesq", "Paraná Pesquisas"), ("parana pesq", "Paraná Pesquisas"),
        ("real time", "Real Time Big Data"), ("realtime", "Real Time Big Data"),
        ("datapoder", "PoderData"), ("fsb", "FSB"), ("futura", "Futura"), ("palver", "Palver"),
        ("nexus", "Nexus"), ("ipespe", "Ipespe"), ("mda", "MDA"), ("vox populi", "Vox Populi"),
        ("genial", "Quaest"), ("ideia", "Ideia"), ("idea", "Ideia"), ("gerp", "Gerp"),
        ("sensus", "Sensus"), ("veritá", "Veritá"), ("verita", "Veritá"), ("ipsos", "Ipsos"),
        ("modalmais", "Futura"), ("brasmarket", "Brasmarket"), ("abc dados", "ABC Dados"),
        ("meio", "Ideia"), ("vox brasil", "Vox Brasil"), ("gpp", "GPP"),
    ]
    for chave, canon in regras:
        if chave in base:
            return canon
    return n


def pesquisas_primeiro_turno(df: pd.DataFrame, candidatos: list[str]) -> pd.DataFrame:
    """Mantém cenários de 1º turno que correspondem à lista final de candidatos.

    Um cenário é aceito se inclui os dois principais candidatos e nenhum nome fora
    da lista final com votação relevante (>2%). Devolve formato largo em votos
    totais (%), mais colunas de 'Outros' e 'Indecisos'.
    """
    linhas = []
    finais = set(candidatos)
    for _, r in df[df.turno == 1].iterrows():
        c = r["candidatos"]
        if not all(k in c for k in candidatos[:2]):
            continue
        intrusos = {k: v for k, v in c.items() if k not in finais}
        if any(v > 2 for v in intrusos.values()):
            continue
        faltam = [k for k in candidatos if k not in c]
        if len(faltam) > 2:
            continue
        linha = {"instituto": r.instituto, "inicio": r.inicio, "fim": r.fim, "meio": r.meio,
                 "amostra": r.get("amostra", np.nan)}
        for k in candidatos:
            linha[k] = c.get(k, np.nan)
        linha["Outros"] = np.nansum([r.get("outros", np.nan) or 0, sum(intrusos.values())])
        linha["Indecisos"] = r.get("indecisos", np.nan)
        linhas.append(linha)
    out = pd.DataFrame(linhas)
    return out.sort_values("meio").reset_index(drop=True)


def unir_fontes(*tabelas: pd.DataFrame, tolerancia_dias: int = 3) -> pd.DataFrame:
    """Une cenários de 1º turno de várias fontes, sem contar duas vezes a mesma pesquisa.

    Duas linhas são a mesma pesquisa quando têm o mesmo instituto e datas finais a até
    `tolerancia_dias` de distância (as páginas às vezes registram o campo de forma diferente).
    Vale a primeira tabela passada; linhas repetidas dentro da mesma fonte são promediadas.
    """
    partes = []
    vistas: dict[str, list[pd.Timestamp]] = {}
    tol = pd.Timedelta(days=tolerancia_dias)
    for t in tabelas:
        if t is None or t.empty:
            continue
        t = t.copy()
        t["_chave"] = list(zip(t.instituto, t.fim.dt.normalize()))
        num = t.select_dtypes("number").columns
        agreg = {c: "mean" for c in num}
        agreg.update({c: "first" for c in t.columns if c not in num and c != "_chave"})
        t = t.groupby("_chave", sort=False).agg(agreg).reset_index(drop=True)
        repetida = t.apply(lambda r: any(abs(r.fim - f) <= tol for f in vistas.get(r.instituto, [])), axis=1)
        t = t[~repetida]
        for inst, fim in zip(t.instituto, t.fim):
            vistas.setdefault(inst, []).append(fim)
        partes.append(t)
    return pd.concat(partes, ignore_index=True).sort_values("meio").reset_index(drop=True)


def pesquisas_segundo_turno(df: pd.DataFrame) -> pd.DataFrame:
    """Confrontos diretos: uma linha por (pesquisa, par de candidatos)."""
    linhas = []
    for _, r in df[df.turno == 2].iterrows():
        c = r["candidatos"]
        if len(c) != 2:
            continue
        (a, va), (b, vb) = sorted(c.items())
        if va + vb <= 0 or va + vb > 101:
            continue
        linhas.append({"instituto": r.instituto, "inicio": r.inicio, "fim": r.fim, "meio": r.meio,
                       "amostra": r.get("amostra", np.nan), "a": a, "b": b, "va": va, "vb": vb,
                       "indecisos": r.get("indecisos", np.nan)})
    colunas = ["instituto", "inicio", "fim", "meio", "amostra", "a", "b", "va", "vb", "indecisos"]
    return pd.DataFrame(linhas, columns=colunas).sort_values("meio").reset_index(drop=True)


# ----------------------------------------------------------------------------- resultados

def _uf_da_celula(txt: str) -> str | None:
    t = re.sub(r"\[.*?\]", "", str(txt)).strip()
    if t in NOME_UF:
        return NOME_UF[t]
    for nome, uf in NOME_UF.items():
        if t.startswith(nome) and not t.startswith(nome + " do") or t == nome:
            return uf
    return None


def _num_br(x) -> float:
    """Contagem de votos (inteiro) escrita com separadores variados: '1 141.607', '3.563'."""
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    s = re.sub(r"\D", "", str(x))
    return float(s) if s else np.nan


# Quais tabelas (índice em pandas.read_html) contêm os resultados por UF.
TABELAS_RESULTADO = {
    2002: {1: 2, 2: 4},
    2006: {1: 10, 2: 11},
    2010: {1: 2, 2: 4},
    2014: {1: 2, 2: 4},
    2018: {1: 5, 2: 7},
    2022: {1: 1, 2: 2},
}


def ler_resultados_uf() -> pd.DataFrame:
    """Votos por candidato e UF, 1º e 2º turnos, 2002–2022 (formato longo).

    Lê as páginas brutas quando existem (após `atualizar_fontes(anos_resultados=...)`);
    caso contrário, usa o CSV já processado que acompanha o repositório.
    """
    csv = PROCESSADOS / "resultados_uf_2002_2022.csv"
    if csv.exists() and not all((BRUTOS / f"res_{a}.html").exists() for a in TABELAS_RESULTADO):
        return pd.read_csv(csv)
    linhas = []
    for ano, tabs in TABELAS_RESULTADO.items():
        caminho = BRUTOS / f"res_{ano}.html"
        if not caminho.exists():
            continue
        todas = pd.read_html(caminho, decimal=",", thousands=".")
        for turno, idx in tabs.items():
            t = todas[idx]
            t.columns = [str(c) for c in t.columns]
            col_uf = t.columns[0]
            # colunas de candidatos: cabeçalho sem '%' seguido de uma coluna '%'
            ignorar = ("eleitorado", "abstenção", "votos", "brancos", "nulos", "estado", "comparecimento")
            nomes = [c for c in t.columns[1:] if not c.startswith("%") and
                     not any(k in c.lower() for k in ignorar)]
            for _, r in t.iterrows():
                uf = _uf_da_celula(r[col_uf])
                if uf is None:
                    continue
                for nome in nomes:
                    v = _num_br(r[nome])
                    if np.isnan(v):
                        continue
                    linhas.append({"ano": ano, "turno": turno, "uf": uf, "candidato": nome, "votos": v})
    df = pd.DataFrame(linhas)
    # 2010 (2º turno) traz linhas duplicadas por UF na página; mantém a primeira.
    return df.drop_duplicates(["ano", "turno", "uf", "candidato"]).reset_index(drop=True)
