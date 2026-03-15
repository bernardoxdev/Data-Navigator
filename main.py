from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from difflib import get_close_matches
from google import genai
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

# =========================================================
# CONFIG
# =========================================================
DEBUG = True

# Gemini
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# Diretórios
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "inputs"
OUTPUT_DIR = BASE_DIR / "outputs"
CREDENTIALS_FILE = BASE_DIR / "credentials" / "service_account.json"

# Escopos para Sheets e Drive
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

SYSTEM_PROMPT = """
Você é um assistente de planilhas.
Sua função é interpretar pedidos do usuário e retornar SOMENTE um JSON válido.
Não escreva explicações fora do JSON.
Nunca use markdown.
Nunca use blocos de código.

Ações permitidas:
- destacar_coluna
- explicar_coluna
- buscar_texto
- resumir_aba
- listar_abas
- listar_colunas

Formato esperado:
{
  "acao": "nome_da_acao",
  "aba": "nome_da_aba_ou_null",
  "coluna": "nome_da_coluna_ou_null",
  "texto": "texto_ou_null"
}

Regras:
- Se o usuário pedir para destacar uma coluna, use "destacar_coluna".
- Se o usuário pedir explicação sobre uma coluna, use "explicar_coluna".
- Se o usuário pedir para procurar um protocolo, código, texto ou termo, use "buscar_texto".
- Se o usuário pedir um resumo da aba, use "resumir_aba".
- Se o usuário pedir abas disponíveis, use "listar_abas".
- Se o usuário pedir colunas disponíveis, use "listar_colunas".
- Se não souber a aba, retorne aba como null.
- Se não souber a coluna, retorne coluna como null.
- Se não houver texto, retorne texto como null.
- Responda sempre em JSON válido.
""".strip()


# =========================================================
# AUTH / CLIENTS
# =========================================================
def garantir_diretorios() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CREDENTIALS_FILE.parent.mkdir(parents=True, exist_ok=True)


def get_google_credentials() -> Credentials:
    if not CREDENTIALS_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo de credenciais não encontrado em: {CREDENTIALS_FILE}"
        )

    return Credentials.from_service_account_file(
        str(CREDENTIALS_FILE),
        scopes=SCOPES,
    )


def get_sheets_service():
    creds = get_google_credentials()
    return build("sheets", "v4", credentials=creds)


def get_drive_service():
    creds = get_google_credentials()
    return build("drive", "v3", credentials=creds)


def configurar_gemini() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Defina a variável GEMINI_API_KEY no ambiente.\n"
            "Exemplo:\n"
            'export GEMINI_API_KEY="SUA_CHAVE"'
        )
    return api_key


# =========================================================
# HELPERS
# =========================================================
def encontrar_nome_proximo(
    opcoes: list[str],
    termo: str | None,
    cutoff: float = 0.5,
) -> str | None:
    if termo is None:
        return None

    opcoes_str = [str(x) for x in opcoes]
    termo_lower = termo.lower().strip()

    for opc in opcoes_str:
        if opc.lower().strip() == termo_lower:
            return opc

    for opc in opcoes_str:
        if termo_lower in opc.lower():
            return opc

    matches = get_close_matches(
        termo_lower,
        [x.lower() for x in opcoes_str],
        n=1,
        cutoff=cutoff,
    )
    if not matches:
        return None

    for opc in opcoes_str:
        if opc.lower() == matches[0]:
            return opc

    return None


def extrair_json_bruto(texto: str) -> dict[str, Any]:
    texto = texto.strip()

    try:
        return json.loads(texto)
    except Exception:
        pass

    match = re.search(r"\{.*\}", texto, re.DOTALL)
    if match:
        trecho = match.group(0)
        try:
            return json.loads(trecho)
        except Exception:
            pass

    raise ValueError(f"Não foi possível extrair JSON válido:\n{texto}")


def sanitizar_nome_arquivo(texto: str) -> str:
    texto = texto.strip()
    texto = re.sub(r"[^\w\-_.]+", "_", texto, flags=re.UNICODE)
    texto = re.sub(r"_+", "_", texto)
    return texto[:120].strip("_") or "saida"


def col_idx_to_a1(col_idx_zero_based: int) -> str:
    col_num = col_idx_zero_based + 1
    result = ""
    while col_num > 0:
        col_num, rem = divmod(col_num - 1, 26)
        result = chr(65 + rem) + result
    return result


# =========================================================
# GEMINI AGENT
# =========================================================
class SpreadsheetGeminiAgent:
    def __init__(self, model_name: str = GEMINI_MODEL_NAME):
        api_key = configurar_gemini()
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

        if DEBUG:
            print(f"[DEBUG] Gemini carregado: {model_name}")

    def interpretar(
        self,
        comando_usuario: str,
        nomes_abas: list[str],
        colunas_por_aba: dict[str, list[str]],
    ) -> dict[str, Any]:
        contexto = {
            "abas_disponiveis": nomes_abas,
            "colunas_por_aba": colunas_por_aba,
            "pedido_usuario": comando_usuario,
        }

        prompt = (
            SYSTEM_PROMPT
            + "\n\nContexto:\n"
            + json.dumps(contexto, ensure_ascii=False, indent=2)
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

        texto = (response.text or "").strip()

        if DEBUG:
            print("\n[DEBUG] Resposta bruta do Gemini:")
            print(texto)

        if not texto:
            raise ValueError("O Gemini retornou uma resposta vazia.")

        acao = extrair_json_bruto(texto)
        acao.setdefault("acao", None)
        acao.setdefault("aba", None)
        acao.setdefault("coluna", None)
        acao.setdefault("texto", None)
        return acao


# =========================================================
# GOOGLE SHEETS
# =========================================================
def obter_metadata_planilha(sheets_service, spreadsheet_id: str) -> dict[str, Any]:
    return sheets_service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()


def normalizar_headers(headers: list[str]) -> list[str]:
    usados: dict[str, int] = {}
    saida: list[str] = []

    for i, h in enumerate(headers):
        nome = str(h).strip() if h is not None else ""
        if not nome:
            nome = f"coluna_{i + 1}"

        if nome in usados:
            usados[nome] += 1
            nome_final = f"{nome}_{usados[nome]}"
        else:
            usados[nome] = 1
            nome_final = nome

        saida.append(nome_final)

    return saida


def listar_abas_e_colunas(
    sheets_service,
    spreadsheet_id: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    metadata = obter_metadata_planilha(sheets_service, spreadsheet_id)

    planilhas: dict[str, pd.DataFrame] = {}
    sheet_ids: dict[str, int] = {}

    for sheet in metadata["sheets"]:
        props = sheet["properties"]
        nome_aba = props["title"]
        sheet_ids[nome_aba] = props["sheetId"]

        result = sheets_service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=nome_aba,
        ).execute()

        values = result.get("values", [])

        if not values:
            planilhas[nome_aba] = pd.DataFrame()
            continue

        header = normalizar_headers(values[0])
        rows = values[1:] if len(values) > 1 else []

        max_cols = len(header)
        normalized_rows = []
        for row in rows:
            row = list(row) + [""] * (max_cols - len(row))
            normalized_rows.append(row[:max_cols])

        planilhas[nome_aba] = pd.DataFrame(normalized_rows, columns=header)

    return planilhas, sheet_ids


def criar_planilha_google(sheets_service, titulo: str) -> tuple[str, dict[str, int]]:
    body = {
        "properties": {
            "title": titulo,
        }
    }

    spreadsheet = sheets_service.spreadsheets().create(body=body).execute()
    spreadsheet_id = spreadsheet["spreadsheetId"]

    metadata = obter_metadata_planilha(sheets_service, spreadsheet_id)
    sheet_ids = {
        s["properties"]["title"]: s["properties"]["sheetId"]
        for s in metadata["sheets"]
    }

    return spreadsheet_id, sheet_ids


def escrever_dataframe_em_aba(
    sheets_service,
    spreadsheet_id: str,
    aba: str,
    df: pd.DataFrame,
) -> None:
    values = [list(df.columns)] + df.fillna("").astype(str).values.tolist()

    sheets_service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=f"{aba}!A1",
        valueInputOption="RAW",
        body={"values": values},
    ).execute()


def renomear_aba(
    sheets_service,
    spreadsheet_id: str,
    sheet_id: int,
    novo_nome: str,
) -> None:
    body = {
        "requests": [
            {
                "updateSheetProperties": {
                    "properties": {
                        "sheetId": sheet_id,
                        "title": novo_nome,
                    },
                    "fields": "title",
                }
            }
        ]
    }

    sheets_service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body=body,
    ).execute()


def adicionar_aba(sheets_service, spreadsheet_id: str, nome_aba: str) -> int:
    body = {
        "requests": [
            {
                "addSheet": {
                    "properties": {
                        "title": nome_aba,
                    }
                }
            }
        ]
    }

    resp = sheets_service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body=body,
    ).execute()

    return resp["replies"][0]["addSheet"]["properties"]["sheetId"]


def destacar_coluna_google(
    sheets_service,
    spreadsheet_id: str,
    sheet_id: int,
    idx_coluna_zero_based: int,
    total_linhas: int,
) -> None:
    body = {
        "requests": [
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 0,
                        "endRowIndex": total_linhas,
                        "startColumnIndex": idx_coluna_zero_based,
                        "endColumnIndex": idx_coluna_zero_based + 1,
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "backgroundColor": {
                                "red": 1.0,
                                "green": 0.96,
                                "blue": 0.62,
                            }
                        }
                    },
                    "fields": "userEnteredFormat.backgroundColor",
                }
            },
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 0,
                        "endRowIndex": 1,
                        "startColumnIndex": idx_coluna_zero_based,
                        "endColumnIndex": idx_coluna_zero_based + 1,
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "backgroundColor": {
                                "red": 1.0,
                                "green": 0.84,
                                "blue": 0.31,
                            },
                            "textFormat": {
                                "bold": True,
                            },
                        }
                    },
                    "fields": "userEnteredFormat(backgroundColor,textFormat)",
                }
            },
        ]
    }

    sheets_service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body=body,
    ).execute()


# =========================================================
# ANALYSIS
# =========================================================
def explicar_coluna(df: pd.DataFrame, nome_coluna: str) -> str:
    if nome_coluna not in df.columns:
        return f"A coluna '{nome_coluna}' não foi encontrada."

    serie = df[nome_coluna]
    texto = []
    texto.append(f"Coluna analisada: {nome_coluna}")
    texto.append(f"Quantidade de registros: {len(serie)}")
    texto.append(f"Valores vazios: {int(serie.isna().sum())}")

    serie_num = pd.to_numeric(serie, errors="coerce")
    if serie_num.notna().sum() > 0:
        texto.append("Tipo: numérico")
        texto.append(f"Menor valor: {serie_num.min()}")
        texto.append(f"Maior valor: {serie_num.max()}")
        texto.append(f"Média: {serie_num.mean():.2f}")
        texto.append(f"Soma: {serie_num.sum():.2f}")
    else:
        texto.append("Tipo: texto/categórico")
        exemplos = list(serie.dropna().astype(str).unique()[:10])
        texto.append(f"Exemplos de valores: {exemplos}")

    return "\n".join(texto)


def buscar_texto_em_dataframe(df: pd.DataFrame, texto: str) -> pd.DataFrame:
    if df.empty:
        return df

    texto = str(texto).strip()
    if not texto:
        return df.iloc[0:0]

    mask = df.astype(str).apply(
        lambda col: col.str.contains(re.escape(texto), case=False, na=False, regex=True)
    )
    return df[mask.any(axis=1)]


def resumir_aba(df: pd.DataFrame, nome_aba: str) -> str:
    linhas, colunas = df.shape
    texto = [f"Resumo da aba '{nome_aba}':"]
    texto.append(f"- Linhas: {linhas}")
    texto.append(f"- Colunas: {colunas}")
    texto.append(f"- Nomes das colunas: {list(df.columns)}")

    if df.empty:
        return "\n".join(texto)

    numericas = []
    for col in df.columns:
        serie_num = pd.to_numeric(df[col], errors="coerce")
        if serie_num.notna().sum() > 0:
            numericas.append(col)

    categoricas = [c for c in df.columns if c not in numericas]

    texto.append(f"- Colunas numéricas: {numericas}")
    texto.append(f"- Colunas categóricas/textuais: {categoricas}")

    if numericas:
        texto.append("- Estatísticas básicas:")
        for col in numericas[:10]:
            serie_num = pd.to_numeric(df[col], errors="coerce")
            texto.append(
                f"  * {col}: min={serie_num.min()}, max={serie_num.max()}, média={serie_num.mean():.2f}"
            )

    return "\n".join(texto)


# =========================================================
# EXECUTION
# =========================================================
def resolver_aba(planilhas: dict[str, pd.DataFrame], aba_sugerida: str | None) -> str:
    abas = list(planilhas.keys())

    if not abas:
        raise ValueError("Nenhuma aba foi encontrada na planilha.")

    if len(abas) == 1:
        return abas[0]

    encontrada = encontrar_nome_proximo(abas, aba_sugerida)
    if encontrada:
        return encontrada

    return abas[0]


def resolver_coluna(
    df: pd.DataFrame,
    coluna_sugerida: str | None,
    fallback_keywords: list[str] | None = None,
) -> str | None:
    colunas = [str(c) for c in df.columns]

    encontrada = encontrar_nome_proximo(colunas, coluna_sugerida)
    if encontrada:
        return encontrada

    if fallback_keywords:
        for kw in fallback_keywords:
            encontrada = encontrar_nome_proximo(colunas, kw)
            if encontrada:
                return encontrada

    return None


def salvar_texto_output(nome_arquivo: str, conteudo: str) -> Path:
    caminho = OUTPUT_DIR / nome_arquivo
    caminho.write_text(conteudo, encoding="utf-8")
    return caminho


def executar_acao(
    acao: dict[str, Any],
    spreadsheet_id: str,
    planilhas: dict[str, pd.DataFrame],
    sheet_ids_origem: dict[str, int],
    sheets_service,
) -> None:
    tipo_acao = acao.get("acao")
    aba_sugerida = acao.get("aba")
    coluna_sugerida = acao.get("coluna")
    texto_sugerido = acao.get("texto")

    if tipo_acao == "listar_abas":
        print("Abas disponíveis:")
        for aba in planilhas.keys():
            print(f"- {aba}")
        return

    if tipo_acao == "listar_colunas":
        aba_real = resolver_aba(planilhas, aba_sugerida)
        print(f"Colunas da aba '{aba_real}':")
        for col in planilhas[aba_real].columns:
            print(f"- {col}")
        return

    if tipo_acao == "resumir_aba":
        aba_real = resolver_aba(planilhas, aba_sugerida)
        resumo = resumir_aba(planilhas[aba_real], aba_real)
        print(resumo)
        salvar_texto_output(
            f"resumo_{sanitizar_nome_arquivo(aba_real)}.txt",
            resumo,
        )
        return

    if tipo_acao == "explicar_coluna":
        aba_real = resolver_aba(planilhas, aba_sugerida)
        df = planilhas[aba_real]

        coluna_real = resolver_coluna(
            df,
            coluna_sugerida,
            fallback_keywords=[
                "lucro",
                "profit",
                "margem",
                "resultado",
                "faturamento",
                "receita",
                "status",
                "protocolo",
            ],
        )

        if coluna_real is None:
            print("Não foi possível identificar a coluna para explicar.")
            return

        explicacao = explicar_coluna(df, coluna_real)
        print(explicacao)
        salvar_texto_output(
            f"explicacao_{sanitizar_nome_arquivo(aba_real)}_{sanitizar_nome_arquivo(coluna_real)}.txt",
            explicacao,
        )
        return

    if tipo_acao == "buscar_texto":
        aba_real = resolver_aba(planilhas, aba_sugerida)
        df = planilhas[aba_real]

        if not texto_sugerido:
            print("Nenhum texto foi informado para busca.")
            return

        encontrados = buscar_texto_em_dataframe(df, texto_sugerido)
        print(
            f"Foram encontradas {len(encontrados)} linha(s) com o texto "
            f"'{texto_sugerido}' na aba '{aba_real}'."
        )

        if len(encontrados) > 0:
            print(encontrados.head(20).to_string(index=False))
            caminho_saida = OUTPUT_DIR / (
                f"busca_{sanitizar_nome_arquivo(aba_real)}_"
                f"{sanitizar_nome_arquivo(str(texto_sugerido))}.csv"
            )
            encontrados.to_csv(caminho_saida, index=False)
            print(f"\nResultado salvo em: {caminho_saida}")
        return

    if tipo_acao == "destacar_coluna":
        aba_real = resolver_aba(planilhas, aba_sugerida)
        df = planilhas[aba_real]

        coluna_real = resolver_coluna(
            df,
            coluna_sugerida,
            fallback_keywords=[
                "lucro",
                "profit",
                "margem",
                "resultado",
                "faturamento",
                "receita",
                "status",
                "protocolo",
            ],
        )

        if coluna_real is None:
            print("Não foi possível identificar a coluna a ser destacada.")
            return

        novo_titulo = sanitizar_nome_arquivo(f"destacado_{aba_real}_{coluna_real}")
        novo_spreadsheet_id, _ = criar_planilha_google(sheets_service, novo_titulo)

        metadata_nova = obter_metadata_planilha(sheets_service, novo_spreadsheet_id)
        aba_padrao = metadata_nova["sheets"][0]["properties"]["title"]
        id_aba_padrao = metadata_nova["sheets"][0]["properties"]["sheetId"]

        if aba_padrao != aba_real:
            renomear_aba(sheets_service, novo_spreadsheet_id, id_aba_padrao, aba_real)

        escrever_dataframe_em_aba(
            sheets_service=sheets_service,
            spreadsheet_id=novo_spreadsheet_id,
            aba=aba_real,
            df=df,
        )

        idx_coluna = list(df.columns).index(coluna_real)
        total_linhas = len(df) + 1

        metadata_atualizada = obter_metadata_planilha(sheets_service, novo_spreadsheet_id)
        sheet_id_real = None

        for s in metadata_atualizada["sheets"]:
            if s["properties"]["title"] == aba_real:
                sheet_id_real = s["properties"]["sheetId"]
                break

        if sheet_id_real is None:
            raise ValueError(
                f"Não foi possível localizar a aba '{aba_real}' na nova planilha."
            )

        destacar_coluna_google(
            sheets_service=sheets_service,
            spreadsheet_id=novo_spreadsheet_id,
            sheet_id=sheet_id_real,
            idx_coluna_zero_based=idx_coluna,
            total_linhas=total_linhas,
        )

        link = f"https://docs.google.com/spreadsheets/d/{novo_spreadsheet_id}"
        texto_saida = (
            f"Coluna '{coluna_real}' destacada com sucesso na aba '{aba_real}'.\n"
            f"Planilha criada: {link}"
        )
        print(texto_saida)
        salvar_texto_output(
            f"planilha_destacada_{sanitizar_nome_arquivo(aba_real)}_{sanitizar_nome_arquivo(coluna_real)}.txt",
            texto_saida,
        )
        return

    print("Ação não reconhecida ou não suportada.")
    print(json.dumps(acao, ensure_ascii=False, indent=2))


# =========================================================
# MAIN
# =========================================================
def main():
    garantir_diretorios()

    print("Assistente Google Sheets + Gemini\n")
    spreadsheet_id = input("Digite o spreadsheet_id da planilha Google: ").strip()

    if not spreadsheet_id:
        raise ValueError("Você precisa informar um spreadsheet_id válido.")

    sheets_service = get_sheets_service()
    planilhas, sheet_ids = listar_abas_e_colunas(sheets_service, spreadsheet_id)

    print("\nPlanilha carregada com sucesso.")
    print("Abas disponíveis:")
    for nome_aba, df in planilhas.items():
        print(f"- {nome_aba} ({df.shape[0]} linhas, {df.shape[1]} colunas)")

    colunas_por_aba = {
        aba: [str(c) for c in df.columns]
        for aba, df in planilhas.items()
    }

    agente = SpreadsheetGeminiAgent()

    while True:
        comando = input("\nDigite seu comando (ou 'sair'): ").strip()
        if comando.lower() in {"sair", "exit", "quit"}:
            print("Encerrando assistente.")
            break

        if not comando:
            print("Digite um comando válido.")
            continue

        try:
            acao = agente.interpretar(
                comando_usuario=comando,
                nomes_abas=list(planilhas.keys()),
                colunas_por_aba=colunas_por_aba,
            )

            if DEBUG:
                print("\n[DEBUG] JSON interpretado:")
                print(json.dumps(acao, ensure_ascii=False, indent=2))

            executar_acao(
                acao=acao,
                spreadsheet_id=spreadsheet_id,
                planilhas=planilhas,
                sheet_ids_origem=sheet_ids,
                sheets_service=sheets_service,
            )

        except Exception as e:
            print(f"Erro ao processar comando: {e}")


if __name__ == "__main__":
    main()