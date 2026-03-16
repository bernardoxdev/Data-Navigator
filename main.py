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

# Directories
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "inputs"
OUTPUT_DIR = BASE_DIR / "outputs"
CREDENTIALS_FILE = BASE_DIR / "credentials" / "service_account.json"

# Scopes for Sheets and Drive
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

SYSTEM_PROMPT = """
You are a spreadsheet assistant.
Your job is to interpret user requests and return ONLY valid JSON.
Do not write explanations outside the JSON.
Never use markdown.
Never use code blocks.

Allowed actions:
- highlight_column
- explain_column
- search_text
- summarize_sheet
- list_sheets
- list_columns

Expected format:
{
  "action": "action_name",
  "sheet": "sheet_name_or_null",
  "column": "column_name_or_null",
  "text": "text_or_null"
}

Rules:
- If the user asks to highlight a column, use "highlight_column".
- If the user asks for an explanation about a column, use "explain_column".
- If the user asks to search for a protocol, code, text, or term, use "search_text".
- If the user asks for a summary of the sheet, use "summarize_sheet".
- If the user asks for available sheets, use "list_sheets".
- If the user asks for available columns, use "list_columns".
- If the sheet is unknown, return sheet as null.
- If the column is unknown, return column as null.
- If there is no text, return text as null.
- Always respond with valid JSON.
""".strip()


# =========================================================
# AUTH / CLIENTS
# =========================================================
def ensure_directories() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CREDENTIALS_FILE.parent.mkdir(parents=True, exist_ok=True)


def get_google_credentials() -> Credentials:
    if not CREDENTIALS_FILE.exists():
        raise FileNotFoundError(
            f"Credentials file not found at: {CREDENTIALS_FILE}"
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


def configure_gemini() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set the GEMINI_API_KEY environment variable.\n"
            "Example:\n"
            'export GEMINI_API_KEY="YOUR_KEY"'
        )
    return api_key


# =========================================================
# HELPERS
# =========================================================
def find_closest_name(
    options: list[str],
    term: str | None,
    cutoff: float = 0.5,
) -> str | None:
    if term is None:
        return None

    options_str = [str(x) for x in options]
    term_lower = term.lower().strip()

    for option in options_str:
        if option.lower().strip() == term_lower:
            return option

    for option in options_str:
        if term_lower in option.lower():
            return option

    matches = get_close_matches(
        term_lower,
        [x.lower() for x in options_str],
        n=1,
        cutoff=cutoff,
    )
    if not matches:
        return None

    for option in options_str:
        if option.lower() == matches[0]:
            return option

    return None


def extract_raw_json(text: str) -> dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        snippet = match.group(0)
        try:
            return json.loads(snippet)
        except Exception:
            pass

    raise ValueError(f"Could not extract valid JSON:\n{text}")


def sanitize_filename(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[^\w\-_.]+", "_", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text)
    return text[:120].strip("_") or "output"


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
        api_key = configure_gemini()
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

        if DEBUG:
            print(f"[DEBUG] Gemini loaded: {model_name}")

    def interpret(
        self,
        user_command: str,
        sheet_names: list[str],
        columns_by_sheet: dict[str, list[str]],
    ) -> dict[str, Any]:
        context = {
            "available_sheets": sheet_names,
            "columns_by_sheet": columns_by_sheet,
            "user_request": user_command,
        }

        prompt = (
            SYSTEM_PROMPT
            + "\n\nContext:\n"
            + json.dumps(context, ensure_ascii=False, indent=2)
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

        text = (response.text or "").strip()

        if DEBUG:
            print("\n[DEBUG] Raw Gemini response:")
            print(text)

        if not text:
            raise ValueError("Gemini returned an empty response.")

        action = extract_raw_json(text)
        action.setdefault("action", None)
        action.setdefault("sheet", None)
        action.setdefault("column", None)
        action.setdefault("text", None)
        return action


# =========================================================
# GOOGLE SHEETS
# =========================================================
def get_spreadsheet_metadata(sheets_service, spreadsheet_id: str) -> dict[str, Any]:
    return sheets_service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()


def normalize_headers(headers: list[str]) -> list[str]:
    used: dict[str, int] = {}
    output: list[str] = []

    for i, h in enumerate(headers):
        name = str(h).strip() if h is not None else ""
        if not name:
            name = f"column_{i + 1}"

        if name in used:
            used[name] += 1
            final_name = f"{name}_{used[name]}"
        else:
            used[name] = 1
            final_name = name

        output.append(final_name)

    return output


def list_sheets_and_columns(
    sheets_service,
    spreadsheet_id: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    metadata = get_spreadsheet_metadata(sheets_service, spreadsheet_id)

    spreadsheets: dict[str, pd.DataFrame] = {}
    sheet_ids: dict[str, int] = {}

    for sheet in metadata["sheets"]:
        props = sheet["properties"]
        sheet_name = props["title"]
        sheet_ids[sheet_name] = props["sheetId"]

        result = sheets_service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=sheet_name,
        ).execute()

        values = result.get("values", [])

        if not values:
            spreadsheets[sheet_name] = pd.DataFrame()
            continue

        header = normalize_headers(values[0])
        rows = values[1:] if len(values) > 1 else []

        max_cols = len(header)
        normalized_rows = []
        for row in rows:
            row = list(row) + [""] * (max_cols - len(row))
            normalized_rows.append(row[:max_cols])

        spreadsheets[sheet_name] = pd.DataFrame(normalized_rows, columns=header)

    return spreadsheets, sheet_ids


def create_google_spreadsheet(sheets_service, title: str) -> tuple[str, dict[str, int]]:
    body = {
        "properties": {
            "title": title,
        }
    }

    spreadsheet = sheets_service.spreadsheets().create(body=body).execute()
    spreadsheet_id = spreadsheet["spreadsheetId"]

    metadata = get_spreadsheet_metadata(sheets_service, spreadsheet_id)
    sheet_ids = {
        s["properties"]["title"]: s["properties"]["sheetId"]
        for s in metadata["sheets"]
    }

    return spreadsheet_id, sheet_ids


def write_dataframe_to_sheet(
    sheets_service,
    spreadsheet_id: str,
    sheet: str,
    df: pd.DataFrame,
) -> None:
    values = [list(df.columns)] + df.fillna("").astype(str).values.tolist()

    sheets_service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=f"{sheet}!A1",
        valueInputOption="RAW",
        body={"values": values},
    ).execute()


def rename_sheet(
    sheets_service,
    spreadsheet_id: str,
    sheet_id: int,
    new_name: str,
) -> None:
    body = {
        "requests": [
            {
                "updateSheetProperties": {
                    "properties": {
                        "sheetId": sheet_id,
                        "title": new_name,
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


def add_sheet(sheets_service, spreadsheet_id: str, sheet_name: str) -> int:
    body = {
        "requests": [
            {
                "addSheet": {
                    "properties": {
                        "title": sheet_name,
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


def highlight_column_google(
    sheets_service,
    spreadsheet_id: str,
    sheet_id: int,
    column_idx_zero_based: int,
    total_rows: int,
) -> None:
    body = {
        "requests": [
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 0,
                        "endRowIndex": total_rows,
                        "startColumnIndex": column_idx_zero_based,
                        "endColumnIndex": column_idx_zero_based + 1,
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
                        "startColumnIndex": column_idx_zero_based,
                        "endColumnIndex": column_idx_zero_based + 1,
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
def explain_column(df: pd.DataFrame, column_name: str) -> str:
    if column_name not in df.columns:
        return f"The column '{column_name}' was not found."

    series = df[column_name]
    text = []
    text.append(f"Analyzed column: {column_name}")
    text.append(f"Number of records: {len(series)}")
    text.append(f"Empty values: {int(series.isna().sum())}")

    series_num = pd.to_numeric(series, errors="coerce")
    if series_num.notna().sum() > 0:
        text.append("Type: numeric")
        text.append(f"Minimum value: {series_num.min()}")
        text.append(f"Maximum value: {series_num.max()}")
        text.append(f"Average: {series_num.mean():.2f}")
        text.append(f"Sum: {series_num.sum():.2f}")
    else:
        text.append("Type: text/categorical")
        examples = list(series.dropna().astype(str).unique()[:10])
        text.append(f"Example values: {examples}")

    return "\n".join(text)


def search_text_in_dataframe(df: pd.DataFrame, text: str) -> pd.DataFrame:
    if df.empty:
        return df

    text = str(text).strip()
    if not text:
        return df.iloc[0:0]

    mask = df.astype(str).apply(
        lambda col: col.str.contains(re.escape(text), case=False, na=False, regex=True)
    )
    return df[mask.any(axis=1)]


def summarize_sheet(df: pd.DataFrame, sheet_name: str) -> str:
    rows, columns = df.shape
    text = [f"Summary of sheet '{sheet_name}':"]
    text.append(f"- Rows: {rows}")
    text.append(f"- Columns: {columns}")
    text.append(f"- Column names: {list(df.columns)}")

    if df.empty:
        return "\n".join(text)

    numeric_columns = []
    for col in df.columns:
        series_num = pd.to_numeric(df[col], errors="coerce")
        if series_num.notna().sum() > 0:
            numeric_columns.append(col)

    categorical_columns = [c for c in df.columns if c not in numeric_columns]

    text.append(f"- Numeric columns: {numeric_columns}")
    text.append(f"- Categorical/text columns: {categorical_columns}")

    if numeric_columns:
        text.append("- Basic statistics:")
        for col in numeric_columns[:10]:
            series_num = pd.to_numeric(df[col], errors="coerce")
            text.append(
                f"  * {col}: min={series_num.min()}, max={series_num.max()}, average={series_num.mean():.2f}"
            )

    return "\n".join(text)


# =========================================================
# EXECUTION
# =========================================================
def resolve_sheet(spreadsheets: dict[str, pd.DataFrame], suggested_sheet: str | None) -> str:
    sheets = list(spreadsheets.keys())

    if not sheets:
        raise ValueError("No sheets were found in the spreadsheet.")

    if len(sheets) == 1:
        return sheets[0]

    found = find_closest_name(sheets, suggested_sheet)
    if found:
        return found

    return sheets[0]


def resolve_column(
    df: pd.DataFrame,
    suggested_column: str | None,
    fallback_keywords: list[str] | None = None,
) -> str | None:
    columns = [str(c) for c in df.columns]

    found = find_closest_name(columns, suggested_column)
    if found:
        return found

    if fallback_keywords:
        for kw in fallback_keywords:
            found = find_closest_name(columns, kw)
            if found:
                return found

    return None


def save_text_output(filename: str, content: str) -> Path:
    path = OUTPUT_DIR / filename
    path.write_text(content, encoding="utf-8")
    return path


def execute_action(
    action: dict[str, Any],
    spreadsheet_id: str,
    spreadsheets: dict[str, pd.DataFrame],
    source_sheet_ids: dict[str, int],
    sheets_service,
) -> None:
    action_type = action.get("action")
    suggested_sheet = action.get("sheet")
    suggested_column = action.get("column")
    suggested_text = action.get("text")

    if action_type == "list_sheets":
        print("Available sheets:")
        for sheet in spreadsheets.keys():
            print(f"- {sheet}")
        return

    if action_type == "list_columns":
        real_sheet = resolve_sheet(spreadsheets, suggested_sheet)
        print(f"Columns in sheet '{real_sheet}':")
        for col in spreadsheets[real_sheet].columns:
            print(f"- {col}")
        return

    if action_type == "summarize_sheet":
        real_sheet = resolve_sheet(spreadsheets, suggested_sheet)
        summary = summarize_sheet(spreadsheets[real_sheet], real_sheet)
        print(summary)
        save_text_output(
            f"summary_{sanitize_filename(real_sheet)}.txt",
            summary,
        )
        return

    if action_type == "explain_column":
        real_sheet = resolve_sheet(spreadsheets, suggested_sheet)
        df = spreadsheets[real_sheet]

        real_column = resolve_column(
            df,
            suggested_column,
            fallback_keywords=[
                "profit",
                "margin",
                "result",
                "revenue",
                "income",
                "status",
                "protocol",
            ],
        )

        if real_column is None:
            print("Could not identify the column to explain.")
            return

        explanation = explain_column(df, real_column)
        print(explanation)
        save_text_output(
            f"explanation_{sanitize_filename(real_sheet)}_{sanitize_filename(real_column)}.txt",
            explanation,
        )
        return

    if action_type == "search_text":
        real_sheet = resolve_sheet(spreadsheets, suggested_sheet)
        df = spreadsheets[real_sheet]

        if not suggested_text:
            print("No text was provided for searching.")
            return

        found_rows = search_text_in_dataframe(df, suggested_text)
        print(
            f"{len(found_rows)} row(s) were found with the text "
            f"'{suggested_text}' in sheet '{real_sheet}'."
        )

        if len(found_rows) > 0:
            print(found_rows.head(20).to_string(index=False))
            output_path = OUTPUT_DIR / (
                f"search_{sanitize_filename(real_sheet)}_"
                f"{sanitize_filename(str(suggested_text))}.csv"
            )
            found_rows.to_csv(output_path, index=False)
            print(f"\nResult saved to: {output_path}")
        return

    if action_type == "highlight_column":
        real_sheet = resolve_sheet(spreadsheets, suggested_sheet)
        df = spreadsheets[real_sheet]

        real_column = resolve_column(
            df,
            suggested_column,
            fallback_keywords=[
                "profit",
                "margin",
                "result",
                "revenue",
                "income",
                "status",
                "protocol",
            ],
        )

        if real_column is None:
            print("Could not identify the column to highlight.")
            return

        new_title = sanitize_filename(f"highlighted_{real_sheet}_{real_column}")
        new_spreadsheet_id, _ = create_google_spreadsheet(sheets_service, new_title)

        new_metadata = get_spreadsheet_metadata(sheets_service, new_spreadsheet_id)
        default_sheet = new_metadata["sheets"][0]["properties"]["title"]
        default_sheet_id = new_metadata["sheets"][0]["properties"]["sheetId"]

        if default_sheet != real_sheet:
            rename_sheet(sheets_service, new_spreadsheet_id, default_sheet_id, real_sheet)

        write_dataframe_to_sheet(
            sheets_service=sheets_service,
            spreadsheet_id=new_spreadsheet_id,
            sheet=real_sheet,
            df=df,
        )

        column_idx = list(df.columns).index(real_column)
        total_rows = len(df) + 1

        updated_metadata = get_spreadsheet_metadata(sheets_service, new_spreadsheet_id)
        real_sheet_id = None

        for s in updated_metadata["sheets"]:
            if s["properties"]["title"] == real_sheet:
                real_sheet_id = s["properties"]["sheetId"]
                break

        if real_sheet_id is None:
            raise ValueError(
                f"Could not locate the sheet '{real_sheet}' in the new spreadsheet."
            )

        highlight_column_google(
            sheets_service=sheets_service,
            spreadsheet_id=new_spreadsheet_id,
            sheet_id=real_sheet_id,
            column_idx_zero_based=column_idx,
            total_rows=total_rows,
        )

        link = f"https://docs.google.com/spreadsheets/d/{new_spreadsheet_id}"
        output_text = (
            f"Column '{real_column}' highlighted successfully in sheet '{real_sheet}'.\n"
            f"Spreadsheet created: {link}"
        )
        print(output_text)
        save_text_output(
            f"highlighted_spreadsheet_{sanitize_filename(real_sheet)}_{sanitize_filename(real_column)}.txt",
            output_text,
        )
        return

    print("Unrecognized or unsupported action.")
    print(json.dumps(action, ensure_ascii=False, indent=2))


# =========================================================
# MAIN
# =========================================================
def main():
    ensure_directories()

    print("Google Sheets + Gemini Assistant\n")
    spreadsheet_id = input("Enter the Google spreadsheet_id: ").strip()

    if not spreadsheet_id:
        raise ValueError("You must provide a valid spreadsheet_id.")

    sheets_service = get_sheets_service()
    spreadsheets, sheet_ids = list_sheets_and_columns(sheets_service, spreadsheet_id)

    print("\nSpreadsheet loaded successfully.")
    print("Available sheets:")
    for sheet_name, df in spreadsheets.items():
        print(f"- {sheet_name} ({df.shape[0]} rows, {df.shape[1]} columns)")

    columns_by_sheet = {
        sheet: [str(c) for c in df.columns]
        for sheet, df in spreadsheets.items()
    }

    agent = SpreadsheetGeminiAgent()

    while True:
        command = input("\nEnter your command (or 'exit'): ").strip()
        if command.lower() in {"exit", "quit"}:
            print("Closing assistant.")
            break

        if not command:
            print("Enter a valid command.")
            continue

        try:
            action = agent.interpret(
                user_command=command,
                sheet_names=list(spreadsheets.keys()),
                columns_by_sheet=columns_by_sheet,
            )

            if DEBUG:
                print("\n[DEBUG] Parsed JSON:")
                print(json.dumps(action, ensure_ascii=False, indent=2))

            execute_action(
                action=action,
                spreadsheet_id=spreadsheet_id,
                spreadsheets=spreadsheets,
                source_sheet_ids=sheet_ids,
                sheets_service=sheets_service,
            )

        except Exception as e:
            print(f"Error processing command: {e}")


if __name__ == "__main__":
    main()