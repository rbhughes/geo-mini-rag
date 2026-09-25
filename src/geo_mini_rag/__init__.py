"""geo-mini-rag: a small RAG pipeline that reads E&P file formats."""


def main() -> None:
    import sys

    from geo_mini_rag.cli import app, console
    from geo_mini_rag.errors import UserError

    try:
        app()
    except UserError as exc:
        console.print(f"[red]Error:[/] {exc}")
        sys.exit(1)
