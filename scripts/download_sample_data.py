"""Download the community-maintained Qlib example dataset used by this project."""

from pathlib import Path

from qlib.tests.data import GetData


def main() -> None:
    target = Path("data/qlib/cn_data")
    target.mkdir(parents=True, exist_ok=True)
    GetData(delete_zip_file=True).qlib_data(
        name="qlib_data_simple",
        target_dir=str(target),
        interval="1d",
        region="cn",
        exists_skip=True,
    )


if __name__ == "__main__":
    main()
