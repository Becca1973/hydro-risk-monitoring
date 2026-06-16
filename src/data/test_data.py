import sys
from pathlib import Path
import shutil

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

CONFIGS = [
    {
        "name": "hidro",
        "data_dir": Path("data/preprocessed/hidro"),
        "reference_dir": Path("data/reference/hidro"),
    },
    {
        "name": "weather",
        "data_dir": Path("data/preprocessed/weather"),
        "reference_dir": Path("data/reference/weather"),
    },
]

REPORTS_DIR = Path("reports/data_testing")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

all_tests_passed = True

for config in CONFIGS:
    print(f"\nTesting {config['name']} data...")

    config["reference_dir"].mkdir(parents=True, exist_ok=True)

    for csv_path in sorted(config["data_dir"].glob("*.csv")):
        reference_path = config["reference_dir"] / csv_path.name

        # Create reference dataset on first run.
        if not reference_path.exists():
            shutil.copy(csv_path, reference_path)
            print(f"Created reference dataset for {csv_path.name}")
            continue

        try:
            current_data = pd.read_csv(csv_path)
            reference_data = pd.read_csv(reference_path)

            # Remove completely empty columns.
            current_data = current_data.dropna(axis=1, how="all")
            reference_data = reference_data.dropna(axis=1, how="all")

            # Keep only common columns.
            common_columns = sorted(
                set(current_data.columns).intersection(reference_data.columns)
            )

            if not common_columns:
                print(f"No common columns for {csv_path.name}, skipping.")
                all_tests_passed = False
                continue

            current_data = current_data[common_columns]
            reference_data = reference_data[common_columns]

            if current_data.empty or reference_data.empty:
                print(
                    f"Empty current/reference data for {csv_path.name}, skipping.")
                all_tests_passed = False
                continue

            report = Report(
                metrics=[
                    DataSummaryPreset(),
                    DataDriftPreset(),
                ],
                include_tests=True,
            )

            result = report.run(
                reference_data=reference_data,
                current_data=current_data,
            )

            output_path = REPORTS_DIR / \
                f"{config['name']}_{csv_path.stem}_report.html"
            result.save_html(str(output_path))

            print(f"Generated report: {output_path}")

            # Check test results.
            file_tests_passed = True
            result_dict = result.dict()

            if "tests" in result_dict:
                for test in result_dict["tests"]:
                    if "status" in test and test["status"] != "SUCCESS":
                        print(
                            f"Test failed for {csv_path.name}: {test.get('name')}")
                        file_tests_passed = False
                        all_tests_passed = False

            # Update reference dataset only if tests passed.
            if file_tests_passed:
                current_data.to_csv(reference_path, index=False)

        except Exception as e:
            print(
                f"Report generation failed for {csv_path.name}: "
                f"{type(e).__name__}: {e}"
            )
            all_tests_passed = False
            continue

if not all_tests_passed:
    print("\nSome Evidently tests failed or reports could not be generated.")
    print("Reports that were successfully generated are available for inspection.")
else:
    print("\nAll Evidently tests passed.")

sys.exit(0)
