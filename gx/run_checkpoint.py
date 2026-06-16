import sys
from pathlib import Path

import great_expectations as gx

context = gx.get_context()

VALIDATION_CONFIGS = [
    {
        "name": "hidro",
        "datasource": "hidro_fluent",
        "suite": "hidro_suite",
        "directory": Path("data/preprocessed/hidro"),
    },
    {
        "name": "weather",
        "datasource": "weather_fluent",
        "suite": "weather_suite",
        "directory": Path("data/preprocessed/weather"),
    },
]

all_success = True

for config in VALIDATION_CONFIGS:
    print(f"\nValidating {config['name']} data...")

    datasource = context.get_datasource(config["datasource"])

    for csv_path in sorted(config["directory"].glob("*.csv")):

        asset_name = f"{config['name']}_{csv_path.stem}".replace("-", "_")

        print(f"\n=== Processing: {csv_path.name} ===")
        print(f"Asset name: {asset_name}")

        try:
            datasource.add_csv_asset(
                name=asset_name,
                batching_regex=rf"{csv_path.name}"
            )
        except Exception:
            pass

        asset = context.get_datasource(
            config["datasource"]
        ).get_asset(asset_name)

        batch_request = asset.build_batch_request()

        checkpoint_name = f"{asset_name}_checkpoint"

        checkpoint = context.add_or_update_checkpoint(
            name=checkpoint_name,
            validations=[
                {
                    "batch_request": batch_request,
                    "expectation_suite_name": config["suite"],
                }
            ],
        )

        try:
            result = checkpoint.run(
                run_id=f"{asset_name}_run"
            )

            if result["success"]:
                print(f"{csv_path.name} passed")
            else:
                print(f"{csv_path.name} failed")
                all_success = False

        except Exception as e:
            print("\n================ ERROR ================")
            print(f"File: {csv_path}")
            print(f"File name: {csv_path.name}")
            print(f"Asset: {asset_name}")
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception message: {e}")
            print("=======================================\n")
            raise

context.build_data_docs()

if all_success:
    print("\nValidation passed for all files!")
    sys.exit(0)
else:
    print(
        "\nValidation completed. Some files failed validation. Check Data Docs report."
    )
    sys.exit(0)
