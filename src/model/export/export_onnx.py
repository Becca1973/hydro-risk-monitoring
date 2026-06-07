from pathlib import Path

import tensorflow as tf
import tf2onnx


def export_keras_to_onnx(
    model_path: str,
    output_path: str,
    input_shape: tuple[int, ...],
):
    model = tf.keras.models.load_model(model_path)

    @tf.function(
        input_signature=[
            tf.TensorSpec(input_shape, tf.float32, name="input")
        ]
    )
    def serving_fn(input):
        return {"output": model(input)}

    tf2onnx.convert.from_function(
        serving_fn,
        input_signature=[
            tf.TensorSpec(input_shape, tf.float32, name="input")
        ],
        output_path=output_path,
    )

    print(f"Exported ONNX model to {output_path}")


def main():
    onnx_dir = Path("models/onnx")
    onnx_dir.mkdir(parents=True, exist_ok=True)

    export_keras_to_onnx(
        model_path="models/hidro_global/model_hidro_global.keras",
        output_path="models/onnx/hidro_global_water_level_forecaster.onnx",
        input_shape=(None, 24, 8),
    )

    export_keras_to_onnx(
        model_path="models/risk/risk_classifier.keras",
        output_path="models/onnx/hidro_risk_classifier.onnx",
        input_shape=(None, 13),
    )


if __name__ == "__main__":
    main()
