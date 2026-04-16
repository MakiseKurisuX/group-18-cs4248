"""Single-headline, interactive, and batch prediction interface for BERT sarcasm detection."""

import argparse
import json

from core.cli_args import add_domain_context_args
from core.config import DEFAULT_BERT_MODE, MAX_LENGTH
from core.inference import (
    load_sequence_classifier,
    predict_single,
    predict_single_with_components,
    resolve_bert_model_reference,
)
from evaluate import evaluate_split


def format_prediction(text: str, predicted_label: int, positive_probability: float) -> str:
    sarcastic_probability = positive_probability
    not_sarcastic_probability = 1.0 - positive_probability
    predicted_name = "sarcastic" if predicted_label == 1 else "not sarcastic"
    return "\n".join(
        [
            f"Input: {text}",
            f"Prediction: {predicted_name}",
            f"Sarcastic probability: {sarcastic_probability:.4f}",
            f"Not sarcastic probability: {not_sarcastic_probability:.4f}",
        ]
    )


def interactive_loop(
    model_path=None,
    mode: str | None = None,
    max_length: int = MAX_LENGTH,
) -> None:
    model_reference = resolve_bert_model_reference(mode=mode, model_path=model_path)
    tokenizer, model, device = load_sequence_classifier(model_reference=model_reference)

    print("Enter a headline to classify. Type 'exit' or press Enter on an empty line to quit.")
    while True:
        text = input("headline> ").strip()
        if not text or text.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        predicted_label, positive_probability = predict_single_with_components(
            text,
            tokenizer,
            model,
            device,
            max_length=max_length,
        )
        print(format_prediction(text, predicted_label, positive_probability))
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict sarcasm with a BERT model.")
    parser.add_argument("--text", default=None, help="Optional text to classify. If omitted, interactive mode is used.")
    parser.add_argument("--input-path", default=None, help="Optional batch input file (CSV/JSON/JSONL) for CSV export.")
    parser.add_argument("--model-path", default=None, help="Optional override for the saved model path.")
    parser.add_argument("--mode", default=DEFAULT_BERT_MODE, help="BERT experiment variant to use for inference.")
    parser.add_argument("--dataset-name", default=None, help="Optional dataset label to stamp into batch exports.")
    parser.add_argument("--output-path", default=None, help="Optional output CSV path for batch exports.")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH, help="Tokenizer max length for BERT inference.")
    add_domain_context_args(parser)
    parser.add_argument("--skip-predictions", action="store_true", help="Batch mode only: compute metrics without exporting the per-example CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_path if args.model_path else None

    if args.input_path:
        metrics = evaluate_split(
            split_or_path=args.input_path,
            model_path=model_path,
            mode=args.mode,
            dataset_name=args.dataset_name,
            output_path=args.output_path,
            max_length=args.max_length,
            use_domain_context=args.use_domain_context,
            save_predictions_output=not args.skip_predictions,
        )
        print(json.dumps(metrics, indent=2))
        return

    if args.text:
        predicted_label, positive_probability = predict_single(
            args.text,
            model_reference=resolve_bert_model_reference(mode=args.mode, model_path=model_path),
            max_length=args.max_length,
        )
        print(format_prediction(args.text, predicted_label, positive_probability))
        return

    interactive_loop(model_path=model_path, mode=args.mode, max_length=args.max_length)


if __name__ == "__main__":
    main()
