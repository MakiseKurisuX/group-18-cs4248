"""Shared argparse helpers for the baseline BERT command-line entrypoints."""

from .config import USE_DOMAIN_CONTEXT


def add_domain_context_args(parser) -> None:
    parser.add_argument(
        "--use-domain-context",
        action="store_true",
        default=USE_DOMAIN_CONTEXT,
        help="Prepend article source domain to headline as a second BERT segment.",
    )
    parser.add_argument(
        "--no-domain-context",
        dest="use_domain_context",
        action="store_false",
        help="Disable domain context injection.",
    )
