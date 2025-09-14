#!/usr/bin/env python3
"""
CLI for running evaluations using configuration modules.

Usage:
    python scripts/run_evaluation.py --config_module=cfgs/my_config.py --cfg_var_name=eval_cfg --model_path=model.json --output_path=results.json
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
try:
    from loguru import logger  # type: ignore
except Exception:  # pragma: no cover - fallback when loguru missing
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
from sl.evaluation.data_models import Evaluation
from sl.evaluation import services as evaluation_services
from sl.llm.data_models import Model
from sl.utils import module_utils, file_utils


async def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation using a configuration module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_evaluation.py --config_module=cfgs/preference_numbers/cfgs.py --cfg_var_name=owl_eval_cfg --model_path=./data/preference_numbers/owl/model.json --output_path=./data/preference_numbers/owl/evaluation_results.json
        """,
    )

    parser.add_argument(
        "--config_module",
        required=True,
        help="Path to Python module containing evaluation configuration",
    )

    parser.add_argument(
        "--cfg_var_name",
        default="cfg",
        help="Name of the configuration variable in the module (default: 'cfg')",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model_path",
        help="Path to the model JSON file (output from fine-tuning)",
    )
    group.add_argument(
        "--use_reference_model",
        action="store_true",
        help="Use the reference_model defined in the config module",
    )

    parser.add_argument(
        "--output_path",
        required=True,
        help="Path where evaluation results will be saved",
    )
    parser.add_argument(
        "--use_system_prompt",
        action="store_true",
        help="If set, read 'system_prompt' from the config module and use it during evaluation",
    )

    args = parser.parse_args()

    # Validate config file exists
    config_path = Path(args.config_module)
    if not config_path.exists():
        logger.error(f"Config module {args.config_module} does not exist")
        sys.exit(1)

    # Validate model path if provided
    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.exists():
            logger.error(f"Model file {args.model_path} does not exist")
            sys.exit(1)

    try:
        # Load configuration from module
        logger.info(
            f"Loading configuration from {args.config_module} (variable: {args.cfg_var_name})..."
        )
        eval_cfg = module_utils.get_obj(args.config_module, args.cfg_var_name)
        # Support either a single Evaluation or a list of Evaluations
        if isinstance(eval_cfg, Evaluation):
            eval_cfgs = [eval_cfg]
        elif isinstance(eval_cfg, list):
            assert all(isinstance(e, Evaluation) for e in eval_cfg), (
                "cfg_var must be an Evaluation or a list[Evaluation]"
            )
            eval_cfgs = eval_cfg
        else:
            raise AssertionError(
                "cfg_var must be an Evaluation or a list[Evaluation]"
            )

        # Load model
        if args.model_path:
            logger.info(f"Loading model from {args.model_path}...")
            with open(args.model_path, "r") as f:
                model_data = json.load(f)
            model = Model.model_validate(model_data)
            logger.info(f"Loaded model: {model.id} (type: {model.type})")
        else:
            logger.info("Loading reference_model from config module...")
            reference_model = module_utils.get_obj(args.config_module, "reference_model")
            assert isinstance(reference_model, Model), (
                "reference_model in config must be of type Model"
            )
            model = reference_model
            logger.info(f"Loaded reference model: {model.id} (type: {model.type})")

        # Run evaluation(s)
        logger.info(
            f"Starting evaluation for {len(eval_cfgs)} configuration(s)..."
        )
        # Optionally load system_prompt from the config module
        system_prompt = None
        if args.use_system_prompt:
            try:
                system_prompt = module_utils.get_obj(
                    args.config_module, "system_prompt"
                )
            except Exception:
                raise Exception("system_prompt not found in config module")

        results_lists = await asyncio.gather(
            *[
                evaluation_services.run_evaluation(
                    model, e, system_prompt=system_prompt
                )
                for e in eval_cfgs
            ]
        )
        # Flatten all question groups across evaluations
        evaluation_results = [row for sub in results_lists for row in sub]
        logger.info(
            f"Completed evaluation with {len(evaluation_results)} total question groups"
        )

        # Save results
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        serializable_results = [row.model_dump() for row in evaluation_results]
        file_utils.save_jsonl(serializable_results, str(output_path), "w")
        logger.info(f"Saved evaluation results to {output_path}")

        logger.info("Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
