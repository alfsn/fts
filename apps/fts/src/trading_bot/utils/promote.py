import argparse

from trading_bot.core.database import SessionLocal
from trading_bot.core.repository import ModelRepository


def main():
    parser = argparse.ArgumentParser(
        description="Promote a candidate model to production."
    )
    parser.add_argument(
        "model_id", type=str, help="The unique model ID of the model to promote."
    )
    parser.add_argument(
        "-y", "--yes", action="store_true", help="Bypass promotion confirmation prompt."
    )
    args = parser.parse_args()

    db = SessionLocal()
    model_repo = ModelRepository(db)

    model = model_repo.get_model(args.model_id)
    if not model:
        print(f"Error: Model with model_id '{args.model_id}' not found.")
        db.close()
        return

    print(f"Found candidate model:")
    print(f"  Model ID:    {model.model_id}")
    print(f"  Type:        {model.model_type}")
    print(
        f"  Market:      {model.market_id} ({model.interval}, horizon {model.horizon})"
    )
    print(f"  Metrics:     {model.metrics}")
    print(f"  ONNX Path:   {model.onnx_path}")

    promote = args.yes
    if not promote:
        confirm = input("Promote this model to production? [y/N]: ").strip().lower()
        promote = confirm == "y"

    if promote:
        try:
            model_repo.promote_to_production(args.model_id)
            db.commit()  # Decoupled commit managed at caller level
            print(f"Successfully promoted model '{args.model_id}' to production!")
            print(
                "Previous production models matching this logical signature are now archived."
            )
        except Exception as e:
            db.rollback()  # Decoupled rollback managed at caller level
            print(f"Error promoting model: {e}")
    else:
        print("Promotion cancelled.")

    db.close()


if __name__ == "__main__":
    main()
