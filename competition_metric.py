import typer
from metric import competition


def main(unlearn_path: str, retrain_path: str):
    score = competition(
        model="ResNet",
        unlearn_path=unlearn_path,
        retrain_path=retrain_path,
        retain="/content/NeurlsIPSUnlearning/data/validation_0/retain",
        val="/content/NeurlsIPSUnlearning/data/val",
        forget="/content/NeurlsIPSUnlearning/data/validation_0/forget",
        test="/content/NeurlsIPSUnlearning/data/test",
        device="cuda:0"
    )
    print(f"Final score: {score: .4f}")


if __name__ == "__main__":
    typer.run(main)
