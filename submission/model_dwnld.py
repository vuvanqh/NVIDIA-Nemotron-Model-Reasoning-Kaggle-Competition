import kagglehub


MODEL_HANDLE = "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"


def main() -> None:
    path = kagglehub.model_download(MODEL_HANDLE)
    print(path)


if __name__ == "__main__":
    main()
