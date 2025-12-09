from transformers import pipeline

MODEL_NAME = "dslim/distilbert-NER"


def main() -> None:
    classifier = pipeline(
        task="token-classification",
        model=MODEL_NAME,
        framework="pt",
    )

    text = "Tim Cook presented the new iPhone in Las Vegas on Tuesday."
    result = classifier(text)  # pipeline returns a list of dictionaries

    for token in result:
        print(f"Token: {token['word']}")
        print(f"BIO tag: {token['entity']}")
        print(f"Softmax probability:{token['score']:.6f}")
        print()


if __name__ == "__main__":
    main()
