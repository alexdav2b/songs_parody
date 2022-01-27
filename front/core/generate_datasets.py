import opendatasets as od
import csv


def download_dataset(url, name, path):
    od.download(url)


def clean_sentence(sentence):
    return sentence.replace("@", "")


def generate_datasets(filename):
    data = {
        "love": [],
        "hate": [],
        "happiness": [],
        "sadness": [],
    }

    fieldsnames = [
        "text"
    ]

    with open(filename + ".csv", "r") as fi:
        reader = csv.DictReader(fi)
        for row in reader:
            cleaned_data = {
                "text": clean_sentence(row["content"])
            }
            if row["sentiment"] in ["love"]:
                data["love"].append(cleaned_data)
            elif row["sentiment"] in ["hate", "anger"]:
                data["hate"].append(cleaned_data)
            elif row["sentiment"] in ["happiness", "relief", "fun", "surprise", "enthusiasm"]:
                data["happiness"].append(cleaned_data)
            elif row["sentiment"] in ["empty", "sadness", "worry", "boredom"]:
                data["sadness"].append(cleaned_data)

    for category in data:
        print(f"{category} : {len(data[category])}")
        with open(f"{filename}_{category}.csv", "w") as fo:
            writer = csv.DictWriter(fo, fieldsnames)
            writer.writeheader()
            for sentence in data[category]:
                writer.writerow(sentence)


def split_dataset(filename, params):
    if params["train"] + params["test"] != 100:
        raise ValueError("Train + tets % must be 100%")

    test = 0
    train = 0
    valid = 0
    with open(filename + ".csv", "r") as fi:
        with open(f"{filename}_train.csv", "w") as trainfile:
            with open(f"{filename}_test.csv", "w") as testfile:
                with open(f"{filename}_valid.csv", "w") as validfile:
                    for ind, row in enumerate(fi):
                        if ind == 0:
                            validfile.write(row)
                            testfile.write(row)
                            trainfile.write(row)
                        elif ind % 101 < params["test"] / params["valid"] or ind % 101 > params["train"] - params["train"] / params["valid"]:
                            validfile.write(row)
                            valid += 1
                        elif ind % 101 < params["test"]:
                            testfile.write(row)
                            test += 1
                        else:
                            trainfile.write(row)
                            train +=1

    print(f"train: {train} - {train / (train + test + valid)}%")
    print(f"test: {test} - {test / (train + test + valid)}%")
    print(f"valid: {valid} - {valid / (train + test + valid)}%")


def main(download=False, generate=False, split=False):
    if download:
        download_dataset("https://www.kaggle.com/pashupatigupta/emotion-detection-from-text", "raw_dataset", "")

    if generate:
        generate_datasets('emotion-detection-from-text/tweet_emotions')

    if split:
        params = {
            "train": 80,
            "test": 20,
            "valid": 10
        }

        split_dataset('emotion-detection-from-text/tweet_emotions_love', params)
        split_dataset('emotion-detection-from-text/tweet_emotions_hate', params)
        split_dataset('emotion-detection-from-text/tweet_emotions_happiness', params)
        split_dataset('emotion-detection-from-text/tweet_emotions_sadness', params)


if __name__ == "__main__":
    main(generate=True, split=True)
