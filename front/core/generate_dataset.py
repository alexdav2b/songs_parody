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
        "id", "sentence"
    ]

    with open(filename + ".csv", "r") as fi:
        reader = csv.DictReader(fi)
        for row in reader:
            cleaned_data = {
                "id": row["tweet_id"],
                "sentence": clean_sentence(row["content"])
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


def main():
    download_dataset("https://www.kaggle.com/pashupatigupta/emotion-detection-from-text", "raw_dataset", "")

    generate_datasets('emotion-detection-from-text/tweet_emotions')


if __name__ == "__main__":
    main()
