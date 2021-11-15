"""
Title: Main file that reproduces the results of the binary classification.
"""
from utils import data_loader


def main():
    # load data

    X_train_scaled, y_train, X_test_scaled, y_test = data_loader(pca=False)
    X_train_pca, y_train, X_test_pca, y_test = data_loader(pca=True)



if __name__ == "__main__":
    main()
