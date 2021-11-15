"""
Title: Main file that reproduces the results of the binary classification.
"""
from utils import data_loader
from models.train_nb_svc_models import train_model
from models.predict_models import predict_model


def main():
    # load data

    X_train_scaled, y_train, X_test_scaled, y_test = data_loader(pca=False)
    X_train_pca, y_train, X_test_pca, y_test = data_loader(pca=True)

    # train Gaussian naive bayes classifier with scaled data
    clf_nb = train_model(X_train=X_train_scaled, y_train=y_train, clf_v='nb', cv_m=5, pca=False)
    nb_results = predict_model(X_test=X_test_scaled, y_test=y_test, clf_v='nb', pca=False)
    print(nb_results)

    # train Gaussian naive bayes classifier with scaled + pca data
    clf_nb_pca = train_model(X_train=X_train_pca, y_train=y_train, clf_v='nb', cv_m=5, pca=True)
    nb_results_pca = predict_model(X_test=X_test_pca, y_test=y_test, clf_v='nb', pca=True)
    print(nb_results_pca)

    # train support vector machine classifier with scaled data
    clf_svc = train_model(X_train=X_train_scaled, y_train=y_train, clf_v='svc', cv_m=5, pca=False)
    svc_results = predict_model(X_test=X_test_scaled, y_test=y_test, clf_v='svc', pca=False)
    print(svc_results)

    # train support vector machine classifier with scaled + pca data
    clf_svc_pca = train_model(X_train=X_train_pca, y_train=y_train, clf_v='svc', cv_m=5, pca=True)
    svc_results_pca = predict_model(X_test=X_test_pca, y_test=y_test, clf_v='svc', pca=True)
    print(svc_results_pca)


if __name__ == "__main__":
    main()
