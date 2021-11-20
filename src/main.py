"""
Title: Main file that reproduces the results of the binary classification.
"""
import numpy as np
from utils import data_loader
from models.train_models import train_model
from models.predict_models import predict_model
from visualization_class_results import plot_roc_curve
from data_preprocessing import data_preprocessing


def main():
    # preprocess data
    data_preprocessing()

    # load data
    X_train_scaled, y_train, X_test_scaled, y_test = data_loader(pca=False)
    X_train_pca, y_train, X_test_pca, y_test = data_loader(pca=True)

    # train Gaussian naive bayes classifier with scaled data
    clf_nb, clf_nb_time = train_model(X_train=X_train_scaled, y_train=y_train, clf_v='nb', cv_m=5, pca=False)
    nb_results, nb_y_pred_proba = predict_model(X_test=X_test_scaled,
                                                y_test=y_test,
                                                time=clf_nb_time,
                                                clf_v='nb',
                                                pca=False)
    print(nb_results)

    # train Gaussian naive bayes classifier with scaled + pca data
    clf_nb_pca, clf_nb_pca_time = train_model(X_train=X_train_pca, y_train=y_train, clf_v='nb', cv_m=5, pca=True)
    nb_results_pca, nb_pca_y_pred_proba = predict_model(X_test=X_test_pca,
                                                        y_test=y_test,
                                                        time=clf_nb_pca_time,
                                                        clf_v='nb',
                                                        pca=True)
    print(nb_results_pca)

    # train support vector machine classifier with scaled data
    clf_svc, clf_svc_time = train_model(X_train=X_train_scaled, y_train=y_train, clf_v='svc', cv_m=5, pca=False)
    svc_results, svc_y_pred_proba = predict_model(X_test=X_test_scaled,
                                                  y_test=y_test,
                                                  time=clf_svc_time,
                                                  clf_v='svc',
                                                  pca=False)
    print(svc_results)

    # train support vector machine classifier with scaled + pca data
    clf_svc_pca, clf_svc_pca_time = train_model(X_train=X_train_pca, y_train=y_train, clf_v='svc', cv_m=5, pca=True)
    svc_results_pca, svc_pca_y_pred_proba = predict_model(X_test=X_test_pca,
                                                          y_test=y_test,
                                                          time=clf_svc_pca_time,
                                                          clf_v='svc',
                                                          pca=True)
    print(svc_results_pca)

    # train decision tree classifier with scaled data
    clf_dt, clf_dt_time = train_model(X_train=X_train_scaled, y_train=y_train, clf_v='dt', cv_m=5, pca=False)
    dt_results, dt_y_pred_proba = predict_model(X_test=X_test_scaled,
                                                y_test=y_test,
                                                time=clf_dt_time,
                                                clf_v='dt',
                                                pca=False)
    print(dt_results)

    # train decision tree classifier with scaled + pca data
    clf_dt_pca, clf_dt_pca_time = train_model(X_train=X_train_pca, y_train=y_train, clf_v='dt', cv_m=5, pca=True)
    dt_results_pca, dt_pca_y_pred_proba = predict_model(X_test=X_test_pca,
                                                        y_test=y_test,
                                                        time=clf_dt_pca_time,
                                                        clf_v='dt',
                                                        pca=True)
    print(dt_results_pca)

    # train random forest classifier with scaled data
    clf_rf, clf_rf_time= train_model(X_train=X_train_scaled, y_train=y_train, clf_v='rf', cv_m=5, pca=False)
    rf_results, rf_y_pred_proba = predict_model(X_test=X_test_scaled,
                                                y_test=y_test,
                                                time=clf_rf_time,
                                                clf_v='rf',
                                                pca=False)
    print(rf_results)

    # train random forest classifier with scaled + pca data
    clf_rf_pca, clf_rf_pca_time = train_model(X_train=X_train_pca, y_train=y_train, clf_v='rf', cv_m=5, pca=True)
    rf_results_pca, rf_pca_y_pred_proba = predict_model(X_test=X_test_pca,
                                                        y_test=y_test,
                                                        time=clf_rf_pca_time,
                                                        clf_v='rf',
                                                        pca=True)
    print(rf_results_pca)

    # plot ROC-Curve
    y_pred_probas = [nb_y_pred_proba, nb_pca_y_pred_proba, svc_y_pred_proba, svc_pca_y_pred_proba, 
      dt_y_pred_proba, dt_pca_y_pred_proba, rf_y_pred_proba, rf_pca_y_pred_proba]
    clfs_names = ['NB + Scaled', 'NB + PCA', 'SVC + Scaled', 'SVC + PCA', 'DT + Scaled', 'DT + PCA', 'RF + Scaled', 'RF + PCA']
    plot_roc_curve(y_test=y_test, y_pred_probas=y_pred_probas, clfs_names=clfs_names)

if __name__ == "__main__":
    main()
