/*
    llda.c
    Labeled-Latent Dirichlet Allocation, main driver.
*/
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "llda.h"
#include "learn.h"
#include "writer.h"
#include "feature.h"
#include "dmatrix.h"
#include "imatrix.h"
#include "util.h"

void calculate(int maxiter, double alpha, double beta, char *data_path, char *label_path, char *output_path){
    document *data;
    FILE *pp, *tp, *n_mzp, *n_wzp, *likp; // for phi, theta, n_mz, n_zw, likelihood
    char c;
    int nlex, dlenmax;
    int ndoc;
    int nclass = -1;
    double **phi;
    double **theta;
    int **n_mz;
    int **n_zw;

    if(( maxiter == 0 )||( alpha == 0 )||( beta == 0 ))
        usage();

    //// open data
    if((data = feature_matrix(data_path, &nlex, &dlenmax, &ndoc)) == NULL){
        fprintf(stderr, "llda:: cannot open training data.\n");
        exit(1);
    }

    // open label data
    if((assign_labels(label_path, data, &nclass) != 1) || (nclass == -1)){
        fprintf(stderr, "llda:: cannot open label data.\n");
        exit(1);
    }

    // open model output
    if(((pp = fopen(strconcat(output_path, ".phi"),"w")) == NULL)
    || ((tp = fopen(strconcat(output_path, ".theta"),"w")) == NULL)
    || ((n_mzp = fopen(strconcat(output_path, ".n_mz"),"w")) == NULL)
    || ((n_wzp = fopen(strconcat(output_path, ".n_wz"),"w")) == NULL)
    || ((likp = fopen(strconcat(output_path, ".lik"),"w")) == NULL)){
        fprintf(stderr, "llda:: cannot open model outputs.\n");
        exit(1);
    }

    // allocate parameters
    if((phi = dmatrix(nlex, nclass)) == NULL){
        fprintf(stderr, "llda:: cannot allocate phi.\n");
        exit(1);
    }
    if((theta = dmatrix(ndoc, nclass)) == NULL){
        fprintf(stderr, "llda:: cannot allocate theta.\n");
        exit(1);
    }
    // n_mz ... number of times document and topic z co-occur
    if((n_mz = imatrix(ndoc, nclass)) == NULL){
        fprintf(stderr, "llda:: cannot allocate n_mz.\n");
        exit(1);
    }
    // n_zw ... number of times topic and word w co-occur
    if((n_zw = imatrix(nclass, nlex)) == NULL){
        fprintf(stderr, "llda:: cannot allocate n_zw.\n");
        exit(1);
    }

    llda_learn(data, alpha, beta, nclass, nlex, dlenmax, maxiter, phi, theta, n_mz, n_zw, likp);
    llda_write(pp, tp, n_mzp, n_wzp, phi, theta, n_mz, n_zw, nclass, nlex, ndoc);

    free_feature_matrix(data);
    free_dmatrix(phi, nlex);
    free_dmatrix(theta, ndoc);
    free_imatrix(n_mz, ndoc);
    free_imatrix(n_zw, nclass);

    fclose(pp);
    fclose(tp);
    fclose(n_mzp);
    fclose(n_wzp);
}

void usage(void){
    printf("usage: %s [-I maxiter] [-A alpha] [-B beta] train label model\n", "llda");
}
