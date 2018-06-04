/*
    ldapf.c
    Latent Dirichlet Allocation with particle filter estimation, main driver.
*/
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "ldapf.h"
#include "learn.h"
#include "writer.h"
#include "feature.h"
#include "dmatrix.h"
#include "imatrix.h"
#include "util.h"

void calculate(int nparticle, int ess, int rejuvenation, double alpha, double beta, char *data_path, char *model_prefix){
    document *data;
    FILE *tp;
    char c;
    int nlex, dlenmax;
    int prelex, preclass;
    int ndoc, nclass;
    double **theta;
    int **n_zw;
    int i,j;
    int *tmp;

    if(nparticle==0){
        nparticle = NPARTICLE_DEFAULT;
    }else if(ess==0){
        ess = ESS_DEFAULT;
    }else if(rejuvenation==0){
        rejuvenation = REJUVENATION_DEFAULT;
    }else if(alpha==0){
        alpha = ALPHA_DEFAULT;
    }else if(beta==0){
        beta = BETA_DEFAULT;
    }

    if(nparticle > 300){
        fprintf(stderr,"ldapf:: Too many particles. The number of particles should be fewer than 300.\n");
        exit(1);
    }

    // open data
    if((data = feature_matrix(data_path, &nlex, &dlenmax, &ndoc)) == NULL){
        fprintf(stderr, "ldapf:: cannot open training data.\n");
        exit(1);
    }
    // open output
    if((tp = fopen(strconcat(data_path, ".theta"),"w")) == NULL){
        fprintf(stderr, "ldapf:: cannot open output.\n");
        exit(1);
    }
    // load model
    if((n_zw = load_n_wz(strconcat(model_prefix,".n_wz"), &prelex, &preclass)) == NULL){
        fprintf(stderr, "ldapf:: cannot load model.\n");
        exit(1);
    }
    nclass = preclass;
    // realloc n_zw to allocate words, occurred only in a new document, to topics.
    if(nlex > prelex){
        for(i = 0;i < nclass;i++){
            tmp = (int *)realloc(n_zw[i], sizeof(int)*nlex);
            if(tmp == NULL){
                fprintf(stderr, "ldapf:: cannot re-allocate n_zw.\n");
                exit(1);
            }else{
                n_zw[i] = tmp;
                for(j = prelex;j < nlex;j++)
                    n_zw[i][j] = 0;
            }
        }
    }else{
        nlex = prelex;
    }

    // allocate parameters
    if((theta = dmatrix(ndoc, nclass)) == NULL){
        fprintf(stderr, "ldapf:: cannot allocate theta.\n");
        exit(1);
    }

    ldapf_learn(data, alpha, beta, ndoc, nclass, nlex, dlenmax, nparticle, ess, rejuvenation, n_zw, theta);
    lda_write(tp, theta, nclass, ndoc);

    free_feature_matrix(data);
    free_dmatrix(theta, ndoc);
    for(i = 0;i < nclass;i++)
        free(n_zw[i]);
    free(n_zw);
    fclose(tp);

}

void usage(void){
    printf("usage: %s [-P particles] [-E effective sample size] [-R rejuvenation steps] [-A alpha] [-B beta]  test model\n", "ldapf");
    exit(0);
}
