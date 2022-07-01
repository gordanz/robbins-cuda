#include <inttypes.h>
#include <limits.h>
#define _USE_MATH_DEFINES
#define REAL double
#define SECINT int
#define SEC_MAX INT_MAX

__global__
void simulate(int nsim, int n,
    uint64_t *sd, REAL *x, SECINT *left, SECINT *right, SECINT *nlch,
    double *res)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t t;
    const REAL dx = (1. / (UINT64_C(1) << 53));
    uint64_t x_int;
    uint64_t s[4];
    REAL oonsim = 1.0/nsim;
    REAL x_unif;
    REAL runningValue=0;
    SECINT ROOT = index * n ;
    SECINT NOWHERE = SEC_MAX;
    SECINT now;
    SECINT rank;


    s[0]=sd[4*index]; s[1]=sd[4*index+ 1];
    s[2]=sd[4*index+ 2]; s[3]=sd[4*index+ 3];

    for (int sim = 0; sim < nsim ; sim++)
    {
        double thisRunValue = 0;
        SECINT togo = 0;

        for (int i = 0; i < n; i++ )
        {

            // The RNG
            t = s[1] << 17; s[2] ^= s[0]; s[3] ^= s[1];
            s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t;
            s[3] = (s[3] << 45) | (s[3] >> 19);
            x_int = s[0]+s[3]; x_unif = ( x_int  >> 11) * dx;

            // The tree
            rank = 1;
            if (0 == i) { // Empty
                x[ROOT]     = x_unif;
                left[ROOT]  = NOWHERE;
                right[ROOT] = NOWHERE;
                nlch[ROOT]  = 0;
            }
            else { // Ready to be filled
                x[ROOT+i]     = x_unif;
                left[ROOT+i]  = NOWHERE;
                right[ROOT+i] = NOWHERE;
                nlch[ROOT+i]  = 0;

                now = ROOT; //start search at root
                while (1) {
                    if (x_unif > x[now]) { // need to go down the right leg
                        rank += nlch[now] + 1;
                        if (right[now] == NOWHERE) { // found a spot
                            right[now] = ROOT+i;
                            break;
                        }
                        else { // keep going right
                            now = right[now];
                        }
                    }
                    else { // need to go down the left leg
                        nlch[now]++;
                        if (left[now] == NOWHERE) { // found a spot
                            left[now] = ROOT + i;
                            break;
                        }
                        else { // keep going left
                            now = left[now];
                        }
                    }
                }
            }
            // condition for stopping
            if ((x_unif < 2.0/(n-i)) && (i > n/10.0)) {
               togo = (n-i);
               break;
            }
        } // end n

        // expected (final) rank
        thisRunValue = 1.0*rank + x_unif*togo;

        // expected score
        // thisRunValue = x_unif;

        runningValue += thisRunValue;
    } // end nsim

    res[index] = runningValue * oonsim;

}
