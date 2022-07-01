#pragma once
#include "stdio.h"
#include "math.h"
#include "time.h"
#include "inttypes.h"

#define _USE_MATH_DEFINES
#define MICRO "u"
#define PREFIX_START (-24)
/* Smallest power of then for which there is a prefix defined.
   If the set of prefixes will be extended, change this constant
   and update the table "prefix". */

char *eng(double value, int digits, int numeric)
{
  const char *prefix[] = {
  "y", "z", "a", "f", "p", "n", MICRO, "m", "",
  "k", "M", "G", "T", "P", "E", "Z", "Y"
  };
  static char zero[] = "0.0";

#define PREFIX_END (PREFIX_START+\
(int)((sizeof(prefix)/sizeof(char *)-1)*3))

      int expof10;
      static char result[100];
      char *res = result;

      if (value < 0.)
        {
            *res++ = '-';
            value = -value;
        }
      if (value == 0.)
        {
        return zero;
        }

      expof10 = (int) log10(value);
      if(expof10 > 0)
        expof10 = (expof10/3)*3;
      else
        expof10 = (-expof10+3)/3*(-3);

      value *= pow(10,-expof10);

      if (value >= 1000.)
         { value /= 1000.0; expof10 += 3; }
      else if(value >= 100.0)
         digits -= 2;
      else if(value >= 10.0)
         digits -= 1;

      if(numeric || (expof10 < PREFIX_START) ||
                    (expof10 > PREFIX_END))
        sprintf(res, "%.*fe%d", digits-1, value, expof10);
      else
        sprintf(res, "%.*f %s", digits-1, value,
          prefix[(expof10-PREFIX_START)/3]);
      return result;
}

void timer()
{
    static int firstRun = 1;
    static clock_t beginTime;
    double timeSpent;
    if (firstRun) {
        beginTime = clock();
        firstRun = 0;
    }
    else {
        timeSpent = (double)(clock() - beginTime) / CLOCKS_PER_SEC;
        printf("Time spent : %ss\n", eng(timeSpent,3,0));
        beginTime = clock();
    }
}


void dhline()
{
    printf("==========================================================\n");
}

void hline()
{
    printf("----------------------------------------------------------\n");
}

void nhline()
{
    printf("\n----------------------------------------------------------\n");
}

void printBinary(uint64_t m, int k)
{
    if (k > 64) {
        printf("Error: k must be <= 64\n");
        exit(1);
    }
    int b[64];
    for (int i = 0; i < k; i++)
    {
        b[k - 1 - i] = m & 1;
        m = m >> 1;
    }
    for (int i = 0; i < k; i++)
    {
        printf("%d", b[i]);
    }
    printf("\n");
}
void print64(uint64_t m)
{
    printf("%" PRIu64 "", m);
}

void printi(int m)
{
    printf("%d", m);
}

void printd(double x)
{
    printf("%f", x);
}

void ln()
{
    printf("\n");
}

void mln(int n)
{
    for (int i = 0; i < n; ++i)
    {
        ln();
    }
}

void statistics(double *val, int valSize)
{
    double Sm = 0, Ss = 0;
    double mean, stdev, se;

    for (int i = 0; i < valSize; i++)
    {
        Sm += val[i];
    };
    mean = Sm / valSize;

    for (int i = 0; i < valSize; i++)
    {
        Ss += (val[i] - mean)*(val[i] - mean);
    };
    stdev = sqrt(Ss / (valSize - 1));
    se = stdev/sqrt(valSize);
    printf("\n nval = %d,  val =  %.5f, se = (%.6f)\n", valSize, mean, se);
}

void statistics_compare(double *val, int valSize, double trueval)
{
    double Sm = 0, Ss = 0;
    double mean, stdev, se;

    for (int i = 0; i < valSize; i++)
    {
        Sm += val[i];
    };
    mean = Sm / valSize;

    for (int i = 0; i < valSize; i++)
    {
        Ss += (val[i] - mean)*(val[i] - mean);
    };
    stdev = sqrt(Ss / (valSize - 1));
    se = stdev/sqrt(valSize);
    printf(" nval = %d", valSize);
    printf("\n  val =  %.5f, trueval = %f, se = (%.6f)\n  err/se = %f, err (rel) = %f%%\n",
     mean, trueval, se, (mean - trueval) / se, 100 * (mean - trueval) /
     trueval);
}


