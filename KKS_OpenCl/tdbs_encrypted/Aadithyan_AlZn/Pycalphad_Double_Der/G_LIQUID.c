/******************************************************************************
 *                      Code generated with sympy 1.5.1                       *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                       This file is part of 'project'                       *
 ******************************************************************************/
#include "G_LIQUID.h"
#include <math.h>

void G_LIQUID(double T, double *y, double *dG) {

   (*dG) = 8.3145000000000007*T*(1.0*((y[0] > 1.0000000000000001e-16) ? (
      log(y[0])*y[0]
   )
   : (
      0
   ))/(1.0*y[0] + 1.0*y[1]) + 1.0*((y[1] > 1.0000000000000001e-16) ? (
      log(y[1])*y[1]
   )
   : (
      0
   ))/(1.0*y[0] + 1.0*y[1])) + (10465.5 - 3.3925900000000002*T)*y[0]*y[1]/(1.0*y[0] + 1.0*y[1]) + (((T >= 298.13999999999999 && T < 692.70000000000005) ? (
      -3.5896e-19*pow(T, 7) - 10.29299*T + ((T >= 298.0 && T < 692.70000000000005) ? (
         -1.264963e-6*pow(T, 3) - 0.0017120340000000001*pow(T, 2) - 23.701309999999999*T*log(T) + 118.4693*T - 7285.7870000000003
      )
      : ((T >= 692.70000000000005 && T < 1700.0) ? (
         -31.379999999999999*T*log(T) + 172.3449*T - 11070.6 + 4.7065699999999997e+26/pow(T, 9)
      )
      : (
         0
      ))) + 7157.2129999999997
   )
   : ((T >= 692.70000000000005 && T < 1700.0) ? (
      -10.737066*T + ((T >= 298.0 && T < 692.70000000000005) ? (
         -1.264963e-6*pow(T, 3) - 0.0017120340000000001*pow(T, 2) - 23.701309999999999*T*log(T) + 118.4693*T - 7285.7870000000003
      )
      : ((T >= 692.70000000000005 && T < 1700.0) ? (
         -31.379999999999999*T*log(T) + 172.3449*T - 11070.6 + 4.7065699999999997e+26/pow(T, 9)
      )
      : (
         0
      ))) + 7450.1679999999997 - 4.7050999999999997e+26/pow(T, 9)
   )
   : (
      0
   )))*y[1] + ((T >= 298.0 && T < 700.0) ? (
      7.9401000000000005e-20*pow(T, 7) - 8.7766400000000004e-7*pow(T, 3) - 0.0018846620000000001*pow(T, 2) - 24.3672*T*log(T) + 125.2307*T + 3029.4029999999998 + 74092/T
   )
   : ((T >= 700.0 && T < 933.60000000000002) ? (
      7.9401000000000005e-20*pow(T, 7) - 5.7642270000000003e-6*pow(T, 3) + 0.01853198*pow(T, 2) - 38.584429999999998*T*log(T) + 211.18610000000001*T - 270.68599999999998 + 74092/T
   )
   : ((T >= 933.60000000000002 && T < 2900.0) ? (
      -31.748190000000001*T*log(T) + 177.41*T - 795.70899999999995
   )
   : (
      0
   ))))*y[0])/(1.0*y[0] + 1.0*y[1]);

}