#include<stdio.h>

typedef union {
    int i;
    float f;
}INTORFLOAT;

INTORFLOAT n;
INTORFLOAT bias;

int main(void){

    bias.i = (23 + 127) << 23;

    printf("%x\n", bias);

    n.f = 8.25f;

    printf("%x\n", n);

    n.f += bias.f;
    printf("%x\n", n);
    n.i -= bias.f;
    printf("%x\n", n);




}

