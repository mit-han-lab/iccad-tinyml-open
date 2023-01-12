#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include "main.h"

#define TRUE 1
#define FALSE 0
#define THRESHOLD 10

#ifndef ELEM_SWAP(a,b)
#define ELEM_SWAP(a,b) { register int t=(a);(a)=(b);(b)=t; }
#endif

int quick_select_median(int arr[], uint16_t n)
{
    uint16_t low, high ;
    uint16_t median;
    uint16_t middle, ll, hh;
    low = 0 ; high = n-1 ; median = (low + high) / 2;
    for (;;) {
    if (high <= low) /* One element only */
    return arr[median] ;
    if (high == low + 1) { /* Two elements only */
    if (arr[low] > arr[high])
    ELEM_SWAP(arr[low], arr[high]) ;
    return arr[median] ;
    }
    /* Find median of low, middle and high items; swap into position low */
    middle = (low + high) / 2;
    if (arr[middle] > arr[high])
    ELEM_SWAP(arr[middle], arr[high]) ;
    if (arr[low] > arr[high])
    ELEM_SWAP(arr[low], arr[high]) ;
    if (arr[middle] > arr[low])
    ELEM_SWAP(arr[middle], arr[low]) ;
    /* Swap low item (now in position middle) into position (low+1) */
    ELEM_SWAP(arr[middle], arr[low+1]) ;
    /* Nibble from each end towards middle, swapping items when stuck */
    ll = low + 1;
    hh = high;
    for (;;) {
    do ll++; while (arr[low] > arr[ll]) ;
    do hh--; while (arr[hh] > arr[low]) ;
    if (hh < ll)
    break;
    ELEM_SWAP(arr[ll], arr[hh]) ;
    }
    /* Swap middle item (in position low) back into correct position */
    ELEM_SWAP(arr[low], arr[hh]) ;
    /* Re-set active partition */
    if (hh <= median)
    low = ll;
    if (hh >= median)
    high = hh - 1;
    }

    return arr[median] ;
}

void aiRun(float *input, float *result) {
	// Count std
	int i;
	float mean_val = input[0];
	float std_val = pow(input[0], 2);

	for (i=1; i<1250; i++) {
		mean_val += input[i];
		std_val += pow(input[i], 2);
	}
	mean_val /= 1250;
	std_val = sqrt(std_val/1250 - pow(mean_val, 2));

	// Count numPeaks (countPeaks)
    int flag = TRUE;
    int delay_steps = 0;
    int peak_cnt = 0;
    int peak_sampled[65];
    int sampled_count = 0;

    for (i=0; i<1250; i++){
    	float std_threshold = std_val * 3;
    	if (input[i] > std_threshold && flag){
    		peak_sampled[sampled_count] = i;
            peak_cnt += 1;
            delay_steps = 0;
            flag = FALSE;
            sampled_count += 1;
    	}

    	if (!flag) {
    		delay_steps += 1;
            if (delay_steps > 20) {
                flag = TRUE;
            }
    	}
    }

    if (sampled_count < 2) {
        result[0] = 1;
    }

    int peak_diff[64];
    for (i=0; i<sampled_count-1; i++) {
    	peak_diff[i] = peak_sampled[i+1] - peak_sampled[i];
    }

    // reject_outliers
    int d[64];
    int peak_diff_median = quick_select_median(peak_diff, sampled_count-1);
    for (i=0; i<sampled_count-1; i++) {
    	d[i] = abs(peak_diff[i] - peak_diff_median);
    }

    int mdev = quick_select_median(d, sampled_count-1);
    float s[64];
    int trunc_peak_diff[64];
    int trunc_peak_diff_count = 0;
    int trunc_peak_diff_sum = 0;
    for (i=0; i<sampled_count-1; i++) {
    	if (mdev != 0) {
    		s[i] = d[i] / mdev;
    	}
    	else {
    		s[i] = 0.f;
    	}

    	if (s[i] < 2.f) {
    		trunc_peak_diff[i] = peak_diff[i];
    		trunc_peak_diff_count += 1;
    		trunc_peak_diff_sum += peak_diff[i];
    	}
    	else {
    		trunc_peak_diff[i] = 0;
    	}
    }

    // Back to Count numPeaks (countPeaks)
    if (trunc_peak_diff_count == 1) {
				result[0] = 1;
    }
    float robustAvgInterval = trunc_peak_diff_sum / trunc_peak_diff_count;
	int numPeaks = (int)round(1250 / robustAvgInterval);

    // Output result
	if (numPeaks <= THRESHOLD) {
      result[0] = 1;
	}
	else {
      result[1] = 1;
	}
}
