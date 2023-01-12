#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include "main.h"

#define TRUE 1
#define FALSE 0
#define THRESHOLD 9.f
#define FACTOR 2.5f
#define INP_LEN 1250

int compare (const void * a, const void * b)
{
  return ( *(float*)a - *(float*)b );
}

int aiRun(const float *input, float *result) {
    // Count std
    int i;
    float mean_val = input[0];
    float std_val = input[0] * input[0];

    for (i=1; i<INP_LEN; i++) {
        mean_val += input[i];
        std_val += input[i] * input[i];
    }
    mean_val /= INP_LEN;
    std_val = sqrt(std_val/INP_LEN - mean_val*mean_val);

    // Count numPeaks (countPeaks)
    int detect = TRUE;
    int delay_steps = 0;
    int peak_cnt = 0;
    int peak_sampled[65];

    for (i=0; i<INP_LEN; i++){
        float std_threshold = std_val * FACTOR;
        if (input[i] > std_threshold && detect){
            peak_sampled[peak_cnt] = i;
            peak_cnt += 1;
            delay_steps = 0;
            detect = FALSE;
        }

        if (!detect) {
            delay_steps += 1;
            if (delay_steps > 20) {
                detect = TRUE;
            }
        }
    }

    if (peak_cnt < 3) {
        result[0] = 1;
				result[1] = 0; 
        return 0;
    }
    peak_cnt -= 1;

    float peak_diff[peak_cnt];
    for (i=0; i<peak_cnt; i++) {
        peak_diff[i] = peak_sampled[i+1] - peak_sampled[i];
    }

    // reject_outliers
    float d[peak_cnt];
    float peak_diff_tmp[peak_cnt];
    memcpy(peak_diff_tmp, peak_diff, peak_cnt*4);
    qsort (peak_diff_tmp, peak_cnt, sizeof(float), compare);
    float peak_diff_median = (peak_cnt%2) ? peak_diff_tmp[peak_cnt/2] : (peak_diff_tmp[peak_cnt/2]+peak_diff_tmp[peak_cnt/2 - 1]) / 2;
    for (i=0; i<peak_cnt; i++) {
        d[i] = fabsf(peak_diff[i] - peak_diff_median);
    }

    float d_tmp[peak_cnt];
    memcpy(d_tmp, d, peak_cnt*4);
    qsort (d_tmp, peak_cnt, sizeof(float), compare);
    float mdev = (peak_cnt%2) ? d_tmp[peak_cnt/2] : (d_tmp[peak_cnt/2]+d_tmp[peak_cnt/2 - 1]) / 2;
    float s;
    int trunc_peak_diff_count = 0;
    float trunc_peak_diff_sum = 0;
		for (i=0; i<peak_cnt; i++) {
        s = mdev ? d[i]/mdev : 0.f;

        if (s < 2.f) {
            trunc_peak_diff_count += 1;
            trunc_peak_diff_sum += peak_diff[i];
        }
    }

    // Back to Count numPeaks (countPeaks)
    if (trunc_peak_diff_count == 1) {
        result[0] = 1;
				result[1] = 0; 
        return 0;
    }
    float robustAvgInterval = trunc_peak_diff_sum / trunc_peak_diff_count;
    float numPeaks = INP_LEN / robustAvgInterval;

    // Output result
    if (numPeaks <= THRESHOLD) {
        result[0] = 1;
				result[1] = 0; 
        return 0;
    }
    else {
				result[0] = 0; 
        result[1] = 1;
        return 0;
    }
}
