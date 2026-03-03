#include "yin.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float clampf(float x, float a, float b)
{
    return (x < a) ? a : (x > b) ? b
                                 : x;
}

// ---------- Step 2: difference function d(tau) ----------
static void difference_function(
    const float *x, // frame pointer (must have W + tau_max samples available)
    int W,
    int tau_max,
    float *d_out // [0..tau_max]
)
{
    d_out[0] = 0.0f;
    for (int tau = 1; tau <= tau_max; ++tau)
    {
        float sum = 0.0f;
        const float *a = x;
        const float *b = x + tau;
        for (int j = 0; j < W; ++j)
        {
            float diff = a[j] - b[j];
            sum += diff * diff;
        }
        d_out[tau] = sum;
    }
}

// ---------- Step 3: CMNDF d'(tau) ----------
static void cmndf(const float *d, int tau_max, float *dprime_out)
{
    dprime_out[0] = 1.0f;

    float cumsum = 0.0f;
    const float eps = 1e-12f;

    for (int tau = 1; tau <= tau_max; ++tau)
    {
        cumsum += d[tau];
        float mean = cumsum / (float)tau;
        if (mean > eps)
        {
            dprime_out[tau] = d[tau] / mean;
        }
        else
        {
            dprime_out[tau] = 1.0f; // silence / degenerate
        }
    }
}

// ---------- Step 4: absolute threshold + local min ----------
static int absolute_threshold(
    const float *dprime,
    int tau_min,
    int tau_max,
    float thresh)
{
    int tau = -1;

    for (int t = tau_min; t <= tau_max; ++t)
    {
        if (dprime[t] < thresh)
        {
            tau = t;
            break;
        }
    }

    if (tau < 0)
    {
        // fallback: global min in range
        int best = tau_min;
        float bestv = dprime[tau_min];
        for (int t = tau_min + 1; t <= tau_max; ++t)
        {
            if (dprime[t] < bestv)
            {
                bestv = dprime[t];
                best = t;
            }
        }
        return best;
    }

    // move to local minimum (while decreasing)
    while (tau + 1 <= tau_max && dprime[tau + 1] < dprime[tau])
    {
        tau++;
    }

    return tau;
}

// ---------- Step 5: parabolic interpolation ----------
static float parabolic_interpolation(const float *y, int n, int i)
{
    if (i <= 0 || i >= n - 1)
        return (float)i;

    float y0 = y[i - 1];
    float y1 = y[i];
    float y2 = y[i + 1];

    float denom = (y0 - 2.0f * y1 + y2);
    if (fabsf(denom) < 1e-20f)
        return (float)i;

    float delta = 0.5f * (y0 - y2) / denom;
    // delta typically in [-0.5, 0.5] for well-shaped minima
    return (float)i + delta;
}

static int yin_one_frame(
    YinContext *ctx,
    const float *signal,
    int length,
    float sr,
    int t,
    int W,
    int tau_min,
    int tau_max,
    float thresh,
    YinPitch *out)
{
    // Need samples: t .. t + W + tau_max - 1
    int needed = t + W + tau_max;
    if (t < 0 || needed > length)
        return 0;

    const float *x = signal + t;

    difference_function(x, W, tau_max, ctx->d);
    cmndf(ctx->d, tau_max, ctx->dprime);

    int tau_int = absolute_threshold(ctx->dprime, tau_min, tau_max, thresh);
    float tau_hat = parabolic_interpolation(ctx->dprime, tau_max + 1, tau_int);

    float ap = ctx->dprime[tau_int];
    float conf = 1.0f - ap;
    conf = clampf(conf, 0.0f, 1.0f);

    out->tau = tau_hat;
    out->f0 = (tau_hat > 0.0f) ? (sr / tau_hat) : 0.0f;
    out->aperiodicity = ap;
    out->confidence = conf;
    out->t_used = t;
    return 1;
}

// ---------- Public API ----------
YinContext *yin_create(int max_tau)
{
    if (max_tau < 1)
        return NULL;

    YinContext *ctx = (YinContext *)calloc(1, sizeof(YinContext));
    if (!ctx)
        return NULL;

    ctx->max_tau = max_tau;
    ctx->d = (float *)malloc((size_t)(max_tau + 1) * sizeof(float));
    ctx->dprime = (float *)malloc((size_t)(max_tau + 1) * sizeof(float));

    if (!ctx->d || !ctx->dprime)
    {
        yin_destroy(ctx);
        return NULL;
    }
    return ctx;
}

void yin_destroy(YinContext *ctx)
{
    if (!ctx)
        return;
    free(ctx->d);
    free(ctx->dprime);
    free(ctx);
}

int yin_detect_pitch(
    YinContext *ctx,
    const float *signal,
    int length,
    float sr,
    int t,
    int W,
    int tau_min,
    int tau_max,
    float thresh,
    int local_radius,
    int local_step,
    int refine_radius,
    YinPitch *out)
{
    if (!ctx || !signal || !out)
        return 0;
    if (sr <= 0.0f || length <= 0)
        return 0;
    if (W <= 0)
        return 0;
    if (tau_min < 1)
        tau_min = 1;
    if (tau_max < tau_min)
        return 0;
    if (tau_max > ctx->max_tau)
        return 0;
    if (local_step <= 0)
        local_step = 1;
    if (refine_radius < 0)
        refine_radius = 0;

    // ----- Step 6: best local estimate -----
    if (local_radius > 0)
    {
        YinPitch best;
        int have_best = 0;

        int start = t - local_radius;
        int end = t + local_radius;

        for (int u = start; u <= end; u += local_step)
        {
            YinPitch tmp;
            if (!yin_one_frame(ctx, signal, length, sr, u, W, tau_min, tau_max, thresh, &tmp))
                continue;

            if (!have_best || tmp.aperiodicity < best.aperiodicity)
            {
                best = tmp;
                have_best = 1;
            }
        }

        if (!have_best)
            return 0;

        // Refine around best tau (second pass with tighter tau bounds)
        int center = (int)lroundf(best.tau);
        int tau_min2 = center - refine_radius;
        int tau_max2 = center + refine_radius;

        if (tau_min2 < tau_min)
            tau_min2 = tau_min;
        if (tau_max2 > tau_max)
            tau_max2 = tau_max;
        if (tau_max2 < tau_min2)
        {
            *out = best;
            return 1;
        }

        YinPitch refined;
        if (!yin_one_frame(ctx, signal, length, sr, best.t_used, W, tau_min2, tau_max2, thresh, &refined))
        {
            *out = best;
            return 1;
        }

        *out = refined;
        return 1;
    }

    // Step 6 disabled: single frame
    return yin_one_frame(ctx, signal, length, sr, t, W, tau_min, tau_max, thresh, out);
}