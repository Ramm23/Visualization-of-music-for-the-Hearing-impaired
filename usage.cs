using System.Runtime.InteropServices;

[StructLayout(LayoutKind.Sequential)]
public struct YinPitch
{
    public float f0, tau, confidence, aperiodicity;
    public int t_used;
}

public static class YinNative
{
    [DllImport("yin_plugin")]
    public static extern System.IntPtr yin_create(int max_tau);

    [DllImport("yin_plugin")]
    public static extern void yin_destroy(System.IntPtr ctx);

    [DllImport("yin_plugin")]
    public static extern int yin_detect_pitch(
        System.IntPtr ctx,
        float[] signal,
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
        out YinPitch result
    );
}